use std::{ptr, time::Instant};

use half::f16;
use itertools::Itertools;
use luminal::{
    dtype::DType,
    graph::{BucketLLIR, DimBucket, LLIRGraph, SelectedSchedule},
    hlir::{Input, Output, ReferenceData},
    op::{ExecutionStats, Runtime, RuntimeStats, TimingMethod},
    prelude::{
        DynMap, FxHashMap, NodeIndex, Symbol, ToId,
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
    },
};
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, ClMem},
    types::CL_BLOCKING,
};

use crate::{
    device::{OpenClDevice, OpenClDeviceInfo, enumerate_devices},
    kernel::{OpenClKernelOp, clear_dyn_dims_order, dyn_dims_order},
};

struct DeviceBuffer {
    buffer: Buffer<u8>,
    logical_bytes: usize,
}

struct ExecutionStep {
    node: NodeIndex,
    input_nodes: Vec<NodeIndex>,
    output_dtype: DType,
    kernels: Vec<Kernel>,
}

struct CompiledBucket {
    bucket_indices: DynMap,
    graph: LLIRGraph,
    llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    node_dtypes: FxHashMap<NodeIndex, DType>,
    output_data: FxHashMap<NodeIndex, NodeIndex>,
    steps: Vec<ExecutionStep>,
    buffers: FxHashMap<NodeIndex, DeviceBuffer>,
}

/// OpenCL runtime optimized for the native Qualcomm Adreno driver exposed by
/// Snapdragon X Elite systems.
pub struct OpenClRuntime {
    context: Context,
    queue: CommandQueue,
    device_info: OpenClDeviceInfo,
    input_data: FxHashMap<NodeIndex, ReferenceData>,
    input_buffers: FxHashMap<NodeIndex, DeviceBuffer>,
    dyn_buffer: Buffer<i32>,
    dyn_dims: Vec<Symbol>,
    dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    buckets: Vec<CompiledBucket>,
    active_bucket: usize,
    selected_schedule: Option<SelectedSchedule>,
}

impl OpenClRuntime {
    /// Create a runtime for one logical device from [`crate::device::available_devices`].
    pub fn try_initialize(device_index: usize) -> Result<Self, String> {
        let devices = enumerate_devices()?;
        let OpenClDevice { info, device } = devices
            .get(device_index)
            .cloned()
            .ok_or_else(|| format!("OpenCL device index {device_index} does not exist"))?;
        let context = Context::from_device(&device)
            .map_err(|e| format!("failed to create OpenCL context for {}: {e}", info.name))?;
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|e| format!("failed to create OpenCL queue for {}: {e}", info.name))?;
        let dyn_buffer = unsafe {
            Buffer::<i32>::create(&context, CL_MEM_READ_ONLY, 1, ptr::null_mut())
                .map_err(|e| format!("failed to allocate OpenCL dynamic-dim buffer: {e}"))?
        };

        Ok(Self {
            context,
            queue,
            device_info: info,
            input_data: FxHashMap::default(),
            input_buffers: FxHashMap::default(),
            dyn_buffer,
            dyn_dims: Vec::new(),
            dim_buckets: FxHashMap::default(),
            buckets: Vec::new(),
            active_bucket: 0,
            selected_schedule: None,
        })
    }

    pub fn device_info(&self) -> &OpenClDeviceInfo {
        &self.device_info
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl Into<ReferenceData>) {
        let id = id.to_id();
        let data = data.into();
        self.input_data.insert(id, data.clone());
        if let Some(dtype) = self.input_dtype(id) {
            self.upload_input(id, &data, dtype);
        }
    }

    /// Copy an output tensor back into a persistent input without crossing the
    /// host boundary. This is useful for recurrent state such as a KV cache.
    pub fn copy_output_to_input(&mut self, output: impl ToId, input: impl ToId) {
        let output = output.to_id();
        let input = input.to_id();
        let bucket = self
            .buckets
            .get(self.active_bucket)
            .expect("OpenCL runtime has not been compiled");
        let data_node = *bucket
            .output_data
            .get(&output)
            .unwrap_or_else(|| panic!("cannot find OpenCL output {output:?}"));
        let source_input = bucket.llir_to_hlir.get(&data_node).copied();
        if source_input == Some(input) {
            return;
        }

        let mut destination = self
            .input_buffers
            .remove(&input)
            .unwrap_or_else(|| panic!("OpenCL input buffer {input:?} is not set"));
        let source_size = if let Some(hlir_id) = source_input {
            self.input_buffers
                .get(&hlir_id)
                .unwrap_or_else(|| panic!("OpenCL input buffer {hlir_id:?} is not set"))
                .buffer
                .size()
                .expect("failed to query OpenCL output size")
        } else {
            bucket
                .buffers
                .get(&data_node)
                .unwrap_or_else(|| panic!("OpenCL output buffer {data_node:?} is not allocated"))
                .buffer
                .size()
                .expect("failed to query OpenCL output size")
        };
        let destination_size = destination
            .buffer
            .size()
            .expect("failed to query OpenCL input size");
        assert_eq!(
            source_size, destination_size,
            "OpenCL recurrent state size changed"
        );
        if source_size > 0 {
            unsafe {
                if let Some(hlir_id) = source_input {
                    self.queue
                        .enqueue_copy_buffer(
                            &self
                                .input_buffers
                                .get(&hlir_id)
                                .unwrap_or_else(|| {
                                    panic!("OpenCL input buffer {hlir_id:?} is not set")
                                })
                                .buffer,
                            &mut destination.buffer,
                            0,
                            0,
                            source_size,
                            &[],
                        )
                        .expect("failed to copy OpenCL output into input");
                } else {
                    self.queue
                        .enqueue_copy_buffer(
                            &bucket
                                .buffers
                                .get(&data_node)
                                .unwrap_or_else(|| {
                                    panic!("OpenCL output buffer {data_node:?} is not allocated")
                                })
                                .buffer,
                            &mut destination.buffer,
                            0,
                            0,
                            source_size,
                            &[],
                        )
                        .expect("failed to copy OpenCL output into input");
                }
            }
        }
        self.input_buffers.insert(input, destination);
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let (bytes, dtype) = self.read_output(id.to_id());
        match dtype {
            DType::F32 => decode::<f32>(&bytes),
            DType::F16 => decode::<f16>(&bytes)
                .into_iter()
                .map(|value| value.to_f32())
                .collect(),
            DType::Int => decode::<i32>(&bytes)
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            DType::Bool => bytes
                .into_iter()
                .map(|value| if value == 0 { 0.0 } else { 1.0 })
                .collect(),
            unsupported => panic!("cannot read OpenCL {unsupported:?} output as f32"),
        }
    }

    pub fn get_f16(&self, id: impl ToId) -> Vec<f16> {
        let (bytes, dtype) = self.read_output(id.to_id());
        assert_eq!(dtype, DType::F16, "output is not F16");
        decode(&bytes)
    }

    pub fn get_i32(&self, id: impl ToId) -> Vec<i32> {
        let (bytes, dtype) = self.read_output(id.to_id());
        assert_eq!(dtype, DType::Int, "output is not Int");
        decode(&bytes)
    }

    pub fn get_bool(&self, id: impl ToId) -> Vec<bool> {
        let (bytes, dtype) = self.read_output(id.to_id());
        assert_eq!(dtype, DType::Bool, "output is not Bool");
        bytes.into_iter().map(|value| value != 0).collect()
    }

    fn input_dtype(&self, id: NodeIndex) -> Option<DType> {
        self.buckets.iter().find_map(|bucket| {
            bucket.graph.node_indices().find_map(|node| {
                bucket.graph[node]
                    .to_op::<Input>()
                    .and_then(|input| (input.node == id.index()).then_some(input.dtype))
            })
        })
    }

    fn upload_input(&mut self, id: NodeIndex, data: &ReferenceData, dtype: DType) {
        let bytes = reference_bytes(data, dtype);
        let mut buffer = create_buffer(&self.context, bytes.len(), CL_MEM_READ_ONLY);
        if !bytes.is_empty() {
            unsafe {
                self.queue
                    .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &bytes, &[])
                    .unwrap_or_else(|e| panic!("failed to upload OpenCL input {id:?}: {e}"));
            }
        }
        self.input_buffers.insert(
            id,
            DeviceBuffer {
                buffer,
                logical_bytes: bytes.len(),
            },
        );
    }

    fn refresh_inputs(&mut self) {
        let pending: Vec<_> = self
            .input_data
            .iter()
            .filter_map(|(&id, data)| self.input_dtype(id).map(|dtype| (id, data.clone(), dtype)))
            .collect();
        for (id, data, dtype) in pending {
            self.upload_input(id, &data, dtype);
        }
    }

    fn compile_bucket(&self, bucket_indices: DynMap, graph: &LLIRGraph) -> CompiledBucket {
        let graph = graph.clone();
        let mut llir_to_hlir = FxHashMap::default();
        let mut node_dtypes = FxHashMap::default();
        let mut output_data = FxHashMap::default();
        let mut steps = Vec::new();
        let topo = toposort(&graph, None).expect("OpenCL LLIR graph has a cycle");

        for &node in &topo {
            if let Some(input) = graph[node].to_op::<Input>() {
                node_dtypes.insert(node, input.dtype);
                llir_to_hlir.insert(node, NodeIndex::new(input.node));
                continue;
            }
            if graph[node].to_op::<Output>().is_some() {
                continue;
            }

            let op = graph[node]
                .to_dialect::<dyn OpenClKernelOp>()
                .unwrap_or_else(|| panic!("OpenCL runtime found unlowered LLIR node {node:?}"));
            let input_nodes: Vec<_> = graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|edge| edge.id())
                .map(|edge| edge.source())
                .collect();
            let input_dtypes: Vec<_> = input_nodes
                .iter()
                .map(|input| {
                    node_dtypes
                        .get(input)
                        .copied()
                        .unwrap_or_else(|| panic!("missing dtype for OpenCL input node {input:?}"))
                })
                .collect();
            let output_dtype = op.infer_output_dtype(&input_dtypes);
            if !self.device_info.supports_fp16
                && input_dtypes
                    .iter()
                    .chain([&output_dtype])
                    .any(|dtype| *dtype == DType::F16)
            {
                panic!(
                    "OpenCL device {} does not expose cl_khr_fp16",
                    self.device_info.name
                );
            }
            let kernels = op.compile(&self.context, &input_dtypes, output_dtype);
            node_dtypes.insert(node, output_dtype);
            steps.push(ExecutionStep {
                node,
                input_nodes,
                output_dtype,
                kernels,
            });
        }

        for node in topo {
            if let Some(Output {
                node: hlir_node, ..
            }) = graph[node].to_op::<Output>()
                && let Some(data_node) = graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|edge| edge.id())
                    .next()
                    .map(|edge| edge.source())
            {
                output_data.insert(NodeIndex::new(*hlir_node), data_node);
            }
        }

        CompiledBucket {
            bucket_indices,
            graph,
            llir_to_hlir,
            node_dtypes,
            output_data,
            steps,
            buffers: FxHashMap::default(),
        }
    }

    fn rebuild_dyn_buffer(&mut self) {
        self.dyn_dims = dyn_dims_order();
        self.dyn_buffer = unsafe {
            Buffer::<i32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                self.dyn_dims.len().max(1),
                ptr::null_mut(),
            )
            .expect("failed to allocate OpenCL dynamic-dim buffer")
        };
    }

    fn select_bucket(&mut self, dyn_map: &DynMap) {
        if self.buckets.len() <= 1 {
            self.active_bucket = 0;
            return;
        }
        self.active_bucket = self
            .buckets
            .iter()
            .position(|bucket| {
                self.dim_buckets.iter().all(|(dim, ranges)| {
                    let value = dyn_map.get(dim).copied().unwrap_or(0);
                    let index = bucket.bucket_indices.get(dim).copied().unwrap_or(0);
                    ranges
                        .get(index)
                        .map(|range| range.contains(value))
                        .unwrap_or(true)
                })
            })
            .unwrap_or_else(|| panic!("no OpenCL bucket matches dynamic dimensions {dyn_map:?}"));
    }

    fn update_dyn_buffer(&mut self, dyn_map: &DynMap) {
        let mut values = Vec::with_capacity(self.dyn_dims.len().max(1));
        for (slot, dim) in self.dyn_dims.iter().enumerate() {
            values.push(*dyn_map.get(dim).unwrap_or_else(|| {
                panic!("OpenCL kernel reads dynamic dim {dim} from slot {slot}, but it is unbound")
            }) as i32);
        }
        if values.is_empty() {
            values.push(0);
        }
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.dyn_buffer, CL_BLOCKING, 0, &values, &[])
                .expect("failed to update OpenCL dynamic-dim buffer");
        }
    }

    fn allocate_intermediates(&mut self, dyn_map: &DynMap) {
        let plans: Vec<_> = {
            let bucket = &self.buckets[self.active_bucket];
            bucket
                .steps
                .iter()
                .map(|step| {
                    let op = bucket.graph[step.node]
                        .to_dialect::<dyn OpenClKernelOp>()
                        .expect("OpenCL execution step lost its dialect op");
                    let elements = op.output_size().exec(dyn_map).unwrap_or(0);
                    let bytes = elements * step.output_dtype.bits().div_ceil(8);
                    (step.node, bytes)
                })
                .collect()
        };

        let bucket = &mut self.buckets[self.active_bucket];
        for (node, logical_bytes) in plans {
            let needs_allocation = bucket
                .buffers
                .get(&node)
                .is_none_or(|buffer| buffer.logical_bytes != logical_bytes);
            if needs_allocation {
                bucket.buffers.insert(
                    node,
                    DeviceBuffer {
                        buffer: create_buffer(&self.context, logical_bytes, CL_MEM_READ_WRITE),
                        logical_bytes,
                    },
                );
            }
        }
    }

    fn read_output(&self, id: NodeIndex) -> (Vec<u8>, DType) {
        let bucket = self
            .buckets
            .get(self.active_bucket)
            .expect("OpenCL runtime has not been compiled");
        let data_node = *bucket
            .output_data
            .get(&id)
            .unwrap_or_else(|| panic!("cannot find OpenCL output {id:?}"));
        let dtype = *bucket
            .node_dtypes
            .get(&data_node)
            .unwrap_or_else(|| panic!("cannot find dtype for OpenCL output {id:?}"));
        let device_buffer = if let Some(hlir_id) = bucket.llir_to_hlir.get(&data_node) {
            self.input_buffers
                .get(hlir_id)
                .unwrap_or_else(|| panic!("OpenCL input buffer {hlir_id:?} is not set"))
        } else {
            bucket
                .buffers
                .get(&data_node)
                .unwrap_or_else(|| panic!("OpenCL output buffer {data_node:?} is not allocated"))
        };
        let mut bytes = vec![0u8; device_buffer.logical_bytes];
        if !bytes.is_empty() {
            unsafe {
                self.queue
                    .enqueue_read_buffer(&device_buffer.buffer, CL_BLOCKING, 0, &mut bytes, &[])
                    .expect("failed to read OpenCL output");
            }
        }
        (bytes, dtype)
    }
}

impl Runtime for OpenClRuntime {
    type Ops = crate::kernel::OpenClOps;
    type CompileArg = usize;
    type ExecReturn = ();

    fn initialize(device_index: Self::CompileArg) -> Self {
        Self::try_initialize(device_index).unwrap_or_else(|error| panic!("{error}"))
    }

    fn compile(
        &mut self,
        space: &luminal::search::SearchSpace,
        dyn_map: &DynMap,
        _options: &luminal::graph::CompileOptions,
        rng: &mut dyn luminal::prelude::RngCore,
    ) {
        let contexts = space.bucket_contexts(dyn_map);
        let selected: Vec<_> = contexts
            .iter()
            .map(|context| luminal::search::extract_one_selected(space, context, rng))
            .collect();
        let schedule = SelectedSchedule::from_search(space, &selected);
        let buckets: Vec<_> = selected
            .into_iter()
            .map(|program| program.into_bucket_llir())
            .collect();
        self.load_llir_buckets(&space.dim_buckets, &buckets);
        self.selected_schedule = schedule;
    }

    fn selected_schedule(&self) -> Option<SelectedSchedule> {
        self.selected_schedule.clone()
    }

    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.load_llir_buckets(
            &FxHashMap::default(),
            &[(DynMap::default(), DynMap::default(), llir_graph.clone())],
        );
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        assert!(!bucket_llirs.is_empty(), "OpenCL received no LLIR buckets");
        clear_dyn_dims_order();
        self.dim_buckets = dim_buckets.clone();
        self.buckets = bucket_llirs
            .iter()
            .map(|(indices, _, graph)| self.compile_bucket(indices.clone(), graph))
            .collect();
        self.active_bucket = 0;
        self.rebuild_dyn_buffer();
        self.refresh_inputs();
    }

    fn execute(&mut self, dyn_map: &DynMap) -> Self::ExecReturn {
        self.select_bucket(dyn_map);
        self.allocate_intermediates(dyn_map);
        self.update_dyn_buffer(dyn_map);

        let bucket = &self.buckets[self.active_bucket];
        for step in &bucket.steps {
            let op = bucket.graph[step.node]
                .to_dialect::<dyn OpenClKernelOp>()
                .expect("OpenCL execution step lost its dialect op");
            let inputs: Vec<_> = step
                .input_nodes
                .iter()
                .map(|node| {
                    if let Some(hlir_id) = bucket.llir_to_hlir.get(node) {
                        &self
                            .input_buffers
                            .get(hlir_id)
                            .unwrap_or_else(|| panic!("OpenCL input {hlir_id:?} is not set"))
                            .buffer
                    } else {
                        &bucket
                            .buffers
                            .get(node)
                            .unwrap_or_else(|| {
                                panic!("OpenCL intermediate buffer {node:?} is missing")
                            })
                            .buffer
                    }
                })
                .collect();
            let output = &bucket
                .buffers
                .get(&step.node)
                .expect("OpenCL output buffer is missing")
                .buffer;
            op.enqueue(
                &self.queue,
                &step.kernels,
                &inputs,
                output,
                &self.dyn_buffer,
                dyn_map,
            );
        }
        self.queue.finish().expect("OpenCL queue execution failed");
    }
}

impl RuntimeStats for OpenClRuntime {
    fn execute_with_stats(&mut self, dyn_map: &DynMap) -> Option<ExecutionStats> {
        let start = Instant::now();
        self.execute(dyn_map);
        Some(ExecutionStats::with_timing_method(
            start.elapsed().as_secs_f64() * 1_000_000.0,
            0,
            0,
            0,
            TimingMethod::WallClock,
        ))
    }
}

fn create_buffer(context: &Context, logical_bytes: usize, flags: u64) -> Buffer<u8> {
    unsafe {
        Buffer::<u8>::create(context, flags, logical_bytes.max(1), ptr::null_mut())
            .expect("failed to allocate OpenCL buffer")
    }
}

fn reference_bytes(data: &ReferenceData, dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => bytemuck::cast_slice(data.to_f32_vec().as_slice()).to_vec(),
        DType::F16 => bytemuck::cast_slice(data.to_f16_vec().as_slice()).to_vec(),
        DType::Int => bytemuck::cast_slice(data.to_i32_vec().as_slice()).to_vec(),
        DType::Bool => data.to_bool_vec().into_iter().map(u8::from).collect(),
        unsupported => panic!("OpenCL input dtype {unsupported:?} is unsupported"),
    }
}

fn decode<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
    let width = std::mem::size_of::<T>();
    assert_eq!(
        bytes.len() % width,
        0,
        "misaligned OpenCL output byte count"
    );
    bytes
        .chunks_exact(width)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
