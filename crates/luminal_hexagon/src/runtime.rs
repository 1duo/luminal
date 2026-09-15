use std::{
    ffi::{CString, c_char, c_int, c_void},
    ptr::NonNull,
    time::Instant,
};

use itertools::Itertools;
use libloading::Library;
use luminal::{
    dtype::DType,
    graph::{BucketLLIR, DimBucket, LLIRGraph, SelectedSchedule},
    hlir::{Input, Output, ReferenceData},
    op::{ExecutionStats, Runtime, RuntimeStats, TimingMethod},
    prelude::{
        DynMap, FxHashMap, NodeIndex, Symbol, ToId,
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
    },
    shape::Expression,
};

use crate::{codegen::BinaryOp, config::HexagonConfig, kernel::HexagonKernelOp};

const RPCMEM_HEAP_ID_SYSTEM: c_int = 25;
const RPCMEM_DEFAULT_FLAGS: u32 = 1;
const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;
const COMPUTE_METHOD_ID: u32 = 2;

type RemoteHandle64 = u64;

#[repr(C)]
#[derive(Clone, Copy)]
struct RemoteBuf {
    pv: *mut c_void,
    n_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
union RemoteArg {
    buf: RemoteBuf,
    h64: RemoteHandle64,
}

#[repr(C)]
struct UnsignedModuleControl {
    domain: i32,
    enable: i32,
}

type RemoteOpen = unsafe extern "C" fn(*const c_char, *mut RemoteHandle64) -> c_int;
type RemoteClose = unsafe extern "C" fn(RemoteHandle64) -> c_int;
type RemoteInvoke = unsafe extern "C" fn(RemoteHandle64, u32, *mut RemoteArg) -> c_int;
type SessionControl = unsafe extern "C" fn(u32, *mut c_void, u32) -> c_int;
type RpcmemInit = unsafe extern "C" fn();
type RpcmemDeinit = unsafe extern "C" fn();
type RpcmemAlloc2 = unsafe extern "C" fn(c_int, u32, usize) -> *mut c_void;
type RpcmemFree = unsafe extern "C" fn(*mut c_void);

struct FastRpcApi {
    // The library must outlive all copied function pointers.
    _library: Library,
    remote_open: RemoteOpen,
    remote_close: RemoteClose,
    remote_invoke: RemoteInvoke,
    session_control: SessionControl,
    rpcmem_init: RpcmemInit,
    rpcmem_deinit: RpcmemDeinit,
    rpcmem_alloc2: RpcmemAlloc2,
    rpcmem_free: RpcmemFree,
}

impl FastRpcApi {
    fn load(config: &HexagonConfig) -> Result<Self, String> {
        let library_name = config
            .rpc_library
            .clone()
            .unwrap_or_else(|| "libcdsprpc.dll".into());
        // SAFETY: loading the user-selected FastRPC library is the explicit
        // device-runtime boundary. Symbols are copied while the library is
        // retained in this struct.
        let library = unsafe { Library::new(&library_name) }.map_err(|error| {
            format!(
                "failed to load FastRPC library {}: {error}; set LUMINAL_HEXAGON_RPC_DLL",
                library_name.display()
            )
        })?;

        unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
            // SAFETY: callers provide the ABI type documented by Hexagon SDK;
            // the library remains alive in FastRpcApi.
            unsafe { library.get::<T>(name) }
                .map(|symbol| *symbol)
                .map_err(|error| {
                    format!(
                        "FastRPC library is missing {}: {error}",
                        String::from_utf8_lossy(name)
                    )
                })
        }

        Ok(Self {
            remote_open: unsafe { symbol(&library, b"remote_handle64_open\0") }?,
            remote_close: unsafe { symbol(&library, b"remote_handle64_close\0") }?,
            remote_invoke: unsafe { symbol(&library, b"remote_handle64_invoke\0") }?,
            session_control: unsafe { symbol(&library, b"remote_session_control\0") }?,
            rpcmem_init: unsafe { symbol(&library, b"rpcmem_init\0") }?,
            rpcmem_deinit: unsafe { symbol(&library, b"rpcmem_deinit\0") }?,
            rpcmem_alloc2: unsafe { symbol(&library, b"rpcmem_alloc2\0") }?,
            rpcmem_free: unsafe { symbol(&library, b"rpcmem_free\0") }?,
            _library: library,
        })
    }
}

struct FastRpcSession {
    api: FastRpcApi,
    handle: RemoteHandle64,
}

impl FastRpcSession {
    fn open(config: &HexagonConfig) -> Result<Self, String> {
        let api = FastRpcApi::load(config)?;
        // SAFETY: the SDK documents rpcmem_init as the process-level setup
        // call and it has no arguments or return value.
        unsafe { (api.rpcmem_init)() };

        if config.allow_unsigned_module {
            let mut control = UnsignedModuleControl {
                domain: config.domain,
                enable: 1,
            };
            // SAFETY: `control` has the exact SDK struct layout and remains
            // valid for the synchronous call.
            let error = unsafe {
                (api.session_control)(
                    DSPRPC_CONTROL_UNSIGNED_MODULE,
                    (&mut control as *mut UnsignedModuleControl).cast(),
                    std::mem::size_of::<UnsignedModuleControl>() as u32,
                )
            };
            if error != 0 {
                return Err(format!(
                    "remote_session_control(unsigned module, domain {}) failed: 0x{error:08x}",
                    config.domain
                ));
            }
        }

        let uri = CString::new(config.skel_uri.as_str())
            .map_err(|_| "Hexagon skel URI contains an interior NUL".to_string())?;
        let mut handle = 0;
        // SAFETY: URI and output handle are valid for the synchronous SDK call.
        let error = unsafe { (api.remote_open)(uri.as_ptr(), &mut handle) };
        if error != 0 {
            return Err(format!(
                "remote_handle64_open({}) failed: 0x{error:08x}",
                config.skel_uri
            ));
        }
        Ok(Self { api, handle })
    }

    fn allocate(&self, logical_bytes: usize) -> Result<DeviceBuffer, String> {
        let allocation_bytes = logical_bytes.max(1);
        // SAFETY: the SDK allocator accepts the system heap and returns a
        // shared host/DSP buffer or null.
        let ptr = unsafe {
            (self.api.rpcmem_alloc2)(
                RPCMEM_HEAP_ID_SYSTEM,
                RPCMEM_DEFAULT_FLAGS,
                allocation_bytes,
            )
        };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| format!("rpcmem_alloc2 failed for {logical_bytes} logical bytes"))?;
        Ok(DeviceBuffer { ptr, logical_bytes })
    }

    unsafe fn free(&self, buffer: DeviceBuffer) {
        // SAFETY: the pointer came from this SDK allocator and is released
        // exactly once by the owning runtime.
        unsafe { (self.api.rpcmem_free)(buffer.ptr.as_ptr()) };
    }

    fn compute(
        &self,
        op: BinaryOp,
        n: usize,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        out: &DeviceBuffer,
    ) -> Result<(), String> {
        let n =
            u32::try_from(n).map_err(|_| "Hexagon dispatch exceeds u32 elements".to_string())?;
        let mut primitive = [
            op as u32,
            n,
            u32::try_from(a.logical_bytes)
                .map_err(|_| "Hexagon input buffer exceeds u32 bytes".to_string())?,
            u32::try_from(b.logical_bytes)
                .map_err(|_| "Hexagon input buffer exceeds u32 bytes".to_string())?,
            u32::try_from(out.logical_bytes)
                .map_err(|_| "Hexagon output buffer exceeds u32 bytes".to_string())?,
        ];
        let mut args = [
            RemoteArg {
                buf: RemoteBuf {
                    pv: primitive.as_mut_ptr().cast(),
                    n_len: std::mem::size_of_val(&primitive),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: a.ptr.as_ptr().cast(),
                    n_len: a.logical_bytes,
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: b.ptr.as_ptr().cast(),
                    n_len: b.logical_bytes,
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: out.ptr.as_ptr().cast(),
                    n_len: out.logical_bytes,
                },
            },
        ];
        // The primitive scalar block is not included in the buffer counts;
        // the two inputs and one rout output are the three RPC buffers.
        let scalars = (COMPUTE_METHOD_ID << 24) | (3 << 16) | (1 << 8);
        // SAFETY: all four descriptors and their backing shared buffers remain
        // alive until the synchronous FastRPC invocation returns.
        let error = unsafe { (self.api.remote_invoke)(self.handle, scalars, args.as_mut_ptr()) };
        if error != 0 {
            return Err(format!(
                "Hexagon compute op={op:?} n={n} failed: 0x{error:08x}"
            ));
        }
        Ok(())
    }
}

impl Drop for FastRpcSession {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: the handle was returned by remote_handle64_open and is
            // closed once during session teardown.
            unsafe { (self.api.remote_close)(self.handle) };
        }
        // SAFETY: matches the process-level rpcmem_init above.
        unsafe { (self.api.rpcmem_deinit)() };
    }
}

struct DeviceBuffer {
    ptr: NonNull<c_void>,
    logical_bytes: usize,
}

struct ExecutionStep {
    node: NodeIndex,
    input_nodes: Vec<NodeIndex>,
    output_size: Expression,
    op: BinaryOp,
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

/// Direct FastRPC runtime for the Hexagon v73 HTP slice.
pub struct HexagonRuntime {
    session: FastRpcSession,
    input_data: FxHashMap<NodeIndex, ReferenceData>,
    input_buffers: FxHashMap<NodeIndex, DeviceBuffer>,
    dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    buckets: Vec<CompiledBucket>,
    active_bucket: usize,
    selected_schedule: Option<SelectedSchedule>,
}

impl HexagonRuntime {
    pub fn try_initialize(config: HexagonConfig) -> Result<Self, String> {
        Ok(Self {
            session: FastRpcSession::open(&config)?,
            input_data: FxHashMap::default(),
            input_buffers: FxHashMap::default(),
            dim_buckets: FxHashMap::default(),
            buckets: Vec::new(),
            active_bucket: 0,
            selected_schedule: None,
        })
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl Into<ReferenceData>) {
        let id = id.to_id();
        let data = data.into();
        self.input_data.insert(id, data.clone());
        if let Some(dtype) = self.input_dtype(id) {
            self.upload_input(id, &data, dtype);
        }
    }

    /// Copy a device-resident output into a persistent input buffer. This is a
    /// host-issued pointer copy between two `rpcmem` mappings; no tensor bytes
    /// cross through an ordinary pageable host allocation or a DSP RPC payload.
    pub fn copy_output_to_input(&mut self, output: impl ToId, input: impl ToId) {
        let output = output.to_id();
        let input = input.to_id();
        let bucket = self
            .buckets
            .get(self.active_bucket)
            .expect("Hexagon runtime has not been compiled");
        let data_node = *bucket
            .output_data
            .get(&output)
            .unwrap_or_else(|| panic!("cannot find Hexagon output {output:?}"));
        let source_input = bucket.llir_to_hlir.get(&data_node).copied();
        if source_input == Some(input) {
            return;
        }

        let source = if let Some(hlir_id) = source_input {
            self.input_buffers
                .get(&hlir_id)
                .unwrap_or_else(|| panic!("Hexagon input buffer {hlir_id:?} is not set"))
        } else {
            bucket
                .buffers
                .get(&data_node)
                .unwrap_or_else(|| panic!("Hexagon output buffer {data_node:?} is not allocated"))
        };
        let destination = self
            .input_buffers
            .get(&input)
            .unwrap_or_else(|| panic!("Hexagon input buffer {input:?} is not set"));
        assert_eq!(
            source.logical_bytes, destination.logical_bytes,
            "Hexagon recurrent state size changed"
        );
        if source.logical_bytes > 0 {
            // SAFETY: both pointers refer to valid non-overlapping rpcmem
            // allocations of the checked size.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.ptr.as_ptr().cast::<u8>(),
                    destination.ptr.as_ptr().cast::<u8>(),
                    source.logical_bytes,
                );
            }
        }
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let (ptr, bytes, dtype) = self.output_buffer(id.to_id());
        assert_eq!(dtype, DType::F32, "Hexagon output is not F32");
        assert_eq!(bytes % std::mem::size_of::<f32>(), 0);
        let mut output = vec![0.0; bytes / std::mem::size_of::<f32>()];
        if bytes > 0 {
            // SAFETY: output is correctly sized and the shared buffer is valid
            // after the synchronous compute call.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr().cast::<u8>(),
                    output.as_mut_ptr().cast::<u8>(),
                    bytes,
                );
            }
        }
        output
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
        let needs_allocation = self
            .input_buffers
            .get(&id)
            .is_none_or(|buffer| buffer.logical_bytes != bytes.len());
        if needs_allocation {
            if let Some(buffer) = self.input_buffers.remove(&id) {
                // SAFETY: the removed buffer is no longer reachable.
                unsafe { self.session.free(buffer) };
            }
            self.input_buffers.insert(
                id,
                self.session
                    .allocate(bytes.len())
                    .unwrap_or_else(|error| panic!("failed to allocate Hexagon input: {error}")),
            );
        }
        if !bytes.is_empty() {
            let buffer = self.input_buffers.get(&id).unwrap();
            // SAFETY: destination allocation is at least `bytes.len()`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.ptr.as_ptr().cast::<u8>(),
                    bytes.len(),
                );
            }
        }
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
        let topo = toposort(&graph, None).expect("Hexagon LLIR graph has a cycle");

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
                .to_dialect::<dyn HexagonKernelOp>()
                .unwrap_or_else(|| panic!("Hexagon found unlowered LLIR node {node:?}"));
            let input_nodes: Vec<_> = graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|edge| edge.id())
                .map(|edge| edge.source())
                .collect();
            if input_nodes.len() != 2 || !op.is_contiguous() {
                panic!("Hexagon v73 MVP requires contiguous binary F32 ops at {node:?}");
            }
            let input_dtypes: Vec<_> = input_nodes
                .iter()
                .map(|input| {
                    node_dtypes
                        .get(input)
                        .copied()
                        .unwrap_or_else(|| panic!("missing dtype for Hexagon input {input:?}"))
                })
                .collect();
            if input_dtypes.iter().any(|dtype| *dtype != DType::F32) {
                panic!("Hexagon v73 MVP only supports F32 binary ops at {node:?}");
            }
            node_dtypes.insert(node, DType::F32);
            steps.push(ExecutionStep {
                node,
                input_nodes,
                output_size: op.output_size(),
                op: op.binary_op(),
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

    fn clear_buffers(&mut self) {
        for (_, buffer) in self.input_buffers.drain() {
            // SAFETY: drained buffers are being released exactly once.
            unsafe { self.session.free(buffer) };
        }
        let buckets = std::mem::take(&mut self.buckets);
        for bucket in buckets {
            for (_, buffer) in bucket.buffers {
                // SAFETY: bucket-owned buffers are no longer reachable.
                unsafe { self.session.free(buffer) };
            }
        }
    }

    fn allocate_intermediates(&mut self, dyn_map: &DynMap) {
        let plans: Vec<_> = {
            let bucket = &self.buckets[self.active_bucket];
            bucket
                .steps
                .iter()
                .map(|step| {
                    let elements = step.output_size.exec(dyn_map).unwrap_or(0);
                    (
                        step.node,
                        elements.saturating_mul(std::mem::size_of::<f32>()),
                    )
                })
                .collect()
        };
        let mut replacements = Vec::new();
        for (node, logical_bytes) in plans {
            let needs_allocation = self.buckets[self.active_bucket]
                .buffers
                .get(&node)
                .is_none_or(|buffer| buffer.logical_bytes != logical_bytes);
            if needs_allocation {
                if let Some(buffer) = self.buckets[self.active_bucket].buffers.remove(&node) {
                    // SAFETY: removed buffer is no longer reachable.
                    unsafe { self.session.free(buffer) };
                }
                replacements.push((
                    node,
                    self.session
                        .allocate(logical_bytes)
                        .unwrap_or_else(|error| {
                            panic!("failed to allocate Hexagon intermediate: {error}")
                        }),
                ));
            }
        }
        self.buckets[self.active_bucket]
            .buffers
            .extend(replacements);
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
            .unwrap_or_else(|| panic!("no Hexagon bucket matches dynamic dimensions {dyn_map:?}"));
    }

    fn output_buffer(&self, id: NodeIndex) -> (NonNull<c_void>, usize, DType) {
        let bucket = self
            .buckets
            .get(self.active_bucket)
            .expect("Hexagon runtime has not been compiled");
        let data_node = *bucket
            .output_data
            .get(&id)
            .unwrap_or_else(|| panic!("cannot find Hexagon output {id:?}"));
        let dtype = *bucket
            .node_dtypes
            .get(&data_node)
            .unwrap_or_else(|| panic!("cannot find Hexagon dtype for output {id:?}"));
        let buffer = if let Some(hlir_id) = bucket.llir_to_hlir.get(&data_node) {
            self.input_buffers
                .get(hlir_id)
                .unwrap_or_else(|| panic!("Hexagon input buffer {hlir_id:?} is not set"))
        } else {
            bucket
                .buffers
                .get(&data_node)
                .unwrap_or_else(|| panic!("Hexagon output buffer {data_node:?} is not allocated"))
        };
        (buffer.ptr, buffer.logical_bytes, dtype)
    }
}

impl Drop for HexagonRuntime {
    fn drop(&mut self) {
        self.clear_buffers();
    }
}

impl Runtime for HexagonRuntime {
    type Ops = crate::kernel::HexagonOps;
    type CompileArg = HexagonConfig;
    type ExecReturn = ();

    fn initialize(config: Self::CompileArg) -> Self {
        Self::try_initialize(config).unwrap_or_else(|error| panic!("{error}"))
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
        self.selected_schedule = SelectedSchedule::from_search(space, &selected);
        let buckets: Vec<_> = selected
            .into_iter()
            .map(|program| program.into_bucket_llir())
            .collect();
        self.load_llir_buckets(&space.dim_buckets, &buckets);
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
        assert!(!bucket_llirs.is_empty(), "Hexagon received no LLIR buckets");
        self.clear_buffers();
        self.dim_buckets = dim_buckets.clone();
        self.buckets = bucket_llirs
            .iter()
            .map(|(indices, _, graph)| self.compile_bucket(indices.clone(), graph))
            .collect();
        self.active_bucket = 0;
        self.refresh_inputs();
    }

    fn execute(&mut self, dyn_map: &DynMap) -> Self::ExecReturn {
        self.select_bucket(dyn_map);
        self.allocate_intermediates(dyn_map);

        let dispatches: Vec<_> = {
            let bucket = &self.buckets[self.active_bucket];
            bucket
                .steps
                .iter()
                .map(|step| {
                    let n = step.output_size.exec(dyn_map).unwrap_or(0);
                    let inputs: Vec<_> = step
                        .input_nodes
                        .iter()
                        .map(|node| {
                            if let Some(hlir_id) = bucket.llir_to_hlir.get(node) {
                                self.input_buffers.get(hlir_id).unwrap_or_else(|| {
                                    panic!("Hexagon input {hlir_id:?} is not set")
                                })
                            } else {
                                bucket.buffers.get(node).unwrap_or_else(|| {
                                    panic!("Hexagon intermediate buffer {node:?} is missing")
                                })
                            }
                        })
                        .collect::<Vec<_>>();
                    let output = bucket
                        .buffers
                        .get(&step.node)
                        .unwrap_or_else(|| panic!("Hexagon output buffer is missing"));
                    (step.op, n, inputs, output)
                })
                .collect()
        };
        for (op, n, inputs, output) in dispatches {
            self.session
                .compute(op, n, inputs[0], inputs[1], output)
                .unwrap_or_else(|error| panic!("{error}"));
        }
    }
}

impl RuntimeStats for HexagonRuntime {
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

fn reference_bytes(data: &ReferenceData, dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => bytemuck::cast_slice(data.to_f32_vec().as_slice()).to_vec(),
        unsupported => panic!("Hexagon input dtype {unsupported:?} is unsupported"),
    }
}
