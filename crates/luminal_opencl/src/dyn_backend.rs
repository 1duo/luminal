//! Dynamic backend wrapper for the OpenCL runtime.

use luminal::{
    dtype::DType,
    dyn_backend::{BackendCompileArgs, DynBackend, bytes_to_reference_data, compile_backend},
    hlir::{Cast, ConstantF64, Input},
    prelude::*,
};

use crate::runtime::OpenClRuntime;

pub struct OpenClDynBackend {
    pub runtime: OpenClRuntime,
}

impl DynBackend for OpenClDynBackend {
    fn name(&self) -> &str {
        "opencl"
    }

    // OpenCL execution is device-side, but this backend intentionally exposes
    // a host-copy boundary. Reporting "cpu" keeps PyTorch tensors on the host
    // until OpenCL external-memory interop is implemented.
    fn device_type(&self) -> &str {
        "cpu"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, dtype: DType) {
        self.runtime
            .set_data(node, bytes_to_reference_data(bytes, dtype));
    }

    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }

    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        self.runtime.get_f32(node)
    }

    fn get_output_f16(&self, node: NodeIndex) -> Vec<half::f16> {
        self.runtime.get_f16(node)
    }

    fn get_output_i32(&self, node: NodeIndex) -> Vec<i32> {
        self.runtime.get_i32(node)
    }

    fn get_output_bool(&self, node: NodeIndex) -> Vec<bool> {
        self.runtime.get_bool(node)
    }

    fn execute(&mut self, dyn_map: &DynMap, _stream: Option<u64>) {
        self.runtime.execute(dyn_map);
    }
}

fn validate_graph(graph: &Graph) -> Result<(), String> {
    let supported = |dtype| matches!(dtype, DType::F32 | DType::F16 | DType::Int | DType::Bool);
    for node in graph.graph.node_indices() {
        if let Some(input) = (*graph.graph[node]).as_any().downcast_ref::<Input>()
            && !supported(input.dtype)
        {
            return Err(format!(
                "OpenCL backend does not support {:?} input `{}`; supported dtypes are F32, F16, Int, and Bool",
                input.dtype, input.label
            ));
        }
        if let Some(cast) = (*graph.graph[node]).as_any().downcast_ref::<Cast>()
            && !supported(cast.1)
        {
            return Err(format!(
                "OpenCL backend does not support casts to {:?}; supported dtypes are F32, F16, Int, and Bool",
                cast.1
            ));
        }
        if (*graph.graph[node]).as_any().is::<ConstantF64>() {
            return Err(
                "OpenCL backend does not support F64 scalar constants; use F32 constants instead"
                    .to_string(),
            );
        }
    }
    Ok(())
}

pub fn opencl_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    if !args.device_ptrs.is_empty() {
        return Err(
            "OpenCL backend accepts host tensors only; CUDA device pointers were provided"
                .to_string(),
        );
    }
    validate_graph(graph)?;
    let device_index = args.device_index.unwrap_or(0);
    compile_backend::<OpenClRuntime>(
        graph,
        args,
        || OpenClRuntime::try_initialize(device_index),
        |runtime, node, bytes, dtype| {
            runtime.set_data(node, bytes_to_reference_data(bytes, dtype));
        },
        None,
        |_, _| Ok(()),
        |runtime| Box::new(OpenClDynBackend { runtime }),
    )
}
