use luminal::{
    dtype::DType,
    dyn_backend::{BackendCompileArgs, DynBackend, bytes_to_reference_data, compile_backend},
    graph::Graph,
    hlir::{ConstantF64, Input},
    prelude::*,
};

use crate::{HexagonConfig, HexagonRuntime};

/// Dynamic-backend adapter used by the Python frontend and other consumers
/// that do not want to name the concrete Runtime type.
pub struct HexagonDynBackend {
    pub runtime: HexagonRuntime,
}

impl DynBackend for HexagonDynBackend {
    fn name(&self) -> &str {
        "hexagon"
    }

    // Tensor interop is deliberately a later boundary. The current backend
    // accepts host tensors, uploads them once into rpcmem, and keeps all
    // intermediate buffers shared with the DSP.
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

    fn execute(&mut self, dyn_map: &DynMap, stream: Option<u64>) {
        assert!(
            stream.is_none(),
            "Hexagon backend does not accept CUDA streams"
        );
        self.runtime.execute(dyn_map);
    }
}

fn validate_graph(graph: &Graph) -> Result<(), String> {
    for node in graph.graph.node_indices() {
        if let Some(input) = (*graph.graph[node]).as_any().downcast_ref::<Input>()
            && input.dtype != DType::F32
        {
            return Err(format!(
                "Hexagon v73 backend only supports F32 inputs; `{}` is {:?}",
                input.label, input.dtype
            ));
        }
        if (*graph.graph[node]).as_any().is::<ConstantF64>() {
            return Err(
                "Hexagon backend does not support F64 scalar constants; use tensor F32 inputs"
                    .to_string(),
            );
        }
    }
    Ok(())
}

pub fn hexagon_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    if !args.device_ptrs.is_empty() {
        return Err(
            "Hexagon backend accepts host tensors only; external device pointers are unsupported"
                .to_string(),
        );
    }
    validate_graph(graph)?;
    compile_backend::<HexagonRuntime>(
        graph,
        args,
        || HexagonRuntime::try_initialize(HexagonConfig::default()),
        |runtime, node, bytes, dtype| {
            runtime.set_data(node, bytes_to_reference_data(bytes, dtype));
        },
        None,
        |_, _| Ok(()),
        |runtime| Box::new(HexagonDynBackend { runtime }),
    )
}
