use crate::{
    BinaryOp, HexagonRuntime, codegen::V73_CFLAGS, emit_dsp_source, kernel::HexagonKernelOp,
};
use luminal::prelude::*;

#[test]
fn v73_codegen_contains_hvx_binary_kernels() {
    let source = emit_dsp_source(&[BinaryOp::AddF32, BinaryOp::MulF32]);
    assert!(source.contains("Q6_Vqf32_vadd_VsfVsf"));
    assert!(source.contains("Q6_Vqf32_vmpy_VsfVsf"));
    assert!(source.contains("remote_handle64"));
    assert!(source.contains("LUMINAL_HEXAGON_ADD_F32"));
    assert!(source.contains("LUMINAL_HEXAGON_MUL_F32"));
}

#[test]
fn codegen_deduplicates_operation_bodies() {
    let source = emit_dsp_source(&[BinaryOp::AddF32, BinaryOp::AddF32, BinaryOp::MulF32]);
    assert_eq!(source.matches("static void luminal_add_f32").count(), 1);
    assert_eq!(source.matches("static void luminal_mul_f32").count(), 1);
}

#[test]
fn x_elite_compiler_flags_are_explicit() {
    assert_eq!(V73_CFLAGS, &["-mv73", "-mhvx", "-mhvx-length=128B"]);
}

#[test]
fn f32_binary_graph_extracts_hexagon_dialect() {
    let mut graph = Graph::new();
    let a = graph.tensor(64);
    let b = graph.tensor(64);
    let output = (a + b).output();
    graph.build_search_space::<HexagonRuntime>(CompileOptions::default());

    let space = graph.search_space().expect("Hexagon search space missing");
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let mut rng = rand::rng();
    let selected = luminal::search::extract_one_selected(space, &contexts[0], &mut rng);
    assert!(selected.llir.node_indices().any(|node| {
        selected.llir[node]
            .to_dialect::<dyn HexagonKernelOp>()
            .is_some()
    }));
    assert!(output.shape.is_contiguous());
}
