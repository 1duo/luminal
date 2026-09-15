use crate::{
    BinaryOp, HexagonRuntime,
    codegen::V73_CFLAGS,
    emit_dsp_source,
    kernel::{HexagonDispatch, HexagonKernelOp, HexagonMatmulI8, MatmulVariant},
    runtime::reference_bytes,
};
use luminal::{
    hlir::{Input, ReferenceData},
    op::EgglogOp,
    prelude::*,
    shape::Expression,
};
use rand::SeedableRng;

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
fn i8_matmul_codegen_contains_scalar_and_hvx_paths() {
    let source = emit_dsp_source(&[BinaryOp::MatmulI8]);
    assert!(source.contains("LUMINAL_HEXAGON_MATMUL_I8"));
    assert!(source.contains("luminal_dot_i8_scalar"));
    assert!(source.contains("luminal_dot_i8_hvx"));
    assert!(source.contains("Q6_Vw_vrmpy_VbVb"));
    assert!(source.contains("uint32_t m, uint32_t k, uint32_t flags"));
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
fn i8_matmul_rewrites_keep_both_target_schedule_choices() {
    let rewrites = HexagonMatmulI8::default().rewrites();
    assert_eq!(rewrites.len(), 2);

    let rendered: Vec<_> = rewrites
        .iter()
        .map(|rewrite| rewrite.to_egglog_string())
        .collect();
    assert!(rendered.iter().all(|rewrite| {
        rewrite.contains("HexagonMatmulI8") && rewrite.contains("(I8)") && rewrite.contains("(Int)")
    }));
    assert!(
        rendered
            .iter()
            .any(|rewrite| rewrite.contains("HexagonMatmulI8") && rewrite.contains(" 0)"))
    );
    assert!(
        rendered
            .iter()
            .any(|rewrite| rewrite.contains("HexagonMatmulI8") && rewrite.contains(" 1)"))
    );
}

#[test]
fn i8_matmul_extracts_contiguous_i32_hexagon_op() {
    let mut graph = Graph::new();
    let lhs = graph.tensor((2, 4)).as_dtype(DType::I8);
    let rhs = graph.tensor((3, 4)).as_dtype(DType::I8);
    let output = lhs
        .cast(DType::Int)
        .matmul(rhs.cast(DType::Int).permute((1, 0)))
        .output();
    assert_eq!(output.dtype, DType::Int);
    assert!(output.shape.is_contiguous());

    graph.build_search_space::<HexagonRuntime>(CompileOptions::default());
    let space = graph.search_space().expect("Hexagon search space missing");
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x0048_4558_4147_4F4E);
    let selected = luminal::search::extract_one_selected(space, &contexts[0], &mut rng);

    let (matmul_node, dispatch) = selected
        .llir
        .node_indices()
        .find_map(|node| {
            selected.llir[node]
                .to_dialect::<dyn HexagonKernelOp>()
                .map(|op| (node, op.dispatch()))
        })
        .expect("I8 matmul was not lowered to Hexagon dialect");
    assert!(
        selected.llir[matmul_node]
            .to_dialect::<dyn HexagonKernelOp>()
            .unwrap()
            .is_contiguous()
    );
    match dispatch {
        HexagonDispatch::MatmulI8 { m, n, k, variant } => {
            assert_eq!(m, Expression::from(2));
            assert_eq!(n, Expression::from(3));
            assert_eq!(k, Expression::from(4));
            assert!(matches!(
                variant,
                MatmulVariant::Scalar | MatmulVariant::Hvx
            ));
        }
        other => panic!("unexpected Hexagon dispatch: {other:?}"),
    }

    let input_nodes: Vec<_> = selected
        .llir
        .neighbors_directed(matmul_node, petgraph::Direction::Incoming)
        .collect();
    assert_eq!(input_nodes.len(), 2);
    assert!(input_nodes.iter().all(|node| {
        selected.llir[*node]
            .to_op::<Input>()
            .is_some_and(|input| input.dtype == DType::I8)
    }));
}

#[test]
fn integer_input_bytes_match_storage_dtype() {
    assert_eq!(
        reference_bytes(&ReferenceData::I8(vec![-128, -1, 0, 127]), DType::I8),
        vec![0x80, 0xff, 0x00, 0x7f]
    );

    let values = vec![1_i32, -2_i32];
    assert_eq!(
        reference_bytes(&ReferenceData::Int(values.clone()), DType::Int),
        bytemuck::cast_slice::<i32, u8>(&values).to_vec()
    );
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
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF32_0B5);
    let selected = luminal::search::extract_one_selected(space, &contexts[0], &mut rng);
    assert!(selected.llir.node_indices().any(|node| {
        selected.llir[node]
            .to_dialect::<dyn HexagonKernelOp>()
            .is_some()
    }));
    assert!(output.shape.is_contiguous());
}
