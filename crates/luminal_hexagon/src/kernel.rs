use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{Args, Rule, SortDef, Term as EggTerm, app, eq, rule, union},
        base::{SORTS, dtype, new_op_call, op_term},
    },
    hlir::{Add, Mul, binary_sort},
    op::{EgglogOp, LLIROp},
    prelude::*,
};

use crate::codegen::BinaryOp;

pub type HexagonOps = (HexagonAdd, HexagonMul);

fn call_sort_from_args(sort: &SortDef, args: &Args) -> EggTerm {
    let mut filtered = Args::new();
    for field in &sort.fields {
        filtered.add(&field.name, args[field.name.as_str()].clone());
    }
    sort.call(filtered)
}

fn f32_binary_rewrite(hlir_sort: &SortDef, hexagon_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp_a", "inp_b"]);
    let hexagon_op = op_term(
        call_sort_from_args(hexagon_sort, &args),
        args["__inputs"].clone(),
    );
    let f32 = app(&SORTS.f32_dt, vec![]);
    rule(union(hlir_match.clone(), hexagon_op.clone()))
        .subsume(hlir_match)
        .set(dtype(hexagon_op), f32.clone())
        .fact(eq(dtype(args["inp_a"].clone()), f32.clone()))
        .fact(eq(dtype(args["inp_b"].clone()), f32))
        .ruleset("kernel_lower")
}

/// An extracted operation for the direct Hexagon dispatcher. Pattern matches
/// that create this dialect live in egglog; the runtime only validates its
/// layout contract and dispatches it.
pub trait HexagonKernelOp: EgglogOp {
    fn output_size(&self) -> Expression;
    fn binary_op(&self) -> BinaryOp;

    /// The first backend slice uses one contiguous vector stream per input.
    fn is_contiguous(&self) -> bool;
}

luminal::impl_into_ops!(HexagonKernelOp);

macro_rules! hexagon_binary_op {
    ($name:ident, $sort_name:literal, $hlir:ty, $kind:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            a_strides: Vec<Expression>,
            b_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef {
                binary_sort($sort_name)
            }

            fn rewrites(&self) -> Vec<Rule> {
                vec![f32_binary_rewrite(&<$hlir>::default().sort(), &self.sort())]
            }

            fn cleanup(&self) -> bool {
                false
            }

            fn extract<'a>(
                &'a self,
                egraph: &'a SerializedEGraph,
                children: &[&'a ENodeId],
                input_enodes: Vec<&'a ENodeId>,
                list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
            ) -> (LLIROp, Vec<&'a ENodeId>) {
                use luminal::egglog_utils::extract_expr_list;
                (
                    LLIROp::new::<dyn HexagonKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                            .unwrap(),
                        a_strides: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                            .unwrap(),
                        b_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                            .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            children[3],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    input_enodes,
                )
            }
        }

        impl HexagonKernelOp for $name {
            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .copied()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn binary_op(&self) -> BinaryOp {
                $kind
            }

            fn is_contiguous(&self) -> bool {
                self.a_strides == self.output_strides && self.b_strides == self.output_strides
            }
        }
    };
}

hexagon_binary_op!(HexagonAdd, "HexagonAdd", Add, BinaryOp::AddF32);
hexagon_binary_op!(HexagonMul, "HexagonMul", Mul, BinaryOp::MulF32);
