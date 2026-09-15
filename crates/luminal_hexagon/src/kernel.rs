use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{
            Args, Rule, SortDef, Term as EggTerm, app, eq, i64 as lit_i64, rule, sort, union, v,
        },
        base::{
            EXPRESSION, I64, OP_KIND, SORTS, cons, dtype, ilist, iter, mul, new_op_call, nil, num,
            op_term,
        },
    },
    hlir::{Add, Cast, Mul, SumReduce, binary_sort},
    op::{EgglogOp, LLIROp},
    prelude::*,
};

use crate::codegen::BinaryOp;

pub type HexagonOps = (HexagonAdd, HexagonMul, HexagonMatmulI8);

/// The schedule selected for the first HTP GEMM kernel.
///
/// Both variants implement the same I8×I8→I32 contract. Keeping the choice in
/// the e-graph lets Luminal's normal search measure it on the target instead
/// of baking a host-side architecture heuristic into the runtime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MatmulVariant {
    #[default]
    Scalar = 0,
    Hvx = 1,
}

impl MatmulVariant {
    fn from_i64(value: i64) -> Self {
        match value {
            0 => Self::Scalar,
            1 => Self::Hvx,
            other => panic!("invalid Hexagon matmul schedule variant {other}"),
        }
    }
}

/// Dispatch information carried by an extracted Hexagon op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexagonDispatch {
    F32(BinaryOp),
    MatmulI8 {
        m: Expression,
        n: Expression,
        k: Expression,
        variant: MatmulVariant,
    },
}

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

/// Match the canonical 2D contraction emitted by `GraphTensor::matmul` for
/// I8 storage with explicit I32 input casts:
///
/// ```text
/// Sum(Mul(Cast(Int, A[I8]), Cast(Int, B[I8])))
/// ```
///
/// The physical RHS is `[n, k]`, which is the ordinary row-major weight
/// layout for a logical `A[m, k] × B[k, n]`. No transpose buffer is created.
fn matmul_i8_rewrites(hexagon_sort: &SortDef) -> Vec<Rule> {
    let m = v("?hexagon_i8_m");
    let n = v("?hexagon_i8_n");
    let k = v("?hexagon_i8_k");
    let lhs = v("?hexagon_i8_lhs");
    let rhs = v("?hexagon_i8_rhs");
    let lhs_cast_size = v("?hexagon_i8_lhs_cast_size");
    let rhs_cast_size = v("?hexagon_i8_rhs_cast_size");
    let zero = num(lit_i64(0));
    let z = iter();
    let int = app(&SORTS.int_dt, vec![]);
    let i8 = app(&SORTS.i8_dt, vec![]);

    let lhs_cast = op_term(
        Cast::default()
            .sort()
            .call([("size", lhs_cast_size), ("dtype", int.clone())]),
        ilist(vec![lhs.clone()]),
    );
    let rhs_cast = op_term(
        Cast::default()
            .sort()
            .call([("size", rhs_cast_size), ("dtype", int.clone())]),
        ilist(vec![rhs.clone()]),
    );

    let mul_shape = cons(m.clone(), cons(n.clone(), cons(k.clone(), nil())));
    // The extracted kernel consumes row-major contiguous buffers. Luminal's
    // row-major constructor starts from the reserved unit stride z, so the
    // [m, n, k] product has [z*k*n, z*k, z] strides. Reducing k leaves
    // [z*k*n, z*k] as the input strides of the Sum.
    let k_stride = mul(z.clone(), k.clone());
    let mul_output_strides = cons(
        mul(k_stride.clone(), n.clone()),
        cons(k_stride.clone(), cons(z.clone(), nil())),
    );
    let lhs_strides = cons(k_stride.clone(), cons(zero.clone(), cons(z.clone(), nil())));
    let rhs_strides = cons(zero.clone(), cons(k_stride.clone(), cons(z.clone(), nil())));
    let mul_op = op_term(
        Mul::default().sort().call([
            ("shape", mul_shape),
            ("a_strides", lhs_strides),
            ("b_strides", rhs_strides),
            ("out_strides", mul_output_strides),
        ]),
        ilist(vec![lhs_cast, rhs_cast]),
    );

    let out_shape = cons(m.clone(), cons(n.clone(), nil()));
    let sum_input_strides = cons(mul(k_stride.clone(), n.clone()), cons(k_stride, nil()));
    let out_strides = cons(mul(z.clone(), n.clone()), cons(z.clone(), nil()));
    let sum_op = op_term(
        SumReduce::default().sort().call([
            ("shape", out_shape),
            ("iters", k.clone()),
            ("strides", sum_input_strides),
            ("iter_stride", z),
            ("out_strides", out_strides),
        ]),
        ilist(vec![mul_op.clone()]),
    );

    let make_hexagon_op = |variant: MatmulVariant| {
        op_term(
            hexagon_sort.call([
                ("m", m.clone()),
                ("n", n.clone()),
                ("k", k.clone()),
                ("variant", lit_i64(variant as i64)),
            ]),
            ilist(vec![lhs.clone(), rhs.clone()]),
        )
    };

    [MatmulVariant::Scalar, MatmulVariant::Hvx]
        .into_iter()
        .map(|variant| {
            let hexagon_op = make_hexagon_op(variant);
            rule(union(sum_op.clone(), hexagon_op.clone()))
                .set(dtype(hexagon_op), int.clone())
                .fact(eq(dtype(sum_op.clone()), int.clone()))
                .fact(eq(dtype(lhs.clone()), i8.clone()))
                .fact(eq(dtype(rhs.clone()), i8.clone()))
                .ruleset("kernel_lower")
                .name(match variant {
                    MatmulVariant::Scalar => "hexagon-i8-matmul-scalar",
                    MatmulVariant::Hvx => "hexagon-i8-matmul-hvx",
                })
        })
        .collect()
}

/// An extracted operation for the direct Hexagon dispatcher. Pattern matches
/// that create this dialect live in egglog; the runtime only validates the
/// resulting layout and dispatches the selected kernel.
pub trait HexagonKernelOp: EgglogOp {
    fn output_size(&self) -> Expression;
    fn dispatch(&self) -> HexagonDispatch;

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

            fn dispatch(&self) -> HexagonDispatch {
                HexagonDispatch::F32($kind)
            }

            fn is_contiguous(&self) -> bool {
                self.a_strides == self.output_strides && self.b_strides == self.output_strides
            }
        }
    };
}

hexagon_binary_op!(HexagonAdd, "HexagonAdd", Add, BinaryOp::AddF32);
hexagon_binary_op!(HexagonMul, "HexagonMul", Mul, BinaryOp::MulF32);

#[derive(Debug, Default, Clone)]
pub struct HexagonMatmulI8 {
    m: Expression,
    n: Expression,
    k: Expression,
    variant: MatmulVariant,
}

impl EgglogOp for HexagonMatmulI8 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "HexagonMatmulI8",
            &[
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("variant", I64),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        matmul_i8_rewrites(&self.sort())
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        let variant = egraph.enodes[children[3]]
            .0
            .parse::<i64>()
            .unwrap_or_else(|error| panic!("invalid Hexagon matmul variant: {error}"));
        (
            LLIROp::new::<dyn HexagonKernelOp>(Box::new(Self {
                m: extract_expr(egraph, children[0], expr_cache).unwrap(),
                n: extract_expr(egraph, children[1], expr_cache).unwrap(),
                k: extract_expr(egraph, children[2], expr_cache).unwrap(),
                variant: MatmulVariant::from_i64(variant),
            })),
            input_enodes,
        )
    }
}

impl HexagonKernelOp for HexagonMatmulI8 {
    fn output_size(&self) -> Expression {
        (self.m * self.n).max(Expression::from(1))
    }

    fn dispatch(&self) -> HexagonDispatch {
        HexagonDispatch::MatmulI8 {
            m: self.m,
            n: self.n,
            k: self.k,
            variant: self.variant,
        }
    }

    fn is_contiguous(&self) -> bool {
        true
    }
}
