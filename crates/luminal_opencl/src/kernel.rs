use std::cell::RefCell;

use luminal::{
    dtype::DType,
    egglog_utils::{
        SerializedEGraph,
        api::{Args, Rule, SortDef, Term as EggTerm, app, eq, rule, sort, union, v},
        base::{DTYPE, ELIST, EXPRESSION, F64, IR, SORTS, dtype, ilist, new_op_call, op_term},
    },
    hlir::{
        Add, Cast, Constant, Gather, Iota, LessThan, MaxReduce, Mod, Mul, Scatter, SumReduce,
        binary_sort, reduce_sort, unary_sort,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::flatten_strides,
};
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    kernel::{ExecuteKernel, Kernel},
    memory::Buffer,
    program::Program,
};

pub type OpenClOps = (
    OpenClExp2,
    OpenClLog2,
    OpenClSin,
    OpenClSqrt,
    OpenClRecip,
    OpenClAdd,
    OpenClMul,
    OpenClMod,
    OpenClLessThan,
    OpenClSumReduce,
    OpenClMaxReduce,
    OpenClMatmul,
    OpenClConstant,
    OpenClIota,
    OpenClGather,
    OpenClScatter,
    OpenClCast,
);

thread_local! {
    static DYN_DIMS_ORDER: RefCell<Vec<Symbol>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn clear_dyn_dims_order() {
    DYN_DIMS_ORDER.with(|order| order.borrow_mut().clear());
}

pub(crate) fn dyn_dims_order() -> Vec<Symbol> {
    DYN_DIMS_ORDER.with(|order| order.borrow().clone())
}

fn dyn_slot(symbol: &Symbol) -> usize {
    DYN_DIMS_ORDER.with(|order| {
        if let Some(slot) = order.borrow().iter().position(|dim| dim == symbol) {
            return slot;
        }
        let mut order = order.borrow_mut();
        order.push(*symbol);
        order.len() - 1
    })
}

fn lower_expression(expr: &Expression, index_var: &str) -> String {
    expr.to_kernel_with(index_var, &|symbol| format!("dyn[{}]", dyn_slot(symbol)))
}

fn source_header(dtypes: impl IntoIterator<Item = DType>) -> &'static str {
    if dtypes.into_iter().any(|dtype| dtype == DType::F16) {
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    } else {
        ""
    }
}

fn buffer_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Int => "int",
        DType::Bool => "uchar",
        unsupported => panic!("OpenCL dtype {unsupported:?} is not supported"),
    }
}

fn numeric_read(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 => format!("{buffer}[{index}]"),
        DType::F16 => format!("convert_float({buffer}[{index}])"),
        DType::Int | DType::Bool => format!("convert_float({buffer}[{index}])"),
        unsupported => panic!("OpenCL dtype {unsupported:?} is not supported"),
    }
}

fn exact_read(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 | DType::F16 | DType::Int | DType::Bool => {
            format!("{buffer}[{index}]")
        }
        unsupported => panic!("OpenCL dtype {unsupported:?} is not supported"),
    }
}

fn numeric_write(dtype: DType, value: &str) -> String {
    match dtype {
        DType::F32 => value.to_string(),
        DType::F16 => format!("convert_half({value})"),
        DType::Int => format!("convert_int_rtz({value})"),
        DType::Bool => format!("(uchar)(({value}) != 0)"),
        unsupported => panic!("OpenCL dtype {unsupported:?} is not supported"),
    }
}

fn binary_values(
    output_dtype: DType,
    a_dtype: DType,
    b_dtype: DType,
    a_index: &str,
    b_index: &str,
) -> (String, String) {
    if output_dtype == DType::Int {
        (
            exact_read(a_dtype, "a", a_index),
            exact_read(b_dtype, "b", b_index),
        )
    } else {
        (
            numeric_read(a_dtype, "a", a_index),
            numeric_read(b_dtype, "b", b_index),
        )
    }
}

fn compile_program(context: &Context, source: &str, names: &[&str]) -> Vec<Kernel> {
    let program = Program::create_and_build_from_source(context, source, "")
        .unwrap_or_else(|error| panic!("OpenCL program build failed: {error}\n{source}"));
    names
        .iter()
        .map(|name| {
            Kernel::create(&program, name)
                .unwrap_or_else(|error| panic!("OpenCL kernel `{name}` creation failed: {error}"))
        })
        .collect()
}

fn call_sort_from_args(sort: &SortDef, args: &Args) -> EggTerm {
    let mut filtered = Args::new();
    for field in &sort.fields {
        filtered.add(&field.name, args[field.name.as_str()].clone());
    }
    sort.call(filtered)
}

fn unary_dtype_rewrite(hlir_sort: &SortDef, opencl_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp"]);
    let opencl_op = op_term(
        call_sort_from_args(opencl_sort, &args),
        args["__inputs"].clone(),
    );
    let dt = v("?__dt");
    rule(union(hlir_match.clone(), opencl_op.clone()))
        .subsume(hlir_match)
        .set(dtype(opencl_op), dt.clone())
        .fact(eq(dt, dtype(args["inp"].clone())))
        .ruleset("kernel_lower")
}

fn binary_dtype_rewrite(hlir_sort: &SortDef, opencl_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp_a", "inp_b"]);
    let opencl_op = op_term(
        call_sort_from_args(opencl_sort, &args),
        args["__inputs"].clone(),
    );
    let dt = v("?__dt");
    rule(union(hlir_match.clone(), opencl_op.clone()))
        .subsume(hlir_match)
        .set(dtype(opencl_op), dt.clone())
        .fact(eq(dt, dtype(args["inp_a"].clone())))
        .ruleset("kernel_lower")
}

/// An extracted OpenCL operation. Every pattern match that creates one lives
/// in egglog; the runtime only compiles and dispatches the extracted operation.
pub trait OpenClKernelOp: EgglogOp {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel>;

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression;

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    );
}

luminal::impl_into_ops!(OpenClKernelOp);

macro_rules! opencl_unary_op {
    ($name:ident, $sort_name:literal, $hlir_name:literal, $body:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            input_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef {
                unary_sort($sort_name)
            }

            fn rewrites(&self) -> Vec<Rule> {
                vec![unary_dtype_rewrite(&unary_sort($hlir_name), &self.sort())]
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
                    LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                            .unwrap(),
                        input_strides: extract_expr_list(
                            egraph,
                            children[1],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    input_enodes,
                )
            }
        }

        impl OpenClKernelOp for $name {
            fn compile(
                &self,
                context: &Context,
                input_dtypes: &[DType],
                output_dtype: DType,
            ) -> Vec<Kernel> {
                let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
                let input_ty = buffer_type(input_dtype);
                let output_ty = buffer_type(output_dtype);
                let input_index = flatten_strides(&self.shape, &self.input_strides);
                let output_index = flatten_strides(&self.shape, &self.output_strides);
                let input_index = lower_expression(&input_index, "idx");
                let output_index = lower_expression(&output_index, "idx");
                let input_value = numeric_read(input_dtype, "input", &input_index);
                let value = ($body)(&input_value);
                let value = numeric_write(output_dtype, &value);
                let header = source_header([input_dtype, output_dtype]);
                let source = format!(
                    r#"{header}
                    __kernel void unary_kernel(
                        __global const {input_ty} *input,
                        __global {output_ty} *output,
                        __global const int *dyn,
                        uint n_elements)
                    {{
                        int idx = (int)get_global_id(0);
                        if ((uint)idx < n_elements) {{
                            output[{output_index}] = {value};
                        }}
                    }}
                    "#,
                );
                compile_program(context, &source, &["unary_kernel"])
            }

            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .copied()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn enqueue(
                &self,
                queue: &CommandQueue,
                kernels: &[Kernel],
                inputs: &[&Buffer<u8>],
                output: &Buffer<u8>,
                dyn_buffer: &Buffer<i32>,
                dyn_map: &DynMap,
            ) {
                let n = self.output_size().exec(dyn_map).unwrap_or(0) as u32;
                if n == 0 {
                    return;
                }
                unsafe {
                    ExecuteKernel::new(&kernels[0])
                        .set_arg(inputs[0])
                        .set_arg(output)
                        .set_arg(dyn_buffer)
                        .set_arg(&n)
                        .set_global_work_size(n as usize)
                        .enqueue_nd_range(queue)
                        .expect("OpenCL unary dispatch failed");
                }
            }
        }
    };
}

opencl_unary_op!(OpenClExp2, "OpenClExp2", "Exp2", |x: &str| format!(
    "exp2({x})"
));
opencl_unary_op!(OpenClLog2, "OpenClLog2", "Log2", |x: &str| format!(
    "log2({x})"
));
opencl_unary_op!(OpenClSin, "OpenClSin", "Sin", |x: &str| format!("sin({x})"));
opencl_unary_op!(OpenClSqrt, "OpenClSqrt", "Sqrt", |x: &str| format!(
    "sqrt({x})"
));
opencl_unary_op!(OpenClRecip, "OpenClRecip", "Recip", |x: &str| format!(
    "1.0f / ({x})"
));

#[allow(clippy::too_many_arguments)]
fn compile_binary(
    context: &Context,
    shape: &[Expression],
    a_strides: &[Expression],
    b_strides: &[Expression],
    output_strides: &[Expression],
    input_dtypes: &[DType],
    output_dtype: DType,
    expression: impl FnOnce(&str, &str, DType) -> String,
) -> Vec<Kernel> {
    let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
    let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
    let a_ty = buffer_type(a_dtype);
    let b_ty = buffer_type(b_dtype);
    let output_ty = buffer_type(output_dtype);
    let a_index = lower_expression(&flatten_strides(shape, a_strides), "idx");
    let b_index = lower_expression(&flatten_strides(shape, b_strides), "idx");
    let output_index = lower_expression(&flatten_strides(shape, output_strides), "idx");
    let (a_value, b_value) = binary_values(output_dtype, a_dtype, b_dtype, &a_index, &b_index);
    let value = numeric_write(output_dtype, &expression(&a_value, &b_value, output_dtype));
    let header = source_header([a_dtype, b_dtype, output_dtype]);
    let source = format!(
        r#"{header}
        __kernel void binary_kernel(
            __global const {a_ty} *a,
            __global const {b_ty} *b,
            __global {output_ty} *output,
            __global const int *dyn,
            uint n_elements)
        {{
            int idx = (int)get_global_id(0);
            if ((uint)idx < n_elements) {{
                output[{output_index}] = {value};
            }}
        }}
        "#,
    );
    compile_program(context, &source, &["binary_kernel"])
}

fn enqueue_binary(
    queue: &CommandQueue,
    kernel: &Kernel,
    inputs: &[&Buffer<u8>],
    output: &Buffer<u8>,
    dyn_buffer: &Buffer<i32>,
    n: usize,
) {
    if n == 0 {
        return;
    }
    let n = n as u32;
    unsafe {
        ExecuteKernel::new(kernel)
            .set_arg(inputs[0])
            .set_arg(inputs[1])
            .set_arg(output)
            .set_arg(dyn_buffer)
            .set_arg(&n)
            .set_global_work_size(n as usize)
            .enqueue_nd_range(queue)
            .expect("OpenCL binary dispatch failed");
    }
}

macro_rules! opencl_binary_op {
    ($name:ident, $sort_name:literal, $hlir:ty, $expression:expr) => {
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
                vec![binary_dtype_rewrite(
                    &<$hlir>::default().sort(),
                    &self.sort(),
                )]
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
                    LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
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

        impl OpenClKernelOp for $name {
            fn compile(
                &self,
                context: &Context,
                input_dtypes: &[DType],
                output_dtype: DType,
            ) -> Vec<Kernel> {
                compile_binary(
                    context,
                    &self.shape,
                    &self.a_strides,
                    &self.b_strides,
                    &self.output_strides,
                    input_dtypes,
                    output_dtype,
                    $expression,
                )
            }

            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .copied()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn enqueue(
                &self,
                queue: &CommandQueue,
                kernels: &[Kernel],
                inputs: &[&Buffer<u8>],
                output: &Buffer<u8>,
                dyn_buffer: &Buffer<i32>,
                dyn_map: &DynMap,
            ) {
                enqueue_binary(
                    queue,
                    &kernels[0],
                    inputs,
                    output,
                    dyn_buffer,
                    self.output_size().exec(dyn_map).unwrap_or(0),
                );
            }
        }
    };
}

opencl_binary_op!(OpenClAdd, "OpenClAdd", Add, |a: &str, b: &str, _| {
    format!("({a}) + ({b})")
});
opencl_binary_op!(OpenClMul, "OpenClMul", Mul, |a: &str, b: &str, _| {
    format!("({a}) * ({b})")
});
opencl_binary_op!(OpenClMod, "OpenClMod", Mod, |a: &str, b: &str, dtype| {
    if dtype == DType::Int {
        format!("({a}) % ({b})")
    } else {
        format!("fmod(({a}), ({b}))")
    }
});

#[derive(Debug, Default, Clone)]
pub struct OpenClLessThan {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for OpenClLessThan {
    fn sort(&self) -> SortDef {
        binary_sort("OpenClLessThan")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&LessThan::default().sort(), &["inp_a", "inp_b"]);
        let opencl_op = op_term(
            call_sort_from_args(&self.sort(), &args),
            args["__inputs"].clone(),
        );
        vec![
            rule(union(hlir_match.clone(), opencl_op.clone()))
                .subsume(hlir_match)
                .set(dtype(opencl_op), app(&SORTS.bool_dt, vec![]))
                .ruleset("kernel_lower"),
        ]
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
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[1], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl OpenClKernelOp for OpenClLessThan {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel> {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = buffer_type(a_dtype);
        let b_ty = buffer_type(b_dtype);
        let output_ty = buffer_type(output_dtype);
        let a_index = lower_expression(&flatten_strides(&self.shape, &self.a_strides), "idx");
        let b_index = lower_expression(&flatten_strides(&self.shape, &self.b_strides), "idx");
        let output_index =
            lower_expression(&flatten_strides(&self.shape, &self.output_strides), "idx");
        let a_value = exact_read(a_dtype, "a", &a_index);
        let b_value = exact_read(b_dtype, "b", &b_index);
        let header = source_header([a_dtype, b_dtype]);
        let source = format!(
            r#"{header}
            __kernel void less_than_kernel(
                __global const {a_ty} *a,
                __global const {b_ty} *b,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_elements)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx < n_elements) {{
                    output[{output_index}] = (uchar)(({a_value}) < ({b_value}));
                }}
            }}
            "#,
        );
        compile_program(context, &source, &["less_than_kernel"])
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::Bool
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        enqueue_binary(
            queue,
            &kernels[0],
            inputs,
            output,
            dyn_buffer,
            self.output_size().exec(dyn_map).unwrap_or(0),
        );
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenClSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    input_strides: Vec<Expression>,
    iter_stride: Expression,
    output_strides: Vec<Expression>,
}

impl EgglogOp for OpenClSumReduce {
    fn sort(&self) -> SortDef {
        reduce_sort("OpenClSum")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![unary_dtype_rewrite(
            &SumReduce::default().sort(),
            &self.sort(),
        )]
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
        use luminal::egglog_utils::{extract_expr, extract_expr_list};
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                input_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, children[3], expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl OpenClKernelOp for OpenClSumReduce {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel> {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        assert_ne!(input_dtype, DType::Bool, "cannot sum Bool tensors");
        let input_ty = buffer_type(input_dtype);
        let output_ty = buffer_type(output_dtype);
        let input_start = lower_expression(
            &flatten_strides(&self.out_shape, &self.input_strides),
            "idx",
        );
        let output_index = lower_expression(
            &flatten_strides(&self.out_shape, &self.output_strides),
            "idx",
        );
        let iters = lower_expression(&self.iters, "idx");
        let iter_offset = lower_expression(&self.iter_stride, "i");
        let (acc_ty, input_value, initial) = if output_dtype == DType::Int {
            (
                "int",
                exact_read(
                    input_dtype,
                    "input",
                    &format!("input_start + {iter_offset}"),
                ),
                "0",
            )
        } else {
            (
                "float",
                numeric_read(
                    input_dtype,
                    "input",
                    &format!("input_start + {iter_offset}"),
                ),
                "0.0f",
            )
        };
        let output_value = numeric_write(output_dtype, "sum");
        let header = source_header([input_dtype, output_dtype]);
        let source = format!(
            r#"{header}
            __kernel void sum_kernel(
                __global const {input_ty} *input,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_outputs)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx >= n_outputs) return;
                int input_start = {input_start};
                int reduction_len = {iters};
                {acc_ty} sum = {initial};
                for (int i = 0; i < reduction_len; ++i) {{
                    sum += {input_value};
                }}
                output[{output_index}] = {output_value};
            }}
            "#,
        );
        compile_program(context, &source, &["sum_kernel"])
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        let n = self.output_size().exec(dyn_map).unwrap_or(0) as u32;
        if n == 0 {
            return;
        }
        unsafe {
            ExecuteKernel::new(&kernels[0])
                .set_arg(inputs[0])
                .set_arg(output)
                .set_arg(dyn_buffer)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(queue)
                .expect("OpenCL sum dispatch failed");
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenClMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    input_strides: Vec<Expression>,
    iter_stride: Expression,
    output_strides: Vec<Expression>,
}

impl EgglogOp for OpenClMaxReduce {
    fn sort(&self) -> SortDef {
        reduce_sort("OpenClMax")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![unary_dtype_rewrite(
            &MaxReduce::default().sort(),
            &self.sort(),
        )]
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
        use luminal::egglog_utils::{extract_expr, extract_expr_list};
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                input_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, children[3], expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl OpenClKernelOp for OpenClMaxReduce {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel> {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        assert_ne!(input_dtype, DType::Bool, "cannot max-reduce Bool tensors");
        let input_ty = buffer_type(input_dtype);
        let output_ty = buffer_type(output_dtype);
        let input_start = lower_expression(
            &flatten_strides(&self.out_shape, &self.input_strides),
            "idx",
        );
        let output_index = lower_expression(
            &flatten_strides(&self.out_shape, &self.output_strides),
            "idx",
        );
        let iters = lower_expression(&self.iters, "idx");
        let iter_offset = lower_expression(&self.iter_stride, "i");
        let (acc_ty, input_value, initial, update) = if output_dtype == DType::Int {
            (
                "int",
                exact_read(
                    input_dtype,
                    "input",
                    &format!("input_start + {iter_offset}"),
                ),
                "(-2147483647 - 1)",
                "max_value = max(max_value, value);",
            )
        } else {
            (
                "float",
                numeric_read(
                    input_dtype,
                    "input",
                    &format!("input_start + {iter_offset}"),
                ),
                "-INFINITY",
                "max_value = fmax(max_value, value);",
            )
        };
        let output_value = numeric_write(output_dtype, "max_value");
        let header = source_header([input_dtype, output_dtype]);
        let source = format!(
            r#"{header}
            __kernel void max_kernel(
                __global const {input_ty} *input,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_outputs)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx >= n_outputs) return;
                int input_start = {input_start};
                int reduction_len = {iters};
                {acc_ty} max_value = {initial};
                for (int i = 0; i < reduction_len; ++i) {{
                    {acc_ty} value = {input_value};
                    {update}
                }}
                output[{output_index}] = {output_value};
            }}
            "#,
        );
        compile_program(context, &source, &["max_kernel"])
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        let n = self.output_size().exec(dyn_map).unwrap_or(0) as u32;
        if n == 0 {
            return;
        }
        unsafe {
            ExecuteKernel::new(&kernels[0])
                .set_arg(inputs[0])
                .set_arg(output)
                .set_arg(dyn_buffer)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(queue)
                .expect("OpenCL max dispatch failed");
        }
    }
}

/// Fuses the primitive broadcast-multiply plus sum-reduce matmul shape into a
/// single OpenCL kernel. The match itself remains an egglog rewrite.
#[derive(Debug, Default, Clone)]
pub struct OpenClMatmul {
    out_shape: Vec<Expression>,
    mul_shape: Vec<Expression>,
    k: Expression,
    lhs_strides: Vec<Expression>,
    rhs_strides: Vec<Expression>,
    sum_input_strides: Vec<Expression>,
    sum_iter_stride: Expression,
    out_strides: Vec<Expression>,
}

impl EgglogOp for OpenClMatmul {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "OpenClMatmul",
            &[
                ("out_shape", ELIST),
                ("mul_shape", ELIST),
                ("k", EXPRESSION),
                ("lhs", IR),
                ("lhs_strides", ELIST),
                ("rhs", IR),
                ("rhs_strides", ELIST),
                ("sum_input_strides", ELIST),
                ("sum_iter_stride", EXPRESSION),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let mul_shape = v("?opencl_matmul_mul_shape");
        let out_shape = v("?opencl_matmul_out_shape");
        let k = v("?opencl_matmul_k");
        let lhs = v("?opencl_matmul_lhs");
        let rhs = v("?opencl_matmul_rhs");
        let lhs_strides = v("?opencl_matmul_lhs_strides");
        let rhs_strides = v("?opencl_matmul_rhs_strides");
        let mul_output_strides = v("?opencl_matmul_mul_output_strides");
        let sum_input_strides = v("?opencl_matmul_sum_input_strides");
        let sum_iter_stride = v("?opencl_matmul_sum_iter_stride");
        let out_strides = v("?opencl_matmul_out_strides");

        let mul_op = op_term(
            OpenClMul::default().sort().call([
                ("shape", mul_shape.clone()),
                ("a_strides", lhs_strides.clone()),
                ("b_strides", rhs_strides.clone()),
                ("out_strides", mul_output_strides),
            ]),
            ilist(vec![lhs.clone(), rhs.clone()]),
        );
        let sum_op = op_term(
            OpenClSumReduce::default().sort().call([
                ("shape", out_shape.clone()),
                ("iters", k.clone()),
                ("strides", sum_input_strides.clone()),
                ("iter_stride", sum_iter_stride.clone()),
                ("out_strides", out_strides.clone()),
            ]),
            ilist(vec![mul_op.clone()]),
        );
        let opencl_op = self.sort().call([
            ("out_shape", out_shape),
            ("mul_shape", mul_shape),
            ("k", k),
            ("lhs", lhs),
            ("lhs_strides", lhs_strides),
            ("rhs", rhs),
            ("rhs_strides", rhs_strides),
            ("sum_input_strides", sum_input_strides),
            ("sum_iter_stride", sum_iter_stride),
            ("out_strides", out_strides),
        ]);
        let dt = v("?opencl_matmul_dt");

        vec![
            rule(union(sum_op.clone(), opencl_op.clone()))
                .set(dtype(opencl_op), dt.clone())
                .fact(eq(dt, dtype(sum_op)))
                .ruleset("matmul_backend")
                .name("opencl-fused-matmul"),
            Rule::raw(
                "(rule
                    ((= ?mul (Op (OpenClMul ?shape ?as ?bs ?os) ?inputs))
                     (= ?sum (Op (OpenClSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (= ?sum (OpenClMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos)))
                    ((delete (Op (OpenClSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (delete (Op (OpenClMul ?shape ?as ?bs ?os) ?inputs)))
                    :ruleset cleanup
                    :name \"delete-opencl-broadcast-mul-sum-when-matmul-exists\")",
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_expr, extract_expr_list};
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                mul_shape: extract_expr_list(egraph, children[1], list_cache, expr_cache).unwrap(),
                k: extract_expr(egraph, children[2], expr_cache).unwrap(),
                lhs_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                rhs_strides: extract_expr_list(egraph, children[6], list_cache, expr_cache)
                    .unwrap(),
                sum_input_strides: extract_expr_list(egraph, children[7], list_cache, expr_cache)
                    .unwrap(),
                sum_iter_stride: extract_expr(egraph, children[8], expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[9], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[3], children[5]],
        )
    }
}

impl OpenClKernelOp for OpenClMatmul {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel> {
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let lhs_ty = buffer_type(lhs_dtype);
        let rhs_ty = buffer_type(rhs_dtype);
        let output_ty = buffer_type(output_dtype);
        let sum_base = lower_expression(
            &flatten_strides(&self.out_shape, &self.sum_input_strides),
            "idx",
        );
        let iter_offset = lower_expression(&self.sum_iter_stride, "i");
        let lhs_index = lower_expression(
            &flatten_strides(&self.mul_shape, &self.lhs_strides),
            "mul_idx",
        );
        let rhs_index = lower_expression(
            &flatten_strides(&self.mul_shape, &self.rhs_strides),
            "mul_idx",
        );
        let output_index =
            lower_expression(&flatten_strides(&self.out_shape, &self.out_strides), "idx");
        let iters = lower_expression(&self.k, "idx");
        let (acc_ty, lhs_value, rhs_value, initial) = if output_dtype == DType::Int {
            (
                "int",
                exact_read(lhs_dtype, "lhs", &lhs_index),
                exact_read(rhs_dtype, "rhs", &rhs_index),
                "0",
            )
        } else {
            (
                "float",
                numeric_read(lhs_dtype, "lhs", &lhs_index),
                numeric_read(rhs_dtype, "rhs", &rhs_index),
                "0.0f",
            )
        };
        let output_value = numeric_write(output_dtype, "sum");
        let header = source_header([lhs_dtype, rhs_dtype, output_dtype]);
        let source = format!(
            r#"{header}
            __kernel void matmul_kernel(
                __global const {lhs_ty} *lhs,
                __global const {rhs_ty} *rhs,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_outputs)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx >= n_outputs) return;
                int base_idx = {sum_base};
                int reduction_len = {iters};
                {acc_ty} sum = {initial};
                for (int i = 0; i < reduction_len; ++i) {{
                    int mul_idx = base_idx + {iter_offset};
                    sum += ({lhs_value}) * ({rhs_value});
                }}
                output[{output_index}] = {output_value};
            }}
            "#,
        );
        compile_program(context, &source, &["matmul_kernel"])
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        enqueue_binary(
            queue,
            &kernels[0],
            inputs,
            output,
            dyn_buffer,
            self.output_size().exec(dyn_map).unwrap_or(0),
        );
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenClConstant {
    value: f32,
}

impl EgglogOp for OpenClConstant {
    fn sort(&self) -> SortDef {
        sort(IR, "OpenClConstant", &[("value", F64)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Constant::default().sort(), &[]);
        let opencl_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(hlir_match.clone(), opencl_op.clone()))
                .subsume(hlir_match)
                .set(dtype(opencl_op), app(&SORTS.f32_dt, vec![]))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                value: egraph.enodes[children[0]]
                    .0
                    .replace('"', "")
                    .parse::<f32>()
                    .unwrap(),
            })),
            vec![],
        )
    }
}

impl OpenClKernelOp for OpenClConstant {
    fn compile(
        &self,
        context: &Context,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Vec<Kernel> {
        let bits = self.value.to_bits();
        let source = format!(
            r#"
            __kernel void constant_kernel(__global float *output) {{
                if (get_global_id(0) == 0) output[0] = as_float(0x{bits:08x}u);
            }}
            "#,
        );
        compile_program(context, &source, &["constant_kernel"])
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        _inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        _dyn_buffer: &Buffer<i32>,
        _dyn_map: &DynMap,
    ) {
        unsafe {
            ExecuteKernel::new(&kernels[0])
                .set_arg(output)
                .set_global_work_size(1)
                .enqueue_nd_range(queue)
                .expect("OpenCL constant dispatch failed");
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenClIota {
    expression: Expression,
    range: Expression,
}

impl EgglogOp for OpenClIota {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "OpenClIota",
            &[("expr", EXPRESSION), ("range", EXPRESSION)],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Iota::default().sort(), &[]);
        let opencl_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(hlir_match.clone(), opencl_op.clone()))
                .subsume(hlir_match)
                .set(dtype(opencl_op), app(&SORTS.int_dt, vec![]))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                expression: extract_expr(egraph, children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl OpenClKernelOp for OpenClIota {
    fn compile(
        &self,
        context: &Context,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Vec<Kernel> {
        let expression = lower_expression(&self.expression, "idx");
        let source = format!(
            r#"
            __kernel void iota_kernel(
                __global int *output,
                __global const int *dyn,
                uint n_elements)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx < n_elements) output[idx] = (int)({expression});
            }}
            "#,
        );
        compile_program(context, &source, &["iota_kernel"])
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::Int
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        _inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        let n = self.range.exec(dyn_map).unwrap_or(0) as u32;
        if n == 0 {
            return;
        }
        unsafe {
            ExecuteKernel::new(&kernels[0])
                .set_arg(output)
                .set_arg(dyn_buffer)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(queue)
                .expect("OpenCL iota dispatch failed");
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenClGather {
    out_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
}

impl EgglogOp for OpenClGather {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "OpenClGather",
            &[
                ("out_shape", ELIST),
                ("indexes", IR),
                ("index_strides", ELIST),
                ("data", IR),
                ("data_shape", ELIST),
                ("data_strides", ELIST),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Gather::default().sort(), &["indexes", "data"]);
        let out_strides = SORTS
            .row_major
            .call([("list".to_string(), args["index_shape"].clone())]);
        let opencl_op = self.sort().call([
            ("out_shape".to_string(), args["index_shape"].clone()),
            ("indexes".to_string(), args["indexes"].clone()),
            ("index_strides".to_string(), args["index_strides"].clone()),
            ("data".to_string(), args["data"].clone()),
            ("data_shape".to_string(), args["data_shape"].clone()),
            ("data_strides".to_string(), args["data_strides"].clone()),
            ("out_strides".to_string(), out_strides),
        ]);
        let dt = v("?opencl_gather_dt");
        vec![
            rule(union(hlir_match.clone(), opencl_op.clone()))
                .subsume(hlir_match)
                .set(dtype(opencl_op), dt.clone())
                .fact(eq(dt, dtype(args["data"].clone())))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                index_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                data_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                data_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, children[6], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl OpenClKernelOp for OpenClGather {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel> {
        let data_dtype = input_dtypes.get(1).copied().unwrap_or(DType::F32);
        let data_ty = buffer_type(data_dtype);
        let output_ty = buffer_type(output_dtype);
        let output_index =
            lower_expression(&flatten_strides(&self.out_shape, &self.out_strides), "idx");
        let index_index = lower_expression(
            &flatten_strides(&self.out_shape, &self.index_strides),
            "idx",
        );
        let data_index = lower_expression(
            &flatten_strides(&self.data_shape, &self.data_strides),
            "gathered_index",
        );
        let value = exact_read(data_dtype, "data", &data_index);
        let header = source_header([data_dtype, output_dtype]);
        let source = format!(
            r#"{header}
            __kernel void gather_kernel(
                __global const int *indexes,
                __global const {data_ty} *data,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_elements)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx < n_elements) {{
                    int gathered_index = indexes[{index_index}];
                    output[{output_index}] = {value};
                }}
            }}
            "#,
        );
        compile_program(context, &source, &["gather_kernel"])
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.get(1).copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        enqueue_binary(
            queue,
            &kernels[0],
            inputs,
            output,
            dyn_buffer,
            self.output_size().exec(dyn_map).unwrap_or(0),
        );
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenClScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
}

impl EgglogOp for OpenClScatter {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "OpenClScatter",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("dest", IR),
                ("indexes", IR),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("src", IR),
                ("src_strides", ELIST),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) =
            new_op_call(&Scatter::default().sort(), &["dest", "indexes", "src"]);
        let out_strides = SORTS
            .row_major
            .call([("list".to_string(), args["dest_shape"].clone())]);
        let opencl_op = self.sort().call([
            ("dest_shape".to_string(), args["dest_shape"].clone()),
            ("dest_strides".to_string(), args["dest_strides"].clone()),
            ("dest".to_string(), args["dest"].clone()),
            ("indexes".to_string(), args["indexes"].clone()),
            ("index_shape".to_string(), args["index_shape"].clone()),
            ("index_strides".to_string(), args["index_strides"].clone()),
            ("src".to_string(), args["src"].clone()),
            ("src_strides".to_string(), args["src_strides"].clone()),
            ("out_strides".to_string(), out_strides),
        ]);
        let dt = v("?opencl_scatter_dt");
        vec![
            rule(union(hlir_match.clone(), opencl_op.clone()))
                .subsume(hlir_match)
                .set(dtype(opencl_op), dt.clone())
                .fact(eq(dt, dtype(args["src"].clone())))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                dest_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                dest_strides: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                index_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                src_strides: extract_expr_list(egraph, children[7], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, children[8], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[2], children[3], children[6]],
        )
    }
}

impl OpenClKernelOp for OpenClScatter {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel> {
        let dest_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let src_dtype = input_dtypes.get(2).copied().unwrap_or(output_dtype);
        let dest_ty = buffer_type(dest_dtype);
        let src_ty = buffer_type(src_dtype);
        let output_ty = buffer_type(output_dtype);
        let dest_index = lower_expression(
            &flatten_strides(&self.dest_shape, &self.dest_strides),
            "idx",
        );
        let copy_output_index =
            lower_expression(&flatten_strides(&self.dest_shape, &self.out_strides), "idx");
        let index_index = lower_expression(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let src_index = lower_expression(
            &flatten_strides(&self.index_shape, &self.src_strides),
            "idx",
        );
        let header = source_header([dest_dtype, src_dtype, output_dtype]);
        let source = format!(
            r#"{header}
            __kernel void scatter_copy_kernel(
                __global const {dest_ty} *dest,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_elements)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx < n_elements) output[{copy_output_index}] = dest[{dest_index}];
            }}

            __kernel void scatter_write_kernel(
                __global const int *indexes,
                __global const {src_ty} *src,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_elements)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx < n_elements) {{
                    int output_index = indexes[{index_index}];
                    output[output_index] = src[{src_index}];
                }}
            }}
            "#,
        );
        compile_program(
            context,
            &source,
            &["scatter_copy_kernel", "scatter_write_kernel"],
        )
    }

    fn output_size(&self) -> Expression {
        self.dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        let n_dest = self
            .dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0) as u32;
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0) as u32;
        unsafe {
            if n_dest > 0 {
                ExecuteKernel::new(&kernels[0])
                    .set_arg(inputs[0])
                    .set_arg(output)
                    .set_arg(dyn_buffer)
                    .set_arg(&n_dest)
                    .set_global_work_size(n_dest as usize)
                    .enqueue_nd_range(queue)
                    .expect("OpenCL scatter copy dispatch failed");
            }
            if n_src > 0 {
                ExecuteKernel::new(&kernels[1])
                    .set_arg(inputs[1])
                    .set_arg(inputs[2])
                    .set_arg(output)
                    .set_arg(dyn_buffer)
                    .set_arg(&n_src)
                    .set_global_work_size(n_src as usize)
                    .enqueue_nd_range(queue)
                    .expect("OpenCL scatter write dispatch failed");
            }
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenClCast {
    size: Expression,
    target_dtype: DType,
}

impl EgglogOp for OpenClCast {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "OpenClCast",
            &[("inp", IR), ("size", EXPRESSION), ("dtype", DTYPE)],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Cast::default().sort(), &["inp"]);
        let opencl_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(hlir_match.clone(), opencl_op.clone()))
                .subsume(hlir_match)
                .set(dtype(opencl_op), args["dtype"].clone())
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_dtype, extract_expr};
        (
            LLIROp::new::<dyn OpenClKernelOp>(Box::new(Self {
                size: extract_expr(egraph, children[1], expr_cache).unwrap(),
                target_dtype: extract_dtype(egraph, children[2]),
            })),
            vec![children[0]],
        )
    }
}

impl OpenClKernelOp for OpenClCast {
    fn compile(
        &self,
        context: &Context,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Vec<Kernel> {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = buffer_type(input_dtype);
        let output_ty = buffer_type(output_dtype);
        let value = match (input_dtype, output_dtype) {
            (_, DType::Bool) => "(uchar)(input[idx] != 0)".to_string(),
            (DType::Bool, DType::F32) | (DType::Int, DType::F32) => {
                "convert_float(input[idx])".to_string()
            }
            (DType::Bool, DType::F16) | (DType::Int, DType::F16) => {
                "convert_half(input[idx])".to_string()
            }
            (DType::Bool, DType::Int) => "convert_int(input[idx])".to_string(),
            (DType::F32, DType::Int) | (DType::F16, DType::Int) => {
                "convert_int_rtz(input[idx])".to_string()
            }
            (DType::F32, DType::F16) => "convert_half(input[idx])".to_string(),
            (DType::F16, DType::F32) => "convert_float(input[idx])".to_string(),
            (a, b) if a == b => "input[idx]".to_string(),
            _ => panic!("OpenCL cast from {input_dtype:?} to {output_dtype:?} is unsupported"),
        };
        let header = source_header([input_dtype, output_dtype]);
        let source = format!(
            r#"{header}
            __kernel void cast_kernel(
                __global const {input_ty} *input,
                __global {output_ty} *output,
                __global const int *dyn,
                uint n_elements)
            {{
                int idx = (int)get_global_id(0);
                if ((uint)idx < n_elements) output[idx] = {value};
            }}
            "#,
        );
        compile_program(context, &source, &["cast_kernel"])
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        self.target_dtype
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn enqueue(
        &self,
        queue: &CommandQueue,
        kernels: &[Kernel],
        inputs: &[&Buffer<u8>],
        output: &Buffer<u8>,
        dyn_buffer: &Buffer<i32>,
        dyn_map: &DynMap,
    ) {
        let n = self.size.exec(dyn_map).unwrap_or(0) as u32;
        if n == 0 {
            return;
        }
        unsafe {
            ExecuteKernel::new(&kernels[0])
                .set_arg(inputs[0])
                .set_arg(output)
                .set_arg(dyn_buffer)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(queue)
                .expect("OpenCL cast dispatch failed");
        }
    }
}
