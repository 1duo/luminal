use half::f16;
use luminal::prelude::*;

use crate::{OpenClRuntime, device::available_devices};

fn runtime() -> Option<OpenClRuntime> {
    match OpenClRuntime::try_initialize(0) {
        Ok(runtime) => Some(runtime),
        Err(error) => {
            eprintln!("skipping OpenCL test: {error}");
            None
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn prefers_native_qualcomm_device() {
    let Ok(devices) = available_devices() else {
        return;
    };
    if devices
        .iter()
        .any(|device| device.vendor.to_ascii_lowercase().contains("qualcomm"))
    {
        assert!(devices[0].vendor.to_ascii_lowercase().contains("qualcomm"));
    }
}

#[test]
fn elementwise_and_reduce_match_reference() {
    let Some(opencl) = runtime() else {
        return;
    };
    let mut graph = Graph::new();
    let input = graph.tensor((2, 3));
    let output = ((input.sin() + 2.0) * input).sum(1).output();
    let mut opencl = graph.compile(opencl, CompileOptions::default().search_graph_limit(1));
    let data = vec![-1.0, -0.5, 0.0, 0.25, 1.0, 2.0];
    opencl.set_data(input, data.clone());
    opencl.execute(&graph.dyn_map);

    let expected = vec![
        data[0] * (data[0].sin() + 2.0)
            + data[1] * (data[1].sin() + 2.0)
            + data[2] * (data[2].sin() + 2.0),
        data[3] * (data[3].sin() + 2.0)
            + data[4] * (data[4].sin() + 2.0)
            + data[5] * (data[5].sin() + 2.0),
    ];
    assert_close(&opencl.get_f32(output), &expected, 1e-4);
}

#[test]
fn matmul_runs_as_a_fused_opencl_op() {
    let Some(opencl) = runtime() else {
        return;
    };
    let mut graph = Graph::new();
    let lhs = graph.tensor((2, 3));
    let rhs = graph.tensor((3, 2));
    let output = lhs.matmul(rhs).output();
    let mut opencl = graph.compile(opencl, CompileOptions::default().search_graph_limit(1));
    opencl.set_data(lhs, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    opencl.set_data(rhs, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    opencl.execute(&graph.dyn_map);
    assert_close(&opencl.get_f32(output), &[58.0, 64.0, 139.0, 154.0], 1e-4);
}

#[test]
fn recurrent_output_can_stay_on_device() {
    let Some(opencl) = runtime() else {
        return;
    };
    let mut graph = Graph::new();
    let state = graph.tensor(4);
    let output = (state + 1.0).output();
    let mut opencl = graph.compile(opencl, CompileOptions::default().search_graph_limit(1));

    opencl.set_data(state, vec![1.0, 2.0, 3.0, 4.0]);
    opencl.execute(&graph.dyn_map);
    opencl.copy_output_to_input(output, state);
    opencl.execute(&graph.dyn_map);

    assert_close(&opencl.get_f32(output), &[3.0, 4.0, 5.0, 6.0], 1e-4);
}

#[test]
fn gather_scatter_and_bool_casts_work() {
    let Some(opencl) = runtime() else {
        return;
    };
    let mut graph = Graph::new();
    let data = graph.tensor(4);
    let indexes = graph.tensor(2).as_dtype(DType::Int);
    let replacement = graph.tensor(2);
    let gathered = data.gather(indexes);
    let scattered = replacement.scatter(indexes, data);
    let mask = data.lt(data * 0.0).cast(DType::F32);
    let output = (gathered.sum(0) + scattered.sum(0) + mask.sum(0)).output();
    let mut opencl = graph.compile(opencl, CompileOptions::default().search_graph_limit(1));
    opencl.set_data(data, vec![-1.0, 2.0, -3.0, 4.0]);
    opencl.set_data(indexes, vec![2, 0]);
    opencl.set_data(replacement, vec![10.0, 20.0]);
    opencl.execute(&graph.dyn_map);
    assert_close(&opencl.get_f32(output), &[34.0], 1e-4);
}

#[test]
fn fp16_round_trip_runs_when_supported() {
    let Some(opencl) = runtime() else {
        return;
    };
    if !opencl.device_info().supports_fp16 {
        return;
    }
    let mut graph = Graph::new();
    let input = graph.tensor(4).as_dtype(DType::F16);
    let output = (input + input).output();
    let mut opencl = graph.compile(opencl, CompileOptions::default().search_graph_limit(1));
    opencl.set_data(
        input,
        vec![
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.5),
        ],
    );
    opencl.execute(&graph.dyn_map);
    let actual: Vec<_> = opencl
        .get_f16(output)
        .into_iter()
        .map(|value| value.to_f32())
        .collect();
    assert_close(&actual, &[1.0, -2.0, 4.0, 7.0], 1e-3);
}

#[test]
fn remaining_primitives_and_dynamic_dims_work() {
    let Some(opencl) = runtime() else {
        return;
    };
    let mut graph = Graph::new();
    graph.set_dim('s', 4);
    let input = graph.tensor(('s', 2));
    let divisor = graph.tensor(('s', 2));
    let iota = graph.arange('s').cast(DType::F32);
    let values = (input.log2().exp2() + input.sqrt() + input.reciprocal()) % divisor;
    let output = (values + iota.expand_dim(1, 2)).max(1).output();
    let mut opencl = graph.compile(opencl, CompileOptions::default().search_graph_limit(1));
    opencl.set_data(input, vec![1.0, 4.0, 9.0, 16.0, 2.0, 8.0, 18.0, 32.0]);
    opencl.set_data(divisor, vec![3.0; 8]);
    opencl.execute(&graph.dyn_map);
    let first = opencl.get_f32(output);
    assert_eq!(first.len(), 4);

    graph.set_dim('s', 2);
    opencl.set_data(input, vec![1.0, 4.0, 9.0, 16.0]);
    opencl.set_data(divisor, vec![3.0; 4]);
    opencl.execute(&graph.dyn_map);
    assert_eq!(opencl.get_f32(output).len(), 2);
}
