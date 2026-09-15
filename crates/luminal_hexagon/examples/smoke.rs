use std::time::Instant;

use luminal::prelude::*;
use luminal_hexagon::{HexagonConfig, HexagonRuntime};

fn main() -> Result<(), String> {
    let mut graph = Graph::new();
    let n = 4096;
    let a = graph.tensor(n);
    let b = graph.tensor(n);
    let output = ((a + b) * b).output();
    let config = HexagonConfig::default();
    let runtime = HexagonRuntime::try_initialize(config)?;
    let mut runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
    runtime.set_data(a, vec![1.0; n]);
    runtime.set_data(b, vec![2.0; n]);

    let start = Instant::now();
    for _ in 0..10 {
        runtime.execute(&graph.dyn_map);
    }
    let elapsed = start.elapsed();
    let dispatches_per_run = 2.0;
    let result = runtime.get_f32(output);
    if result.iter().any(|&value| (value - 6.0).abs() > 1e-4) {
        eprintln!("first Hexagon values: {:?}", &result[..result.len().min(8)]);
        return Err("Hexagon smoke result mismatch".to_string());
    }
    runtime.copy_output_to_input(output, a);
    runtime.execute(&graph.dyn_map);
    let recurrent = runtime.get_f32(output);
    if recurrent.iter().any(|&value| (value - 16.0).abs() > 1e-4) {
        eprintln!(
            "recurrent Hexagon values: {:?}",
            &recurrent[..recurrent.len().min(8)]
        );
        return Err("Hexagon recurrent result mismatch".to_string());
    }
    println!(
        "hexagon smoke: {} runs, {:.3} ms/run, {:.2} GiB/s shared-buffer traffic",
        10,
        elapsed.as_secs_f64() * 1_000_000.0 / 10.0 / 1000.0,
        (n as f64 * 4.0 * 3.0 * dispatches_per_run * 10.0)
            / elapsed.as_secs_f64()
            / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}
