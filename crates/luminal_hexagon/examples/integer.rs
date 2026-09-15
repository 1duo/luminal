//! Smallest useful integer HTP model: one I8 token projection.
//!
//! The graph is `[tokens, 144] × [10, 144]^T -> [tokens, 10]`, with signed
//! I32 accumulation. It is deliberately a linear next-token classifier: it
//! exercises the same packed-I8 GEMM contract a larger quantized model needs,
//! without introducing convolution or quantization-scale policy.

use std::time::{Duration, Instant};

use luminal::prelude::*;
use luminal_hexagon::{HexagonConfig, HexagonRuntime};

const TOKENS: usize = 1;
const HIDDEN: usize = 144;
const VOCAB: usize = 10;
const RUNS: usize = 100;

fn reference(input: &[i8], weight: &[i8]) -> Vec<i32> {
    (0..TOKENS)
        .flat_map(|token| {
            (0..VOCAB).map(move |class| {
                (0..HIDDEN)
                    .map(|feature| {
                        i32::from(input[token * HIDDEN + feature])
                            * i32::from(weight[class * HIDDEN + feature])
                    })
                    .sum()
            })
        })
        .collect()
}

fn main() -> Result<(), String> {
    let input: Vec<i8> = (0..TOKENS * HIDDEN)
        .map(|index| (index as i8 % 17) - 8)
        .collect();
    let weight: Vec<i8> = (0..VOCAB * HIDDEN)
        .map(|index| (index as i8 % 11) - 5)
        .collect();

    let mut graph = Graph::new();
    let tokens = graph.tensor((TOKENS, HIDDEN)).as_dtype(DType::I8);
    let weight_tensor = graph.tensor((VOCAB, HIDDEN)).as_dtype(DType::I8);
    let logits = tokens
        .cast(DType::Int)
        // The frontend RHS is logical [hidden, vocab]; its underlying input
        // remains the convenient [vocab, hidden] weight layout.
        .matmul(weight_tensor.cast(DType::Int).permute((1, 0)))
        .output();

    let mut runtime = HexagonRuntime::try_initialize(HexagonConfig::default())?;
    // Preload before compile so Luminal can measure scalar versus HVX on the
    // actual HTP. Callers that upload after compile retain the fallback path.
    runtime.set_data(tokens, input.clone());
    runtime.set_data(weight_tensor, weight.clone());
    let options = CompileOptions::default()
        .search_graph_limit(4)
        .trials(2)
        .search_time_limit(Duration::from_secs(10));
    let mut runtime = graph.compile(runtime, options);

    runtime.execute(&graph.dyn_map);
    let expected = reference(&input, &weight);
    let actual = runtime.get_i32(logits);
    if actual != expected {
        return Err(format!(
            "I8 HTP matmul mismatch: expected {expected:?}, got {actual:?}"
        ));
    }

    let started = Instant::now();
    for _ in 0..RUNS {
        runtime.execute(&graph.dyn_map);
    }
    let elapsed = started.elapsed();
    let tok_per_second = RUNS as f64 * TOKENS as f64 / elapsed.as_secs_f64();
    let macs_per_second = tok_per_second * HIDDEN as f64 * VOCAB as f64;
    println!(
        "hexagon int8 projection: {TOKENS} token/run, {:.3} ms/run, {:.2} tok/s, {:.2} GMAC/s",
        elapsed.as_secs_f64() * 1_000.0 / RUNS as f64,
        tok_per_second,
        macs_per_second / 1e9,
    );
    Ok(())
}
