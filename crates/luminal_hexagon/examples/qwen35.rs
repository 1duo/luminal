//! Run the resident Qwen3.5-0.8B Q4 model through Luminal's Hexagon backend.
//!
//! Required environment:
//!
//! * `LUMINAL_QWEN35_WEIGHTS` — exported QWH9/QHM4 file;
//! * `LUMINAL_HEXAGON_RPC_DLL` — FastRPC library;
//! * `LUMINAL_QWEN35_SKEL_URI` — ABI-compatible `libhexinfer-v73.so` URI.

use std::time::Instant;

use luminal::prelude::*;
use luminal_hexagon::{Qwen35Q4Config, Qwen35Runtime, qwen35_q4_step};

const DECODE_STEPS: usize = 16;

fn main() -> Result<(), String> {
    // These are token IDs, not text tokenization. Keeping tokenization out of
    // the first device bring-up makes the HTP boundary independently testable.
    let prompt_ids: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let mut graph = Graph::new();
    let prompt = graph.tensor(prompt_ids.len()).as_dtype(DType::Int);
    let next_token = qwen35_q4_step(prompt).output();

    let mut runtime = Qwen35Runtime::try_initialize(Qwen35Q4Config::default())?;
    runtime.set_data(prompt, prompt_ids);
    let mut runtime = graph.compile(
        runtime,
        CompileOptions::default().search_graph_limit(1).trials(1),
    );

    let prefill_started = Instant::now();
    let first = runtime.execute(&graph.dyn_map);
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1_000.0;

    let decode_started = Instant::now();
    let mut last = first;
    for _ in 0..DECODE_STEPS {
        last = runtime.execute(&graph.dyn_map);
    }
    let decode_elapsed = decode_started.elapsed().as_secs_f64();
    let decode_tok_per_second = DECODE_STEPS as f64 / decode_elapsed;

    println!(
        "Qwen3.5-0.8B Q4 HTP: prefill {:.3} ms, decode {:.2} tok/s, last token {} (logit {:.5})",
        prefill_ms, decode_tok_per_second, last.id, last.logit
    );
    println!("Luminal output token: {:?}", runtime.get_i32(next_token));
    Ok(())
}
