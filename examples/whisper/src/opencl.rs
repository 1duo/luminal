#![cfg(feature = "opencl")]

// OpenCL benchmark runner for Snapdragon X Elite. The model and audio
// preprocessing are shared with the CUDA example; logits cross the public
// host-copy boundary while recurrent KV state stays on the device.

mod audio;
mod hf;
mod model;

use audio::{
    N_FRAMES, N_MELS, N_SAMPLES, load_wav, load_wav_bytes, log_mel_spectrogram, pad_or_trim,
};
use hf::prepare_hf_model;
use luminal::{dtype::DType, graph::Graph, hlir::Input, prelude::*};
use luminal_opencl::OpenClRuntime;
use model::*;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use std::{
    fs::File,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

const REPO_ID: &str = "openai/whisper-tiny.en";
const DEFAULT_AUDIO_BYTES: &[u8] = include_bytes!("../assets/jfk.wav");

#[derive(Debug, Default)]
struct DecodeStats {
    generated: usize,
    elapsed: Duration,
    prefill: Duration,
    decode: Duration,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn tensor_as_f32(tensor: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    match tensor.dtype() {
        SafeDtype::F32 => bytemuck::cast_slice(tensor.data()).to_vec(),
        SafeDtype::F16 => bytemuck::cast_slice::<u8, half::f16>(tensor.data())
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        SafeDtype::BF16 => bytemuck::cast_slice::<u8, half::bf16>(tensor.data())
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        dtype => panic!("Whisper OpenCL runner cannot load {dtype:?} weights"),
    }
}

fn load_weights(runtime: &mut OpenClRuntime, cx: &Graph, path: &Path) {
    let file = File::open(path).expect("failed to open Whisper weights");
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .expect("failed to map Whisper weights")
    };
    let tensors = SafeTensors::deserialize(&mmap).expect("invalid Whisper safetensors");
    let mut loaded = 0usize;

    for node in cx.graph.node_indices() {
        let Some(input) = (*cx.graph[node]).as_any().downcast_ref::<Input>() else {
            continue;
        };
        if input.dtype != DType::F32 {
            continue;
        }
        let Ok(tensor) = tensors.tensor(&input.label) else {
            continue;
        };
        runtime.set_data(node, tensor_as_f32(&tensor));
        loaded += 1;
    }

    assert!(loaded > 0, "no Whisper weights matched the graph");
    println!("Loaded {loaded} F32 weight tensors");
}

fn reset_cache(runtime: &mut OpenClRuntime, kv_cache: &KVCache, elements: usize) {
    let zeros = vec![0.0f32; elements];
    for (&k_cache, &v_cache) in kv_cache.k_caches.iter().zip(&kv_cache.v_caches) {
        runtime.set_data(k_cache, zeros.clone());
        runtime.set_data(v_cache, zeros.clone());
    }
}

fn update_cache(
    runtime: &mut OpenClRuntime,
    kv_cache: &KVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
) {
    for ((k_cache, v_cache), (k_out, v_out)) in kv_cache
        .k_caches
        .iter()
        .zip(&kv_cache.v_caches)
        .zip(cache_outputs)
    {
        runtime.copy_output_to_input(*k_out, *k_cache);
        runtime.copy_output_to_input(*v_out, *v_cache);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generation(
    runtime: &mut OpenClRuntime,
    cx: &mut Graph,
    input: GraphTensor,
    pos_ids: GraphTensor,
    mel: GraphTensor,
    logits: GraphTensor,
    kv_cache: &KVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
    mel_data: &[f32],
    prompt: &[u32],
    max_tokens: usize,
    tokenizer: &Tokenizer,
    print_text: bool,
) -> DecodeStats {
    let max_target_pos = kv_cache.max_seq;
    let cache_elements = N_TEXT_HEAD * max_target_pos * HEAD_DIM;
    reset_cache(runtime, kv_cache, cache_elements);
    runtime.set_data(mel, mel_data.to_vec());

    let total_start = Instant::now();
    let prefill_start = Instant::now();
    cx.set_dim('s', prompt.len());
    cx.set_dim('p', 0);
    runtime.set_data(
        input,
        prompt.iter().map(|token| *token as i32).collect::<Vec<_>>(),
    );
    runtime.set_data(pos_ids, (0..prompt.len() as i32).collect::<Vec<_>>());
    runtime.execute(&cx.dyn_map);
    let logits_data = runtime.get_f32(logits);
    update_cache(runtime, kv_cache, cache_outputs);
    let prefill = prefill_start.elapsed();

    let mut generated = 0usize;
    let mut next_input = None;
    let last_row = &logits_data[logits_data.len() - N_VOCAB..];
    let next_token = greedy_decode(last_row, true);
    if next_token != TOKEN_EOT && max_tokens > 0 {
        generated = 1;
        next_input = Some(next_token as i32);
        if print_text && let Ok(decoded) = tokenizer.decode(&[next_token], false) {
            print!("{decoded}");
            std::io::stdout().flush().unwrap();
        }
    }

    let decode_start = Instant::now();
    let mut prev_seq = prompt.len();
    while generated < max_tokens && prev_seq < max_target_pos - 1 {
        let Some(current_input) = next_input else {
            break;
        };
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![current_input]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);
        update_cache(runtime, kv_cache, cache_outputs);
        prev_seq += 1;

        let next_token = greedy_decode(&logits_data[logits_data.len() - N_VOCAB..], false);
        if next_token == TOKEN_EOT {
            break;
        }
        generated += 1;
        next_input = Some(next_token as i32);
        if print_text && let Ok(decoded) = tokenizer.decode(&[next_token], false) {
            print!("{decoded}");
            std::io::stdout().flush().unwrap();
        }
    }

    let decode = decode_start.elapsed();
    DecodeStats {
        generated,
        elapsed: total_start.elapsed(),
        prefill,
        decode,
    }
}

/// Greedy argmax with Whisper's special-token suppression.
fn greedy_decode(logits: &[f32], first_step: bool) -> u32 {
    debug_assert_eq!(logits.len(), N_VOCAB);
    logits
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            let index = *index as u32;
            (index < TOKEN_SOT || index == TOKEN_EOT) && !(first_step && index == TOKEN_EOT)
        })
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index as u32)
        .unwrap_or(TOKEN_EOT)
}

fn main() {
    let gen_tokens = env_usize("GEN_TOKENS", 32);
    let warmup_tokens = env_usize("WARMUP_TOKENS", 4);
    let search_graphs = env_usize("SEARCH_GRAPHS", 5);
    let audio_path = std::env::args().nth(1);

    let mut runtime = OpenClRuntime::try_initialize(0).expect("no usable OpenCL GPU");
    let device = runtime.device_info().clone();
    println!(
        "OpenCL device: {} ({}) via {}",
        device.name, device.vendor, device.platform
    );

    let model_dir = std::env::var_os("WHISPER_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| prepare_hf_model(REPO_ID).expect("failed to prepare Whisper model"));
    println!("Using model directory: {}", model_dir.display());
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let audio = match audio_path.as_deref() {
        Some(path) => load_wav(path).expect("failed to load audio"),
        None => load_wav_bytes(DEFAULT_AUDIO_BYTES).expect("failed to decode bundled audio"),
    };
    let audio = pad_or_trim(&audio, N_SAMPLES);
    let mel_data = log_mel_spectrogram(&audio, N_MELS);
    assert_eq!(mel_data.len(), N_MELS * N_FRAMES);

    let max_target_pos = N_TEXT_CTX;
    let mut cx = Graph::default();
    let mel_tensor = cx.named_tensor("mel", (N_MELS, N_FRAMES)).persist();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_target_pos);
    let whisper = Whisper::init(&mut cx);
    let xa = whisper.encoder.forward(mel_tensor);
    let (logits, cache_outputs) = whisper.decoder.forward(input, pos_ids, xa, &kv_cache);
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    let prompt = [TOKEN_SOT, TOKEN_NO_TIMESTAMPS];
    let compile_options = CompileOptions::default()
        .dim_buckets(
            's',
            &[
                DimBucket::new(1, 1),
                DimBucket::new(2, prompt.len()).representative(prompt.len()),
            ],
        )
        .search_dim('p', 0)
        .search_graph_limit(search_graphs);

    load_weights(&mut runtime, &cx, &model_dir.join("model.safetensors"));
    runtime.set_data(mel_tensor, mel_data.clone());
    runtime.set_data(input, vec![1i32; prompt.len()]);
    runtime.set_data(pos_ids, (0..prompt.len() as i32).collect::<Vec<_>>());
    let cache_elements = N_TEXT_HEAD * max_target_pos * HEAD_DIM;
    reset_cache(&mut runtime, &kv_cache, cache_elements);

    println!("Compiling with search_graphs={search_graphs}...");
    cx.set_dim('s', prompt.len());
    cx.set_dim('p', 0);
    runtime = cx.compile(runtime, compile_options);
    println!("Compilation complete");

    if warmup_tokens > 0 {
        let warmup = run_generation(
            &mut runtime,
            &mut cx,
            input,
            pos_ids,
            mel_tensor,
            logits,
            &kv_cache,
            &cache_outputs,
            &mel_data,
            &prompt,
            warmup_tokens,
            &tokenizer,
            false,
        );
        println!(
            "Warmup: {} tokens in {:.2}s",
            warmup.generated,
            warmup.elapsed.as_secs_f64()
        );
    }

    print!("Transcription: ");
    std::io::stdout().flush().unwrap();
    let stats = run_generation(
        &mut runtime,
        &mut cx,
        input,
        pos_ids,
        mel_tensor,
        logits,
        &kv_cache,
        &cache_outputs,
        &mel_data,
        &prompt,
        gen_tokens,
        &tokenizer,
        true,
    );
    println!();
    println!(
        "Generated {} tokens in {:.3}s ({:.2} tok/s, including prefill)",
        stats.generated,
        stats.elapsed.as_secs_f64(),
        stats.generated as f64 / stats.elapsed.as_secs_f64().max(1e-9),
    );
    println!(
        "Prefill: {:.3}s; decode: {:.3}s ({:.2} tok/s after prefill)",
        stats.prefill.as_secs_f64(),
        stats.decode.as_secs_f64(),
        stats.generated.saturating_sub(1) as f64 / stats.decode.as_secs_f64().max(1e-9),
    );
}
