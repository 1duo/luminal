//! Resident Qwen3.5-0.8B Q4 support for the Hexagon backend.
//!
//! The selected checkpoint is the mixed-GGUF export
//! `unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf`.  The HTP runtime does
//! not consume raw GGUF directly: the local Hexagon exporter turns it into a
//! validated QWH9 base store with an embedded QHM4 HMX sidecar.  The base and
//! sidecar stay in rpcmem for the lifetime of a session, while one shared
//! runtime buffer holds activations, KV cache, and GatedDeltaNet state.
//!
//! Luminal represents the whole resident model step as one explicit custom
//! operation.  This is intentionally a coarse boundary: the model is hybrid
//! (GatedDeltaNet plus full attention), and splitting it into ordinary tensor
//! RPCs would evict the very state that makes HTP execution useful.  Internal
//! Q4/HMX kernel selection remains the HexKL/DSP concern; Luminal still owns
//! the graph, extraction, lifecycle, and profiling boundary.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    slice,
    time::Instant,
};

use itertools::Itertools;
use luminal::{
    dtype::DType,
    graph::{BucketLLIR, CompileOptions, DimBucket, LLIRGraph, SelectedSchedule},
    hlir::{Input, Output, ReferenceData},
    op::{CustomOp, ExecutionStats, LLIROp, Runtime, RuntimeStats, TimingMethod},
    prelude::{
        DynMap, FxHashMap, NodeIndex, ToId,
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
    },
    shape::Expression,
};

use crate::{
    config::HexagonConfig,
    runtime::{DeviceBuffer, FastRpcSession},
};

/// Hugging Face source selected for this backend.
pub const HF_REPOSITORY: &str = "unsloth/Qwen3.5-0.8B-GGUF";
/// Hugging Face file selected for the Q4 HTP path.
pub const HF_FILENAME: &str = "Qwen3.5-0.8B-Q4_0.gguf";
/// Exported resident file consumed by the HTP runtime.
pub const EXPORTED_WEIGHT_FILENAME: &str = "qwen3_5_08B_Q4_0_weights.bin";

// Qwen3.5-0.8B text configuration. Vision and MTP components are deliberately
// outside this first device path.
pub const VOCAB_SIZE: usize = 248_320;
pub const HIDDEN_SIZE: usize = 1_024;
pub const NUM_LAYERS: usize = 24;
pub const INTERMEDIATE_SIZE: usize = 3_584;
pub const NUM_FULL_ATTENTION_LAYERS: usize = 6;
pub const NUM_LINEAR_ATTENTION_LAYERS: usize = 18;
pub const FULL_ATTENTION_LAYERS: [usize; NUM_FULL_ATTENTION_LAYERS] = [3, 7, 11, 15, 19, 23];
pub const FULL_ATTENTION_Q_HEADS: usize = 8;
pub const FULL_ATTENTION_KV_HEADS: usize = 2;
pub const FULL_ATTENTION_HEAD_DIM: usize = 256;
pub const GDN_HEADS: usize = 16;
pub const GDN_HEAD_DIM: usize = 128;
pub const GDN_CONV_KERNEL: usize = 4;
pub const ROPE_DIM: usize = 64;
pub const ROPE_THETA: f32 = 10_000_000.0;
pub const MAX_PREFILL_TOKENS: usize = 256;
pub const MAX_DECODE_CONTEXT: usize = 4_096;

// Generated QWH9/QHM4 sizes for the selected exporter/runtime ABI.
pub const QWH9_TENSOR_COUNT: u32 = 326;
pub const QWH9_BASE_BYTES: usize = 560_014_848;
pub const QWH9_DATA_START: usize = 13_440;
pub const QHM4_VERSION: u32 = 8;
pub const QHM4_TENSOR_COUNT: u32 = 150;
pub const QHM4_BYTES: usize = 994_050_176;

// Generated from qwen_runtime_layout.h. Keep this in sync with the DSP
// module's QWEN_RUNTIME_LAYOUT_VERSION and QWEN_RUNTIME_BYTES.
pub const QWEN_RUNTIME_LAYOUT_VERSION: u32 = 4;
pub const QWEN_RUNTIME_MAGIC: u32 = 0x5152_544d;
pub const QWEN_WORKSPACE_BYTES: usize = 40_412_800;
pub const QWEN_RUNTIME_BYTES: usize = 211_195_648;

// The packed hexinfer_stop_diag_t is 144 bytes. Keep one reusable response
// allocation large enough for teardown as well as the Qwen phase results.
const QWEN_RESPONSE_BYTES: usize = 256;

const OP_COMPUTE: u32 = 2;
const OP_STOP: u32 = 3;
const OP_MMAP: u32 = 5;
const OP_MUNMAP: u32 = 6;
const OP_MMAP_HMX: u32 = 7;
const OP_MUNMAP_HMX: u32 = 8;
const OP_MMAP_RUNTIME: u32 = 11;
const OP_MUNMAP_RUNTIME: u32 = 12;
const OP_QWEN_DECODE_BIND: u32 = 284;
const OP_QWEN_FULL_DECODE_STEP: u32 = 285;
const OP_QWEN_FULL_PREFILL: u32 = 297;

const CONTROL_MAGIC_OFFSET: usize = 0;
const CONTROL_LAYOUT_VERSION_OFFSET: usize = 4;
const CONTROL_FLAGS_OFFSET: usize = 8;
const CONTROL_POISONED_OFFSET: usize = 12;
const CONTROL_POSITION_OFFSET: usize = 16;
const CONTROL_CACHE_LEN_OFFSET: usize = 20;
const CONTROL_GENERATED_COUNT_OFFSET: usize = 24;
const CONTROL_GENERATED_CAPACITY_OFFSET: usize = 28;

/// Runtime configuration for the selected Qwen3.5 Q4 export.
#[derive(Debug, Clone)]
pub struct Qwen35Q4Config {
    /// QWH9 file exported from the selected mixed-GGUF checkpoint.
    pub weights: PathBuf,
    /// FastRPC and DSP skel settings. For actual execution the URI must point
    /// to the ABI-compatible resident Qwen module.
    pub rpc: HexagonConfig,
    /// Runtime-side prefill limit. The current QWH9 DSP module supports up to
    /// 256 prompt tokens without changing its generated workspace layout.
    pub max_prefill_tokens: usize,
}

impl Default for Qwen35Q4Config {
    fn default() -> Self {
        let rpc = HexagonConfig {
            skel_uri: std::env::var("LUMINAL_QWEN35_SKEL_URI").unwrap_or_else(|_| {
                "file:///libhexinfer-v73.so?hexinfer_skel_handle_invoke&_modver=1.0&_dom=cdsp"
                    .to_string()
            }),
            ..HexagonConfig::default()
        };
        Self {
            weights: std::env::var_os("LUMINAL_QWEN35_WEIGHTS")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(EXPORTED_WEIGHT_FILENAME)),
            rpc,
            max_prefill_tokens: MAX_PREFILL_TOKENS,
        }
    }
}

impl Qwen35Q4Config {
    pub fn from_weights(path: impl Into<PathBuf>) -> Self {
        Self {
            weights: path.into(),
            ..Self::default()
        }
    }

    pub fn with_weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = path.into();
        self
    }

    pub fn with_skel_uri(mut self, uri: impl Into<String>) -> Self {
        self.rpc.skel_uri = uri.into();
        self
    }

    pub fn with_rpc_library(mut self, path: impl Into<PathBuf>) -> Self {
        self.rpc.rpc_library = Some(path.into());
        self
    }

    pub fn with_max_prefill_tokens(mut self, max_tokens: usize) -> Self {
        assert!((2..=MAX_PREFILL_TOKENS).contains(&max_tokens));
        self.max_prefill_tokens = max_tokens;
        self
    }
}

/// The exact resident-file layout validated before any large rpcmem
/// allocation is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Q4WeightLayout {
    pub file_bytes: u64,
    pub base_bytes: usize,
    pub data_start: usize,
    pub sidecar_offset: usize,
    pub hmx_bytes: usize,
}

/// Parse the fixed QWH9 header. Raw GGUF begins with `GGUF` and is rejected.
pub fn parse_qwh9_header(header: &[u8]) -> Result<(u32, u32, u64), String> {
    if header.len() < 20 {
        return Err("QWH9 header is truncated".to_string());
    }
    if &header[0..4] != b"QWH9" {
        return Err(format!(
            "expected QWH9 resident export, found {:?}; raw GGUF is not an HTP weight file",
            &header[0..4]
        ));
    }
    Ok((
        read_header_u32(header, 4)?,
        read_header_u32(header, 8)?,
        read_header_u64(header, 12)?,
    ))
}

/// Validate the complete QWH9/QHM4 export and return its byte layout.
pub fn validate_qwen35_q4_weights(path: impl AsRef<Path>) -> Result<Qwen35Q4WeightLayout, String> {
    let path = path.as_ref();
    let file_bytes = std::fs::metadata(path)
        .map_err(|error| format!("cannot stat Qwen3.5 Q4 weights {}: {error}", path.display()))?
        .len();

    let mut file = File::open(path)
        .map_err(|error| format!("cannot open Qwen3.5 Q4 weights {}: {error}", path.display()))?;
    let mut header = [0_u8; 20];
    file.read_exact(&mut header)
        .map_err(|error| format!("cannot read QWH9 header from {}: {error}", path.display()))?;
    let (version, tensor_count, base_bytes) = parse_qwh9_header(&header)?;
    if version != 1 || tensor_count != QWH9_TENSOR_COUNT || base_bytes != QWH9_BASE_BYTES as u64 {
        return Err(format!(
            "unsupported QWH9 header in {}: version={version}, tensors={tensor_count}, base_bytes={base_bytes}",
            path.display()
        ));
    }

    let entry_bytes = usize::try_from(tensor_count)
        .unwrap()
        .checked_mul(41)
        .ok_or_else(|| "QWH9 tensor table size overflowed".to_string())?;
    let data_start = align128(20 + entry_bytes);
    if data_start != QWH9_DATA_START {
        return Err(format!(
            "unexpected QWH9 data start: expected {QWH9_DATA_START}, got {data_start}"
        ));
    }
    let mut entries = vec![0_u8; entry_bytes];
    file.read_exact(&mut entries)
        .map_err(|error| format!("cannot read QWH9 tensor table: {error}"))?;
    for index in 0..usize::try_from(tensor_count).unwrap() {
        let entry = &entries[index * 41..(index + 1) * 41];
        let offset = read_header_u64(entry, 4)?;
        let bytes = read_header_u64(entry, 12)?;
        let ndim = entry[20];
        let ggml_type = read_header_u32(entry, 37)?;
        // The table stores the original GGML type IDs, not the compact
        // QWEN_DTYPE_WEIGHT_* IDs used by the generated DSP catalog.
        let supported_ggml_type = matches!(ggml_type, 0 | 2 | 3 | 8 | 13 | 14);
        if ndim == 0 || ndim > 4 || !supported_ggml_type || bytes > u32::MAX as u64 {
            return Err(format!("invalid QWH9 tensor table entry {index}"));
        }
        if offset.checked_add(bytes).is_none_or(|end| end > base_bytes) {
            return Err(format!(
                "QWH9 tensor table entry {index} exceeds base store"
            ));
        }
    }

    let sidecar_offset = data_start
        .checked_add(QWH9_BASE_BYTES)
        .ok_or_else(|| "QWH9 sidecar offset overflowed".to_string())?;
    let mut sidecar_header = [0_u8; 20];
    file.seek(SeekFrom::Start(sidecar_offset as u64))
        .map_err(|error| format!("cannot seek to QHM4 sidecar: {error}"))?;
    file.read_exact(&mut sidecar_header)
        .map_err(|error| format!("cannot read QHM4 sidecar header: {error}"))?;
    if &sidecar_header[0..4] != b"QHM4" {
        return Err("QWH9 export is missing its QHM4 HMX sidecar".to_string());
    }
    let sidecar_version = read_header_u32(&sidecar_header, 4)?;
    let sidecar_count = read_header_u32(&sidecar_header, 8)?;
    let hmx_bytes = read_header_u64(&sidecar_header, 12)?;
    if sidecar_version != QHM4_VERSION
        || sidecar_count != QHM4_TENSOR_COUNT
        || hmx_bytes != QHM4_BYTES as u64
    {
        return Err(format!(
            "unsupported QHM4 sidecar: version={sidecar_version}, tensors={sidecar_count}, bytes={hmx_bytes}"
        ));
    }
    let expected_file_bytes = sidecar_offset
        .checked_add(QHM4_BYTES)
        .ok_or_else(|| "QHM4 file size overflowed".to_string())?
        as u64;
    if file_bytes != expected_file_bytes {
        return Err(format!(
            "QWH9/QHM4 file size mismatch: expected {expected_file_bytes}, got {file_bytes}"
        ));
    }
    Ok(Qwen35Q4WeightLayout {
        file_bytes,
        base_bytes: QWH9_BASE_BYTES,
        data_start,
        sidecar_offset,
        hmx_bytes: QHM4_BYTES,
    })
}

fn read_header_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "integer offset overflowed".to_string())?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| "binary header is truncated".to_string())?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_header_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "integer offset overflowed".to_string())?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| "binary header is truncated".to_string())?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

const fn align128(value: usize) -> usize {
    (value + 127) & !127
}

/// One model-step custom operation. The first execution performs prefill from
/// the token input; subsequent executions perform one resident decode step.
#[derive(Debug, Clone)]
pub struct Qwen35Q4Step {
    prompt_len: Expression,
}

impl Qwen35Q4Step {
    pub fn new(prompt_len: Expression) -> Self {
        Self { prompt_len }
    }

    pub fn prompt_len(&self) -> Expression {
        self.prompt_len
    }
}

/// Add a resident Qwen3.5 Q4 model step to a Luminal graph.
///
/// `prompt` must be a one-dimensional `DType::Int` token-id input. The
/// returned tensor contains one `DType::Int` argmax token per execution.
pub fn qwen35_q4_step(
    prompt: impl Into<luminal::prelude::GraphTensor>,
) -> luminal::prelude::GraphTensor {
    let prompt = prompt.into();
    assert_eq!(
        prompt.dtype,
        DType::Int,
        "Qwen3.5 prompt must use DType::Int"
    );
    assert_eq!(
        prompt.shape.len(),
        1,
        "Qwen3.5 prompt must be one-dimensional"
    );
    let prompt_len = prompt.dims1();
    prompt
        .graph()
        .custom_op(Qwen35Q4Step::new(prompt_len), prompt, 1, DType::Int)
}

/// Dialect boundary for the resident operation. Matching/creation is explicit
/// in the graph's CustomOp; extraction preserves this op without a Rust-side
/// pattern rewrite.
pub trait Qwen35KernelOp: std::fmt::Debug {
    fn prompt_len(&self) -> Expression;
}

impl Qwen35KernelOp for Qwen35Q4Step {
    fn prompt_len(&self) -> Expression {
        self.prompt_len
    }
}

impl CustomOp for Qwen35Q4Step {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn Qwen35KernelOp>(Box::new(self.clone()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35Token {
    pub id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Copy)]
struct CompiledQwen {
    prompt_hlir: NodeIndex,
    output_hlir: NodeIndex,
    prompt_len: Expression,
}

/// Resident host session for the Qwen3.5 Q4 ABI-compatible DSP module.
struct Qwen35Session {
    session: FastRpcSession,
    base_weights: Option<DeviceBuffer>,
    hmx_weights: Option<DeviceBuffer>,
    runtime: Option<DeviceBuffer>,
    request: Option<DeviceBuffer>,
    response: Option<DeviceBuffer>,
    base_host_mapped: bool,
    base_dsp_mapped: bool,
    hmx_host_mapped: bool,
    hmx_dsp_mapped: bool,
    runtime_host_mapped: bool,
    runtime_dsp_mapped: bool,
    bound: bool,
    position: usize,
    cache_len: usize,
    generated_count: usize,
}

impl Qwen35Session {
    fn open(config: &Qwen35Q4Config) -> Result<Self, String> {
        let layout = validate_qwen35_q4_weights(&config.weights)?;
        let session = FastRpcSession::open(&config.rpc)?;
        let mut this = Self {
            session,
            base_weights: None,
            hmx_weights: None,
            runtime: None,
            request: None,
            response: None,
            base_host_mapped: false,
            base_dsp_mapped: false,
            hmx_host_mapped: false,
            hmx_dsp_mapped: false,
            runtime_host_mapped: false,
            runtime_dsp_mapped: false,
            bound: false,
            position: 0,
            cache_len: 0,
            generated_count: 0,
        };

        this.base_weights = Some(this.session.allocate(layout.base_bytes)?);
        this.hmx_weights = Some(this.session.allocate(layout.hmx_bytes)?);
        this.runtime = Some(this.session.allocate(QWEN_RUNTIME_BYTES)?);
        // Prefill is at most 32 + 256 * sizeof(i32), but leave room for a
        // future flag/debug extension without another allocation.
        this.request = Some(this.session.allocate(4096)?);
        this.response = Some(this.session.allocate(QWEN_RESPONSE_BYTES)?);

        load_range(
            &config.weights,
            layout.data_start as u64,
            this.base_weights.as_ref().unwrap(),
        )?;
        load_range(
            &config.weights,
            layout.sidecar_offset as u64,
            this.hmx_weights.as_ref().unwrap(),
        )?;
        zero_buffer(this.runtime.as_ref().unwrap());
        initialize_runtime_control(this.runtime.as_ref().unwrap());

        this.map_weight_store(OP_MMAP, OP_MUNMAP, true)?;
        this.map_weight_store(OP_MMAP_HMX, OP_MUNMAP_HMX, false)?;
        this.map_runtime()?;
        this.bind_runtime()?;
        Ok(this)
    }

    fn map_weight_store(
        &mut self,
        map_method: u32,
        _unmap_method: u32,
        base: bool,
    ) -> Result<(), String> {
        let buffer = if base {
            self.base_weights.as_ref().unwrap()
        } else {
            self.hmx_weights.as_ref().unwrap()
        };
        self.session.map_buffer(buffer)?;
        if base {
            self.base_host_mapped = true;
        } else {
            self.hmx_host_mapped = true;
        }
        let fd = u32::try_from(self.session.buffer_fd(buffer)?).unwrap();
        let bytes = u32::try_from(buffer.logical_bytes)
            .map_err(|_| "Qwen resident mapping exceeds the FastRPC u32 size ABI".to_string())?;
        self.session.invoke_scalars(map_method, &[fd, bytes])?;
        if base {
            self.base_dsp_mapped = true;
        } else {
            self.hmx_dsp_mapped = true;
        }
        Ok(())
    }

    fn map_runtime(&mut self) -> Result<(), String> {
        let buffer = self.runtime.as_ref().unwrap();
        self.session.map_buffer(buffer)?;
        self.runtime_host_mapped = true;
        let fd = u32::try_from(self.session.buffer_fd(buffer)?).unwrap();
        let bytes = u32::try_from(buffer.logical_bytes)
            .map_err(|_| "Qwen runtime buffer exceeds the FastRPC u32 size ABI".to_string())?;
        self.session.invoke_scalars(OP_MMAP_RUNTIME, &[fd, bytes])?;
        self.runtime_dsp_mapped = true;
        Ok(())
    }

    fn bind_runtime(&mut self) -> Result<(), String> {
        let request = self.request.as_ref().unwrap();
        let response = self.response.as_ref().unwrap();
        zero_buffer(response);
        write_u32(request, 0, OP_QWEN_DECODE_BIND);
        write_u32(request, 4, QWEN_RUNTIME_LAYOUT_VERSION);
        write_u32(request, 8, QWEN_RUNTIME_BYTES as u32);
        write_u32(request, 12, 0);
        self.invoke_compute(request, response)?;
        let ack = read_u32(response, 0)?;
        if ack != QWEN_RUNTIME_LAYOUT_VERSION {
            return Err(format!("Qwen runtime bind returned layout ack {ack}"));
        }
        self.bound = true;
        Ok(())
    }

    fn invoke_compute(
        &self,
        request: &DeviceBuffer,
        response: &DeviceBuffer,
    ) -> Result<(), String> {
        self.session.invoke_raw(OP_COMPUTE, request, response)
    }

    fn prefill(&mut self, input_ids: &[i32]) -> Result<Qwen35Token, String> {
        if input_ids.len() < 2 || input_ids.len() > MAX_PREFILL_TOKENS {
            return Err(format!(
                "Qwen3.5 Q4 prefill length must be in [2, {MAX_PREFILL_TOKENS}], got {}",
                input_ids.len()
            ));
        }
        if !self.bound {
            return Err("Qwen runtime was not bound before prefill".to_string());
        }
        let request = self.request.as_ref().unwrap();
        let response = self.response.as_ref().unwrap();
        let request_bytes = 32 + input_ids.len() * 4;
        if request_bytes > request.logical_bytes {
            return Err("Qwen prefill request exceeds resident request buffer".to_string());
        }
        write_u32(request, 0, OP_QWEN_FULL_PREFILL);
        write_u32(request, 4, QWEN_RUNTIME_LAYOUT_VERSION);
        write_u32(request, 8, QWEN_RUNTIME_BYTES as u32);
        write_u32(request, 12, input_ids.len() as u32);
        write_u32(request, 16, 0);
        write_u32(request, 20, u32::MAX);
        write_u32(request, 24, 0);
        write_u32(request, 28, 0);
        for (index, &token) in input_ids.iter().enumerate() {
            write_u32(request, 32 + index * 4, token as u32);
        }
        zero_buffer(response);
        let request_view = DeviceBufferView {
            ptr: request.ptr,
            logical_bytes: request_bytes,
        };
        self.invoke_compute(&request_view.as_buffer(), response)?;
        let status = read_i32(response, 20)?;
        if status != 0 {
            return Err(format!("Qwen HTP prefill returned status {status}"));
        }
        let token = Qwen35Token {
            id: read_u32(response, 0)?,
            logit: f32::from_bits(read_u32(response, 4)?),
        };
        if read_u32(response, 8)? != 0
            || read_u32(response, 12)? != input_ids.len() as u32
            || read_u32(response, 16)? != input_ids.len() as u32
        {
            return Err("Qwen HTP prefill returned an inconsistent position".to_string());
        }
        self.position = input_ids.len();
        self.cache_len = input_ids.len();
        self.generated_count = 1;
        Ok(token)
    }

    fn decode(&mut self) -> Result<Qwen35Token, String> {
        if self.cache_len == 0 {
            return Err("Qwen decode requested before prefill".to_string());
        }
        if self.cache_len >= MAX_DECODE_CONTEXT {
            return Err(format!("Qwen decode context reached {MAX_DECODE_CONTEXT}"));
        }
        let request = self.request.as_ref().unwrap();
        let response = self.response.as_ref().unwrap();
        write_u32(request, 0, OP_QWEN_FULL_DECODE_STEP);
        write_u32(request, 4, self.position as u32);
        write_u32(request, 8, self.generated_count as u32);
        write_u32(request, 12, 0);
        write_u32(request, 16, u32::MAX);
        write_u32(request, 20, 0);
        zero_buffer(response);
        let request_view = DeviceBufferView {
            ptr: request.ptr,
            logical_bytes: 24,
        };
        self.invoke_compute(&request_view.as_buffer(), response)?;
        let status = read_i32(response, 16)?;
        if status != 0 {
            return Err(format!("Qwen HTP decode returned status {status}"));
        }
        let token_index = read_u32(response, 8)? as usize;
        let completed_position = read_u32(response, 12)? as usize;
        if token_index != self.generated_count || completed_position != self.position {
            return Err(format!(
                "Qwen HTP decode position mismatch: token_index={token_index}, expected={}, completed_position={completed_position}, expected_position={}",
                self.generated_count, self.position
            ));
        }
        let token = Qwen35Token {
            id: read_u32(response, 0)?,
            logit: f32::from_bits(read_u32(response, 4)?),
        };
        self.position += 1;
        self.cache_len += 1;
        self.generated_count += 1;
        Ok(token)
    }
}

impl Drop for Qwen35Session {
    fn drop(&mut self) {
        // The resident DSP module owns worker/HMX resources that can outlive
        // the synchronous compute return. Drain it before removing mappings.
        // `hexinfer_stop` is an output-sequence ABI, not a scalar-only call.
        if self.bound
            && let Some(response) = self.response.as_ref()
        {
            zero_buffer(response);
            let _ = self.session.invoke_output(OP_STOP, response);
        }
        self.unmap_resident(
            self.runtime.as_ref(),
            self.runtime_dsp_mapped,
            self.runtime_host_mapped,
            OP_MUNMAP_RUNTIME,
        );
        self.unmap_resident(
            self.hmx_weights.as_ref(),
            self.hmx_dsp_mapped,
            self.hmx_host_mapped,
            OP_MUNMAP_HMX,
        );
        self.unmap_resident(
            self.base_weights.as_ref(),
            self.base_dsp_mapped,
            self.base_host_mapped,
            OP_MUNMAP,
        );
        for buffer in [
            self.response.take(),
            self.request.take(),
            self.runtime.take(),
            self.hmx_weights.take(),
            self.base_weights.take(),
        ]
        .into_iter()
        .flatten()
        {
            // SAFETY: each buffer is owned by this session and is released
            // once after the best-effort DSP/host unmap sequence.
            unsafe { self.session.free(buffer) };
        }
    }
}

impl Qwen35Session {
    fn unmap_resident(
        &self,
        buffer: Option<&DeviceBuffer>,
        dsp_mapped: bool,
        host_mapped: bool,
        method: u32,
    ) {
        let Some(buffer) = buffer else { return };
        if dsp_mapped && let Ok(fd) = self.session.buffer_fd(buffer) {
            let _ = self.session.invoke_scalars(method, &[fd as u32]);
        }
        if host_mapped {
            let _ = self.session.unmap_buffer(buffer);
        }
    }
}

// A short-lived view lets the resident RPC carry only the used prefix of the
// request buffer while retaining one rpcmem allocation for all steps.
struct DeviceBufferView {
    ptr: std::ptr::NonNull<std::ffi::c_void>,
    logical_bytes: usize,
}

impl DeviceBufferView {
    fn as_buffer(&self) -> DeviceBuffer {
        DeviceBuffer {
            ptr: self.ptr,
            logical_bytes: self.logical_bytes,
        }
    }
}

fn load_range(path: &Path, offset: u64, buffer: &DeviceBuffer) -> Result<(), String> {
    let mut file = File::open(path).map_err(|error| {
        format!(
            "cannot open {} for resident upload: {error}",
            path.display()
        )
    })?;
    file.seek(SeekFrom::Start(offset)).map_err(|error| {
        format!(
            "cannot seek {} for resident upload: {error}",
            path.display()
        )
    })?;
    // SAFETY: the slice is exactly the live rpcmem allocation owned by the
    // session and remains valid for the synchronous read.
    let destination = unsafe {
        slice::from_raw_parts_mut(buffer.ptr.as_ptr().cast::<u8>(), buffer.logical_bytes)
    };
    file.read_exact(destination).map_err(|error| {
        format!(
            "cannot upload {} bytes from {}: {error}",
            buffer.logical_bytes,
            path.display()
        )
    })
}

fn zero_buffer(buffer: &DeviceBuffer) {
    // SAFETY: every byte belongs to the live rpcmem allocation.
    unsafe { std::ptr::write_bytes(buffer.ptr.as_ptr().cast::<u8>(), 0, buffer.logical_bytes) };
}

fn initialize_runtime_control(buffer: &DeviceBuffer) {
    write_u32(buffer, CONTROL_MAGIC_OFFSET, QWEN_RUNTIME_MAGIC);
    write_u32(
        buffer,
        CONTROL_LAYOUT_VERSION_OFFSET,
        QWEN_RUNTIME_LAYOUT_VERSION,
    );
    write_u32(buffer, CONTROL_FLAGS_OFFSET, 0);
    write_u32(buffer, CONTROL_POISONED_OFFSET, 0);
    write_u32(buffer, CONTROL_POSITION_OFFSET, 0);
    write_u32(buffer, CONTROL_CACHE_LEN_OFFSET, 0);
    write_u32(buffer, CONTROL_GENERATED_COUNT_OFFSET, 0);
    write_u32(
        buffer,
        CONTROL_GENERATED_CAPACITY_OFFSET,
        MAX_DECODE_CONTEXT as u32,
    );
}

fn write_u32(buffer: &DeviceBuffer, offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    // SAFETY: callers use offsets from a checked ABI layout or request size.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.ptr.as_ptr().cast::<u8>().add(offset),
            bytes.len(),
        )
    };
}

fn read_u32(buffer: &DeviceBuffer, offset: usize) -> Result<u32, String> {
    // SAFETY: response offsets are fixed fields in the Qwen result structs.
    let bytes = unsafe { slice::from_raw_parts(buffer.ptr.as_ptr().cast::<u8>().add(offset), 4) };
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_i32(buffer: &DeviceBuffer, offset: usize) -> Result<i32, String> {
    Ok(read_u32(buffer, offset)? as i32)
}

/// Luminal runtime for the one-operation resident graph.
pub struct Qwen35Runtime {
    session: Qwen35Session,
    input_data: FxHashMap<NodeIndex, ReferenceData>,
    input_buffers: FxHashMap<NodeIndex, DeviceBuffer>,
    compiled: Option<CompiledQwen>,
    output_buffer: Option<DeviceBuffer>,
    dim_buckets: FxHashMap<luminal::prelude::Symbol, Vec<DimBucket>>,
    selected_schedule: Option<SelectedSchedule>,
    started: bool,
    last_token: Option<Qwen35Token>,
    last_execution_us: f64,
    max_prefill_tokens: usize,
}

impl Qwen35Runtime {
    pub fn try_initialize(config: Qwen35Q4Config) -> Result<Self, String> {
        if !(2..=MAX_PREFILL_TOKENS).contains(&config.max_prefill_tokens) {
            return Err(format!(
                "Qwen3.5 prefill limit must be in [2, {MAX_PREFILL_TOKENS}], got {}",
                config.max_prefill_tokens
            ));
        }
        Ok(Self {
            session: Qwen35Session::open(&config)?,
            input_data: FxHashMap::default(),
            input_buffers: FxHashMap::default(),
            compiled: None,
            output_buffer: None,
            dim_buckets: FxHashMap::default(),
            selected_schedule: None,
            started: false,
            last_token: None,
            last_execution_us: 0.0,
            max_prefill_tokens: config.max_prefill_tokens,
        })
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl Into<ReferenceData>) {
        let id = id.to_id();
        let data = data.into();
        self.input_data.insert(id, data.clone());
        if self
            .compiled
            .is_some_and(|compiled| compiled.prompt_hlir == id)
        {
            self.upload_prompt(id, &data);
        }
    }

    pub fn get_token(&self) -> Qwen35Token {
        self.last_token
            .expect("Qwen3.5 has not executed a prefill/decode step")
    }

    pub fn get_i32(&self, id: impl ToId) -> Vec<i32> {
        let id = id.to_id();
        let compiled = self
            .compiled
            .as_ref()
            .expect("Qwen3.5 runtime has not been compiled");
        assert_eq!(id, compiled.output_hlir, "unknown Qwen3.5 output id");
        let buffer = self.output_buffer.as_ref().unwrap();
        vec![read_i32(buffer, 0).unwrap()]
    }

    fn upload_prompt(&mut self, id: NodeIndex, data: &ReferenceData) {
        let buffer_bytes = crate::runtime::reference_bytes(data, DType::Int);
        let needs_allocation = self
            .input_buffers
            .get(&id)
            .is_none_or(|buffer| buffer.logical_bytes != buffer_bytes.len());
        if needs_allocation {
            if let Some(buffer) = self.input_buffers.remove(&id) {
                // SAFETY: the old prompt buffer is no longer reachable.
                unsafe { self.session.session.free(buffer) };
            }
            self.input_buffers.insert(
                id,
                self.session
                    .session
                    .allocate(buffer_bytes.len())
                    .unwrap_or_else(|error| panic!("failed to allocate Qwen prompt: {error}")),
            );
        }
        let buffer = self.input_buffers.get(&id).unwrap();
        // SAFETY: the allocation is exactly large enough for the serialized
        // prompt bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer_bytes.as_ptr(),
                buffer.ptr.as_ptr().cast::<u8>(),
                buffer_bytes.len(),
            )
        };
    }

    fn refresh_prompt(&mut self) {
        let Some(compiled) = self.compiled else {
            return;
        };
        if let Some(data) = self.input_data.get(&compiled.prompt_hlir).cloned() {
            self.upload_prompt(compiled.prompt_hlir, &data);
        }
    }

    fn compile_graph(&self, graph: &LLIRGraph) -> CompiledQwen {
        let topo = toposort(graph, None).expect("Qwen3.5 LLIR graph has a cycle");
        let mut llir_to_hlir = FxHashMap::default();
        let mut step_node = None;
        let mut prompt_hlir = None;
        let mut step_prompt_len = None;
        let mut output_hlir = None;
        let mut output_source = None;
        for &node in &topo {
            if let Some(input) = graph[node].to_op::<Input>() {
                llir_to_hlir.insert(node, NodeIndex::new(input.node));
                continue;
            }
            if let Some(output) = graph[node].to_op::<Output>() {
                let source = graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|edge| edge.id())
                    .next()
                    .map(|edge| edge.source())
                    .expect("Qwen3.5 output has no source");
                output_hlir = Some(NodeIndex::new(output.node));
                output_source = Some(source);
                continue;
            }
            let op = graph[node]
                .to_dialect::<dyn Qwen35KernelOp>()
                .unwrap_or_else(|| {
                    panic!("unrecognized operation in Qwen3.5 resident graph at {node:?}")
                });
            assert!(
                step_node.is_none(),
                "Qwen3.5 resident graph must contain one model step"
            );
            let input_nodes: Vec<_> = graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|edge| edge.id())
                .map(|edge| edge.source())
                .collect();
            assert_eq!(
                input_nodes.len(),
                1,
                "Qwen3.5 step must have one prompt input"
            );
            prompt_hlir = Some(
                *llir_to_hlir
                    .get(&input_nodes[0])
                    .expect("Qwen3.5 prompt input was not an Input node"),
            );
            step_node = Some(node);
            step_prompt_len = Some(op.prompt_len());
        }
        let step_node = step_node.expect("Qwen3.5 graph has no resident model step");
        assert_eq!(
            output_source,
            Some(step_node),
            "Qwen3.5 output does not reference the model step"
        );
        CompiledQwen {
            prompt_hlir: prompt_hlir.expect("Qwen3.5 graph has no prompt input"),
            output_hlir: output_hlir.expect("Qwen3.5 graph has no output marker"),
            prompt_len: step_prompt_len.expect("Qwen3.5 graph has no prompt length"),
        }
    }

    fn clear_compiled(&mut self) {
        self.compiled = None;
        if let Some(buffer) = self.output_buffer.take() {
            // SAFETY: output buffer is owned by this runtime.
            unsafe { self.session.session.free(buffer) };
        }
        for buffer in self.input_buffers.drain().map(|(_, buffer)| buffer) {
            // SAFETY: prompt buffers are owned by this runtime.
            unsafe { self.session.session.free(buffer) };
        }
    }
}

impl Drop for Qwen35Runtime {
    fn drop(&mut self) {
        self.clear_compiled();
    }
}

impl Runtime for Qwen35Runtime {
    type Ops = ();
    type CompileArg = Qwen35Q4Config;
    type ExecReturn = Qwen35Token;

    fn initialize(config: Self::CompileArg) -> Self {
        Self::try_initialize(config).unwrap_or_else(|error| panic!("{error}"))
    }

    fn compile(
        &mut self,
        space: &luminal::search::SearchSpace,
        dyn_map: &DynMap,
        _options: &CompileOptions,
        rng: &mut dyn luminal::prelude::RngCore,
    ) {
        let contexts = space.bucket_contexts(dyn_map);
        assert_eq!(
            contexts.len(),
            1,
            "Qwen3.5 runtime currently uses one prompt bucket"
        );
        let selected = contexts
            .iter()
            .map(|context| luminal::search::extract_one_selected(space, context, rng))
            .collect_vec();
        self.selected_schedule = SelectedSchedule::from_search(space, &selected);
        self.load_llir_buckets(
            &space.dim_buckets,
            &[selected.into_iter().next().unwrap().into_bucket_llir()],
        );
    }

    fn selected_schedule(&self) -> Option<SelectedSchedule> {
        self.selected_schedule.clone()
    }

    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.load_llir_buckets(
            &FxHashMap::default(),
            &[(DynMap::default(), DynMap::default(), llir_graph.clone())],
        );
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<luminal::prelude::Symbol, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        assert_eq!(
            bucket_llirs.len(),
            1,
            "Qwen3.5 runtime currently uses one prompt bucket"
        );
        self.clear_compiled();
        self.dim_buckets = dim_buckets.clone();
        let compiled = self.compile_graph(&bucket_llirs[0].2);
        let prompt_len = compiled.prompt_len;
        self.output_buffer = Some(
            self.session
                .session
                .allocate(std::mem::size_of::<i32>())
                .unwrap_or_else(|error| panic!("failed to allocate Qwen token output: {error}")),
        );
        self.compiled = Some(compiled);
        self.started = false;
        self.last_token = None;
        self.refresh_prompt();
        if prompt_len
            .exec(&DynMap::default())
            .is_some_and(|len| len > self.max_prefill_tokens)
        {
            panic!(
                "Qwen3.5 graph prompt length exceeds configured limit {}",
                self.max_prefill_tokens
            );
        }
    }

    fn execute(&mut self, dyn_map: &DynMap) -> Self::ExecReturn {
        let started = Instant::now();
        let compiled = self
            .compiled
            .expect("Qwen3.5 runtime has not been compiled");
        if !self.started {
            let prompt_len = compiled
                .prompt_len
                .exec(dyn_map)
                .expect("Qwen3.5 prompt length is not resolved");
            assert!(
                prompt_len >= 2 && prompt_len <= self.max_prefill_tokens,
                "Qwen3.5 prompt length must be in [2, {}], got {prompt_len}",
                self.max_prefill_tokens
            );
            let data = self
                .input_data
                .get(&compiled.prompt_hlir)
                .unwrap_or_else(|| {
                    panic!(
                        "Qwen3.5 prompt input {:?} was not set",
                        compiled.prompt_hlir
                    )
                });
            let input_ids = data.to_i32_vec();
            assert_eq!(
                input_ids.len(),
                prompt_len,
                "Qwen3.5 prompt data/shape mismatch"
            );
            let token = self
                .session
                .prefill(&input_ids)
                .unwrap_or_else(|error| panic!("{error}"));
            self.started = true;
            self.last_token = Some(token);
        } else {
            let token = self
                .session
                .decode()
                .unwrap_or_else(|error| panic!("{error}"));
            self.last_token = Some(token);
        }
        let token = self.last_token.unwrap();
        write_u32(self.output_buffer.as_ref().unwrap(), 0, token.id);
        self.last_execution_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        token
    }
}

impl RuntimeStats for Qwen35Runtime {
    fn execute_with_stats(&mut self, dyn_map: &DynMap) -> Option<ExecutionStats> {
        self.execute(dyn_map);
        Some(ExecutionStats::with_timing_method(
            self.last_execution_us,
            0,
            std::mem::size_of::<i32>(),
            0,
            TimingMethod::WallClock,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::prelude::rand::SeedableRng;
    use luminal::prelude::{CompileOptions, Graph, rand};

    #[test]
    fn qwen35_architecture_matches_selected_checkpoint() {
        assert_eq!(NUM_LAYERS, 24);
        assert_eq!(FULL_ATTENTION_LAYERS, [3, 7, 11, 15, 19, 23]);
        assert_eq!(NUM_LINEAR_ATTENTION_LAYERS, 18);
        assert_eq!(QWEN_RUNTIME_BYTES, 211_195_648);
    }

    #[test]
    fn raw_gguf_is_rejected_before_allocation() {
        let error = parse_qwh9_header(b"GGUF\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0").unwrap_err();
        assert!(error.contains("raw GGUF"));
    }

    #[test]
    fn qwen_step_survives_luminal_extraction() {
        let mut graph = Graph::new();
        let prompt = graph.tensor(8).as_dtype(DType::Int);
        let output = qwen35_q4_step(prompt).output();
        graph.build_search_space::<Qwen35Runtime>(CompileOptions::default());
        let space = graph.search_space().expect("Qwen search space missing");
        let contexts = space.bucket_contexts(&graph.dyn_map);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5157_454e_3335);
        let selected = luminal::search::extract_one_selected(space, &contexts[0], &mut rng);
        assert!(selected.llir.node_indices().any(|node| {
            selected.llir[node]
                .to_dialect::<dyn Qwen35KernelOp>()
                .is_some()
        }));
        assert_eq!(output.dtype, DType::Int);
        assert_eq!(output.dims1(), Expression::from(1));
    }
}
