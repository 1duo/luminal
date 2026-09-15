//! Hexagon SDK backend for Luminal.
//!
//! The backend deliberately keeps the host/device contract small:
//!
//! * Rust lowers supported Luminal operations through egglog into the
//!   `Hexagon*` dialect.
//! * The Hexagon SDK builds the v73 DSP shared object in `device/`.
//! * FastRPC dispatches that object and `rpcmem` buffers keep intermediate
//!   tensors shared across calls.
//! * I8 projection GEMM uses signed byte inputs and signed I32 accumulation;
//!   scalar and HVX schedules remain ordinary e-graph alternatives.
//!
//! The SDK installed on the development host has no public HexKL compiler or
//! library. `codegen` is therefore a narrow, replaceable SDK C/HVX emitter;
//! an official HexKL backend can plug into that boundary without moving
//! pattern matching or search out of egglog.
//!
//! The operation set is intentionally narrow so the memory, dtype, and ABI
//! invariants are easy to verify on a real Snapdragon X Elite before adding
//! larger kernels.

pub mod codegen;
pub mod config;
pub mod dyn_backend;
pub mod kernel;
pub mod runtime;

pub use codegen::{BinaryOp, emit_dsp_source};
pub use config::HexagonConfig;
pub use dyn_backend::{HexagonDynBackend, hexagon_factory};
pub use kernel::HexagonOps;
pub use runtime::HexagonRuntime;

#[cfg(test)]
mod tests;
