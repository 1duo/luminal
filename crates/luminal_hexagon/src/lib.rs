//! Hexagon SDK backend for Luminal.
//!
//! The backend deliberately keeps the host/device contract small:
//!
//! * Rust lowers supported Luminal operations through egglog into the
//!   `Hexagon*` dialect.
//! * The Hexagon SDK builds the v73 DSP shared object in `device/`.
//! * FastRPC dispatches that object and `rpcmem` buffers keep intermediate
//!   tensors shared across calls.
//!
//! The initial operation set is contiguous F32 elementwise Add and Mul. It is
//! intentionally narrow so the memory and ABI invariants are easy to verify
//! on a real Snapdragon X Elite before adding larger kernels.

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
