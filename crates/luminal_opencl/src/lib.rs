//! OpenCL backend for Luminal.
//!
//! The first-class target is the Qualcomm Adreno GPU in Snapdragon X Elite
//! systems. The implementation only relies on the system OpenCL driver, so it
//! does not require the Qualcomm AI Engine Direct SDK.

pub mod device;
pub mod dyn_backend;
pub mod kernel;
pub mod runtime;

pub use device::{OpenClDeviceInfo, available_devices};
pub use kernel::OpenClOps;
pub use runtime::OpenClRuntime;

#[cfg(test)]
mod tests;
