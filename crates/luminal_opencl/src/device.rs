use opencl3::{
    device::{CL_DEVICE_TYPE_GPU, Device},
    platform::get_platforms,
    types::cl_device_id,
};

/// One GPU exposed by the system OpenCL loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenClDeviceInfo {
    /// Logical index accepted by [`crate::OpenClRuntime::try_initialize`].
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub platform: String,
    pub supports_fp16: bool,
}

#[derive(Clone)]
pub(crate) struct OpenClDevice {
    pub info: OpenClDeviceInfo,
    pub device: Device,
}

fn preference(device: &OpenClDevice) -> (u8, String, String) {
    let vendor = device.info.vendor.to_ascii_lowercase();
    let name = device.info.name.to_ascii_lowercase();
    let rank = if vendor.contains("qualcomm") && name.contains("adreno") {
        0
    } else if vendor.contains("qualcomm") {
        1
    } else {
        2
    };
    (rank, vendor, name)
}

pub(crate) fn enumerate_devices() -> Result<Vec<OpenClDevice>, String> {
    let mut devices = Vec::new();
    for platform in get_platforms().map_err(|e| format!("OpenCL platform query failed: {e}"))? {
        let platform_name = platform
            .name()
            .unwrap_or_else(|_| "unknown OpenCL platform".to_string());
        let ids: Vec<cl_device_id> = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
        for id in ids {
            let device = Device::new(id);
            let name = device
                .name()
                .unwrap_or_else(|_| "unknown OpenCL GPU".to_string());
            let vendor = device
                .vendor()
                .unwrap_or_else(|_| "unknown vendor".to_string());
            let extensions = device.extensions().unwrap_or_default();
            devices.push(OpenClDevice {
                info: OpenClDeviceInfo {
                    index: 0,
                    name,
                    vendor,
                    platform: platform_name.clone(),
                    supports_fp16: extensions
                        .split_ascii_whitespace()
                        .any(|ext| ext == "cl_khr_fp16"),
                },
                device,
            });
        }
    }

    // Windows on Snapdragon may expose the Adreno twice: once through the
    // native Qualcomm ICD and once through Microsoft's OpenCL-on-D3D12 layer.
    // Prefer the native driver because it exposes FP16 and Qualcomm extensions.
    devices.sort_by_key(preference);
    for (index, device) in devices.iter_mut().enumerate() {
        device.info.index = index;
    }
    Ok(devices)
}

/// Enumerate usable OpenCL GPUs, with the native Qualcomm Adreno ICD first.
pub fn available_devices() -> Result<Vec<OpenClDeviceInfo>, String> {
    enumerate_devices().map(|devices| devices.into_iter().map(|d| d.info).collect())
}
