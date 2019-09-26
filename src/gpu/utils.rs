use ocl::{Device, Platform};
use super::error::{GPUResult, GPUError};

pub const GPU_NVIDIA_PLATFORM_NAME : &str = "NVIDIA CUDA";
pub const CPU_INTEL_PLATFORM_NAME : &str = "Intel(R) CPU Runtime for OpenCL(TM) Applications";

pub fn get_devices(platform_name: &str) -> GPUResult<Vec<Device>> {
    let nvidia = Platform::list().into_iter().find(|&p|
        match p.name() {
            Ok(p) => p == platform_name,
            Err(e) => false
        }).expect("No device found!");
    let devices = Device::list_all(nvidia)?;
    return Ok(devices);
}
