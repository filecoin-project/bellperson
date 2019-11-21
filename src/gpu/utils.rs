use crate::gpu::error::{GPUError, GPUResult};
use ocl::{Device, Platform};

use std::collections::HashMap;
use std::env;

pub const GPU_NVIDIA_PLATFORM_NAME: &str = "NVIDIA CUDA";
// pub const CPU_INTEL_PLATFORM_NAME: &str = "Intel(R) CPU Runtime for OpenCL(TM) Applications";

pub fn get_devices(platform_name: &str) -> GPUResult<Vec<Device>> {
    if env::var("BELLMAN_NO_GPU").is_ok() {
        return Err(GPUError {
            msg: "GPU accelerator is disabled!".to_string(),
        });
    }
    
    for platform in Platform::list()? {
        let name = platform.name()?;
        if name == platform_name {
            debug!("GPU Platform {:?} is supported.", name);
            return Ok(Device::list_all(platform)?)
        } else {
            debug!("GPU Platform {:?} is not supported.", name);
        }
    }
    
    return Err(GPUError { msg: "No working GPUs found!".to_string() });
}

lazy_static::lazy_static! {
    static ref CORE_COUNTS: HashMap<&'static str, usize> = vec![
        ("GeForce RTX 2080 Ti", 4352),
        ("GeForce RTX 2080 SUPER", 3072),
        ("GeForce RTX 2080", 2944),
        ("GeForce GTX 1080 Ti", 3584),
        ("GeForce GTX 1080", 2560),
        ("GeForce GTX 1060", 1280),
    ]
    .into_iter()
    .collect();
}

pub fn get_core_count(d: Device) -> GPUResult<usize> {
    match CORE_COUNTS.get(&d.name()?[..]) {
        Some(&cores) => Ok(cores),
        None => Err(GPUError {
            msg: "Device unknown!".to_string(),
        }),
    }
}
