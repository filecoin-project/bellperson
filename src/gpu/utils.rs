use std::env;

use log::{info, warn};
use rust_gpu_tools::{Device, CUDA_CORES};

const DEFAULT_CORE_COUNT: usize = 2560;
pub fn get_core_count(name: &str) -> usize {
    // Determining the number of cores was moved to rust-gpu-tools, which uses the
    // `RUST_GPU_TOOLS_CUSTOM_GPU` environment variable to set custom GPUs. Users should upgrade
    // using that one instead. Though using `BELLMAN_CUSTOM_GPU` is still supported for backwards
    // compatibility, but will be ignored if `RUST_GPU_TOOLS_CUSTOM_GPU` is also set.
    // Setting `RUST_GPU_TOOLS_CUSTOM_GPU` must happen before the first call to `CUDA_CORES`, as
    // it will be initialized only once for the lifetime of the library.
    if let Ok(custom_gpu) = env::var("BELLMAN_CUSTOM_GPU") {
        match env::var("RUST_GPU_TOOLS_CUSTOM_GPU") {
            Ok(_) => {
                info!("`BELLMAN_CUSTOM_GPU` was ignored as `RUST_GPU_TOOLS_CUSTOM_GPU` is set.");
            }
            Err(_) => {
                info!(
                    "Please use `RUST_GPU_TOOLS_CUSTOM_GPU` instead of `BELLMAN_CUSTOM_GPU`, \
                     their values are fully compatible."
                );
                env::set_var("RUST_GPU_TOOLS_CUSTOM_GPU", custom_gpu)
            }
        }
    }
    match CUDA_CORES.get(name) {
        Some(&cores) => cores,
        None => {
            warn!(
                "Number of CUDA cores for your device ({}) is unknown! Best performance is \
                 only achieved when the number of CUDA cores is known! You can find the \
                 instructions on how to support custom GPUs here: \
                 https://lotu.sh/en+hardware-mining",
                name
            );
            DEFAULT_CORE_COUNT
        }
    }
}

pub fn dump_device_list() {
    for d in Device::all() {
        info!("Device: {:?}", d);
    }
}
