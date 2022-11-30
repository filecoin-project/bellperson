use ec_gpu_gen::EcError;

#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("GPUError: {0}")]
    Simple(&'static str),
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    #[error("No kernel is initialized!")]
    KernelUninitialized,
    #[error("EC GPU error: {0}")]
    EcGpu(#[from] EcError),
    #[error("GPU accelerator is disabled!")]
    GpuDisabled,
}

pub type GpuResult<T> = std::result::Result<T, GpuError>;
