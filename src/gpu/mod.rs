mod error;

pub use self::error::*;

#[cfg(feature = "gpu")]
mod locks;

#[cfg(feature = "gpu")]
pub use self::locks::*;

#[cfg(feature = "gpu")]
mod sources;

#[cfg(feature = "gpu")]
pub use self::sources::*;

#[cfg(feature = "gpu")]
mod utils;

#[cfg(feature = "gpu")]
pub use self::utils::*;

#[cfg(feature = "gpu")]
mod fft;

#[cfg(feature = "gpu")]
pub use self::fft::*;

#[cfg(feature = "gpu")]
mod multiexp;

#[cfg(feature = "gpu")]
pub use self::multiexp::*;

#[cfg(not(feature = "gpu"))]
mod nogpu;

#[cfg(not(feature = "gpu"))]
pub use self::nogpu::*;

#[cfg(feature = "gpu")]
pub use ec_gpu::GpuEngine;
#[cfg(not(feature = "gpu"))]
pub trait GpuEngine {}
#[cfg(not(feature = "gpu"))]
impl<E: pairing::Engine> GpuEngine for E {}
