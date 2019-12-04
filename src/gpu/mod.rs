mod error;
pub use self::error::*;

#[cfg(feature = "gpu")]
mod sources;
#[cfg(feature = "gpu")]
pub use self::sources::*;

#[cfg(feature = "gpu")]
mod utils;
#[cfg(feature = "gpu")]
pub use self::utils::*;

#[cfg(feature = "gpu")]
mod structs;
#[cfg(feature = "gpu")]
pub use self::structs::*;

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
use ocl::Device;
#[cfg(feature = "gpu")]
lazy_static::lazy_static! {
    pub static ref GPU_NVIDIA_DEVICES: Vec<Device> = get_devices(GPU_NVIDIA_PLATFORM_NAME).unwrap_or_default();
}

#[derive(Debug)]
pub struct LockedFile(File);

use std::fs::File;

use std::io;
pub const LOCK_NAME: &str = "/tmp/bellman.lock";
pub fn lock() -> io::Result<LockedFile> {
    let file = File::create(LOCK_NAME)?;

    #[cfg(feature = "gpu")]
    {
        use fs2::FileExt;
        file.lock_exclusive()?;
    }

    Ok(LockedFile(file))
}

pub fn unlock(lock: LockedFile) {
    drop(lock);
}
