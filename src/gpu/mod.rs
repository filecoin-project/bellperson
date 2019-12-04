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

use std::fs::File;
pub const LOCK_NAME: &str = "/tmp/bellman.lock";
pub fn lock() -> File {
    let file = File::create(LOCK_NAME).unwrap();

    #[cfg(feature = "gpu")]
    {
        use std::os::unix::io::AsRawFd;
        use nix::fcntl::{flock, FlockArg};
        let fd = file.as_raw_fd();
        flock(fd, FlockArg::LockExclusive).unwrap();
    }

    file
}

pub fn unlock(lock: File) {
    drop(lock);
}
