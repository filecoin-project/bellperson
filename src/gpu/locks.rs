use std::fs::File;
use std::path::PathBuf;

use ec_gpu_gen::fft::FftKernel;
use ec_gpu_gen::rust_gpu_tools::Device;
use ff::Field;
use fs2::FileExt;
use group::prime::PrimeCurveAffine;
use log::{debug, info, warn};

use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::{CpuGpuMultiexpKernel, GpuName};

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";
fn tmp_path(filename: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(filename);
    p
}

/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct GPULock(File);
impl GPULock {
    pub fn lock() -> GPULock {
        let gpu_lock_file = tmp_path(GPU_LOCK_NAME);
        debug!("Acquiring GPU lock at {:?} ...", &gpu_lock_file);
        let f = File::create(&gpu_lock_file)
            .unwrap_or_else(|_| panic!("Cannot create GPU lock file at {:?}", &gpu_lock_file));
        f.lock_exclusive().unwrap();
        debug!("GPU lock acquired!");
        GPULock(f)
    }
}
impl Drop for GPULock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("GPU lock released!");
    }
}

/// `PrioriyLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        let priority_lock_file = tmp_path(PRIORITY_LOCK_NAME);
        debug!("Acquiring priority lock at {:?} ...", &priority_lock_file);
        let f = File::create(&priority_lock_file).unwrap_or_else(|_| {
            panic!(
                "Cannot create priority lock file at {:?}",
                &priority_lock_file
            )
        });
        f.lock_exclusive().unwrap();
        debug!("Priority lock acquired!");
        PriorityLock(f)
    }

    pub fn wait(priority: bool) {
        if !priority {
            if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .lock_exclusive()
            {
                warn!("failed to create priority log: {:?}", err);
            }
        }
    }

    pub fn should_break(priority: bool) -> bool {
        if priority {
            return false;
        }
        if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME))
            .unwrap()
            .try_lock_shared()
        {
            // Check that the error is actually a locking one
            if err.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
                return true;
            } else {
                warn!("failed to check lock: {:?}", err);
            }
        }
        false
    }
}

impl Drop for PriorityLock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("Priority lock released!");
    }
}

fn create_fft_kernel<'a, F>(priority: bool) -> Option<FftKernel<'a, F>>
where
    F: Field + GpuName,
{
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .ok()?;

    let kernel = if priority {
        FftKernel::create_with_abort(programs, &|| -> bool {
            // We only supply a function in case it is high priority, hence always passing in
            // `true`.
            PriorityLock::should_break(true)
        })
    } else {
        FftKernel::create(programs)
    };
    match kernel {
        Ok(k) => {
            info!("GPU FFT kernel instantiated!");
            Some(k)
        }
        Err(e) => {
            warn!("Cannot instantiate GPU FFT kernel! Error: {}", e);
            None
        }
    }
}

fn create_multiexp_kernel<'a, G>(priority: bool) -> Option<CpuGpuMultiexpKernel<'a, G>>
where
    G: PrimeCurveAffine + GpuName,
{
    let devices = Device::all();
    let kernel = if priority {
        CpuGpuMultiexpKernel::create_with_abort(&devices, &|| -> bool {
            // We only supply a function in case it is high priority, hence always passing in
            // `true`.
            PriorityLock::should_break(true)
        })
    } else {
        CpuGpuMultiexpKernel::create(&devices)
    };
    match kernel {
        Ok(k) => {
            info!("GPU Multiexp kernel instantiated!");
            Some(k)
        }
        Err(e) => {
            warn!("Cannot instantiate GPU Multiexp kernel! Error: {}", e);
            None
        }
    }
}

/// Wrap the kernel so that only a single one runs on the GPU at a time.
macro_rules! locked_kernel {
    ($kernel:ty, $func:ident, $name:expr, pub struct $class:ident<$lifetime:lifetime, $generic:ident>
        where $(
            $bound:ty: $boundvalue:tt $(+ $morebounds:tt )*,
        )+
    ) => {
        pub struct $class<$lifetime, $generic>
        where $(
            $bound : $boundvalue $(+ $morebounds)*,
        )+
        {
            priority: bool,
            // Keep the GPU lock alongside the kernel, so that the lock is automatically dropped
            // if the kernel is dropped.
            kernel_and_lock: Option<($kernel, GPULock)>,
        }

        impl<'a, $generic> $class<$lifetime, $generic>
        where $(
            $bound: $boundvalue $(+ $morebounds)*,
        )+
        {
            pub fn new(priority: bool) -> Self {
                Self {
                    priority,
                    kernel_and_lock: None,
                }
            }

            /// Intialize a kernel.
            ///
            /// On OpenCL that also means that the kernel source is compiled.
            fn init(&mut self) {
                if self.kernel_and_lock.is_none() {
                    PriorityLock::wait(self.priority);
                    info!("GPU is available for {}!", $name);
                    if let Some(kernel) = $func(self.priority) {
                        self.kernel_and_lock = Some((kernel, GPULock::lock()));
                    }
                }
            }

            /// Free kernel resources early.
            ///
            /// When the locked kernel is dropped, it will free the resources automatically. In
            /// case we are waiting for the GPU to be used, we free those resources early.
            fn free(&mut self) {
                if let Some(_) = self.kernel_and_lock.take() {
                    warn!(
                        "GPU acquired by a high priority process! Freeing up {} kernels...",
                        $name
                    );
                }
            }

            /// Execute a function with the kernel.
            ///
            /// This function makes sure that only one things is run on the GPU at a time. It will
            /// block until the GPU is available.
            pub fn with<Fun, R>(&mut self, mut f: Fun) -> GpuResult<R>
            where
                Fun: FnMut(&mut $kernel) -> GpuResult<R>,
            {
                if std::env::var("BELLMAN_NO_GPU").is_ok() {
                    return Err(GpuError::GpuDisabled);
                }

                loop {
                    // `init()` is a possibly blocking call that waits until the GPU is available.
                    self.init();
                    if let Some((ref mut k, ref _gpu_lock)) = self.kernel_and_lock {
                        match f(k) {
                            // Re-trying to run on the GPU is the core of this loop, all other
                            // cases abort the loop.
                            Err(GpuError::GpuTaken) => {
                                self.free();
                            }
                            Err(e) => {
                                warn!("GPU {} failed! Falling back to CPU... Error: {}", $name, e);
                                return Err(e);
                            }
                            Ok(v) => return Ok(v),
                        }
                    } else {
                        return Err(GpuError::KernelUninitialized);
                    }
                }
            }
        }
    };
}

locked_kernel!(
    FftKernel<'a, F>,
    create_fft_kernel,
    "FFT",
    pub struct LockedFftKernel<'a, F> where F: Field + GpuName,
);
locked_kernel!(
    CpuGpuMultiexpKernel<'a, G>,
    create_multiexp_kernel,
    "Multiexp",
    pub struct LockedMultiexpKernel<'a, G>
    where
        G: PrimeCurveAffine + GpuName,
);
