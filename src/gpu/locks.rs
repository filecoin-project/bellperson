use std::fs::File;
use std::path::PathBuf;

use ec_gpu_gen::fft::FftKernel;
use ec_gpu_gen::rust_gpu_tools::{Device, UniqueId};
use ec_gpu_gen::EcError;
use ff::Field;
use fs2::FileExt;
use group::prime::PrimeCurveAffine;
use log::{debug, info, warn};

use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::{CpuGpuMultiexpKernel, GpuName};

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";

fn tmp_path(filename: &str, id: Option<UniqueId>) -> PathBuf {
    let mut p = std::env::temp_dir();
    let mut tmpfile = filename.to_owned();
    if let Some(id_str) = id {
        tmpfile.push_str(&(".".to_owned() + &id_str.to_string()));
    }
    p.push(tmpfile);
    p
}

#[derive(Debug)]
struct LockInfo<'a> {
    file_path: Option<(File, PathBuf)>,
    devices: Vec<&'a Device>,
}

/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct GPULock<'a>(Vec<LockInfo<'a>>);

impl GPULock<'_> {
    pub fn lock() -> Self {
        let devices = Device::all();

        if let Ok(val) = std::env::var("BELLPERSON_GPUS_PER_LOCK") {
            match val.parse::<usize>() {
                Ok(val) if val > 0 => {
                    info!(
                        "BELLPERSON_GPUS_PER_LOCK == {}, try lock {}/{} gpus",
                        val,
                        val,
                        devices.len(),
                    );

                    let mut locks = Vec::new();
                    for (index, device) in devices.iter().enumerate() {
                        let uid = device.unique_id();
                        let path = tmp_path(GPU_LOCK_NAME, Some(uid));
                        debug!("Acquiring GPU lock {}/{} at {:?} ...", index, val, &path);
                        let file = File::create(&path).unwrap_or_else(|_| {
                            panic!("Cannot create GPU {:?} lock file at {:?}", uid, &path)
                        });
                        if file.try_lock_exclusive().is_err() {
                            continue;
                        }
                        debug!("GPU lock acquired at {:?}", path);
                        locks.push(LockInfo {
                            file_path: Some((file, path)),
                            devices: vec![device],
                        });
                        if locks.len() >= val {
                            break;
                        }
                    }

                    return GPULock(locks);
                }
                Ok(val) if val == 0 => {
                    info!("BELLPERSON_GPUS_PER_LOCK == 0, no lock acquired");
                    return GPULock(vec![LockInfo {
                        file_path: None,
                        devices,
                    }]);
                }
                Ok(val) => warn!(
                    "BELLPERSON_GPUS_PER_LOCK has invalid value {}, using all gpus",
                    val,
                ),
                Err(_) => warn!("BELLPERSON_GPUS_PER_LOCK parsing failed, using all gpus"),
            };
        }

        info!("BELLPERSON_GPUS_PER_LOCK fallback to single lock mode");

        // Fallback to create single lock
        let path = tmp_path(GPU_LOCK_NAME, None);
        debug!("Acquiring GPU lock at {:?} ...", &path);
        let file = File::create(&path).unwrap_or_else(|_| {
            panic!("Cannot create GPU lock file at {:?}", &path);
        });
        file.lock_exclusive().unwrap();
        debug!("GPU lock acquired at {:?}", path);
        GPULock(vec![LockInfo {
            file_path: Some((file, path)),
            devices,
        }])
    }
}

impl Drop for GPULock<'_> {
    fn drop(&mut self) {
        for lock_info in &self.0 {
            if let Some((file, path)) = &lock_info.file_path {
                file.unlock().unwrap();
                debug!("GPU lock released at {:?}", path);
            }
        }
    }
}

/// `PrioriyLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub(crate) struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        let priority_lock_file = tmp_path(PRIORITY_LOCK_NAME, None);
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

    fn wait(priority: bool) {
        if !priority {
            if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME, None))
                .unwrap()
                .lock_exclusive()
            {
                warn!("failed to create priority log: {:?}", err);
            }
        }
    }

    /// Returns true if the priority lock is currently taken.
    ///
    /// This is used by low priority proofs to determine whether to run on the GPU or not.
    ///
    /// It also returns `false` in case the state of the lock cannot be determined.
    fn is_taken() -> bool {
        if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME, None))
            .unwrap()
            .try_lock_shared()
        {
            // Check that the error is actually a locking one
            if err.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
                return true;
            }
            warn!("failed to check lock: {:?}", err);
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

fn create_fft_kernel<'a, F>(priority: bool) -> Option<(FftKernel<'a, F>, GPULock<'a>)>
where
    F: Field + GpuName,
{
    let lock = GPULock::lock();
    let mut devices = Vec::new();
    for lock_info in &lock.0 {
        for device in &lock_info.devices {
            devices.push(*device);
        }
    }

    /*
    let devices = lock
        .0
        .iter()
        .map(|LockInfo { devices, .. }| devices)
        .collect::<Vec<&Device>>();
    */

    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .ok()?;

    let kernel = if priority {
        FftKernel::create(programs)
    } else {
        // Low priority kernels may be aborted if a high priority kernel wants to run/is running.
        FftKernel::create_with_abort(programs, &PriorityLock::is_taken)
    };
    match kernel {
        Ok(k) => {
            info!("GPU FFT kernel instantiated!");
            Some((k, lock))
        }
        Err(e) => {
            warn!("Cannot instantiate GPU FFT kernel! Error: {}", e);
            None
        }
    }
}

fn create_multiexp_kernel<'a, G>(
    priority: bool,
) -> Option<(CpuGpuMultiexpKernel<'a, G>, GPULock<'a>)>
where
    G: PrimeCurveAffine + GpuName,
{
    let lock = GPULock::lock();
    let mut devices = Vec::new();
    for lock_info in &lock.0 {
        for device in &lock_info.devices {
            devices.push(*device);
        }
    }
    /*
    let devices = lock
        .0
        .iter()
        .map(|LockInfo { devices, .. }| devices)
        .collect::<Vec<&Device>>();
    */

    let kernel = if priority {
        CpuGpuMultiexpKernel::create(&devices)
    } else {
        // Low priority kernels may be aborted if a high priority kernel wants to run/is running.
        CpuGpuMultiexpKernel::create_with_abort(&devices, &PriorityLock::is_taken)
    };
    match kernel {
        Ok(k) => {
            info!("GPU Multiexp kernel instantiated!");
            Some((k, lock))
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
            kernel_and_lock: Option<($kernel, GPULock<'a>)>,
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
                    if let Some((kernel, lock)) = $func(self.priority) {
                        self.kernel_and_lock = Some((kernel, lock));
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
                            Err(GpuError::EcGpu(EcError::Aborted)) => {
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
