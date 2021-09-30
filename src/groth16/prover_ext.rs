use super::BellTaskType;
use crate::gpu;
use pairing::Engine;

use crate::gpu::{LockedFFTKernel, LockedMultiexpKernel};
use crate::SynthesisError;
#[cfg(any(feature = "cuda", feature = "opencl"))]
use log::warn;

#[cfg(any(feature = "cuda", feature = "opencl"))]
use scheduler_client::{
    list_devices, Client as SClient, Error as ClientError, ResourceAlloc, ResourceMemory,
    ResourceReq, ResourceType, TaskFunc, TaskReqBuilder, TaskResult, TaskType,
};

#[cfg(any(feature = "cuda", feature = "opencl"))]
const TIMEOUT: u64 = 1200;
pub struct Client {
    _task_type: BellTaskType,

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    inner: SClient,
}

impl Client {
    pub fn new(_task_type: Option<BellTaskType>) -> Result<Client, SynthesisError> {
        let _task_type = _task_type.unwrap_or(BellTaskType::MerkleTree);
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        let inner = SClient::register::<SynthesisError>()?;

        Ok(Self {
            _task_type,
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            inner,
        })
    }

    pub fn update_context(&mut self, _name: String, _context: String) {
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        self.inner.set_name(_name);
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        self.inner.set_context(_context);
    }
}

macro_rules! solver {
    ($class:ident, $kern:ident) => {
        pub struct $class<E, F, R>
        where
            for<'a> F: FnMut(&'a mut Option<$kern<E>>) -> Result<R, SynthesisError>,
            E: Engine + gpu::GpuEngine,
        {
            kernel: Option<$kern<E>>,
            _log_d: usize,
            call: F,
        }

        impl<E, F, R> $class<E, F, R>
        where
            for<'a> F: FnMut(&'a mut Option<$kern<E>>) -> Result<R, SynthesisError>,
            E: Engine + gpu::GpuEngine,
        {
            pub fn new(log_d: usize, call: F) -> Self {
                $class::<E, F, R> {
                    _log_d: log_d,
                    kernel: None,
                    call,
                }
            }

            #[cfg(any(feature = "cuda", feature = "opencl"))]
            pub fn solve(&mut self, client: &mut Client) -> Result<(), SynthesisError> {
                use std::time::Duration;

                let task_type = match client._task_type {
                    BellTaskType::WinningPost => TaskType::WinningPost,
                    BellTaskType::WindowPost => TaskType::WindowPost,
                    _ => TaskType::MerkleTree,
                };

                let requirements = {
                    let resouce_req = ResourceReq {
                        resource: ResourceType::Gpu(ResourceMemory::All),
                        quantity: list_devices().gpu_devices().len(),
                        preemptible: true,
                    };
                    let task_req = TaskReqBuilder::new()
                        .resource_req(resouce_req)
                        .with_task_type(task_type);
                    task_req.build()
                };

                let task_type = requirements.task_type;

                let res = client
                    .inner
                    .schedule_one_of(self, requirements, Duration::from_secs(TIMEOUT))
                    .map(|_| ());

                match res {
                    Ok(res) => Ok(res),
                    // fallback to CPU in case of a timeout for winning_post task
                    Err(SynthesisError::Scheduler(ClientError::Timeout))
                        if task_type == Some(TaskType::WinningPost) =>
                    {
                        warn!("WinningPost timeout error -> falling back to CPU");
                        self.use_cpu()
                    }
                    Err(SynthesisError::Scheduler(ClientError::NoGpuResources)) => {
                        warn!("No supported GPU resources -> falling back to CPU");
                        self.use_cpu()
                    }
                    Err(e) => Err(e),
                }
            }

            #[cfg(not(any(feature = "cuda", feature = "opencl")))]
            pub fn solve(&mut self, _client: &mut Client) -> Result<(), SynthesisError> {
                self.use_cpu()
            }

            fn use_cpu(&mut self) -> Result<(), SynthesisError> {
                let mut kernel = self.kernel.take();
                (self.call)(&mut kernel).map(|_| ())
            }
        }

        #[cfg(any(feature = "cuda", feature = "opencl"))]
        impl<E, F, R> TaskFunc for $class<E, F, R>
        where
            for<'a> F: FnMut(&'a mut Option<$kern<E>>) -> Result<R, SynthesisError>,
            E: Engine + gpu::GpuEngine,
        {
            type Output = ();
            type Error = SynthesisError;

            fn init(&mut self, alloc: Option<&ResourceAlloc>) -> Result<Self::Output, Self::Error> {
                self.kernel.replace($kern::<E>::new(self._log_d, alloc));
                Ok(())
            }
            fn end(&mut self, _: Option<&ResourceAlloc>) -> Result<Self::Output, Self::Error> {
                Ok(())
            }
            fn task(&mut self, _alloc: Option<&ResourceAlloc>) -> Result<TaskResult, Self::Error> {
                let mut kernel = self.kernel.take();
                let res = match (self.call)(&mut kernel) {
                    Ok(_) => Ok(TaskResult::Done),
                    Err(e) => Err(e),
                };
                res
            }
        }
    };
}

// FFT Kernels use only one device
solver!(FftSolver, LockedFFTKernel);
// Multiexp kernels use all GPUS in the system
// so does not pass a specific number of gpus to use.
solver!(MultiexpSolver, LockedMultiexpKernel);
