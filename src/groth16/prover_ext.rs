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
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    task_type: TaskType,

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    inner: SClient,
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
impl Client {
    pub fn new(task_type: Option<BellTaskType>) -> Result<Client, SynthesisError> {
        let task_type = match task_type.unwrap_or(BellTaskType::MerkleTree) {
            BellTaskType::WinningPost => TaskType::WinningPost,
            BellTaskType::WindowPost => TaskType::WindowPost,
            _ => TaskType::MerkleTree,
        };
        let inner = SClient::register::<SynthesisError>()?;

        Ok(Self { task_type, inner })
    }

    pub fn update_context(&mut self, _name: String, _context: String) {
        self.inner.set_name(_name);
        self.inner.set_context(_context);
    }
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl Client {
    pub fn new(_task_type: Option<BellTaskType>) -> Result<Client, SynthesisError> {
        Ok(Self)
    }
    pub fn update_context(&mut self, _name: String, _context: String) {}
}

macro_rules! solver {
    ($class:ident, $kern:ident) => {
        pub struct $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine + gpu::GpuEngine,
        {
            kernel: Option<$kern<E>>,
            _log_d: usize,
            call: F,
            index: usize,
        }

        impl<E, F, R> $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine + gpu::GpuEngine,
        {
            pub fn new(log_d: usize, call: F) -> Self {
                $class::<E, F, R> {
                    _log_d: log_d,
                    kernel: None,
                    call,
                    index: 0,
                }
            }

            #[cfg(any(feature = "cuda", feature = "opencl"))]
            pub fn solve(&mut self, client: &mut Client) -> Result<(), SynthesisError> {
                use std::time::Duration;

                let requirements = {
                    let resouce_req = ResourceReq {
                        resource: ResourceType::Gpu(ResourceMemory::All),
                        quantity: list_devices().gpu_devices().len(),
                        preemptible: true,
                    };
                    let task_req = TaskReqBuilder::new()
                        .resource_req(resouce_req)
                        .with_task_type(client.task_type);
                    task_req.build()
                };

                let res = client
                    .inner
                    .schedule_one_of(self, requirements, Duration::from_secs(TIMEOUT))
                    .map(|_| ());

                match res {
                    Ok(res) => Ok(res),
                    // fallback to CPU in case of a timeout for winning_post task
                    Err(SynthesisError::Scheduler(ClientError::Timeout))
                        if client.task_type == TaskType::WinningPost =>
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
                loop {
                    match (self.call)(self.index, &mut self.kernel) {
                        Some(Ok(_)) => {
                            self.index += 1;
                        }
                        Some(Err(e)) => {
                            self.kernel.take();
                            return Err(e);
                        }
                        None => {
                            self.kernel.take();
                            return Ok(());
                        }
                    }
                }
            }
        }

        #[cfg(any(feature = "cuda", feature = "opencl"))]
        impl<E, F, R> TaskFunc for $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
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
                match (self.call)(self.index, &mut self.kernel) {
                    Some(Ok(_)) => {
                        self.index += 1;
                        Ok(TaskResult::Continue)
                    }
                    Some(Err(e)) => {
                        self.kernel.take();
                        Err(e)
                    }
                    None => {
                        self.kernel.take();
                        Ok(TaskResult::Done)
                    }
                }
            }
        }
    };
}

// FFT Kernels use only one device
solver!(FftSolver, LockedFFTKernel);
// Multiexp kernels use all GPUS in the system
// so does not pass a specific number of gpus to use.
solver!(MultiexpSolver, LockedMultiexpKernel);
