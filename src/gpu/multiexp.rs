use super::error::{GPUError, GPUResult};
use super::{locks, program, utils, GpuEngine};
use crate::multicore::Worker;
use crate::multiexp::{multiexp as cpu_multiexp, FullDensity};

use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group};
use log::{error, info};
use pairing::Engine;
use rust_gpu_tools::{program_closures, Device, Program};

use std::any::TypeId;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

const MAX_WINDOW_SIZE: usize = 10;
const LOCAL_WORK_SIZE: usize = 256;
const MEMORY_PADDING: f64 = 0.2f64; // Let 20% of GPU memory be free

pub fn get_cpu_utilization() -> f64 {
    use std::env;
    env::var("BELLMAN_CPU_UTILIZATION")
        .map_or(0f64, |v| match v.parse() {
            Ok(val) => val,
            Err(_) => {
                error!("Invalid BELLMAN_CPU_UTILIZATION! Defaulting to 0...");
                0f64
            }
        })
        .max(0f64)
        .min(1f64)
}

// Multiexp kernel for a single GPU
pub struct SingleMultiexpKernel<E>
where
    E: Engine + GpuEngine,
{
    program: Program,
    core_count: usize,
    n: usize,

    priority: bool,
    _phantom: std::marker::PhantomData<E::Fr>,
}

fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
    // Observations show that we get the best performance when num_groups * num_windows ~= 2 * CUDA_CORES
    2 * core_count / num_windows
}

fn calc_window_size(n: usize, exp_bits: usize, core_count: usize) -> usize {
    // window_size = ln(n / num_groups)
    // num_windows = exp_bits / window_size
    // num_groups = 2 * core_count / num_windows = 2 * core_count * window_size / exp_bits
    // window_size = ln(n / num_groups) = ln(n * exp_bits / (2 * core_count * window_size))
    // window_size = ln(exp_bits * n / (2 * core_count)) - ln(window_size)
    //
    // Thus we need to solve the following equation:
    // window_size + ln(window_size) = ln(exp_bits * n / (2 * core_count))
    let lower_bound = (((exp_bits * n) as f64) / ((2 * core_count) as f64)).ln();
    for w in 0..MAX_WINDOW_SIZE {
        if (w as f64) + (w as f64).ln() > lower_bound {
            return w;
        }
    }

    MAX_WINDOW_SIZE
}

fn calc_best_chunk_size(max_window_size: usize, core_count: usize, exp_bits: usize) -> usize {
    // Best chunk-size (N) can also be calculated using the same logic as calc_window_size:
    // n = e^window_size * window_size * 2 * core_count / exp_bits
    (((max_window_size as f64).exp() as f64)
        * (max_window_size as f64)
        * 2f64
        * (core_count as f64)
        / (exp_bits as f64))
        .ceil() as usize
}

fn calc_chunk_size<E>(mem: u64, core_count: usize) -> usize
where
    E: Engine,
{
    let aff_size = std::mem::size_of::<E::G1Affine>() + std::mem::size_of::<E::G2Affine>();
    let exp_size = exp_size::<E>();
    let proj_size = std::mem::size_of::<E::G1>() + std::mem::size_of::<E::G2>();
    ((((mem as f64) * (1f64 - MEMORY_PADDING)) as usize)
        - (2 * core_count * ((1 << MAX_WINDOW_SIZE) + 1) * proj_size))
        / (aff_size + exp_size)
}

fn exp_size<E: Engine>() -> usize {
    std::mem::size_of::<<E::Fr as ff::PrimeField>::Repr>()
}

impl<E> SingleMultiexpKernel<E>
where
    E: Engine + GpuEngine,
{
    pub fn create(device: &Device, priority: bool) -> GPUResult<SingleMultiexpKernel<E>> {
        let exp_bits = exp_size::<E>() * 8;
        let core_count = utils::get_core_count(&device.name());
        let mem = device.memory();
        let max_n = calc_chunk_size::<E>(mem, core_count);
        let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        let n = std::cmp::min(max_n, best_n);

        let program = program::program::<E>(device)?;

        Ok(SingleMultiexpKernel {
            program,
            core_count,
            n,
            priority,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        bases: &[G],
        exps: &[<G::Scalar as PrimeField>::Repr],
        n: usize,
    ) -> GPUResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine,
    {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let exp_bits = exp_size::<E>() * 8;
        let window_size = calc_window_size(n as usize, exp_bits, self.core_count);
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(self.core_count, num_windows);
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(
            |program, _arg| -> GPUResult<Vec<<G as PrimeCurveAffine>::Curve>> {
                let base_buffer = program.create_buffer_from_slice(bases)?;
                let exp_buffer = program.create_buffer_from_slice(exps)?;

                // It is safe as the GPU will initialize that buffer
                let bucket_buffer = unsafe {
                    program.create_buffer::<<G as PrimeCurveAffine>::Curve>(
                        2 * self.core_count * bucket_len,
                    )?
                };
                // It is safe as the GPU will initialize that buffer
                let result_buffer = unsafe {
                    program.create_buffer::<<G as PrimeCurveAffine>::Curve>(2 * self.core_count)?
                };

                // The global work size follows CUDA's definition and is the number of
                // `LOCAL_WORK_SIZE` sized thread groups.
                let global_work_size =
                    (num_windows * num_groups + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

                let kernel = program.create_kernel(
                    if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                        "G1_bellman_multiexp"
                    } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                        "G2_bellman_multiexp"
                    } else {
                        return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
                    },
                    global_work_size,
                    LOCAL_WORK_SIZE,
                )?;

                kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer)
                    .arg(&result_buffer)
                    .arg(&exp_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_groups as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;

                let mut results =
                    vec![<G as PrimeCurveAffine>::Curve::identity(); 2 * self.core_count];
                program.read_into_buffer(&result_buffer, &mut results)?;

                Ok(results)
            }
        );

        let results = self.program.run(closures, ())?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as PrimeCurveAffine>::Curve::identity();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc = acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

// A struct that containts several multiexp kernels for different devices
pub struct MultiexpKernel<E>
where
    E: Engine + GpuEngine,
{
    kernels: Vec<SingleMultiexpKernel<E>>,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
}

impl<E> MultiexpKernel<E>
where
    E: Engine + GpuEngine,
{
    pub fn create(priority: bool, isWinPost: bool) -> GPUResult<MultiexpKernel<E>> {
        let lock = locks::GPULock::lock(isWinPost);

        let mut kernels_bak: Vec<_> = Device::all()
            .iter()
            .filter_map(|device| {
                let kernel = SingleMultiexpKernel::<E>::create(device, priority);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels_bak.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }
        info!(
            "Multiexp: {} working device(s) selected. (CPU utilization: {})",
            kernels_bak.len(),
            get_cpu_utilization()
        );
        for (i, k) in kernels_bak.iter().enumerate() {
            info!(
                "Multiexp: Device {}: {} (Chunk-size: {})",
                i,
                k.program.device_name(),
                k.n
            );
        }
	
	if kernels_bak.len() > 1 {
	    if isWinPost {
	        kernels_bak.remove(1);
	        let kernels = kernels_bak;
		Ok(MultiexpKernel::<E> {
                    kernels,
                    _lock: lock,
                })
	    } else {
	        kernels_bak.remove(0);
	        let kernels = kernels_bak;
		Ok(MultiexpKernel::<E> {
                    kernels,
                    _lock: lock,
                })
	    }
	} else {
	    let kernels = kernels_bak;
            Ok(MultiexpKernel::<E> {
                kernels,
                _lock: lock,
            })
	}
    }

    pub fn multiexp<G>(
        &mut self,
        pool: &Worker,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        skip: usize,
        n: usize,
    ) -> GPUResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine<Scalar = E::Fr>,
    {
        let num_devices = self.kernels.len();
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + n)];
        let exps = &exps[..n];

        let cpu_n = ((n as f64) * get_cpu_utilization()) as usize;
        let n = n - cpu_n;
        let (cpu_bases, bases) = bases.split_at(cpu_n);
        let (cpu_exps, exps) = exps.split_at(cpu_n);

        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

        let mut results = Vec::new();
        let error = Arc::new(RwLock::new(Ok(())));

        let cpu_acc = pool.scoped(|s| {
            if n > 0 {
                results = vec![<G as PrimeCurveAffine>::Curve::identity(); self.kernels.len()];

                for (((bases, exps), kern), result) in bases
                    .chunks(chunk_size)
                    .zip(exps.chunks(chunk_size))
                    .zip(self.kernels.iter_mut())
                    .zip(results.iter_mut())
                {
                    let error = error.clone();
                    s.execute(move || {
                        let mut acc = <G as PrimeCurveAffine>::Curve::identity();
                        for (bases, exps) in bases.chunks(kern.n).zip(exps.chunks(kern.n)) {
                            if error.read().unwrap().is_err() {
                                break;
                            }
                            match kern.multiexp(bases, exps, bases.len()) {
                                Ok(result) => acc.add_assign(&result),
                                Err(e) => {
                                    *error.write().unwrap() = Err(e);
                                    break;
                                }
                            }
                        }
                        if error.read().unwrap().is_ok() {
                            *result = acc;
                        }
                    });
                }
            }

            cpu_multiexp::<_, _, _, E, _>(
                &pool,
                (Arc::new(cpu_bases.to_vec()), 0),
                FullDensity,
                Arc::new(cpu_exps.to_vec()),
                &mut None,
            )
        });

        Arc::try_unwrap(error)
            .expect("only one ref left")
            .into_inner()
            .unwrap()?;
        let mut acc = <G as PrimeCurveAffine>::Curve::identity();
        for r in results {
            acc.add_assign(&r);
        }

        acc.add_assign(&cpu_acc.wait().unwrap());
        Ok(acc)
    }
}
