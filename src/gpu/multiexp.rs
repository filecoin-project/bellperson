use std::env;
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

use ec_gpu_gen::multiexp::MultiexpKernel;
use ec_gpu_gen::multiexp_cpu::{multiexp_cpu, FullDensity};
use ec_gpu_gen::rust_gpu_tools::Device;
use ec_gpu_gen::threadpool::Worker;
use ec_gpu_gen::EcResult;
use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group};
use log::{error, info};

use crate::gpu::GpuName;

pub fn get_cpu_utilization() -> f64 {
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

/// Set the correct enviornment variables for a custom GPU.
///
/// Determining the number of cores was moved to rust-gpu-tools, which uses the
/// `RUST_GPU_TOOLS_CUSTOM_GPU` environment variable to set custom GPUs. Users should upgrade
/// using that one instead. Though using `BELLMAN_CUSTOM_GPU` is still supported for backwards
/// compatibility, but will be ignored if `RUST_GPU_TOOLS_CUSTOM_GPU` is also set.
/// Setting `RUST_GPU_TOOLS_CUSTOM_GPU` must happen before the first call to
/// [`rust_gpu_tools::CUDA_CORES`], as it will be initialized only once for the lifetime of the
/// library.
fn set_custom_gpu_env_var() {
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
}

/// A Multiexp kernel that can share the workload between the GPU and the CPU.
pub struct CpuGpuMultiexpKernel<'a, G>(MultiexpKernel<'a, G>)
where
    G: PrimeCurveAffine;

impl<'a, G> CpuGpuMultiexpKernel<'a, G>
where
    G: PrimeCurveAffine + GpuName,
{
    /// Create new kernels, one for each given device.
    pub fn create(devices: &[&Device]) -> EcResult<Self> {
        info!("Multiexp: CPU utilization: {}.", get_cpu_utilization());
        set_custom_gpu_env_var();
        let programs = devices
            .iter()
            .map(|device| ec_gpu_gen::program!(device))
            .collect::<Result<_, _>>()?;
        let kernel = MultiexpKernel::create(programs, devices)?;
        Ok(Self(kernel))
    }

    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create_with_abort(
        devices: &[&Device],
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        info!("Multiexp: CPU utilization: {}.", get_cpu_utilization());
        set_custom_gpu_env_var();
        let programs = devices
            .iter()
            .map(|device| ec_gpu_gen::program!(device))
            .collect::<Result<_, _>>()?;
        let kernel = MultiexpKernel::create_with_abort(programs, devices, maybe_abort)?;
        Ok(Self(kernel))
    }

    /// Calculate multiexp.
    pub fn multiexp(
        &mut self,
        pool: &Worker,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        skip: usize,
    ) -> EcResult<G::Curve> {
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + exps.len())];
        let exps = &exps[..];

        let cpu_n = ((exps.len() as f64) * get_cpu_utilization()) as usize;
        let n = exps.len() - cpu_n;
        let (cpu_bases, bases) = bases.split_at(cpu_n);
        let (cpu_exps, exps) = exps.split_at(cpu_n);

        let mut results = Vec::new();
        let error = Arc::new(RwLock::new(Ok(())));

        let cpu_acc = pool.scoped(|s| {
            if n > 0 {
                results = vec![G::Curve::identity(); self.0.num_kernels()];
                self.0
                    .parallel_multiexp(s, bases, exps, &mut results, error.clone());
            }

            multiexp_cpu(
                pool,
                (Arc::new(cpu_bases.to_vec()), 0),
                FullDensity,
                Arc::new(cpu_exps.to_vec()),
            )
        });

        Arc::try_unwrap(error)
            .expect("only one ref left")
            .into_inner()
            .unwrap()?;
        let mut acc = G::Curve::identity();
        for r in results {
            acc.add_assign(&r);
        }

        acc.add_assign(&cpu_acc.wait().unwrap());
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test if `RUST_GPU_TOOLS_CUSTOM_GPU` is set correctly if only `BELLMAN_CUSTOM_GPU` is set.
    #[test]
    fn belllman_custom_gpu_env_var() {
        temp_env::with_vars(
            vec![
                ("BELLMAN_CUSTOM_GPU", Some("My custom GPU:3241")),
                ("RUST_GPU_TOOLS_CUSTOM_GPU", None),
            ],
            || {
                set_custom_gpu_env_var();
                let rust_gpu_tools_custom_gpu = env::var("RUST_GPU_TOOLS_CUSTOM_GPU").expect(
                    "RUST_GPU_TOOLS_CUSTOM_GPU is set after `set_custom_gpu_env_var` was called.",
                );
                assert_eq!(
                    rust_gpu_tools_custom_gpu, "My custom GPU:3241",
                    "`RUST_GPU_TOOLS_CUSTOM_GPU` has the value set by `BELLMAN_CUSTOM_GPU`."
                );
            },
        )
    }

    /// Test if `BELLMAN_CUSTOM_GPU` is correctly ignored if `RUST_GPU_TOOLS_CUSTOM_GPU` is already
    /// set.
    #[test]
    fn belllman_custom_gpu_env_var_ignored() {
        temp_env::with_vars(
            vec![
                ("RUST_GPU_TOOLS_CUSTOM_GPU", Some("My custom GPU:444")),
                ("BELLMAN_CUSTOM_GPU", Some("My custom GPU:3242")),
            ],
            || {
                set_custom_gpu_env_var();
                let rust_gpu_tools_custom_gpu = env::var("RUST_GPU_TOOLS_CUSTOM_GPU").expect(
                    "RUST_GPU_TOOLS_CUSTOM_GPU is set after `set_custom_gpu_env_var` was called.",
                );
                assert_eq!(rust_gpu_tools_custom_gpu, "My custom GPU:444", "`RUST_GPU_TOOLS_CUSTOM_GPU` has its original value, as the value of `BELLMAN_CUSTOM_GPU` was correctly ignored,");
            },
        )
    }
}
