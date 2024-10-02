//! Prover implementation implemented in Rust.

use std::{
    ops::{AddAssign, Mul, MulAssign},
    sync::Arc,
    time::Instant,
};

use bellpepper_core::{Circuit, SynthesisError};
use ec_gpu_gen::{
    multiexp_cpu::FullDensity,
    threadpool::{Worker, THREAD_POOL},
};
use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use log::trace;
use log::{debug, info};
use pairing::MultiMillerLoop;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use super::{ParameterSource, Proof, ProvingAssignment};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use crate::gpu::PriorityLock;
use crate::{
    domain::EvaluationDomain,
    gpu::{GpuError, GpuName, LockedFftKernel, LockedMultiexpKernel},
    multiexp::multiexp,
    BELLMAN_VERSION,
};

#[allow(clippy::type_complexity)]
pub(super) fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    randomization: Option<(Vec<E::Fr>, Vec<E::Fr>)>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    info!("Bellperson {} is being used!", BELLMAN_VERSION);

    let provers = super::synthesize_circuits_batch(circuits)?;
    proof_circuits_batch(provers, params, randomization, priority)
}

#[allow(clippy::type_complexity)]
fn proof_circuits_batch<E, P>(
    mut provers: std::vec::Vec<ProvingAssignment<E::Fr>>,
    params: P,
    randomization: Option<(Vec<E::Fr>, Vec<E::Fr>)>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
    P: ParameterSource<E>,
{
    // Start fft/multiexp prover timer
    let start = Instant::now();
    info!("starting proof timer");

    let input_assignments = provers
        .par_iter_mut()
        .map(|prover| {
            let input_assignment = std::mem::take(&mut prover.input_assignment);
            Arc::new(
                input_assignment
                    .into_iter()
                    .map(|s| s.to_repr())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let aux_assignments = provers
        .par_iter_mut()
        .map(|prover| {
            let aux_assignment = std::mem::take(&mut prover.aux_assignment);
            Arc::new(
                aux_assignment
                    .into_iter()
                    .map(|s| s.to_repr())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let worker = Worker::new();
    let input_len = input_assignments[0].len();
    let vk = params.get_vk(input_len)?.clone();
    let n = provers[0].a.len();
    let a_aux_density_total = provers[0].a_aux_density.get_total_density();
    let b_input_density_total = provers[0].b_input_density.get_total_density();
    let b_aux_density_total = provers[0].b_aux_density.get_total_density();
    let aux_assignment_len = provers[0].aux_assignment.len();
    let num_circuits = provers.len();

    let zk = randomization.is_some();
    let (r_s, s_s) = randomization.unwrap_or((
        vec![E::Fr::ZERO; num_circuits],
        vec![E::Fr::ZERO; num_circuits],
    ));

    // Make sure all circuits have the same input len.
    for prover in &provers {
        assert_eq!(
            prover.a.len(),
            n,
            "only equaly sized circuits are supported"
        );
        debug_assert_eq!(
            a_aux_density_total,
            prover.a_aux_density.get_total_density(),
            "only identical circuits are supported"
        );
        debug_assert_eq!(
            b_input_density_total,
            prover.b_input_density.get_total_density(),
            "only identical circuits are supported"
        );
        debug_assert_eq!(
            b_aux_density_total,
            prover.b_aux_density.get_total_density(),
            "only identical circuits are supported"
        );
    }

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    let prio_lock = if priority {
        trace!("acquiring priority lock");
        Some(PriorityLock::lock())
    } else {
        None
    };

    let mut a_s = Vec::with_capacity(num_circuits);
    let mut params_h = None;
    let worker = &worker;
    let provers_ref = &mut provers;
    let params = &params;

    THREAD_POOL.scoped(|s| -> Result<(), SynthesisError> {
        let params_h = &mut params_h;
        s.execute(move || {
            debug!("get h");
            *params_h = Some(params.get_h(n));
        });

        let mut fft_kern = Some(LockedFftKernel::new(priority));
        for prover in provers_ref {
            a_s.push(execute_fft(worker, prover, &mut fft_kern)?);
        }
        Ok(())
    })?;

    let mut multiexp_g1_kern = LockedMultiexpKernel::<E::G1Affine>::new(priority);
    let params_h = params_h.unwrap()?;

    let mut h_s = Vec::with_capacity(num_circuits);
    let mut params_l = None;

    THREAD_POOL.scoped(|s| {
        let params_l = &mut params_l;
        s.execute(move || {
            debug!("get l");
            *params_l = Some(params.get_l(aux_assignment_len));
        });

        debug!("multiexp h");
        for a in a_s.into_iter() {
            h_s.push(multiexp(
                worker,
                params_h.clone(),
                FullDensity,
                a,
                &mut multiexp_g1_kern,
            ));
        }
    });

    let params_l = params_l.unwrap()?;

    let mut l_s = Vec::with_capacity(num_circuits);
    let mut params_a = None;
    let mut params_b_g1 = None;
    let mut params_b_g2 = None;
    let a_aux_density_total = provers[0].a_aux_density.get_total_density();
    let b_input_density_total = provers[0].b_input_density.get_total_density();
    let b_aux_density_total = provers[0].b_aux_density.get_total_density();

    THREAD_POOL.scoped(|s| {
        let params_a = &mut params_a;
        let params_b_g1 = &mut params_b_g1;
        let params_b_g2 = &mut params_b_g2;
        s.execute(move || {
            debug!("get_a b_g1 b_g2");
            *params_a = Some(params.get_a(input_len, a_aux_density_total));
            if zk {
                *params_b_g1 = Some(params.get_b_g1(b_input_density_total, b_aux_density_total));
            }
            *params_b_g2 = Some(params.get_b_g2(b_input_density_total, b_aux_density_total));
        });

        debug!("multiexp l");
        for aux in aux_assignments.iter() {
            l_s.push(multiexp(
                worker,
                params_l.clone(),
                FullDensity,
                aux.clone(),
                &mut multiexp_g1_kern,
            ));
        }
    });

    debug!("get a b_g1");
    let (a_inputs_source, a_aux_source) = params_a.unwrap()?;
    let params_b_g1_opt = params_b_g1.transpose()?;

    let densities = provers
        .iter_mut()
        .map(|prover| {
            let a_aux_density = std::mem::take(&mut prover.a_aux_density);
            let b_input_density = std::mem::take(&mut prover.b_input_density);
            let b_aux_density = std::mem::take(&mut prover.b_aux_density);
            (
                Arc::new(a_aux_density),
                Arc::new(b_input_density),
                Arc::new(b_aux_density),
            )
        })
        .collect::<Vec<_>>();
    drop(provers);

    debug!("multiexp a b_g1");
    // Collect the data, so that the inputs can be dropped early.
    #[allow(clippy::needless_collect)]
    let inputs_g1 = input_assignments
        .iter()
        .zip(aux_assignments.iter())
        .zip(densities.iter())
        .map(
            |(
                (input_assignment, aux_assignment),
                (a_aux_density, b_input_density, b_aux_density),
            )| {
                let a_inputs = multiexp(
                    worker,
                    a_inputs_source.clone(),
                    FullDensity,
                    input_assignment.clone(),
                    &mut multiexp_g1_kern,
                );

                let a_aux = multiexp(
                    worker,
                    a_aux_source.clone(),
                    a_aux_density.clone(),
                    aux_assignment.clone(),
                    &mut multiexp_g1_kern,
                );

                let b_g1_inputs_aux_opt =
                    params_b_g1_opt
                        .as_ref()
                        .map(|(b_g1_inputs_source, b_g1_aux_source)| {
                            (
                                multiexp(
                                    worker,
                                    b_g1_inputs_source.clone(),
                                    b_input_density.clone(),
                                    input_assignment.clone(),
                                    &mut multiexp_g1_kern,
                                ),
                                multiexp(
                                    worker,
                                    b_g1_aux_source.clone(),
                                    b_aux_density.clone(),
                                    aux_assignment.clone(),
                                    &mut multiexp_g1_kern,
                                ),
                            )
                        });

                (a_inputs, a_aux, b_g1_inputs_aux_opt)
            },
        )
        .collect::<Vec<_>>();
    #[allow(clippy::drop_non_drop)]
    drop(multiexp_g1_kern);
    drop(a_inputs_source);
    drop(a_aux_source);
    drop(params_b_g1_opt);

    // The multiexp kernel for G1 can only be initiated after the kernel for G1 was dropped. Else
    // it would block, trying to acquire the GPU lock.
    let mut multiexp_g2_kern = LockedMultiexpKernel::<E::G2Affine>::new(priority);

    debug!("get b_g2");
    let (b_g2_inputs_source, b_g2_aux_source) = params_b_g2.unwrap()?;

    debug!("multiexp b_g2");
    // Collect the data, so that the inputs can be dropped early.
    #[allow(clippy::needless_collect)]
    let inputs_g2 = input_assignments
        .iter()
        .zip(aux_assignments.iter())
        .zip(densities.iter())
        .map(
            |((input_assignment, aux_assignment), (_, b_input_density, b_aux_density))| {
                let b_g2_inputs = multiexp(
                    worker,
                    b_g2_inputs_source.clone(),
                    b_input_density.clone(),
                    input_assignment.clone(),
                    &mut multiexp_g2_kern,
                );
                let b_g2_aux = multiexp(
                    worker,
                    b_g2_aux_source.clone(),
                    b_aux_density.clone(),
                    aux_assignment.clone(),
                    &mut multiexp_g2_kern,
                );

                (b_g2_inputs, b_g2_aux)
            },
        )
        .collect::<Vec<_>>();
    #[allow(clippy::drop_non_drop)]
    drop(multiexp_g2_kern);
    drop(densities);
    drop(b_g2_inputs_source);
    drop(b_g2_aux_source);

    debug!("proofs");
    let proofs = h_s
        .into_iter()
        .zip(l_s.into_iter())
        .zip(inputs_g1.into_iter())
        .zip(inputs_g2.into_iter())
        .zip(r_s.into_iter())
        .zip(s_s.into_iter())
        .map(
            |(
                ((((h, l), (a_inputs, a_aux, b_g1_inputs_aux_opt)), (b_g2_inputs, b_g2_aux)), r),
                s,
            )| {
                if (vk.delta_g1.is_identity() | vk.delta_g2.is_identity()).into() {
                    // If this element is zero, someone is trying to perform a
                    // subversion-CRS attack.
                    return Err(SynthesisError::UnexpectedIdentity);
                }

                let mut g_a = vk.delta_g1.mul(r);
                g_a.add_assign(&vk.alpha_g1);
                let mut g_b = vk.delta_g2.mul(s);
                g_b.add_assign(&vk.beta_g2);
                let mut a_answer = a_inputs.wait().map_err(GpuError::from)?;
                a_answer.add_assign(&a_aux.wait().map_err(GpuError::from)?);
                g_a.add_assign(&a_answer);
                a_answer.mul_assign(s);
                let mut g_c = a_answer;

                let mut b2_answer = b_g2_inputs.wait().map_err(GpuError::from)?;
                b2_answer.add_assign(&b_g2_aux.wait().map_err(GpuError::from)?);

                g_b.add_assign(&b2_answer);

                if let Some((b_g1_inputs, b_g1_aux)) = b_g1_inputs_aux_opt {
                    let mut b1_answer = b_g1_inputs.wait().map_err(GpuError::from)?;
                    b1_answer.add_assign(&b_g1_aux.wait().map_err(GpuError::from)?);
                    b1_answer.mul_assign(r);
                    g_c.add_assign(&b1_answer);
                    let mut rs = r;
                    rs.mul_assign(&s);
                    g_c.add_assign(vk.delta_g1.mul(rs));
                    g_c.add_assign(&vk.alpha_g1.mul(s));
                    g_c.add_assign(&vk.beta_g1.mul(r));
                }

                g_c.add_assign(&h.wait().map_err(GpuError::from)?);
                g_c.add_assign(&l.wait().map_err(GpuError::from)?);

                Ok(Proof {
                    a: g_a.to_affine(),
                    b: g_b.to_affine(),
                    c: g_c.to_affine(),
                })
            },
        )
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    {
        trace!("dropping priority lock");
        drop(prio_lock);
    }

    let proof_time = start.elapsed();
    info!("prover time: {:?}", proof_time);

    Ok(proofs)
}

fn execute_fft<F>(
    worker: &Worker,
    prover: &mut ProvingAssignment<F>,
    fft_kern: &mut Option<LockedFftKernel<F>>,
) -> Result<Arc<Vec<F::Repr>>, SynthesisError>
where
    F: PrimeField + GpuName,
{
    let mut a = EvaluationDomain::from_coeffs(std::mem::take(&mut prover.a))?;
    let mut b = EvaluationDomain::from_coeffs(std::mem::take(&mut prover.b))?;
    let mut c = EvaluationDomain::from_coeffs(std::mem::take(&mut prover.c))?;

    EvaluationDomain::ifft_many(&mut [&mut a, &mut b, &mut c], worker, fft_kern)?;
    EvaluationDomain::coset_fft_many(&mut [&mut a, &mut b, &mut c], worker, fft_kern)?;

    a.mul_assign(worker, &b);
    drop(b);
    a.sub_assign(worker, &c);
    drop(c);

    a.divide_by_z_on_coset(worker);
    a.icoset_fft(worker, fft_kern)?;

    let a = a.into_coeffs();
    let a_len = a.len() - 1;
    let a = a
        .into_par_iter()
        .take(a_len)
        .map(|s| s.to_repr())
        .collect::<Vec<_>>();
    Ok(Arc::new(a))
}
