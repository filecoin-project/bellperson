use std::ops::{AddAssign, Mul, MulAssign};
use std::sync::Arc;
use std::time::Instant;

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve};
use pairing::MultiMillerLoop;
use rand_core::RngCore;
use rayon::prelude::*;

use super::{ParameterSource, Proof};
use crate::domain::EvaluationDomain;
use crate::gpu::{GpuName, LockedFftKernel, LockedMultiexpKernel};
use crate::multiexp::multiexp;
use crate::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable, BELLMAN_VERSION,
};
use ec_gpu_gen::multiexp_cpu::{DensityTracker, FullDensity};
use ec_gpu_gen::threadpool::{Worker, THREAD_POOL};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use log::trace;
use log::{debug, info};

#[cfg(any(feature = "cuda", feature = "opencl"))]
use crate::gpu::PriorityLock;

struct ProvingAssignment<Scalar: PrimeField> {
    // Density of queries
    a_aux_density: DensityTracker,
    b_input_density: DensityTracker,
    b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    a: Vec<Scalar>,
    b: Vec<Scalar>,
    c: Vec<Scalar>,

    // Assignments of variables
    input_assignment: Vec<Scalar>,
    aux_assignment: Vec<Scalar>,
}
use std::fmt;

impl<Scalar: PrimeField> fmt::Debug for ProvingAssignment<Scalar> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("ProvingAssignment")
            .field("a_aux_density", &self.a_aux_density)
            .field("b_input_density", &self.b_input_density)
            .field("b_aux_density", &self.b_aux_density)
            .field(
                "a",
                &self
                    .a
                    .iter()
                    .map(|v| format!("Fr({:?})", v))
                    .collect::<Vec<_>>(),
            )
            .field(
                "b",
                &self
                    .b
                    .iter()
                    .map(|v| format!("Fr({:?})", v))
                    .collect::<Vec<_>>(),
            )
            .field(
                "c",
                &self
                    .c
                    .iter()
                    .map(|v| format!("Fr({:?})", v))
                    .collect::<Vec<_>>(),
            )
            .field("input_assignment", &self.input_assignment)
            .field("aux_assignment", &self.aux_assignment)
            .finish()
    }
}

impl<Scalar: PrimeField> PartialEq for ProvingAssignment<Scalar> {
    fn eq(&self, other: &ProvingAssignment<Scalar>) -> bool {
        self.a_aux_density == other.a_aux_density
            && self.b_input_density == other.b_input_density
            && self.b_aux_density == other.b_aux_density
            && self.a == other.a
            && self.b == other.b
            && self.c == other.c
            && self.input_assignment == other.input_assignment
            && self.aux_assignment == other.aux_assignment
    }
}

impl<Scalar: PrimeField> ConstraintSystem<Scalar> for ProvingAssignment<Scalar> {
    type Root = Self;

    fn new() -> Self {
        Self {
            a_aux_density: DensityTracker::new(),
            b_input_density: DensityTracker::new(),
            b_aux_density: DensityTracker::new(),
            a: vec![],
            b: vec![],
            c: vec![],
            input_assignment: vec![],
            aux_assignment: vec![],
        }
    }

    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.aux_assignment.push(f()?);
        self.a_aux_density.add_element();
        self.b_aux_density.add_element();

        Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.input_assignment.push(f()?);
        self.b_input_density.add_element();

        Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        let c = c(LinearCombination::zero());

        let input_assignment = &self.input_assignment;
        let aux_assignment = &self.aux_assignment;
        let a_aux_density = &mut self.a_aux_density;
        let b_input_density = &mut self.b_input_density;
        let b_aux_density = &mut self.b_aux_density;

        let a_res = a.eval(
            // Inputs have full density in the A query
            // because there are constraints of the
            // form x * 0 = 0 for each input.
            None,
            Some(a_aux_density),
            input_assignment,
            aux_assignment,
        );

        let b_res = b.eval(
            Some(b_input_density),
            Some(b_aux_density),
            input_assignment,
            aux_assignment,
        );

        let c_res = c.eval(
            // There is no C polynomial query,
            // though there is an (beta)A + (alpha)B + C
            // query for all aux variables.
            // However, that query has full density.
            None,
            None,
            input_assignment,
            aux_assignment,
        );

        self.a.push(a_res);
        self.b.push(b_res);
        self.c.push(c_res);
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn is_extensible() -> bool {
        true
    }

    fn extend(&mut self, other: Self) {
        self.a_aux_density.extend(other.a_aux_density, false);
        self.b_input_density.extend(other.b_input_density, true);
        self.b_aux_density.extend(other.b_aux_density, false);

        self.a.extend(other.a);
        self.b.extend(other.b);
        self.c.extend(other.c);

        self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
        self.aux_assignment.extend(other.aux_assignment);
    }
}

pub fn create_random_proof_batch_priority<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    let r_s = (0..circuits.len())
        .map(|_| E::Fr::random(&mut *rng))
        .collect();
    let s_s = (0..circuits.len())
        .map(|_| E::Fr::random(&mut *rng))
        .collect();

    create_proof_batch_priority::<E, C, P>(circuits, params, r_s, s_s, priority)
}

/// creates a batch of proofs where the randomization vector is set to zero.
/// This allows for optimization of proving.
pub fn create_proof_batch_priority_nonzk<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    create_proof_batch_priority_inner(circuits, params, None, priority)
}

/// creates a batch of proofs where the randomization vector is already
/// predefined
#[allow(clippy::needless_collect)]
pub fn create_proof_batch_priority<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    create_proof_batch_priority_inner(circuits, params, Some((r_s, s_s)), priority)
}

#[allow(clippy::type_complexity)]
#[allow(clippy::needless_collect)]
fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
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

    let (start, mut provers, input_assignments, aux_assignments) =
        synthesize_circuits_batch(circuits)?;

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
        vec![E::Fr::zero(); num_circuits],
        vec![E::Fr::zero(); num_circuits],
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
                let mut a_answer = a_inputs.wait()?;
                a_answer.add_assign(&a_aux.wait()?);
                g_a.add_assign(&a_answer);
                a_answer.mul_assign(s);
                let mut g_c = a_answer;

                let mut b2_answer = b_g2_inputs.wait()?;
                b2_answer.add_assign(&b_g2_aux.wait()?);

                g_b.add_assign(&b2_answer);

                if let Some((b_g1_inputs, b_g1_aux)) = b_g1_inputs_aux_opt {
                    let mut b1_answer = b_g1_inputs.wait()?;
                    b1_answer.add_assign(&b_g1_aux.wait()?);
                    b1_answer.mul_assign(r);
                    g_c.add_assign(&b1_answer);
                    let mut rs = r;
                    rs.mul_assign(&s);
                    g_c.add_assign(vk.delta_g1.mul(rs));
                    g_c.add_assign(&vk.alpha_g1.mul(s));
                    g_c.add_assign(&vk.beta_g1.mul(r));
                }

                g_c.add_assign(&h.wait()?);
                g_c.add_assign(&l.wait()?);

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

#[allow(clippy::type_complexity)]
fn synthesize_circuits_batch<Scalar, C>(
    circuits: Vec<C>,
) -> Result<
    (
        Instant,
        std::vec::Vec<ProvingAssignment<Scalar>>,
        std::vec::Vec<std::sync::Arc<std::vec::Vec<<Scalar as PrimeField>::Repr>>>,
        std::vec::Vec<std::sync::Arc<std::vec::Vec<<Scalar as PrimeField>::Repr>>>,
    ),
    SynthesisError,
>
where
    Scalar: PrimeField,
    C: Circuit<Scalar> + Send,
{
    let start = Instant::now();
    let mut provers = circuits
        .into_par_iter()
        .map(|circuit| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignment::new();

            prover.alloc_input(|| "", || Ok(Scalar::one()))?;

            circuit.synthesize(&mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            Ok(prover)
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!("synthesis time: {:?}", start.elapsed());

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

    Ok((start, provers, input_assignments, aux_assignments))
}

#[cfg(test)]
mod tests {
    use super::*;

    use blstrs::Scalar as Fr;
    use rand::Rng;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_proving_assignment_extend() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for k in &[2, 4, 8] {
            for j in &[10, 20, 50] {
                let count: usize = k * j;

                let mut full_assignment = ProvingAssignment::<Fr>::new();
                full_assignment
                    .alloc_input(|| "one", || Ok(<Fr as Field>::one()))
                    .unwrap();

                let mut partial_assignments = Vec::with_capacity(count / k);
                for i in 0..count {
                    if i % k == 0 {
                        let mut p = ProvingAssignment::new();
                        p.alloc_input(|| "one", || Ok(<Fr as Field>::one()))
                            .unwrap();
                        partial_assignments.push(p)
                    }

                    let index: usize = i / k;
                    let partial_assignment = &mut partial_assignments[index];

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                        partial_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                        partial_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    // TODO: LinearCombination
                }

                let mut combined = ProvingAssignment::new();
                combined
                    .alloc_input(|| "one", || Ok(<Fr as Field>::one()))
                    .unwrap();

                for assignment in partial_assignments.into_iter() {
                    combined.extend(assignment);
                }
                assert_eq!(combined, full_assignment);
            }
        }
    }
}
