#[cfg(not(feature = "cuda-supraseal"))]
mod native;
#[cfg(feature = "cuda-supraseal")]
mod supraseal;

use std::time::Instant;

use bellpepper_core::{Circuit, ConstraintSystem, Index, SynthesisError, Variable};
use ec_gpu_gen::multiexp_cpu::DensityTracker;
use ff::{Field, PrimeField};
use log::info;
use pairing::MultiMillerLoop;
use rand_core::RngCore;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(not(feature = "cuda-supraseal"))]
use self::native as prover;
#[cfg(feature = "cuda-supraseal")]
use self::supraseal as prover;
use super::{ParameterSource, Proof};
use crate::gpu::GpuName;

/// Implement various traits for the proving assignment.
///
/// It's a macro so that it can be used for different types. SupraSeal is using some special memory
/// optimized data structures internally. Using a macro makes sure that the implementation will not
/// diverge over time.
macro_rules! proving_assignment_impls {
    ($type:ty) => {
        use bellpepper_core as bc;
        impl<Scalar: PrimeField> std::fmt::Debug for $type {
            fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                fmt.debug_struct(stringify!($type))
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

        impl<Scalar: PrimeField> PartialEq for $type {
            fn eq(&self, other: &$type) -> bool {
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

        impl<Scalar: PrimeField> bc::ConstraintSystem<Scalar> for $type {
            type Root = Self;

            fn new() -> Self {
                Self::default()
            }

            fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<bc::Variable, bc::SynthesisError>
            where
                F: FnOnce() -> Result<Scalar, bc::SynthesisError>,
                A: FnOnce() -> AR,
                AR: Into<String>,
            {
                self.aux_assignment.push(f()?);
                self.a_aux_density.add_element();
                self.b_aux_density.add_element();

                Ok(bc::Variable(bc::Index::Aux(self.aux_assignment.len() - 1)))
            }

            fn alloc_input<F, A, AR>(
                &mut self,
                _: A,
                f: F,
            ) -> Result<bc::Variable, bc::SynthesisError>
            where
                F: FnOnce() -> Result<Scalar, bc::SynthesisError>,
                A: FnOnce() -> AR,
                AR: Into<String>,
            {
                self.input_assignment.push(f()?);
                self.b_input_density.add_element();

                Ok(bc::Variable(bc::Index::Input(
                    self.input_assignment.len() - 1,
                )))
            }

            fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
            where
                A: FnOnce() -> AR,
                AR: Into<String>,
                LA: FnOnce(bc::LinearCombination<Scalar>) -> bc::LinearCombination<Scalar>,
                LB: FnOnce(bc::LinearCombination<Scalar>) -> bc::LinearCombination<Scalar>,
                LC: FnOnce(bc::LinearCombination<Scalar>) -> bc::LinearCombination<Scalar>,
            {
                let a = a(bc::LinearCombination::zero());
                let b = b(bc::LinearCombination::zero());
                let c = c(bc::LinearCombination::zero());

                let input_assignment = &self.input_assignment;
                let aux_assignment = &self.aux_assignment;
                let a_aux_density = &mut self.a_aux_density;
                let b_input_density = &mut self.b_input_density;
                let b_aux_density = &mut self.b_aux_density;

                let a_res = crate::lc::eval_with_trackers(
                    &a,
                    // Inputs have full density in the A query
                    // because there are constraints of the
                    // form x * 0 = 0 for each input.
                    None,
                    Some(a_aux_density),
                    input_assignment,
                    aux_assignment,
                );

                let b_res = crate::lc::eval_with_trackers(
                    &b,
                    Some(b_input_density),
                    Some(b_aux_density),
                    input_assignment,
                    aux_assignment,
                );

                // There is no C polynomial query,
                // though there is an (beta)A + (alpha)B + C
                // query for all aux variables.
                // However, that query has full density.
                let c_res = crate::lc::eval(&c, input_assignment, aux_assignment);

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

            fn extend(&mut self, other: &Self) {
                self.a_aux_density.extend(&other.a_aux_density, false);
                self.b_input_density.extend(&other.b_input_density, true);
                self.b_aux_density.extend(&other.b_aux_density, false);

                self.a.extend(&other.a);
                self.b.extend(&other.b);
                self.c.extend(&other.c);

                self.input_assignment
                    // Skip first input, which must have been a temporarily allocated one variable.
                    .extend(&other.input_assignment[1..]);
                self.aux_assignment.extend(&other.aux_assignment);
            }
        }
    };
}
pub(crate) use proving_assignment_impls;

#[derive(Default)]
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

proving_assignment_impls!(ProvingAssignment<Scalar>);

pub(super) fn create_random_proof_batch_priority<E, C, R, P: ParameterSource<E>>(
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

/// creates a batch of proofs where the randomization vector is already
/// predefined
pub(super) fn create_proof_batch_priority<E, C, P: ParameterSource<E>>(
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
    prover::create_proof_batch_priority_inner(circuits, params, Some((r_s, s_s)), priority)
}

fn synthesize_circuits_batch<Scalar, C>(
    circuits: Vec<C>,
) -> Result<std::vec::Vec<ProvingAssignment<Scalar>>, SynthesisError>
where
    Scalar: PrimeField,
    C: Circuit<Scalar> + Send,
{
    let start = Instant::now();
    let provers = circuits
        .into_par_iter()
        .map(|circuit| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignment::new();

            prover.alloc_input(|| "", || Ok(Scalar::ONE))?;

            circuit.synthesize(&mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            Ok(prover)
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!("synthesis time: {:?}", start.elapsed());

    Ok(provers)
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
                    .alloc_input(|| "one", || Ok(<Fr as Field>::ONE))
                    .unwrap();

                let mut partial_assignments = Vec::with_capacity(count / k);
                for i in 0..count {
                    if i % k == 0 {
                        let mut p = ProvingAssignment::new();
                        p.alloc_input(|| "one", || Ok(<Fr as Field>::ONE)).unwrap();
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
                    .alloc_input(|| "one", || Ok(<Fr as Field>::ONE))
                    .unwrap();

                for assignment in partial_assignments.iter() {
                    combined.extend(assignment);
                }
                assert_eq!(combined, full_assignment);
            }
        }
    }
}
