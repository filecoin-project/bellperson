//! Prover implementation implemented using SupraSeal (C++).

use std::time::Instant;

use bellpepper_core::{Circuit, ConstraintSystem, Index, SynthesisError, Variable};
use ff::{Field, PrimeField};
use log::info;
use pairing::MultiMillerLoop;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{ParameterSource, Proof, ProvingAssignment};
use crate::{gpu::GpuName, BELLMAN_VERSION};

impl<Scalar> From<&ProvingAssignment<Scalar>> for supraseal_c2::Assignment<Scalar>
where
    Scalar: PrimeField,
{
    fn from(assignment: &ProvingAssignment<Scalar>) -> Self {
        assert_eq!(assignment.a.len(), assignment.b.len());
        assert_eq!(assignment.a.len(), assignment.c.len());

        Self {
            a_aux_density: assignment.a_aux_density.bv.as_raw_slice().as_ptr(),
            a_aux_bit_len: assignment.a_aux_density.bv.len(),
            a_aux_popcount: assignment.a_aux_density.get_total_density(),

            b_inp_density: assignment.b_input_density.bv.as_raw_slice().as_ptr(),
            b_inp_bit_len: assignment.b_input_density.bv.len(),
            b_inp_popcount: assignment.b_input_density.get_total_density(),

            b_aux_density: assignment.b_aux_density.bv.as_raw_slice().as_ptr(),
            b_aux_bit_len: assignment.b_aux_density.bv.len(),
            b_aux_popcount: assignment.b_aux_density.get_total_density(),

            a: assignment.a.as_ptr(),
            b: assignment.b.as_ptr(),
            c: assignment.c.as_ptr(),
            abc_size: assignment.a.len(),

            inp_assignment_data: assignment.input_assignment.as_ptr(),
            inp_assignment_size: assignment.input_assignment.len(),

            aux_assignment_data: assignment.aux_assignment.as_ptr(),
            aux_assignment_size: assignment.aux_assignment.len(),
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    randomization: Option<(Vec<E::Fr>, Vec<E::Fr>)>,
    _priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    info!(
        "Bellperson {} with SupraSeal is being used!",
        BELLMAN_VERSION
    );

    let provers = synthesize_circuits_batch(circuits)?;

    // Start fft/multiexp prover timer
    let start = Instant::now();
    info!("starting proof timer");

    let num_circuits = provers.len();
    let (r_s, s_s) = randomization.unwrap_or((
        vec![E::Fr::ZERO; num_circuits],
        vec![E::Fr::ZERO; num_circuits],
    ));

    // Make sure all circuits have the same input len.
    let n = provers[0].a.len();
    for prover in &provers {
        assert_eq!(
            prover.a.len(),
            n,
            "only equaly sized circuits are supported"
        );
    }

    let provers_c2: Vec<supraseal_c2::Assignment<E::Fr>> =
        provers.iter().map(|p| p.into()).collect();

    let mut proofs: Vec<Proof<E>> = Vec::with_capacity(num_circuits);
    // We call out to C++ code which is unsafe anyway, hence silence this warning.
    #[allow(clippy::uninit_vec)]
    unsafe {
        proofs.set_len(num_circuits);
    }

    let srs = params.get_supraseal_srs().ok_or_else(|| {
        log::error!("SupraSeal SRS wasn't allocated correctly");
        SynthesisError::MalformedSrs
    })?;
    supraseal_c2::generate_groth16_proofs(
        provers_c2.as_slice(),
        r_s.as_slice(),
        s_s.as_slice(),
        proofs.as_mut_slice(),
        srs,
    );

    let proof_time = start.elapsed();
    info!("prover time: {:?}", proof_time);

    Ok(proofs)
}

#[allow(clippy::type_complexity)]
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
