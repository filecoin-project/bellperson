//! Prover implementation implemented using SupraSeal (C++).

use std::time::Instant;

use bellpepper_core::{Circuit, ConstraintSystem, Index, SynthesisError, Variable};
use ff::{Field, PrimeField};
use log::info;
use pairing::MultiMillerLoop;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{ParameterSource, Proof, ProvingAssignment};
use crate::{gpu::GpuName, BELLMAN_VERSION};

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
    for prover in &provers {
        assert_eq!(
            prover.a.len(),
            provers[0].a.len(),
            "only equaly sized circuits are supported"
        );
    }

    impl<Scalar> Into<supraseal_c2::Assignment<Scalar>> for &ProvingAssignment<Scalar>
    where
        Scalar: PrimeField,
    {
        fn into(self) -> supraseal_c2::Assignment<Scalar> {
            assert_eq!(self.a.len(), self.b.len());
            assert_eq!(self.a.len(), self.c.len());

            supraseal_c2::Assignment::<Scalar> {
                a_aux_density: self.a_aux_density.bv.as_raw_slice().as_ptr(),
                a_aux_bit_len: self.a_aux_density.bv.len(),
                a_aux_popcount: self.a_aux_density.get_total_density(),

                b_inp_density: self.b_input_density.bv.as_raw_slice().as_ptr(),
                b_inp_bit_len: self.b_input_density.bv.len(),
                b_inp_popcount: self.b_input_density.get_total_density(),

                b_aux_density: self.b_aux_density.bv.as_raw_slice().as_ptr(),
                b_aux_bit_len: self.b_aux_density.bv.len(),
                b_aux_popcount: self.b_aux_density.get_total_density(),

                a: self.a.as_ptr(),
                b: self.b.as_ptr(),
                c: self.c.as_ptr(),
                abc_size: self.a.len(),

                inp_assignment_data: self.input_assignment.as_ptr(),
                inp_assignment_size: self.input_assignment.len(),

                aux_assignment_data: self.aux_assignment.as_ptr(),
                aux_assignment_size: self.aux_assignment.len(),
            }
        }
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
