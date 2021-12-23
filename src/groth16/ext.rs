use super::{create_proof_batch_priority, create_random_proof_batch_priority};
use super::{ParameterSource, Proof};
use crate::{gpu, Circuit, SynthesisError};
use pairing::MultiMillerLoop;
use rand_core::RngCore;

pub fn create_proof<E, C, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    r: E::Fr,
    s: E::Fr,
) -> Result<Proof<E>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
{
    let proofs = create_proof_batch_priority::<E, C, P>(
        vec![circuit],
        params,
        vec![r],
        vec![s],
        false,
        false,
    )?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_random_proof<E, C, R, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
{
    let proofs =
        create_random_proof_batch_priority::<E, C, R, P>(vec![circuit], params, rng, false, false)?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_proof_batch<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r: Vec<E::Fr>,
    s: Vec<E::Fr>,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
{
    create_proof_batch_priority::<E, C, P>(circuits, params, r, s, false, false)
}

pub fn create_random_proof_batch<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
    is_win_post: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
{
    create_random_proof_batch_priority::<E, C, R, P>(circuits, params, rng, false, is_win_post)
}

pub fn create_proof_in_priority<E, C, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    r: E::Fr,
    s: E::Fr,
) -> Result<Proof<E>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
{
    let proofs = create_proof_batch_priority::<E, C, P>(
        vec![circuit],
        params,
        vec![r],
        vec![s],
        true,
        false,
    )?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_random_proof_in_priority<E, C, R, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
{
    let proofs =
        create_random_proof_batch_priority::<E, C, R, P>(vec![circuit], params, rng, true, false)?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_proof_batch_in_priority<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r: Vec<E::Fr>,
    s: Vec<E::Fr>,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
{
    create_proof_batch_priority::<E, C, P>(circuits, params, r, s, true, false)
}

pub fn create_random_proof_batch_in_priority<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
    is_win_post: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
{
    create_random_proof_batch_priority::<E, C, R, P>(circuits, params, rng, true, is_win_post)
}
