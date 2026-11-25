use bellpepper_core::{Circuit, SynthesisError};

use super::prover::{create_proof_batch_priority, create_random_proof_batch_priority};
use super::{ParameterSource, Proof};
use crate::gpu;
use pairing::MultiMillerLoop;
use rand_core::Rng;

/// Creates a single proof where the randomization vector is already predefined
pub fn create_proof<E, C, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    r: E::Fr,
    s: E::Fr,
) -> Result<Proof<E>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    let proofs =
        create_proof_batch_priority::<E, C, P>(vec![circuit], params, vec![r], vec![s], false)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a single proof.
pub fn create_random_proof<E, C, R, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: Rng,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    let proofs =
        create_random_proof_batch_priority::<E, C, R, P>(vec![circuit], params, rng, false)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a batch of proofs where the randomization vector is already predefined
pub fn create_proof_batch<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r: Vec<E::Fr>,
    s: Vec<E::Fr>,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    create_proof_batch_priority::<E, C, P>(circuits, params, r, s, false)
}

/// Creates a batch of proofs.
pub fn create_random_proof_batch<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: Rng,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    create_random_proof_batch_priority::<E, C, R, P>(circuits, params, rng, false)
}

/// Creates a single proof.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_proof_in_priority<E, C, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    r: E::Fr,
    s: E::Fr,
) -> Result<Proof<E>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    let proofs =
        create_proof_batch_priority::<E, C, P>(vec![circuit], params, vec![r], vec![s], true)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a batch of proofs.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_random_proof_in_priority<E, C, R, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: Rng,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    let proofs =
        create_random_proof_batch_priority::<E, C, R, P>(vec![circuit], params, rng, true)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a batch of proofs where the randomization vector is already predefined.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_proof_batch_in_priority<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r: Vec<E::Fr>,
    s: Vec<E::Fr>,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    create_proof_batch_priority::<E, C, P>(circuits, params, r, s, true)
}

/// Creates a batch of proofs.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_random_proof_batch_in_priority<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: Rng,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    create_random_proof_batch_priority::<E, C, R, P>(circuits, params, rng, true)
}
