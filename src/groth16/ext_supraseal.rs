//! This module is a copy of of the `ext` module for when the `cuda-supraseal` feature is abled.
//! Technically it's not needed, but it's introduce in order to catch bugs at compile-time. The
//! difference is that it doesn't take the `params` per [`ParameterSource`] trait, but as the
//! concrete [`SuprasealParameters`] type. This way we get some kind of compile type checks on the
//! right types, when this public API is used. The lower-level functions only depend on the
//! `ParameterSource` trait and will dynamically do the right thing, in case the `cuda-supraseal`
//! feature is enabled.

use bellpepper_core::{Circuit, SynthesisError};
use pairing::MultiMillerLoop;
use rand_core::RngCore;

use crate::{
    gpu,
    groth16::{
        Proof, SuprasealParameters,
        params::ParameterSource,
        prover::{create_proof_batch_priority, create_random_proof_batch_priority},
    },
};

/// Creates a single proof where the randomization vector is already predefined.
pub fn create_proof<E, C, P: ParameterSource<E>>(
    //pub fn create_proof<E, C>(
    circuit: C,
    params: &SuprasealParameters<E>,
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
    let proofs = create_proof_batch_priority(vec![circuit], params, vec![r], vec![s], false)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a single proof.
pub fn create_random_proof<E, C, R>(
    circuit: C,
    params: &SuprasealParameters<E>,
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    let proofs = create_random_proof_batch_priority(vec![circuit], params, rng, false)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a batch of proofs where the randomization vector is already predefined
pub fn create_proof_batch<E, C>(
    circuits: Vec<C>,
    params: &SuprasealParameters<E>,
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
    create_proof_batch_priority(circuits, params, r, s, false)
}

/// Creates a batch of proofs.
pub fn create_random_proof_batch<E, C, R>(
    circuits: Vec<C>,
    params: &SuprasealParameters<E>,
    rng: &mut R,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    create_random_proof_batch_priority(circuits, params, rng, false)
}

/// Creates a single proof.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_proof_in_priority<E, C>(
    circuit: C,
    params: &SuprasealParameters<E>,
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
    let proofs = create_proof_batch_priority(vec![circuit], params, vec![r], vec![s], true)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a batch of proofs.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_random_proof_in_priority<E, C, R>(
    circuit: C,
    params: &SuprasealParameters<E>,
    rng: &mut R,
) -> Result<Proof<E>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    let proofs = create_random_proof_batch_priority(vec![circuit], params, rng, true)?;
    Ok(proofs.into_iter().next().unwrap())
}

/// Creates a batch of proofs where the randomization vector is already predefined.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_proof_batch_in_priority<E, C>(
    circuits: Vec<C>,
    params: &SuprasealParameters<E>,
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
    create_proof_batch_priority(circuits, params, r, s, true)
}

/// Creates a batch of proofs.
/// When several proofs are run in parallel on the GPU, it will get priority and will never be
/// aborted or pushed down to CPU.
pub fn create_random_proof_batch_in_priority<E, C, R>(
    circuits: Vec<C>,
    params: &SuprasealParameters<E>,
    rng: &mut R,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
    E::Fr: gpu::GpuName,
    E::G1Affine: gpu::GpuName,
    E::G2Affine: gpu::GpuName,
{
    create_random_proof_batch_priority(circuits, params, rng, true)
}
