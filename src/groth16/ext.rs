use super::{create_proof_batch_priority, create_random_proof_batch_priority};
use super::{ParameterSource, Proof};
use crate::groth16::BellTaskType;
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
    let proofs =
        create_proof_batch_priority::<E, C, P>(vec![circuit], params, vec![r], vec![s], None)?;
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
        create_random_proof_batch_priority::<E, C, R, P>(vec![circuit], params, rng, None)?;
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
    create_proof_batch_priority::<E, C, P>(circuits, params, r, s, None)
}

// Winning/window and tree_builders/hashers use this library. as a general use case, we should pass to
// the scheduler the deadlines and task types. The deadline is the priority, the sooner the task must be completed
// the more the scheduler will prioritize it.
// the task_type is used to get the exclusive resources that a  task can use.
// the rest of the API in this file remains the same, the no priority functions bellow, assign None as a
// deadline, the other functions with the in_priority suffix set the deadline to now. both keep the
// task_type as None, indicating there are not exclusive resources for them.

// this function omits the deadline but takes in a task_type, for now the deadline is hardcoded in
// the scheduler-client library according to the task_type. the scheduler will assigned the lowest
// possible deadline to tasks that do not have a type. this function is called by rust-fil-proof
// and it was easier to refactor the functions there to include only one parameter than modifying
// them to take 2(deadline plus task_type)
pub fn create_random_proof_batch_with_type<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
    task_type: Option<BellTaskType>,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
{
    create_random_proof_batch_priority::<E, C, R, P>(circuits, params, rng, task_type)
}

pub fn create_random_proof_with_type<E, C, R, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    rng: &mut R,
    task_type: Option<BellTaskType>,
) -> Result<Proof<E>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
{
    let proofs =
        create_random_proof_batch_priority::<E, C, R, P>(vec![circuit], params, rng, task_type)?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_random_proof_batch<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: gpu::GpuEngine + MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
{
    create_random_proof_batch_priority::<E, C, R, P>(circuits, params, rng, None)
}
