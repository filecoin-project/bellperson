use ff::PrimeField;
use group::prime::PrimeCurveAffine;
use pairing::{MillerLoopResult, MultiMillerLoop};
use rayon::prelude::*;

use crate::groth16::multiscalar::*;
use crate::SynthesisError;

/// Returns the miller loop evaluated on inputs, i.e.
/// e(l_1,r_1)e(l_2,r_2)...
/// NOTE: the result is not in the final subgroup, one must run
/// `E::final_exponentiation` to use the final result.
pub(crate) fn pairing_miller_affine<E: MultiMillerLoop>(
    left: &[E::G1Affine],
    right: &[E::G2Affine],
) -> Result<<E as MultiMillerLoop>::Result, SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::IncompatibleLengthVector(
            "pairing_miller_affine left and right".to_string(),
        ));
    }
    let prepared: Vec<E::G2Prepared> = right.par_iter().map(|&p| p.into()).collect();
    let pairs_ref: Vec<_> = left
        .iter()
        .zip(prepared.iter())
        .map(|(a, b)| (a, b))
        .collect();

    Ok(E::multi_miller_loop(&pairs_ref))
}

/// Returns the miller loop result of the inner pairing product
pub(crate) fn pairing<E: MultiMillerLoop>(
    left: &[E::G1Affine],
    right: &[E::G2Affine],
) -> Result<E::Gt, SynthesisError> {
    Ok(pairing_miller_affine::<E>(left, right)?.final_exponentiation())
}

pub(crate) fn multiexponentiation<G>(
    left: &[G],
    right: &[G::Scalar],
) -> Result<G::Curve, SynthesisError>
where
    G: PrimeCurveAffine,
    <G::Scalar as PrimeField>::Repr: Sync,
{
    if left.len() != right.len() {
        return Err(SynthesisError::IncompatibleLengthVector(
            "multiexponentiation left and right".to_string(),
        ));
    }

    let table = precompute_fixed_window::<G>(&left, WINDOW_SIZE);
    let getter = |i: usize| -> <G::Scalar as PrimeField>::Repr { right[i].to_repr() };
    Ok(par_multiscalar::<_, G>(
        &ScalarList::Getter(getter, right.len()),
        &table,
        std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
    ))
}
