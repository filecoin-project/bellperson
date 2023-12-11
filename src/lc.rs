use std::ops;

use ec_gpu_gen::multiexp_cpu::DensityTracker;
use ff::PrimeField;

use crate::LinearCombination;

/// Copy of `eval` from bellpepper that also works with a
/// [`groth16::prover::superaseal::ScalarVec`].
// `T` is a slice of `Scalar`s. This way it works with `&[Scalar]` as well as `&ScalarVec<Scalar>`
pub(crate) fn eval<'a, Scalar, T>(
    lc: &'a LinearCombination<Scalar>,
    input_assignment: &[Scalar],
    aux_assignment: &'a T,
) -> Scalar
where
    Scalar: PrimeField + ops::AddAssign<T::Output>,
    T: ops::Index<usize>,
    T::Output: PrimeField + std::ops::MulAssign<&'a Scalar>,
{
    let mut acc = Scalar::ZERO;

    let one = Scalar::ONE;

    for (index, coeff) in lc.iter_inputs() {
        let mut tmp = input_assignment[*index];
        if coeff != &one {
            tmp *= coeff;
        }
        acc += tmp;
    }

    for (index, coeff) in lc.iter_aux() {
        let mut tmp = aux_assignment[*index];
        if coeff != &one {
            tmp *= coeff;
        }
        acc += tmp;
    }

    acc
}

// `T` is a slice of `Scalar`s. This way it works with `&[Scalar]` as well as `&ScalarVec<Scalar>`
pub(crate) fn eval_with_trackers<'a, Scalar, T>(
    lc: &'a LinearCombination<Scalar>,
    mut input_density: Option<&'a mut DensityTracker>,
    mut aux_density: Option<&'a mut DensityTracker>,
    input_assignment: &[Scalar],
    aux_assignment: &'a T,
) -> Scalar
where
    Scalar: PrimeField + ops::AddAssign<T::Output>,
    T: ops::Index<usize>,
    T::Output: PrimeField + std::ops::MulAssign<&'a Scalar>,
{
    let mut acc = Scalar::ZERO;

    let one = Scalar::ONE;

    for (index, coeff) in lc.iter_inputs() {
        if !coeff.is_zero_vartime() {
            let mut tmp = input_assignment[*index];
            if coeff != &one {
                tmp *= coeff;
            }
            acc += tmp;

            if let Some(ref mut v) = input_density {
                v.inc(*index);
            }
        }
    }

    for (index, coeff) in lc.iter_aux() {
        if !coeff.is_zero_vartime() {
            let mut tmp = aux_assignment[*index];
            if coeff != &one {
                tmp *= coeff;
            }
            acc += tmp;

            if let Some(ref mut v) = aux_density {
                v.inc(*index);
            }
        }
    }

    acc
}
