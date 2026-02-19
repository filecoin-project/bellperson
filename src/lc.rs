use crate::LinearCombination;
use ec_gpu_gen::multiexp_cpu::DensityTracker;
use ff::PrimeField;

#[allow(dead_code)]
pub fn eval_with_trackers<Scalar: PrimeField>(
    lc: &LinearCombination<Scalar>,
    mut input_density: Option<&mut DensityTracker>,
    mut aux_density: Option<&mut DensityTracker>,
    input_assignment: &[Scalar],
    aux_assignment: &[Scalar],
) -> Scalar {
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
