use std::ops::{AddAssign, Mul, MulAssign};

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use pairing::{Engine, MillerLoopResult, MultiMillerLoop};
use rayon::prelude::*;

use super::{multiscalar, PreparedVerifyingKey, Proof, VerifyingKey};
use crate::{le_bytes_to_u64s, SynthesisError};

/// Generate a prepared verifying key, required to verify a proofs.
pub fn prepare_verifying_key<E: Engine + MultiMillerLoop>(
    vk: &VerifyingKey<E>,
) -> PreparedVerifyingKey<E>
where
    E: MultiMillerLoop,
{
    let neg_gamma = -vk.gamma_g2;
    let neg_delta = -vk.delta_g2;

    let multiscalar = multiscalar::precompute_fixed_window(&vk.ic, multiscalar::WINDOW_SIZE);

    PreparedVerifyingKey {
        alpha_g1_beta_g2: E::pairing(&vk.alpha_g1, &vk.beta_g2),
        neg_gamma_g2: neg_gamma.into(),
        neg_delta_g2: neg_delta.into(),
        gamma_g2: vk.gamma_g2.into(),
        delta_g2: vk.delta_g2.into(),
        ic: vk.ic.clone(),
        multiscalar,
        alpha_g1: vk.alpha_g1.to_curve(),
        beta_g2: vk.beta_g2.into(),
        ic_projective: vk.ic.par_iter().map(|i| i.to_curve()).collect(),
    }
}

/// Verify a single Proof.
pub fn verify_proof<'a, E>(
    pvk: &'a PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs: &[E::Fr],
) -> Result<bool, SynthesisError>
where
    E: MultiMillerLoop,
    <<E as Engine>::Fr as PrimeField>::Repr: Sync,
{
    use multiscalar::MultiscalarPrecomp;

    if (public_inputs.len() + 1) != pvk.ic.len() {
        return Err(SynthesisError::MalformedVerifyingKey);
    }

    // The original verification equation is:
    // A * B = alpha * beta + inputs * gamma + C * delta
    // ... however, we rearrange it so that it is:
    // A * B - inputs * gamma - C * delta = alpha * beta
    // or equivalently:
    // A * B + inputs * (-gamma) + C * (-delta) = alpha * beta
    // which allows us to do a single final exponentiation.

    // Miller Loop for alpha * beta
    let mut ml_a_b = Default::default();
    // Miller Loop for C * (-delta)
    let mut ml_all = <E as MultiMillerLoop>::Result::default();
    // Miller Loop for inputs * (-gamma)
    let mut ml_acc = <E as MultiMillerLoop>::Result::default();

    // Start the two independent miller loops
    rayon::in_place_scope(|s| {
        // - Thread 1: Calculate ML alpha * beta
        let ml_a_b = &mut ml_a_b;
        s.spawn(move |_| {
            *ml_a_b = E::multi_miller_loop(&[(&proof.a, &proof.b.into())]);
        });

        // - Thread 2: Calculate ML C * (-delta)
        let ml_all = &mut ml_all;
        s.spawn(move |_| *ml_all = E::multi_miller_loop(&[(&proof.c, &pvk.neg_delta_g2)]));

        // - Accumulate inputs (on the current thread)
        let subset = pvk.multiscalar.at_point(1);
        let public_inputs_repr: Vec<_> = public_inputs.iter().map(PrimeField::to_repr).collect();

        let mut acc = multiscalar::par_multiscalar::<&multiscalar::Getter<E::G1Affine>, E::G1Affine>(
            &multiscalar::ScalarList::Slice(&public_inputs_repr),
            &subset,
            std::mem::size_of::<<E::Fr as PrimeField>::Repr>() * 8,
        );

        acc.add_assign(&pvk.ic[0]);

        // Calculate ML inputs * (-gamma)
        let acc_aff = acc.to_affine();
        ml_acc = E::multi_miller_loop(&[(&acc_aff, &pvk.neg_gamma_g2)]);
    });
    // Wait for the threaded miller loops to finish

    // Combine the results.
    ml_all += ml_a_b;
    ml_all += ml_acc;

    // Calculate the final exponentiation
    let actual = ml_all.final_exponentiation();

    Ok(actual == pvk.alpha_g1_beta_g2)
}

/// Randomized batch verification - see Appendix B.2 in Zcash spec
pub fn verify_proofs_batch<'a, E, R>(
    pvk: &'a PreparedVerifyingKey<E>,
    rng: &mut R,
    proofs: &[&Proof<E>],
    public_inputs: &[Vec<E::Fr>],
) -> Result<bool, SynthesisError>
where
    E: MultiMillerLoop,
    <E::Fr as PrimeField>::Repr: Sync + Copy,
    R: rand::RngCore,
{
    debug_assert_eq!(proofs.len(), public_inputs.len());

    for pub_input in public_inputs {
        if (pub_input.len() + 1) != pvk.ic.len() {
            return Err(SynthesisError::MalformedVerifyingKey);
        }
    }

    let num_inputs = public_inputs[0].len();
    let num_proofs = proofs.len();

    if num_proofs < 2 {
        return verify_proof(pvk, proofs[0], &public_inputs[0]);
    }

    let proof_num = proofs.len();

    // Choose random coefficients for combining the proofs.
    let mut rand_z_repr: Vec<_> = Vec::with_capacity(proof_num);
    let mut rand_z: Vec<_> = Vec::with_capacity(proof_num);
    let mut accum_y = E::Fr::zero();

    for _ in 0..proof_num {
        use rand::Rng;

        let t: u128 = rng.gen();

        let mut repr = E::Fr::zero().to_repr();
        let mut repr_u64s = le_bytes_to_u64s(&repr.as_ref());
        assert!(repr_u64s.len() > 1);

        repr_u64s[0] = (t & (-1i64 as u128) >> 64) as u64;
        repr_u64s[1] = (t >> 64) as u64;

        for (i, limb) in repr_u64s.iter().enumerate() {
            let start = i * 8;
            let stop = start + 8;
            repr.as_mut()[start..stop].copy_from_slice(&limb.to_le_bytes());
        }

        let fr = E::Fr::from_repr(repr).unwrap();
        let repr = fr.to_repr();

        // calculate sum
        accum_y.add_assign(&fr);
        // store FrRepr
        rand_z_repr.push(repr);
        // store Fr
        rand_z.push(fr);
    }

    // MillerLoop(\sum Accum_Gamma)
    let mut ml_g = <E as MultiMillerLoop>::Result::default();
    // MillerLoop(Accum_Delta)
    let mut ml_d = <E as MultiMillerLoop>::Result::default();
    // MillerLoop(Accum_AB)
    let mut acc_ab = <E as MultiMillerLoop>::Result::default();
    // Y^-Accum_Y
    let mut y = <E as Engine>::Gt::identity();

    let accum_y = &accum_y;
    let rand_z_repr = &rand_z_repr;

    rayon::in_place_scope(|s| {
        // - Thread 1: Calculate MillerLoop(\sum Accum_Gamma)
        let ml_g = &mut ml_g;
        s.spawn(move |_| {
            let scalar_getter = |idx: usize| -> <E::Fr as ff::PrimeField>::Repr {
                if idx == 0 {
                    return accum_y.to_repr();
                }
                let idx = idx - 1;

                // \sum(z_j * aj,i)
                let mut cur_sum = rand_z[0];
                cur_sum.mul_assign(&public_inputs[0][idx]);

                for (pi_mont, mut rand_mont) in
                    public_inputs.iter().zip(rand_z.iter().copied()).skip(1)
                {
                    // z_j * a_j,i
                    let pi_mont = &pi_mont[idx];
                    rand_mont.mul_assign(pi_mont);
                    cur_sum.add_assign(&rand_mont);
                }

                cur_sum.to_repr()
            };

            // \sum Accum_Gamma
            let acc_g_psi = multiscalar::par_multiscalar::<_, E::G1Affine>(
                &multiscalar::ScalarList::Getter(scalar_getter, num_inputs + 1),
                &pvk.multiscalar,
                256,
            );

            // MillerLoop(acc_g_psi, vk.gamma)
            *ml_g = E::multi_miller_loop(&[(&acc_g_psi.to_affine(), &pvk.gamma_g2)]);
        });

        // - Thread 2: Calculate MillerLoop(Accum_Delta)
        let ml_d = &mut ml_d;
        s.spawn(move |_| {
            let points: Vec<_> = proofs.iter().map(|p| p.c).collect();

            // Accum_Delta
            let acc_d: E::G1 = {
                let pre = multiscalar::precompute_fixed_window::<E::G1Affine>(&points, 1);
                multiscalar::multiscalar::<E::G1Affine>(
                    &rand_z_repr,
                    &pre,
                    std::mem::size_of::<<E::Fr as PrimeField>::Repr>() * 8,
                )
            };

            *ml_d = E::multi_miller_loop(&[(&acc_d.to_affine(), &pvk.delta_g2)]);
        });

        // - Thread 3: Calculate MillerLoop(Accum_AB)
        let acc_ab = &mut acc_ab;
        s.spawn(move |_| {
            let accum_ab_mls: Vec<_> = proofs
                .par_iter()
                .zip(rand_z_repr.par_iter())
                .map(|(proof, rand)| {
                    // [z_j] pi_j,A
                    let mul_a = proof.a.mul(E::Fr::from_repr(*rand).unwrap());

                    // -pi_j,B
                    let cur_neg_b = -proof.b.to_curve();

                    E::multi_miller_loop(&[(&mul_a.to_affine(), &cur_neg_b.to_affine().into())])
                })
                .collect();

            // Accum_AB = mul_j(ml((zj*proof_aj), -proof_bj))
            *acc_ab = accum_ab_mls[0];
            for accum in accum_ab_mls.iter().skip(1).take(num_proofs) {
                *acc_ab += accum;
            }
        });

        // Thread 4(current): Calculate Y^-Accum_Y
        // -Accum_Y
        let accum_y_neg = -*accum_y;

        // Y^-Accum_Y
        y = pvk.alpha_g1_beta_g2 * accum_y_neg;
    });

    let mut ml_all = acc_ab;
    ml_all += ml_d;
    ml_all += ml_g;

    let actual = ml_all.final_exponentiation();
    Ok(actual == y)
}
