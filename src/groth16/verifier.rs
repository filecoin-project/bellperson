use crate::bls::{Engine, PairingCurveAffine};
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;

use super::{multiscalar, PreparedVerifyingKey, Proof, VerifyingKey};
use crate::SynthesisError;

/// Generate a prepared verifying key, required to verify a proofs.
pub fn prepare_verifying_key<E: Engine>(vk: &VerifyingKey<E>) -> PreparedVerifyingKey<E> {
    let mut neg_gamma = vk.gamma_g2;
    neg_gamma.negate();
    let mut neg_delta = vk.delta_g2;
    neg_delta.negate();

    let multiscalar = multiscalar::precompute_fixed_window(&vk.ic, multiscalar::WINDOW_SIZE);

    PreparedVerifyingKey {
        alpha_g1_beta_g2: E::pairing(vk.alpha_g1, vk.beta_g2),
        neg_gamma_g2: neg_gamma.prepare(),
        neg_delta_g2: neg_delta.prepare(),
        gamma_g2: vk.gamma_g2.prepare(),
        delta_g2: vk.delta_g2.prepare(),
        ic: vk.ic.clone(),
        multiscalar,
        alpha_g1: vk.alpha_g1.into_projective(),
        beta_g2: vk.beta_g2.prepare(),
        ic_projective: vk.ic.par_iter().map(|i| i.into_projective()).collect(),
    }
}

/// Verify a single Proof.
pub fn verify_proof<'a, E: Engine>(
    pvk: &'a PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs: &[E::Fr],
) -> Result<bool, SynthesisError> {
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
    let mut ml_a_b = E::Fqk::zero();
    // Miller Loop for C * (-delta)
    let mut ml_all = E::Fqk::zero();
    // Miller Loop for inputs * (-gamma)
    let mut ml_acc = E::Fqk::zero();

    // Start the two independent miller loops
    rayon::in_place_scope(|s| {
        // - Thread 1: Calculate ML alpha * beta
        let ml_a_b = &mut ml_a_b;
        s.spawn(move |_| {
            *ml_a_b = E::miller_loop(&[(&proof.a.prepare(), &proof.b.prepare())]);
        });

        // - Thread 2: Calculate ML C * (-delta)
        let ml_all = &mut ml_all;
        s.spawn(move |_| *ml_all = E::miller_loop(&[(&proof.c.prepare(), &pvk.neg_delta_g2)]));

        // - Accumulate inputs (on the current thread)
        let subset = pvk.multiscalar.at_point(1);
        let public_inputs_repr: Vec<_> = public_inputs.iter().map(PrimeField::into_repr).collect();

        let mut acc = multiscalar::par_multiscalar::<&multiscalar::Getter<E::G1Affine>, E::G1Affine>(
            &multiscalar::ScalarList::Slice(&public_inputs_repr),
            &subset,
            std::mem::size_of::<<E::Fr as PrimeField>::Repr>() * 8,
        );

        acc.add_assign_mixed(&pvk.ic[0]);

        // Calculate ML inputs * (-gamma)
        let acc_aff = acc.into_affine();
        ml_acc = E::miller_loop(&[(&acc_aff.prepare(), &pvk.neg_gamma_g2)]);
    });
    // Wait for the threaded miller loops to finish

    // Combine the results.
    ml_all.mul_assign(&ml_a_b);
    ml_all.mul_assign(&ml_acc);

    // Calculate the final exponentiation
    let actual = E::final_exponentiation(&ml_all).unwrap();

    Ok(actual == pvk.alpha_g1_beta_g2)
}

/// Randomized batch verification - see Appendix B.2 in Zcash spec
pub fn verify_proofs_batch<'a, E: Engine, R: rand::RngCore>(
    pvk: &'a PreparedVerifyingKey<E>,
    rng: &mut R,
    proofs: &[&Proof<E>],
    public_inputs: &[Vec<E::Fr>],
) -> Result<bool, SynthesisError>
where
    <<E as ff::ScalarEngine>::Fr as ff::PrimeField>::Repr: From<<E as ff::ScalarEngine>::Fr>,
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
        let mut el = E::Fr::zero().into_repr();
        let el_ref: &mut [u64] = el.as_mut();
        assert!(el_ref.len() > 1);

        el_ref[0] = (t & (-1i64 as u128) >> 64) as u64;
        el_ref[1] = (t >> 64) as u64;

        let fr = E::Fr::from_repr(el).unwrap();

        // calculate sum
        accum_y.add_assign(&fr);
        // store FrRepr
        rand_z_repr.push(el);
        // store Fr
        rand_z.push(fr);
    }

    // MillerLoop(\sum Accum_Gamma)
    let mut ml_g = E::Fqk::zero();
    // MillerLoop(Accum_Delta)
    let mut ml_d = E::Fqk::zero();
    // MillerLoop(Accum_AB)
    let mut acc_ab = E::Fqk::zero();
    // Y^-Accum_Y
    let mut y = E::Fqk::zero();

    let accum_y = &accum_y;
    let rand_z_repr = &rand_z_repr;

    rayon::in_place_scope(|s| {
        // - Thread 1: Calculate MillerLoop(\sum Accum_Gamma)
        let ml_g = &mut ml_g;
        s.spawn(move |_| {
            let scalar_getter = |idx: usize| -> <E::Fr as ff::PrimeField>::Repr {
                if idx == 0 {
                    return accum_y.into_repr();
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

                cur_sum.into_repr()
            };

            // \sum Accum_Gamma
            let acc_g_psi = multiscalar::par_multiscalar::<_, E::G1Affine>(
                &multiscalar::ScalarList::Getter(scalar_getter, num_inputs + 1),
                &pvk.multiscalar,
                256,
            );

            // MillerLoop(acc_g_psi, vk.gamma)
            *ml_g = E::miller_loop(&[(&acc_g_psi.into_affine().prepare(), &pvk.gamma_g2)]);
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

            *ml_d = E::miller_loop(&[(&acc_d.into_affine().prepare(), &pvk.delta_g2)]);
        });

        // - Thread 3: Calculate MillerLoop(Accum_AB)
        let acc_ab = &mut acc_ab;
        s.spawn(move |_| {
            let accum_ab_mls: Vec<_> = proofs
                .par_iter()
                .zip(rand_z_repr.par_iter())
                .map(|(proof, rand)| {
                    // [z_j] pi_j,A
                    let mul_a = proof.a.mul(*rand);

                    // -pi_j,B
                    let mut cur_neg_b = proof.b.into_projective();
                    cur_neg_b.negate();

                    E::miller_loop(&[(
                        &mul_a.into_affine().prepare(),
                        &cur_neg_b.into_affine().prepare(),
                    )])
                })
                .collect();

            // Accum_AB = mul_j(ml((zj*proof_aj), -proof_bj))
            *acc_ab = accum_ab_mls[0];
            for accum in accum_ab_mls.iter().skip(1).take(num_proofs) {
                acc_ab.mul_assign(accum);
            }
        });

        // Thread 4(current): Calculate Y^-Accum_Y
        // -Accum_Y
        let mut accum_y_neg = *accum_y;
        accum_y_neg.negate();

        // Y^-Accum_Y
        y = pvk.alpha_g1_beta_g2.pow(&accum_y_neg.into_repr());
    });

    let mut ml_all = acc_ab;
    ml_all.mul_assign(&ml_d);
    ml_all.mul_assign(&ml_g);

    Ok(E::final_exponentiation(&ml_all).unwrap() == y)
}
