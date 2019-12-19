use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use paired::{Engine, PairingCurveAffine};

use super::{PreparedVerifyingKey, Proof, VerifyingKey};
use crate::SynthesisError;

pub fn prepare_verifying_key<E: Engine>(vk: &VerifyingKey<E>) -> PreparedVerifyingKey<E> {
    let mut gamma = vk.gamma_g2;
    gamma.negate();
    let mut delta = vk.delta_g2;
    delta.negate();

    PreparedVerifyingKey {
        alpha_g1_beta_g2: E::pairing(vk.alpha_g1, vk.beta_g2),
        neg_gamma_g2: gamma.prepare(),
        neg_delta_g2: delta.prepare(),
        ic: vk.ic.clone(),
    }
}

pub fn verify_proof<'a, E: Engine>(
    pvk: &'a PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs: &[E::Fr],
) -> Result<bool, SynthesisError> {
    if (public_inputs.len() + 1) != pvk.ic.len() {
        return Err(SynthesisError::MalformedVerifyingKey);
    }

    let mut acc = pvk.ic[0].into_projective();

    for (i, b) in public_inputs.iter().zip(pvk.ic.iter().skip(1)) {
        acc.add_assign(&b.mul(i.into_repr()));
    }

    // The original verification equation is:
    // A * B = alpha * beta + inputs * gamma + C * delta
    // ... however, we rearrange it so that it is:
    // A * B - inputs * gamma - C * delta = alpha * beta
    // or equivalently:
    // A * B + inputs * (-gamma) + C * (-delta) = alpha * beta
    // which allows us to do a single final exponentiation.

    Ok(E::final_exponentiation(&E::miller_loop(
        [
            (&proof.a.prepare(), &proof.b.prepare()),
            (&acc.into_affine().prepare(), &pvk.neg_gamma_g2),
            (&proof.c.prepare(), &pvk.neg_delta_g2),
        ]
        .iter(),
    ))
    .unwrap()
        == pvk.alpha_g1_beta_g2)
}

// randomized batch verification - see Appendix B.2 in Zcash spec
pub fn verify_proofs<'a, E: Engine, R: rand::RngCore>(
    pvk: &'a PreparedVerifyingKey<E>,
    rng: &mut R,
    proofs: &[Proof<E>],
    public_inputs: &[Vec<E::Fr>],
) -> Result<bool, SynthesisError>
where
    <<E as ff::ScalarEngine>::Fr as ff::PrimeField>::Repr: From<<E as ff::ScalarEngine>::Fr>,
{
    for pub_input in public_inputs {
        if (pub_input.len() + 1) != pvk.ic.len() {
            return Err(SynthesisError::MalformedVerifyingKey);
        }
    }

    let pi_num = pvk.ic.len() - 1;
    let proof_num = proofs.len();

    // choose random coefficients for combining the proofs
    let mut r = vec![];
    for _ in 0..proof_num {
        r.push(E::Fr::random(rng));
    }

    // create corresponding scalars for public input vk elements
    let mut pi_scalars = vec![];

    for i in 0..pi_num {
        pi_scalars.push(E::Fr::zero());
        for j in 0..proof_num {
            let mut tmp = r[j];
            tmp.mul_assign(&public_inputs[j][i]);
            pi_scalars[i].add_assign(&tmp);
        }
    }

    // create group element corresponding to public input combination
    // This roughly corresponds to Accum_Gamma in spec
    let mut acc_pi = pvk.ic[0].into_projective();

    for (i, b) in pi_scalars.iter().zip(pvk.ic.iter().skip(1)) {
        acc_pi.add_assign(&b.mul(i.into_repr()));
    }

    // TODO: why is this not used?
    let acc_pi = acc_pi.into_affine().prepare();

    let mut sum_r = E::Fr::zero();
    for i in r.iter() {
        sum_r.add_assign(i);
    }
    sum_r.negate();
    let acc_y = pvk.alpha_g1_beta_g2.pow(&sum_r.into_repr());

    // This corresponds to Accum_Delta
    let mut acc_c = E::G1::zero();
    for (rand_coeff, proof) in r.iter().zip(proofs.iter()) {
        let mut tmp: E::G1 = proof.c.into();
        tmp.mul_assign(*rand_coeff);
        acc_c.add_assign(&tmp);
    }

    let acc_c = acc_c.into_affine().prepare();

    let mut ml_g1 = vec![];
    let mut ml_g2 = vec![];
    for (rand_coeff, proof) in r.into_iter().zip(proofs.iter()) {
        let mut tmp: E::G1 = proof.a.into();
        tmp.mul_assign(rand_coeff);
        ml_g1.push(tmp.into_affine().prepare());
        let mut tmp: E::G2 = proof.b.into();
        tmp.negate();
        ml_g2.push(tmp.into_affine().prepare());
    }
    let parts = ml_g1.iter().zip(ml_g2.iter()).collect::<Vec<_>>();
    let mut acc_ab = E::miller_loop(&parts);
    acc_ab.mul_assign(&E::miller_loop(&[(&acc_c, &pvk.neg_delta_g2)]));

    Ok(E::final_exponentiation(&acc_ab).unwrap() == acc_y)
}
