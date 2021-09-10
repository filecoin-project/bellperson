use blstrs::Compress;
use crossbeam_channel::bounded;
use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use log::debug;
use log::*;
use pairing::{Engine, MultiMillerLoop};
use rayon::prelude::*;
use serde::Serialize;

use super::{
    accumulator::PairingChecks,
    inner_product,
    prove::polynomial_evaluation_product_form_from_transcript,
    structured_scalar_power,
    transcript::{Challenge, Transcript},
    AggregateProof, KZGOpening, VerifierSRS,
};
use crate::groth16::{
    multiscalar::{par_multiscalar, MultiscalarPrecomp, ScalarList},
    PreparedVerifyingKey,
};
use crate::SynthesisError;

use std::default::Default;
use std::ops::{AddAssign, MulAssign, SubAssign};
use std::time::Instant;

/// Verifies the aggregated proofs thanks to the Groth16 verifying key, the
/// verifier SRS from the aggregation scheme, all the public inputs of the
/// proofs and the aggregated proof.
///
/// WARNING: transcript_include represents everything that should be included in
/// the transcript from outside the boundary of this function. This is especially
/// relevant for ALL public inputs of ALL individual proofs. In the regular case,
/// one should input ALL public inputs from ALL proofs aggregated. However, IF ALL the
/// public inputs are **fixed, and public before the aggregation time**, then there is
/// no need to hash those. The reason we specify this extra assumption is because hashing
/// the public inputs from the decoded form can take quite some time depending on the
/// number of proofs and public inputs (+100ms in our case). In the case of Filecoin, the only
/// non-fixed part of the public inputs are the challenges derived from a seed. Even though this
/// seed comes from a random beeacon, we are hashing this as a safety precaution.
pub fn verify_aggregate_proof<E, R>(
    ip_verifier_srs: &VerifierSRS<E>,
    pvk: &PreparedVerifyingKey<E>,
    rng: R,
    public_inputs: &[Vec<E::Fr>],
    proof: &AggregateProof<E>,
    transcript_include: &[u8],
) -> Result<bool, SynthesisError>
where
    E: MultiMillerLoop + std::fmt::Debug,
    E::Fr: Serialize,
    <E as Engine>::Gt: Compress + Serialize,
    E::G1: Serialize,
    E::G1Affine: Serialize,
    E::G2Affine: Serialize,
    R: rand_core::RngCore + Send,
{
    info!("verify_aggregate_proof");
    proof.parsing_check()?;
    for pub_input in public_inputs {
        if (pub_input.len() + 1) != pvk.ic.len() {
            return Err(SynthesisError::MalformedVerifyingKey);
        }
    }

    if public_inputs.len() != proof.tmipp.gipa.nproofs as usize {
        return Err(SynthesisError::MalformedProofs(
            "public inputs length does not match nproofs".to_string(),
        ));
    }

    let hcom = Transcript::<E>::new("hcom")
        .write(&proof.com_ab)
        .write(&proof.com_c)
        .into_challenge();

    // Random linear combination of proofs
    let r = Transcript::<E>::new("random-r")
        .write(&hcom)
        .write(&transcript_include)
        .into_challenge();

    let pairing_checks = PairingChecks::new(rng);
    let pairing_checks_copy = &pairing_checks;

    // 1.Check TIPA proof ab
    // 2.Check TIPA proof c
    //        s.spawn(move |_| {
    let now = Instant::now();
    verify_tipp_mipp::<E, R>(
        ip_verifier_srs,
        proof,
        &r, // we give the extra r as it's not part of the proof itself - it is simply used on top for the groth16 aggregation
        pairing_checks_copy,
        &hcom,
    );
    debug!("TIPP took {} ms", now.elapsed().as_millis(),);

    // Check aggregate pairing product equation
    // SUM of a geometric progression
    // SUM a^i = (1 - a^n) / (1 - a) = -(1-a^n)/-(1-a)
    // = (a^n - 1) / (a - 1)
    info!("checking aggregate pairing");
    let mut r_sum = r.pow_vartime(&[public_inputs.len() as u64]);
    r_sum.sub_assign(&E::Fr::one());
    let b = (*r - E::Fr::one()).invert().unwrap();
    r_sum.mul_assign(&b);

    // The following parts 3 4 5 are independently computing the parts of the Groth16
    // verification equation
    // NOTE From this point on, we are only checking *one* pairing check (the Groth16
    // verification equation) so we don't need to randomize as all other checks are being
    // randomized already. When merging all pairing checks together, this will be the only one
    // non-randomized.
    //
    let (r_vec_sender, r_vec_receiver) = bounded(1);

    let now = Instant::now();
    r_vec_sender
        .send(structured_scalar_power(public_inputs.len(), &*r))
        .unwrap();
    let elapsed = now.elapsed().as_millis();
    debug!("generation of r vector: {}ms", elapsed);

    par! {
        // 3. Compute left part of the final pairing equation
        let left = {
            let mut alpha_g1_r_sum = pvk.alpha_g1;
            alpha_g1_r_sum.mul_assign(r_sum);

            E::multi_miller_loop(&[(&alpha_g1_r_sum.to_affine(), &pvk.beta_g2)])
        },
        // 4. Compute right part of the final pairing equation
        let right = {
            E::multi_miller_loop(&[(
                // e(c^r vector form, h^delta)
                // let agg_c = inner_product::multiexponentiation::<E::G1Affine>(&c, r_vec)
                &proof.agg_c.to_affine(),
                &pvk.delta_g2,
            )])
        },
        // 5. compute the middle part of the final pairing equation, the one
        //    with the public inputs
        let middle = {
            // We want to compute MUL(i:0 -> l) S_i ^ (SUM(j:0 -> n) ai,j * r^j)
            // this table keeps tracks of incremental computation of each i-th
            // exponent to later multiply with S_i
            // The index of the table is i, which is an index of the public
            // input element
            // We incrementally build the r vector and the table
            // NOTE: in this version it's not r^2j but simply r^j

            let l = public_inputs[0].len();
            let mut g_ic = pvk.ic_projective[0];
            g_ic.mul_assign(r_sum);

            let powers = r_vec_receiver.recv().unwrap();

            let now = Instant::now();
            // now we do the multi exponentiation
            let getter = |i: usize| -> <E::Fr as PrimeField>::Repr {
                // i denotes the column of the public input, and j denotes which public input
                let mut c = public_inputs[0][i];
                for j in 1..public_inputs.len() {
                    let mut ai = public_inputs[j][i];
                    ai.mul_assign(&powers[j]);
                    c.add_assign(&ai);
                }
                c.to_repr()
            };

            let totsi = par_multiscalar::<_, E::G1Affine>(
                &ScalarList::Getter(getter, l),
                &pvk.multiscalar.at_point(1),
                std::mem::size_of::<<E::Fr as PrimeField>::Repr>() * 8,
            );

            g_ic.add_assign(&totsi);

            let ml = E::multi_miller_loop(&[(&g_ic.to_affine(), &pvk.gamma_g2)]);
            let elapsed = now.elapsed().as_millis();
            debug!("table generation: {}ms", elapsed);

            ml
        }
    };

    pairing_checks_copy.merge_nonrandom(
        vec![left, middle, right],
        // final value ip_ab is what we want to compare in the groth16
        // aggregated equation A * B
        proof.ip_ab,
    );

    let res = pairing_checks.verify();
    info!("aggregate verify done");
    res
}

/// verify_tipp_mipp returns a pairing equation to check the tipp proof.  $r$ is
/// the randomness used to produce a random linear combination of A and B and
/// used in the MIPP part with C
fn verify_tipp_mipp<E, R>(
    v_srs: &VerifierSRS<E>,
    proof: &AggregateProof<E>,
    r_shift: &E::Fr,
    pairing_checks: &PairingChecks<E, R>,
    hcom: &Challenge<E>,
) where
    E: MultiMillerLoop,
    E::Fr: Serialize,
    <E as Engine>::Gt: Compress + Serialize,
    E::G1: Serialize,
    E::G1Affine: Serialize,
    E::G2Affine: Serialize,
    R: rand_core::RngCore + Send,
{
    info!("verify with srs shift");
    let now = Instant::now();
    // (T,U), Z for TIPP and MIPP  and all challenges
    let (final_res, final_r, challenges, challenges_inv) =
        gipa_verify_tipp_mipp(&proof, r_shift, hcom);
    debug!(
        "TIPP verify: gipa verify tipp {}ms",
        now.elapsed().as_millis()
    );

    // Verify commitment keys wellformed
    let fvkey = proof.tmipp.gipa.final_vkey;
    let fwkey = proof.tmipp.gipa.final_wkey;
    // KZG challenge point
    let c = Transcript::<E>::new("random-z")
        .write(&challenges[0])
        .write(&fvkey.0)
        .write(&fvkey.1)
        .write(&fwkey.0)
        .write(&fwkey.1)
        .into_challenge();

    // we take reference so they are able to be copied in the par! macro
    let final_a = &proof.tmipp.gipa.final_a;
    let final_b = &proof.tmipp.gipa.final_b;
    let final_c = &proof.tmipp.gipa.final_c;
    let final_zab = &final_res.zab;
    let final_tab = &final_res.tab;
    let final_uab = &final_res.uab;
    let final_tc = &final_res.tc;
    let final_uc = &final_res.uc;

    let now = Instant::now();
    par! {
        // check the opening proof for v
        let _vtuple = verify_kzg_v(
            v_srs,
            &fvkey,
            &proof.tmipp.vkey_opening,
            &challenges_inv,
            &c,
            pairing_checks,
        ),
        // check the opening proof for w - note that w has been rescaled by $r^{-1}$
        let _wtuple = verify_kzg_w(
            v_srs,
            &fwkey,
            &proof.tmipp.wkey_opening,
            &challenges,
            &r_shift.invert().unwrap(),
            &c,
            pairing_checks,
        ),
        //
        // We create a sequence of pairing tuple that we aggregate together at
        // the end to perform only once the final exponentiation.
        //
        // TIPP
        // z = e(A,B)
        let _check_z = pairing_checks.merge_miller_inputs(&[(final_a, final_b)], final_zab),
        //  final_aB.0 = T = e(A,v1)e(w1,B)
        let _check_ab0 = pairing_checks.merge_miller_inputs(&[(final_a, &fvkey.0),(&fwkey.0, final_b)], final_tab),

        //  final_aB.1 = U = e(A,v2)e(w2,B)
        let _check_ab1 = pairing_checks.merge_miller_inputs(&[(final_a, &fvkey.1),(&fwkey.1, final_b)], final_uab),

        // MIPP
        // Verify base inner product commitment
        // Z ==  c ^ r
        let final_z =
            inner_product::multiexponentiation::<E::G1Affine>(&[*final_c],
            &[final_r]),
        // Check commiment correctness
        // T = e(C,v1)
        let _check_t = pairing_checks.merge_miller_inputs(&[(final_c,&fvkey.0)], final_tc),
        // U = e(A,v2)
        let _check_u = pairing_checks.merge_miller_inputs(&[(final_c,&fvkey.1)], final_uc)
    };
    match final_z {
        Err(e) => pairing_checks.report_err(e),
        Ok(z) => {
            debug!(
                "TIPP verify: parallel checks before merge: {}ms",
                now.elapsed().as_millis(),
            );
            let b = z == final_res.zc;
            // only check that doesn't require pairing so we can give a tuple
            // that will render the equation wrong in case it's false
            if !b {
                pairing_checks.invalidate();
            }
        }
    }
}

/// gipa_verify_tipp_mipp recurse on the proof and statement and produces the final
/// values to be checked by TIPP and MIPP verifier, namely, for TIPP for example:
/// * T,U: the final commitment values of A and B
/// * Z the final product between A and B.
/// * Challenges are returned in inverse order as well to avoid
/// repeating the operation multiple times later on.
/// * There are T,U,Z vectors as well for the MIPP relationship. Both TIPP and
/// MIPP share the same challenges however, enabling to re-use common operations
/// between them, such as the KZG proof for commitment keys.
#[allow(clippy::type_complexity)]
fn gipa_verify_tipp_mipp<E>(
    proof: &AggregateProof<E>,
    r_shift: &E::Fr,
    hcom: &E::Fr,
) -> (GipaTUZ<E>, E::Fr, Vec<E::Fr>, Vec<E::Fr>)
where
    E: MultiMillerLoop,
    E::Fr: Serialize,
    <E as Engine>::Gt: Compress + Serialize,
    E::G1: Serialize,
{
    info!("gipa verify TIPP");
    let gipa = &proof.tmipp.gipa;
    // COM(A,B) = PROD e(A,B) given by prover
    let comms_ab = &gipa.comms_ab;
    // COM(C,r) = SUM C^r given by prover
    let comms_c = &gipa.comms_c;
    // Z vectors coming from the GIPA proofs
    let zs_ab = &gipa.z_ab;
    let zs_c = &gipa.z_c;

    let now = Instant::now();

    let mut challenges = Vec::new();
    let mut challenges_inv = Vec::new();

    let mut c_inv: E::Fr = *Transcript::<E>::new("gipa-0")
        .write(hcom)
        .write(&proof.ip_ab)
        .write(&proof.agg_c)
        .write(&r_shift)
        .into_challenge();
    let mut c = c_inv.invert().unwrap();

    // We first generate all challenges as this is the only consecutive process
    // that can not be parallelized then we scale the commitments in a
    // parallelized way
    for (i, ((comm_ab, z_ab), (comm_c, z_c))) in comms_ab
        .iter()
        .zip(zs_ab.iter())
        .zip(comms_c.iter().zip(zs_c.iter()))
        .enumerate()
    {
        let (tab_l, tab_r) = comm_ab;
        let (zab_l, zab_r) = z_ab;
        let (tc_l, tc_r) = comm_c;
        let (zc_l, zc_r) = z_c;

        // Fiat-Shamir challenge
        if i == 0 {
            // already generated c_inv and c outside of the loop
        } else {
            c_inv = *Transcript::<E>::new(&format!("gipa-{}", i))
                .write(&c_inv)
                .write(&zab_l)
                .write(&zab_r)
                .write(&zc_l)
                .write(&zc_r)
                .write(&tab_l.0)
                .write(&tab_l.1)
                .write(&tab_r.0)
                .write(&tab_r.1)
                .write(&tc_l.0)
                .write(&tc_l.1)
                .write(&tc_r.0)
                .write(&tc_r.1)
                .into_challenge();
            c = c_inv.invert().unwrap();
        }
        challenges.push(c);
        challenges_inv.push(c_inv);
    }

    debug!(
        "TIPP verify: gipa challenge gen took {}ms",
        now.elapsed().as_millis()
    );

    let now = Instant::now();
    // output of the pair commitment T and U in TIPP -> COM((v,w),A,B)
    let (t_ab, u_ab) = proof.com_ab;
    let z_ab = proof.ip_ab; // in the end must be equal to Z = A^r * B

    // COM(v,C)
    let (t_c, u_c) = proof.com_c;
    let z_c = proof.agg_c; // in the end must be equal to Z = C^r

    let mut final_res = GipaTUZ {
        tab: t_ab,
        uab: u_ab,
        zab: z_ab,
        tc: t_c,
        uc: u_c,
        zc: z_c,
    };

    // we first multiply each entry of the Z U and L vectors by the respective
    // challenges independently
    // Since at the end we want to multiple all "t" values together, we do
    // multiply all of them in parrallel and then merge then back at the end.
    // same for u and z.
    #[allow(clippy::upper_case_acronyms)]
    enum Op<'a, E>
    where
        E: MultiMillerLoop,
    {
        TAB(&'a <E as Engine>::Gt, &'a E::Fr),
        UAB(&'a <E as Engine>::Gt, &'a E::Fr),
        ZAB(&'a <E as Engine>::Gt, &'a E::Fr),
        TC(&'a <E as Engine>::Gt, &'a E::Fr),
        UC(&'a <E as Engine>::Gt, &'a E::Fr),
        ZC(&'a E::G1, &'a E::Fr),
    }

    let res = comms_ab
        .par_iter()
        .zip(zs_ab.par_iter())
        .zip(comms_c.par_iter().zip(zs_c.par_iter()))
        .zip(challenges.par_iter().zip(challenges_inv.par_iter()))
        .flat_map(|(((comm_ab, z_ab), (comm_c, z_c)), (c, c_inv))| {
            // T and U values for right and left for AB part
            let ((tab_l, uab_l), (tab_r, uab_r)) = comm_ab;
            let (zab_l, zab_r) = z_ab;
            // T and U values for right and left for C part
            let ((tc_l, uc_l), (tc_r, uc_r)) = comm_c;
            let (zc_l, zc_r) = z_c;

            // we multiple left side by x and right side by x^-1
            vec![
                Op::TAB::<E>(tab_l, c),
                Op::TAB(tab_r, c_inv),
                Op::UAB(uab_l, c),
                Op::UAB(uab_r, c_inv),
                Op::ZAB(zab_l, c),
                Op::ZAB(zab_r, c_inv),
                Op::TC::<E>(tc_l, c),
                Op::TC(tc_r, c_inv),
                Op::UC(uc_l, c),
                Op::UC(uc_r, c_inv),
                Op::ZC(zc_l, c),
                Op::ZC(zc_r, c_inv),
            ]
        })
        .fold(GipaTUZ::<E>::default, |mut res, op: Op<E>| {
            match op {
                Op::TAB(tx, c) => {
                    let tx = *tx * c;
                    res.tab += tx;
                }
                Op::UAB(ux, c) => {
                    let ux = *ux * c;
                    res.uab += ux;
                }
                Op::ZAB(zx, c) => {
                    let zx = *zx * c;
                    res.zab += zx;
                }
                Op::TC(tx, c) => {
                    let tx = *tx * c;
                    res.tc += tx;
                }
                Op::UC(ux, c) => {
                    let ux = *ux * c;
                    res.uc += ux;
                }
                Op::ZC(zx, c) => {
                    let zx = *zx * c;
                    res.zc += zx;
                }
            }
            res
        })
        .reduce(GipaTUZ::default, |mut acc_res, res| {
            acc_res.merge(&res);
            acc_res
        });
    // we reverse the order because the polynomial evaluation routine expects
    // the challenges in reverse order.Doing it here allows us to compute the final_r
    // in log time. Challenges are used as well in the KZG verification checks.
    challenges.reverse();
    challenges_inv.reverse();

    let ref_final_res = &mut final_res;
    let ref_challenges_inv = &challenges_inv;

    ref_final_res.merge(&res);
    let final_r = polynomial_evaluation_product_form_from_transcript(
        ref_challenges_inv,
        r_shift,
        &E::Fr::one(),
    );

    debug!(
        "TIPP verify: gipa prep and accumulate took {}ms",
        now.elapsed().as_millis()
    );
    (final_res, final_r, challenges, challenges_inv)
}

/// verify_kzg_opening_g2 takes a KZG opening, the final commitment key, SRS and
/// any shift (in TIPP we shift the v commitment by r^-1) and returns a pairing
/// tuple to check if the opening is correct or not.
pub fn verify_kzg_v<E, R>(
    v_srs: &VerifierSRS<E>,
    final_vkey: &(E::G2Affine, E::G2Affine),
    vkey_opening: &KZGOpening<E::G2Affine>,
    challenges: &[E::Fr],
    kzg_challenge: &E::Fr,
    pairing_checks: &PairingChecks<E, R>,
) where
    E: MultiMillerLoop,
    R: rand_core::RngCore + Send,
{
    // f_v(z)
    let vpoly_eval_z = polynomial_evaluation_product_form_from_transcript(
        challenges,
        kzg_challenge,
        &E::Fr::one(),
    );
    // -g such that when we test a pairing equation we only need to check if
    // it's equal 1 at the end:
    // e(a,b) = e(c,d) <=> e(a,b)e(-c,d) = 1
    // e(A,B) = e(C,D) <=> e(A,B)e(-C,D) == 1 <=> e(A,B)e(C,D)^-1 == 1
    let ng = (-v_srs.g).to_affine();

    par! {
        // e(g, C_f * h^{-y}) == e(v1 * g^{-x}, \pi) = 1
        let _check1 = kzg_check_v::<E, R>(
            v_srs,
            ng,
            *kzg_challenge,
            vpoly_eval_z,
            final_vkey.0.to_curve(),
            v_srs.g_alpha,
            vkey_opening.0,
            pairing_checks,
        ),

        // e(g, C_f * h^{-y}) == e(v2 * g^{-x}, \pi) = 1
        let _check2 = kzg_check_v::<E, R>(
            v_srs,
            ng,
            *kzg_challenge,
            vpoly_eval_z,
            final_vkey.1.to_curve(),
            v_srs.g_beta,
            vkey_opening.1,
            pairing_checks,
        )
    };
}

#[allow(clippy::too_many_arguments)]
fn kzg_check_v<E, R>(
    v_srs: &VerifierSRS<E>,
    ng: E::G1Affine,
    x: E::Fr,
    y: E::Fr,
    cf: E::G2,
    vk: E::G1,
    pi: E::G2Affine,
    pairing_checks: &PairingChecks<E, R>,
) where
    E: MultiMillerLoop,
    R: rand_core::RngCore + Send,
{
    // KZG Check: e(g, C_f * h^{-y}) = e(vk * g^{-x}, \pi)
    // Transformed, such that
    // e(-g, C_f * h^{-y}) * e(vk * g^{-x}, \pi) = 1

    // C_f - (y * h)
    let b = (cf - (v_srs.h * y)).to_affine();

    // vk - (g * x)
    let c = (vk - (v_srs.g * x)).to_affine();

    pairing_checks.merge_miller_inputs(&[(&ng, &b), (&c, &pi)], &<E as Engine>::Gt::generator());
}

/// Similar to verify_kzg_opening_g2 but for g1.
pub fn verify_kzg_w<E, R>(
    v_srs: &VerifierSRS<E>,
    final_wkey: &(E::G1Affine, E::G1Affine),
    wkey_opening: &KZGOpening<E::G1Affine>,
    challenges: &[E::Fr],
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
    pairing_checks: &PairingChecks<E, R>,
) where
    E: MultiMillerLoop,
    R: rand_core::RngCore + Send,
{
    // compute in parallel f(z) and z^n and then combines into f_w(z) = z^n * f(z)
    par! {
        let fz = polynomial_evaluation_product_form_from_transcript(challenges, kzg_challenge, r_shift),
        let zn = kzg_challenge.pow_vartime(&[v_srs.n as u64])
    };

    let mut fwz = fz;
    fwz.mul_assign(&zn);

    let nh = -v_srs.h;
    let nh = nh.to_affine();

    par! {
        // e(C_f * g^{-y}, h) = e(\pi, w1 * h^{-x})
        let _check1 = kzg_check_w::<E, R>(
            v_srs,
            nh,
            *kzg_challenge,
            fwz,
            final_wkey.0.to_curve(),
            v_srs.h_alpha,
            wkey_opening.0,
            pairing_checks,
        ),

        // e(C_f * g^{-y}, h) = e(\pi, w2 * h^{-x})
        let _check2 = kzg_check_w::<E, R>(
            v_srs,
            nh,
            *kzg_challenge,
            fwz,
            final_wkey.1.to_curve(),
            v_srs.h_beta,
            wkey_opening.1,
            pairing_checks,
        )
    };
}

#[allow(clippy::too_many_arguments)]
fn kzg_check_w<E, R>(
    v_srs: &VerifierSRS<E>,
    nh: E::G2Affine,
    x: E::Fr,
    y: E::Fr,
    cf: E::G1,
    wk: E::G2,
    pi: E::G1Affine,
    pairing_checks: &PairingChecks<E, R>,
) where
    E: MultiMillerLoop,
    R: rand_core::RngCore + Send,
{
    // KZG Check: e(C_f * g^{-y}, h) = e(\pi, wk * h^{-x})
    // Transformed, such that
    // e(C_f * g^{-y}, -h) * e(\pi, wk * h^{-x}) = 1

    // C_f - (y * g)
    let a = (cf - (v_srs.g * y)).to_affine();

    // wk - (x * h)
    let d = (wk - (v_srs.h * x)).to_affine();

    pairing_checks.merge_miller_inputs(&[(&a, &nh), (&pi, &d)], &<E as Engine>::Gt::generator());
}

/// Keeps track of the variables that have been sent by the prover and must
/// be multiplied together by the verifier. Both MIPP and TIPP are merged
/// together.
#[allow(clippy::upper_case_acronyms)]
struct GipaTUZ<E>
where
    E: MultiMillerLoop,
{
    pub tab: <E as Engine>::Gt,
    pub uab: <E as Engine>::Gt,
    pub zab: <E as Engine>::Gt,
    pub tc: <E as Engine>::Gt,
    pub uc: <E as Engine>::Gt,
    pub zc: E::G1,
}

impl<E> Default for GipaTUZ<E>
where
    E: MultiMillerLoop,
{
    fn default() -> Self {
        Self {
            tab: <E as Engine>::Gt::identity(),
            uab: <E as Engine>::Gt::identity(),
            zab: <E as Engine>::Gt::identity(),
            tc: <E as Engine>::Gt::identity(),
            uc: <E as Engine>::Gt::identity(),
            zc: E::G1::identity(),
        }
    }
}

impl<E> GipaTUZ<E>
where
    E: MultiMillerLoop,
{
    fn merge(&mut self, other: &Self) {
        self.tab += &other.tab;
        self.uab += &other.uab;
        self.zab += &other.zab;
        self.tc += &other.tc;
        self.uc += &other.uc;
        self.zc += &other.zc;
    }
}
