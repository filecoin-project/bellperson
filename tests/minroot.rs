extern crate ff;
extern crate rand;

use ff::Field;

// For randomness (during paramgen and proof generation)
//use self::rand::Rng;

use blstrs::Bls12;
use pairing::Engine;
// We'll use these interfaces to construct our circuit.
use bellperson::groth16::{
    aggregate::AggregateVersion, create_random_proof, generate_random_parameters,
    prepare_verifying_key, verify_proof, Parameters, Proof,
};
use bellperson::{Circuit, ConstraintSystem, SynthesisError};

pub const MINROOT_ROUNDS: usize = 10;

//
pub fn minroot<E: Engine>(mut xl: E::Fr, mut xr: E::Fr) -> (E::Fr, E::Fr) {
    for _ in 0..MINROOT_ROUNDS {
        let mut tmp1 = xl;
        tmp1 += xr;

        //    power equals (2 * p - 1) / 5.  Don't delete this, was very hard to figure out.
        let tmp2 = tmp1.pow_vartime([
            0x33333332CCCCCCCD,
            0x217F0E679998F199,
            0xE14A56699D73F002,
            0x2E5F0FBADD72321C,
        ]);

        xr = xl;
        xl = tmp2;
    }

    (xl, xr)
}

fn fifth_root<E: Engine>(x: E::Fr) -> Option<E::Fr> {
    Some(x.pow_vartime([
        0x33333332CCCCCCCD,
        0x217F0E679998F199,
        0xE14A56699D73F002,
        0x2E5F0FBADD72321C,
    ]))
}

// proving that I know x1, y1 such that x2^3  == (x1 + y1)
#[derive(Clone)]
pub struct MinRoot<E: Engine> {
    pub xl: Option<E::Fr>,
    pub xr: Option<E::Fr>,
}

/// Our circuit implements this `Circuit` trait which
/// is used during paramgen and proving in order to
/// synthesize the constraint system.
impl<E: Engine> Circuit<E::Fr> for MinRoot<E> {
    fn synthesize<CS: ConstraintSystem<E::Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        // Allocate the first component of the preimage.
        let mut xl_value = self.xl;
        let mut xl = cs.alloc_input(
            || "preimage xl",
            || xl_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        // Allocate the second component of the preimage.
        let mut xr_value = self.xr;
        let mut xr = cs.alloc_input(
            || "preimage xr",
            || xr_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        for i in 0..MINROOT_ROUNDS {
            // xL, xR := (xL + xR)^(1/5), xL
            let cs = &mut cs.namespace(|| format!("round {}", i));

            // power equals (2 * p - 1) / 5.

            let mut new_xl_value = None;
            if let Some(xl_value) = xl_value {
                let mut tmp1 = xl_value;
                tmp1 += &xr_value.unwrap();
                new_xl_value = fifth_root::<E>(tmp1);
            }

            let new_xl = if i == (MINROOT_ROUNDS - 1) {
                // This is the last round, xL is our image and so
                // we allocate a public input.
                cs.alloc_input(
                    || "image_xl",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            } else {
                cs.alloc(
                    || "new_xl",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            };

            // tmp2 = (xl_(i+1))^2
            let tmp2 = new_xl_value.map(|mut e| {
                e = e.square();
                e
            });

            // tmp3 = (xl_(i+1))^4
            let tmp3 = tmp2.map(|mut e| {
                e = e.square();
                e
            });

            let tmp2 = cs.alloc(|| "tmp2", || tmp2.ok_or(SynthesisError::AssignmentMissing))?;

            let tmp3 = cs.alloc(|| "tmp3", || tmp3.ok_or(SynthesisError::AssignmentMissing))?;

            let new_xr = if i == (MINROOT_ROUNDS - 1) {
                // This is the last round, xR is our image and so
                // we allocate a public input.
                cs.alloc_input(
                    || "image_xr",
                    || xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            } else {
                cs.alloc(
                    || "new_xr",
                    || xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            };

            // enforce that tmp2 = tmp1^2
            cs.enforce(
                || "tmp2 = new_xl^2",
                |lc| lc + new_xl,
                |lc| lc + new_xl,
                |lc| lc + tmp2,
            );

            // enforce that tmp3 = tmp2^2
            cs.enforce(
                || "tmp3 = tmp2^2",
                |lc| lc + tmp2,
                |lc| lc + tmp2,
                |lc| lc + tmp3,
            );

            // tmp3 * new_xl = new_xl^5 = xl + xr
            cs.enforce(
                || "new_xL^5 = xl + xr",
                |lc| lc + tmp3,
                |lc| lc + new_xl,
                |lc| lc + xl + xr,
            );

            // update xl and xr for next round
            xr = new_xr;
            xr_value = xl_value;

            xl = new_xl;
            xl_value = new_xl_value;
        }

        Ok(())
    }
}

#[test]
fn minroot_test() {
    let rng = &mut rand_core::OsRng;

    // Create parameters for our circuit
    let params: Parameters<Bls12> = {
        let c = MinRoot::<Bls12> { xl: None, xr: None };

        generate_random_parameters(c, rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    let mut proof_vec = vec![];
    let mut proofs = vec![];
    let mut images = vec![];

    // Generate a random preimage and compute the image
    let xl = <Bls12 as Engine>::Fr::random(*rng);
    let xr = <Bls12 as Engine>::Fr::random(*rng);
    let (image_xl, image_xr) = minroot::<Bls12>(xl, xr);

    proof_vec.truncate(0);

    {
        // Create an instance of our circuit (with the
        // witness)
        let c = MinRoot::<Bls12> {
            xl: Some(xl),
            xr: Some(xr),
        };

        // Create a groth16 proof with our parameters.
        let proof = create_random_proof(c, &params, rng).unwrap();

        proof.write(&mut proof_vec).unwrap();
    }

    let proof = Proof::read(&proof_vec[..]).unwrap();

    // Check the proof
    assert!(verify_proof(&pvk, &proof, &[xl, xr, image_xl, image_xr]).unwrap());
    proofs.push(proof);
    images.push(vec![xl, xr, image_xl, image_xr]);
}

use bellperson::groth16::aggregate::{
    aggregate_proofs_and_instances, setup_fake_srs, verify_aggregate_proof_and_aggregate_instances,
    GenericSRS,
};
use blstrs::Scalar as Fr;
use rand_core::SeedableRng;

#[test]
fn minroot_aggregate_proof() {
    minroot_aggregate_proof_inner(AggregateVersion::V1);
    minroot_aggregate_proof_inner(AggregateVersion::V2);
}

fn minroot_aggregate_proof_inner(version: AggregateVersion) {
    let nb_proofs: usize = 128;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);

    // CRS for aggregation
    let generic: GenericSRS<Bls12> = setup_fake_srs(&mut rng, nb_proofs);

    // Create parameters for our circuit
    let params = {
        let c = MinRoot::<Bls12> { xl: None, xr: None };

        generate_random_parameters(c, &mut rng).unwrap()
    };

    // verification key for indivdual verification of proof
    let pvk = prepare_verifying_key(&params.vk);

    let mut xl = <Bls12 as Engine>::Fr::random(&mut rng);
    let mut xr = <Bls12 as Engine>::Fr::random(&mut rng);

    let public_inputs = [xl, xr].to_vec();

    let mut proofs: Vec<Proof<Bls12>> = Vec::new();
    let mut statements: Vec<Vec<Fr>> = Vec::new();

    let mut statement_circuit: (Vec<Fr>, MinRoot<Bls12>);

    for _ in 0..nb_proofs {
        statement_circuit = generate_proof(xl, xr);

        xl = statement_circuit.0[2];
        xr = statement_circuit.0[3];

        let proof = create_random_proof(statement_circuit.1, &params, &mut rng).unwrap();

        assert!(verify_proof(&pvk, &proof, &statement_circuit.0).unwrap());
        proofs.push(proof);
        statements.push(statement_circuit.0);
    }

    let public_outputs = [xl, xr].to_vec();

    let mut buf = Vec::new();
    proofs[0].write(&mut buf).expect("buffer");
    let inclusion = vec![1, 2, 3];

    env_logger::try_init().ok();

    let (pk, vk) = generic.specialize_input_aggregation(nb_proofs);

    // Aggregate proofs using inner product proofs
    let aggregate_proof_and_instance = aggregate_proofs_and_instances::<Bls12>(
        &pk,
        &inclusion,
        &statements[..nb_proofs].to_vec(),
        &proofs[..nb_proofs],
        version,
    )
    .expect("failed to aggregate proofs");

    let verified = verify_aggregate_proof_and_aggregate_instances(
        &vk,
        &pvk,
        &mut rng,
        &public_inputs,
        &public_outputs,
        &aggregate_proof_and_instance,
        &inclusion,
        version,
    )
    .unwrap();

    assert!(verified, "failed to verify aggregate proof");
}

fn generate_proof(
    xl_old: <Bls12 as Engine>::Fr,
    xr_old: <Bls12 as Engine>::Fr,
) -> (Vec<Fr>, MinRoot<Bls12>) {
    let (image_xl, image_xr) = minroot::<Bls12>(xl_old, xr_old);

    // Create an instance of our circuit (with the witness)
    let c = MinRoot::<Bls12> {
        xl: Some(xl_old),
        xr: Some(xr_old),
    };

    (vec![xl_old, xr_old, image_xl, image_xr], c)
}
