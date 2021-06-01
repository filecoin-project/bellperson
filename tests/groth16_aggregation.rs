use bellperson::bls::{Bls12, Engine, Fr, FrRepr};
use bellperson::gadgets::num::AllocatedNum;
use bellperson::groth16::{
    aggregate::{
        aggregate_proofs, setup_fake_srs, verify_aggregate_proof, AggregateProof, GenericSRS,
    },
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
    verify_proofs_batch, Parameters, Proof,
};
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use ff::{Field, PrimeField, ScalarEngine};
use groupy::CurveProjective;
use itertools::Itertools;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use std::default::Default;
use std::time::{Duration, Instant};

const MIMC_ROUNDS: usize = 322;

/// This is an implementation of MiMC, specifically a
/// variant named `LongsightF322p3` for BLS12-381.
/// See http://eprint.iacr.org/2016/492 for more
/// information about this construction.
///
/// ```
/// function LongsightF322p3(xL ⦂ Fp, xR ⦂ Fp) {
///     for i from 0 up to 321 {
///         xL, xR := xR + (xL + Ci)^3, xL
///     }
///     return xL
/// }
/// ```
fn mimc<E: Engine>(mut xl: E::Fr, mut xr: E::Fr, constants: &[E::Fr]) -> E::Fr {
    assert_eq!(constants.len(), MIMC_ROUNDS);

    for i in 0..MIMC_ROUNDS {
        let mut tmp1 = xl;
        tmp1.add_assign(&constants[i]);
        let mut tmp2 = tmp1;
        tmp2.square();
        tmp2.mul_assign(&tmp1);
        tmp2.add_assign(&xr);
        xr = xl;
        xl = tmp2;
    }

    xl
}

#[derive(Clone)]
struct MiMCDemo<'a, E: Engine> {
    xl: Option<E::Fr>,
    xr: Option<E::Fr>,
    constants: &'a [E::Fr],
}

impl<'a, E: Engine> Circuit<E> for MiMCDemo<'a, E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        assert_eq!(self.constants.len(), MIMC_ROUNDS);

        // Allocate the first component of the preimage.
        let mut xl_value = self.xl;
        let mut xl = cs.alloc(
            || "preimage xl",
            || xl_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        // Allocate the second component of the preimage.
        let mut xr_value = self.xr;
        let mut xr = cs.alloc(
            || "preimage xr",
            || xr_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        for i in 0..MIMC_ROUNDS {
            // xL, xR := xR + (xL + Ci)^3, xL
            let cs = &mut cs.namespace(|| format!("round {}", i));

            // tmp = (xL + Ci)^2
            let tmp_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.square();
                e
            });
            let tmp = cs.alloc(
                || "tmp",
                || tmp_value.ok_or(SynthesisError::AssignmentMissing),
            )?;

            cs.enforce(
                || "tmp = (xL + Ci)^2",
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + tmp,
            );

            // new_xL = xR + (xL + Ci)^3
            // new_xL = xR + tmp * (xL + Ci)
            // new_xL - xR = tmp * (xL + Ci)
            let new_xl_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.mul_assign(&tmp_value.unwrap());
                e.add_assign(&xr_value.unwrap());
                e
            });

            let new_xl = if i == (MIMC_ROUNDS - 1) {
                // This is the last round, xL is our image and so
                // we allocate a public input.
                cs.alloc_input(
                    || "image",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            } else {
                cs.alloc(
                    || "new_xl",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            };

            cs.enforce(
                || "new_xL = xR + (xL + Ci)^3",
                |lc| lc + tmp,
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + new_xl - xr,
            );

            // xR = xL
            xr = xl;
            xr_value = xl_value;

            // xL = new_xL
            xl = new_xl;
            xl_value = new_xl_value;
        }

        Ok(())
    }
}

#[derive(Clone)]
struct TestCircuit<E: Engine> {
    public_inputs: Vec<Option<E::Fr>>,
    witness_input: Option<E::Fr>,
    public_product: Option<E::Fr>,
}
impl<E: Engine> Circuit<E> for TestCircuit<E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, mut cs: &mut CS) -> Result<(), SynthesisError> {
        let input_variables: Vec<_> = self
            .public_inputs
            .iter()
            .enumerate()
            .map(|(_i, input)| -> Result<AllocatedNum<_>, SynthesisError> {
                let num = AllocatedNum::alloc(&mut cs, || {
                    input.ok_or(SynthesisError::AssignmentMissing)
                })?;
                num.inputize(&mut cs)?;
                Ok(num)
            })
            .collect::<Result<_, _>>()?;
        let product = AllocatedNum::alloc(&mut cs, || {
            self.public_product.ok_or(SynthesisError::AssignmentMissing)
        })?;
        product.inputize(&mut cs)?;
        let witness = AllocatedNum::alloc(&mut cs, || {
            self.witness_input.ok_or(SynthesisError::AssignmentMissing)
        })?;
        let mut computed_product = witness;
        for x in &input_variables {
            computed_product = computed_product.mul(&mut cs, x)?;
        }
        cs.enforce(
            || "product = computed product",
            |lc| lc + CS::one(),
            |lc| lc + computed_product.get_variable(),
            |lc| lc + product.get_variable(),
        );

        Ok(())
    }
}

#[test]
fn test_groth16_srs_io() {
    use memmap::MmapOptions;
    use std::fs::File;
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::NamedTempFile;

    const NUM_PROOFS_TO_AGGREGATE: usize = 16;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);

    println!("Creating parameters...");

    // Generate parameters for inner product aggregation
    let srs: GenericSRS<Bls12> = setup_fake_srs(&mut rng, NUM_PROOFS_TO_AGGREGATE);

    // Write out parameters to a temp file
    let mut cache_file = NamedTempFile::new().expect("failed to create temp cache file");
    srs.write(&mut cache_file).expect("failed to write out srs");
    cache_file.flush().expect("failed to flush srs write");

    println!("cache file written to");

    // Read back parameters from the temp file
    cache_file
        .seek(SeekFrom::Start(0))
        .expect("failed to rewind tmp file");

    let srs2 =
        GenericSRS::<Bls12>::read(&mut cache_file).expect("failed to read srs from cache file");

    // Ensure that the parameters match
    assert_eq!(srs, srs2);

    let cache_path = cache_file.into_temp_path();
    let mapped_file = File::open(&cache_path).expect("failed to open file");
    let mmap = unsafe {
        MmapOptions::new()
            .map(&mapped_file)
            .expect("failed to mmap")
    };

    let max_len = (2 << 19) + 1;
    let srs3 =
        GenericSRS::<Bls12>::read_mmap(&mmap, max_len).expect("failed to read srs from cache file");

    // Ensure that the parameters match
    assert_eq!(srs, srs3);

    // Remove temp file
    cache_path.close().expect("failed to close temp path");
}

// structure to write to CSV file
#[derive(Debug, Serialize, Default)]
struct Record {
    nproofs: u32,              // number of proofs that have been verified
    aggregate_create_ms: u32,  // time to create the aggregated proof
    aggregate_verify_ms: u32,  // time to verify the aggregate proof
    batch_verify_ms: u32,      // time to verify all proofs via batching of 10
    batch_all_ms: u32,         // time ot verify all proofs via batching at once
    aggregate_size_bytes: u32, // size of the aggregated proof
    batch_size_bytes: u32,     // size of the batch of proof
}

impl Record {
    pub fn average(records: &[Record]) -> Record {
        let mut agg: Record = records.iter().fold(Default::default(), |mut acc, r| {
            acc.nproofs += r.nproofs;
            acc.aggregate_create_ms += r.aggregate_create_ms;
            acc.aggregate_verify_ms += r.aggregate_verify_ms;
            acc.batch_verify_ms += r.batch_verify_ms;
            acc.batch_all_ms += r.batch_all_ms;
            acc.aggregate_size_bytes += r.aggregate_size_bytes;
            acc.batch_size_bytes += r.batch_size_bytes;
            acc
        });
        let n = records.len() as u32;
        agg.nproofs /= n;
        agg.aggregate_create_ms /= n;
        agg.aggregate_verify_ms /= n;
        agg.batch_verify_ms /= n;
        agg.batch_all_ms /= n;
        agg.aggregate_size_bytes /= n;
        agg.batch_size_bytes /= n;
        agg
    }
}

#[test]
#[ignore]
fn test_groth16_bench() {
    let n_average = 3; // number of times we do the benchmarking to average out results
    let nb_proofs = vec![8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
    let max = nb_proofs.last().unwrap().clone();
    let public_inputs = 350; // roughly what a prove commit needs
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    // CRS for aggregation
    let generic = setup_fake_srs(&mut rng, max);
    // Create parameters for our circuit
    let params = {
        let c = TestCircuit::<Bls12> {
            public_inputs: vec![Default::default(); public_inputs],
            public_product: Default::default(),
            witness_input: Default::default(),
        };
        generate_random_parameters(c, &mut rng).unwrap()
    };
    // verification key for indivdual verification of proof
    let pvk = prepare_verifying_key(&params.vk);

    let (proofs, statements): (Vec<Proof<Bls12>>, Vec<Vec<Fr>>) = (0..max)
        .map(|_| generate_proof(public_inputs, &params, &mut rng))
        .unzip();

    let mut writer = csv::Writer::from_path("aggregation.csv").expect("unable to open csv writer");

    println!("Generating {} Groth16 proofs...", max);

    let mut buf = Vec::new();
    proofs[0].write(&mut buf).expect("buffer");
    let proof_size = buf.len();
    let inclusion = vec![1, 2, 3];
    for i in nb_proofs {
        let mut records = Vec::new();
        for _ in 0..n_average {
            println!("Proofs {}", i);
            let (pk, vk) = generic.specialize(i);
            // Aggregate proofs using inner product proofs
            let start = Instant::now();
            println!("\t-Aggregation...");
            let aggregate_proof = aggregate_proofs::<Bls12>(&pk, &inclusion, &proofs[..i])
                .expect("failed to aggregate proofs");
            let prover_time = start.elapsed().as_millis();
            println!("\t-Aggregate Verification ...");

            let mut buffer = Vec::new();
            aggregate_proof.write(&mut buffer).unwrap();
            let start = Instant::now();
            let deserialized =
                AggregateProof::<Bls12>::read(std::io::Cursor::new(&buffer)).unwrap();

            let result = verify_aggregate_proof(
                &vk,
                &pvk,
                &mut rng,
                &statements[..i],
                &deserialized,
                &inclusion,
            );
            assert!(result.unwrap());
            let verifier_time = start.elapsed().as_millis();

            println!("\t-Batch per 10 packets verification...");
            let batches: Vec<_> = proofs
                .iter()
                .cloned()
                .take(i)
                .zip(statements.iter().cloned().take(i))
                .chunks(10)
                .into_iter()
                .map(|s| s.collect())
                .collect::<Vec<Vec<(Proof<Bls12>, Vec<Fr>)>>>();
            let start = Instant::now();
            batches.par_iter().for_each(|batch| {
                let batch_proofs = batch.iter().by_ref().map(|(p, _)| p).collect::<Vec<_>>();
                let batch_statements = batch
                    .iter()
                    .map(|(_, state)| state.clone())
                    .collect::<Vec<_>>();
                let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
                assert!(
                    verify_proofs_batch(&pvk, &mut rng, &batch_proofs, &batch_statements).unwrap()
                )
            });
            let batch_verifier_time = start.elapsed().as_millis();

            println!("\t-Batch all-in verification...");
            let proofs_serialized: Vec<Vec<_>> = proofs
                .iter()
                .take(i)
                .map(|p| {
                    let mut buff = Vec::new();
                    p.write(&mut buff).unwrap();
                    buff
                })
                .collect::<Vec<Vec<_>>>();
            let start = Instant::now();
            let proofs: Vec<_> = proofs_serialized
                .into_iter()
                .map(|buff| Proof::<Bls12>::read(std::io::Cursor::new(&buff)).unwrap())
                .collect::<Vec<_>>();
            let proofs_ref: Vec<_> = proofs.iter().collect();

            assert!(verify_proofs_batch(&pvk, &mut rng, &proofs_ref, &statements[..i]).unwrap());
            let batch_all_time = start.elapsed().as_millis();
            let agg_size = buffer.len();
            records.push(Record {
                nproofs: i as u32,
                aggregate_create_ms: prover_time as u32,
                aggregate_verify_ms: verifier_time as u32,
                aggregate_size_bytes: agg_size as u32,
                batch_verify_ms: batch_verifier_time as u32,
                batch_size_bytes: (proof_size * i) as u32,
                batch_all_ms: batch_all_time as u32,
            });
        }
        let average = Record::average(&records);
        writer
            .serialize(average)
            .expect("unable to write result to csv");
    }
    writer.flush().expect("failed to flush");
}

fn generate_proof<R: SeedableRng + RngCore>(
    publics: usize,
    p: &Parameters<Bls12>,
    mut rng: &mut R,
) -> (Proof<Bls12>, Vec<Fr>) {
    // Generate random inputs to product together
    let mut public_inputs = Vec::new();
    let mut statement = Vec::new();
    let mut prod = Fr::one();
    for _i in 0..publics {
        let x = Fr::from_str("4").unwrap();
        public_inputs.push(Some(x));
        statement.push(x);
        prod.mul_assign(&x);
    }
    let w = Fr::from_repr(FrRepr::from(3)).unwrap();
    let mut product: Fr = w.clone();
    product.mul_assign(&prod);
    statement.push(product.clone());

    let c = TestCircuit {
        public_inputs,
        public_product: Some(product),
        witness_input: Some(w),
    };
    (create_random_proof(c, p, &mut rng).unwrap(), statement)
}

/// This test creates and aggregates some valid proofs, then tries a bunch of
/// failing test case scenarios
#[test]
fn test_groth16_aggregation() {
    const NUM_PUBLIC_INPUTS: usize = 50; //1000;
    const NUM_PROOFS: usize = 8; //1024;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);

    println!("Creating parameters...");

    // Generate parameters for inner product aggregation
    let generic = setup_fake_srs(&mut rng, NUM_PROOFS);
    let (pk, vk) = generic.specialize(NUM_PROOFS);

    // Create parameters for our circuit
    let params = {
        let c = TestCircuit::<Bls12> {
            public_inputs: vec![Default::default(); NUM_PUBLIC_INPUTS],
            public_product: Default::default(),
            witness_input: Default::default(),
        };

        generate_random_parameters(c, &mut rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    println!("Creating proofs...");

    // Generate proofs
    println!("Generating {} Groth16 proofs...", NUM_PROOFS);

    let mut proofs = Vec::new();
    let mut statements = Vec::new();
    let mut generation_time = Duration::new(0, 0);
    let mut individual_verification_time = Duration::new(0, 0);

    for _ in 0..NUM_PROOFS {
        // Generate random inputs to product together
        let mut public_inputs = Vec::new();
        let mut statement = Vec::new();
        let mut prod = Fr::one();
        for _i in 0..NUM_PUBLIC_INPUTS {
            let x = Fr::from_str("4").unwrap();
            public_inputs.push(Some(x));
            statement.push(x);
            prod.mul_assign(&x);
        }
        let w = Fr::from_repr(FrRepr::from(3)).unwrap();

        let mut product: Fr = w.clone();
        product.mul_assign(&prod);
        statement.push(product.clone());

        let start = Instant::now();
        // Create an instance of our circuit (with the
        // witness)
        let c = TestCircuit {
            public_inputs,
            public_product: Some(product),
            witness_input: Some(w),
        };

        // Create a groth16 proof with our parameters.
        let proof = create_random_proof(c, &params, &mut rng).unwrap();
        generation_time += start.elapsed();

        assert!(verify_proof(&pvk, &proof, &statement).unwrap());
        individual_verification_time += start.elapsed();

        proofs.push(proof);
        statements.push(statement);
    }
    let to_include = vec![1, 2, 3];
    // 1. Valid proofs
    println!("Aggregating {} Groth16 proofs...", proofs.len());
    let mut aggregate_proof =
        aggregate_proofs::<Bls12>(&pk, &to_include, &proofs).expect("failed to aggregate proofs");
    let result = verify_aggregate_proof(
        &vk,
        &pvk,
        &mut rng,
        &statements,
        &aggregate_proof,
        &to_include,
    )
    .expect("these proofs should have been valid");
    assert!(result);

    // Invalid transcript inclusion
    assert_eq!(
        verify_aggregate_proof(
            &vk,
            &pvk,
            &mut rng,
            &statements,
            &aggregate_proof,
            &vec![4, 5, 6],
        )
        .unwrap(),
        false
    );

    // 2. Non power of two
    let err = aggregate_proofs::<Bls12>(&pk, &to_include, &proofs[0..NUM_PROOFS - 1]).unwrap_err();
    assert!(match err {
        SynthesisError::NonPowerOfTwo => true,
        _ => false,
    });

    // 3. aggregate invalid proof content (random A, B, and C)
    let old_a = proofs[0].a.clone();
    proofs[0].a = <Bls12 as Engine>::G1::random(&mut rng).into_affine();
    let invalid_agg = aggregate_proofs::<Bls12>(&pk, &to_include, &proofs)
        .expect("I should be able to aggregate");
    let res = verify_aggregate_proof(&vk, &pvk, &mut rng, &statements, &invalid_agg, &to_include)
        .expect("no synthesis");
    assert!(res == false);
    proofs[0].a = old_a;

    let old_b = proofs[0].b.clone();
    proofs[0].b = <Bls12 as Engine>::G2::random(&mut rng).into_affine();
    let invalid_agg = aggregate_proofs::<Bls12>(&pk, &to_include, &proofs)
        .expect("I should be able to aggregate");
    let res = verify_aggregate_proof(&vk, &pvk, &mut rng, &statements, &invalid_agg, &to_include)
        .expect("no synthesis");
    assert!(res == false);
    proofs[0].b = old_b;

    let old_c = proofs[0].c.clone();
    proofs[0].c = <Bls12 as Engine>::G1::random(&mut rng).into_affine();
    let invalid_agg = aggregate_proofs::<Bls12>(&pk, &to_include, &proofs)
        .expect("I should be able to aggregate");
    let res = verify_aggregate_proof(&vk, &pvk, &mut rng, &statements, &invalid_agg, &to_include)
        .expect("no synthesis");
    assert!(res == false);
    proofs[0].c = old_c;

    // 4. verify with invalid aggregate proof
    // first invalid commitment
    let old_aggc = aggregate_proof.agg_c.clone();
    aggregate_proof.agg_c = <Bls12 as Engine>::G1::random(&mut rng);
    let res = verify_aggregate_proof(
        &vk,
        &pvk,
        &mut rng,
        &statements,
        &aggregate_proof,
        &to_include,
    )
    .expect("no synthesis");
    assert!(res == false);
    aggregate_proof.agg_c = old_aggc;

    // 5. invalid gipa element
    let old_finala = aggregate_proof.tmipp.gipa.final_a.clone();
    aggregate_proof.tmipp.gipa.final_a = <Bls12 as Engine>::G1::random(&mut rng).into_affine();
    let res = verify_aggregate_proof(
        &vk,
        &pvk,
        &mut rng,
        &statements,
        &aggregate_proof,
        &to_include,
    )
    .expect("no synthesis");
    assert!(res == false);
    aggregate_proof.tmipp.gipa.final_a = old_finala;
}

#[test]
fn test_groth16_aggregation_mimc() {
    const NUM_PROOFS_TO_AGGREGATE: usize = 8; //1024;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);

    // Generate the MiMC round constants
    let constants = (0..MIMC_ROUNDS)
        .map(|_| <Bls12 as ScalarEngine>::Fr::random(&mut rng))
        .collect::<Vec<_>>();

    println!("Creating parameters...");

    // Create parameters for our circuit
    let params = {
        let c = MiMCDemo::<Bls12> {
            xl: None,
            xr: None,
            constants: &constants,
        };

        generate_random_parameters(c, &mut rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    // Generate parameters for inner product aggregation
    // first generic SRS then specialized to the correct size
    let generic = setup_fake_srs(&mut rng, NUM_PROOFS_TO_AGGREGATE);
    let (pk, vk) = generic.specialize(NUM_PROOFS_TO_AGGREGATE);

    println!("Creating proofs...");

    // Generate proofs
    println!("Generating {} Groth16 proofs...", NUM_PROOFS_TO_AGGREGATE);

    let mut proofs = Vec::new();
    let mut images = Vec::new();
    let mut generation_time = Duration::new(0, 0);
    let mut individual_verification_time = Duration::new(0, 0);

    for _ in 0..NUM_PROOFS_TO_AGGREGATE {
        // Generate a random preimage and compute the image
        let xl = <Bls12 as ScalarEngine>::Fr::random(&mut rng);
        let xr = <Bls12 as ScalarEngine>::Fr::random(&mut rng);
        let image = mimc::<Bls12>(xl, xr, &constants);

        let start = Instant::now();
        // Create an instance of our circuit (with the
        // witness)
        let c = MiMCDemo {
            xl: Some(xl),
            xr: Some(xr),
            constants: &constants,
        };

        // Create a groth16 proof with our parameters.
        let proof = create_random_proof(c, &params, &mut rng).unwrap();
        generation_time += start.elapsed();

        assert!(verify_proof(&pvk, &proof, &[image]).unwrap());
        individual_verification_time += start.elapsed();

        proofs.push(proof);
        images.push(vec![image]);
    }
    let inclusion = vec![1, 2, 3];

    // Aggregate proofs using inner product proofs
    let start = Instant::now();
    println!("Aggregating {} Groth16 proofs...", NUM_PROOFS_TO_AGGREGATE);
    let aggregate_proof =
        aggregate_proofs::<Bls12>(&pk, &inclusion, &proofs).expect("failed to aggregate proofs");
    let prover_time = start.elapsed().as_millis();

    println!("Verifying aggregated proof...");
    let start = Instant::now();
    let result =
        verify_aggregate_proof(&vk, &pvk, &mut rng, &images, &aggregate_proof, &inclusion).unwrap();
    let verifier_time = start.elapsed().as_millis();
    assert!(result);

    let start = Instant::now();
    let proofs: Vec<_> = proofs.iter().collect();
    assert!(verify_proofs_batch(&pvk, &mut rng, &proofs, &images).unwrap());
    let batch_verifier_time = start.elapsed().as_millis();

    println!("Proof generation time: {} ms", generation_time.as_millis());
    println!("Proof aggregation time: {} ms", prover_time);
    println!("Proof aggregation verification time: {} ms", verifier_time);
    println!(
        "Proof individual verification time: {} ms",
        individual_verification_time.as_millis()
    );

    println!("Proof batch verification time: {} ms", batch_verifier_time);
}
