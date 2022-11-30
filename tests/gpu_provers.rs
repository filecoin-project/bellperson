use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use ff::PrimeField;

#[derive(Clone)]
pub struct DummyDemo {
    pub interations: u64,
}

impl<Scalar: PrimeField> Circuit<Scalar> for DummyDemo {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let mut x_val = Scalar::from(2u64);
        let mut x = cs.alloc(|| "", || Ok(x_val))?;

        for _ in 0..self.interations {
            // Allocate: x * x = x2
            let x2_val = x_val.square();

            let x2 = cs.alloc(|| "", || Ok(x2_val))?;

            // Enforce: x * x = x2
            cs.enforce(|| "", |lc| lc + x, |lc| lc + x, |lc| lc + x2);

            x = x2;
            x_val = x2_val;
        }

        cs.enforce(
            || "",
            |lc| lc + (x_val, CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + x,
        );

        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
#[test]
pub fn test_parallel_prover() {
    use bellperson::groth16::{
        create_random_proof, create_random_proof_in_priority, generate_random_parameters,
        prepare_verifying_key, verify_proof,
    };
    use blstrs::Bls12;
    use rand::thread_rng;
    use std::thread;
    use std::time::{Duration, Instant};

    let _ = env_logger::try_init();
    let rng = &mut thread_rng();

    println!("Initializing circuit...");
    println!("Creating parameters...");

    // Higher prio circuit
    let c = DummyDemo {
        #[cfg(not(feature = "_coverage"))]
        interations: 10_000,
        #[cfg(feature = "_coverage")]
        interations: 100,
    };
    // Lower prio circuit
    let c2 = DummyDemo {
        #[cfg(not(feature = "_coverage"))]
        interations: 500_000,
        #[cfg(feature = "_coverage")]
        interations: 5000,
    };

    let params = generate_random_parameters::<Bls12, _, _>(c.clone(), rng).unwrap();
    let params2 = generate_random_parameters::<Bls12, _, _>(c2.clone(), rng).unwrap();

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);
    let pvk2 = prepare_verifying_key(&params2.vk);

    let higher_thread = thread::spawn(move || {
        for _ in 0..20 {
            let now = Instant::now();

            let rng = &mut thread_rng();
            let proof_higher = create_random_proof_in_priority(c.clone(), &params, rng).unwrap();
            assert!(verify_proof(&pvk, &proof_higher, &[]).unwrap());

            println!(
                "Higher proof gen finished in {}s and {}ms",
                now.elapsed().as_secs(),
                now.elapsed().subsec_millis()
            );

            // Sleep in between higher proofs so that LOWER thread can acquire GPU again
            thread::sleep(Duration::from_millis(600));
        }
    });

    // Start lower proofs after a few seconds
    thread::sleep(Duration::from_millis(2000));
    println!("Starting low priority proof gen...");
    {
        for _ in 0..10 {
            let now = Instant::now();

            let proof_lower = create_random_proof(c2.clone(), &params2, rng).unwrap();
            assert!(verify_proof(&pvk2, &proof_lower, &[]).unwrap());

            println!(
                "Lower proof gen finished in {}s and {}ms",
                now.elapsed().as_secs(),
                now.elapsed().subsec_millis()
            );
        }
    }

    higher_thread.join().unwrap();
}
