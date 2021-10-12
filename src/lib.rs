#![allow(clippy::suspicious_arithmetic_impl)]
//! `bellperson` is a crate for building zk-SNARK circuits. It provides circuit
//! traits and and primitive structures, as well as basic gadget implementations
//! such as booleans and number abstractions.
//!
//! # Example circuit
//!
//! Say we want to write a circuit that proves we know the preimage to some hash
//! computed using SHA-256d (calling SHA-256 twice). The preimage must have a
//! fixed length known in advance (because the circuit parameters will depend on
//! it), but can otherwise have any value. We take the following strategy:
//!
//! - Witness each bit of the preimage.
//! - Compute `hash = SHA-256d(preimage)` inside the circuit.
//! - Expose `hash` as a public input using multiscalar packing.
//!
//! ```no_run
//! use bellperson::{
//!     gadgets::{
//!         boolean::{AllocatedBit, Boolean},
//!         multipack,
//!         sha256::sha256,
//!     },
//!     groth16, Circuit, ConstraintSystem, SynthesisError,
//! };
//! use blstrs::Bls12;
//! use ff::PrimeField;
//! use pairing::Engine;
//! use rand::rngs::OsRng;
//! use sha2::{Digest, Sha256};
//!
//! /// Our own SHA-256d gadget. Input and output are in little-endian bit order.
//! fn sha256d<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
//!     mut cs: CS,
//!     data: &[Boolean],
//! ) -> Result<Vec<Boolean>, SynthesisError> {
//!     // Flip endianness of each input byte
//!     let input: Vec<_> = data
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect();
//!
//!     let mid = sha256(cs.namespace(|| "SHA-256(input)"), &input)?;
//!     let res = sha256(cs.namespace(|| "SHA-256(mid)"), &mid)?;
//!
//!     // Flip endianness of each output byte
//!     Ok(res
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect())
//! }
//!
//! struct MyCircuit {
//!     /// The input to SHA-256d we are proving that we know. Set to `None` when we
//!     /// are verifying a proof (and do not have the witness data).
//!     preimage: Option<[u8; 80]>,
//! }
//!
//! impl<Scalar: PrimeField> Circuit<Scalar> for MyCircuit {
//!     fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
//!         // Compute the values for the bits of the preimage. If we are verifying a proof,
//!         // we still need to create the same constraints, so we return an equivalent-size
//!         // Vec of None (indicating that the value of each bit is unknown).
//!         let bit_values = if let Some(preimage) = self.preimage {
//!             preimage
//!                 .iter()
//!                 .map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
//!                 .flatten()
//!                 .map(|b| Some(b))
//!                 .collect()
//!         } else {
//!             vec![None; 80 * 8]
//!         };
//!         assert_eq!(bit_values.len(), 80 * 8);
//!
//!         // Witness the bits of the preimage.
//!         let preimage_bits = bit_values
//!             .into_iter()
//!             .enumerate()
//!             // Allocate each bit.
//!             .map(|(i, b)| {
//!                 AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {}", i)), b)
//!             })
//!             // Convert the AllocatedBits into Booleans (required for the sha256 gadget).
//!             .map(|b| b.map(Boolean::from))
//!             .collect::<Result<Vec<_>, _>>()?;
//!
//!         // Compute hash = SHA-256d(preimage).
//!         let hash = sha256d(cs.namespace(|| "SHA-256d(preimage)"), &preimage_bits)?;
//!
//!         // Expose the vector of 32 boolean variables as compact public inputs.
//!         multipack::pack_into_inputs(cs.namespace(|| "pack hash"), &hash)
//!     }
//! }
//!
//! // Create parameters for our circuit. In a production deployment these would
//! // be generated securely using a multiparty computation.
//! let params = {
//!     let c = MyCircuit { preimage: None };
//!     groth16::generate_random_parameters::<Bls12, _, _>(c, &mut OsRng).unwrap()
//! };
//!
//! // Prepare the verification key (for proof verification).
//! let pvk = groth16::prepare_verifying_key(&params.vk);
//!
//! // Pick a preimage and compute its hash.
//! let preimage = [42; 80];
//! let hash = Sha256::digest(&Sha256::digest(&preimage));
//!
//! // Create an instance of our circuit (with the preimage as a witness).
//! let c = MyCircuit {
//!     preimage: Some(preimage),
//! };
//!
//! // Create a Groth16 proof with our parameters.
//! let proof = groth16::create_random_proof(c, &params, &mut OsRng).unwrap();
//!
//! // Pack the hash as inputs for proof verification.
//! let hash_bits = multipack::bytes_to_bits_le(&hash);
//! let inputs = multipack::compute_multipacking::<<Bls12 as Engine>::Fr>(&hash_bits);
//!
//! // Check the proof!
//! assert!(groth16::verify_proof(&pvk, &proof, &inputs).unwrap());
//! ```
//!
//! # Roadmap
//!
//! `bellperson` is being refactored into a generic proving library. Currently it
//! is pairing-specific, and different types of proving systems need to be
//! implemented as sub-modules. After the refactor, `bellperson` will be generic
//! using the [`ff`] and [`group`] crates, while specific proving systems will
//! be separate crates that pull in the dependencies they require.

// Requires nightly for aarch64
#![cfg_attr(target_arch = "aarch64", feature(stdsimd))]

#[cfg(test)]
#[macro_use]
extern crate hex_literal;

pub mod domain;
pub mod gadgets;
pub mod gpu;
#[cfg(feature = "groth16")]
pub mod groth16;
pub mod multicore;
pub mod multiexp;
#[cfg(test)]
pub mod test_utils;
pub mod util_cs;

mod lc;
pub use lc::{Index, LinearCombination, Variable};
mod constraint_system;
pub use constraint_system::{Circuit, ConstraintSystem, Namespace, SynthesisError};

pub const BELLMAN_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "groth16")]
pub(crate) fn le_bytes_to_u64s(le_bytes: &[u8]) -> Vec<u64> {
    use std::convert::TryInto;

    assert_eq!(
        le_bytes.len() % 8,
        0,
        "length must be divisible by u64 byte length (8-bytes)"
    );
    le_bytes
        .chunks(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
