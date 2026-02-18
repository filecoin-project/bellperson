//! Helpers for packing vectors of bits into scalar field elements.

use ff::PrimeField;

use super::Assignment;
use super::boolean::Boolean;
use super::num::{AllocatedNum, Num};
use bellpepper_core::{ConstraintSystem, SynthesisError};

/// Takes a sequence of booleans and exposes them as compact
/// public inputs
pub fn pack_into_inputs<Scalar, CS>(mut cs: CS, bits: &[Boolean]) -> Result<(), SynthesisError>
where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
{
    for (i, bits) in bits.chunks(Scalar::CAPACITY as usize).enumerate() {
        let mut num = Num::<Scalar>::zero();
        let mut coeff = Scalar::ONE;
        for bit in bits {
            num = num.add_bool_with_coeff(CS::one(), bit, coeff);

            coeff = coeff.double();
        }

        let input = cs.alloc_input(|| format!("input {}", i), || Ok(*num.get_value().get()?))?;

        // num * 1 = input
        cs.enforce(
            || format!("packing constraint {}", i),
            |_| num.lc(Scalar::ONE),
            |lc| lc + CS::one(),
            |lc| lc + input,
        );
    }

    Ok(())
}

pub fn bytes_to_bits(bytes: &[u8]) -> Vec<bool> {
    bytes
        .iter()
        .flat_map(|&v| (0..8).rev().map(move |i| (v >> i) & 1 == 1))
        .collect()
}

pub fn bytes_to_bits_le(bytes: &[u8]) -> Vec<bool> {
    bytes
        .iter()
        .flat_map(|&v| (0..8).map(move |i| (v >> i) & 1 == 1))
        .collect()
}

pub fn compute_multipacking<Scalar: PrimeField>(bits: &[bool]) -> Vec<Scalar> {
    let mut result = vec![];

    for bits in bits.chunks(Scalar::CAPACITY as usize) {
        let mut cur = Scalar::ZERO;
        let mut coeff = Scalar::ONE;

        for bit in bits {
            if *bit {
                cur.add_assign(&coeff);
            }

            coeff = coeff.double();
        }

        result.push(cur);
    }

    result
}

/// Takes a sequence of booleans and exposes them as a single compact Num.
pub fn pack_bits<Scalar, CS>(
    mut cs: CS,
    bits: &[Boolean],
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
{
    let mut num = Num::<Scalar>::zero();
    let mut coeff = Scalar::ONE;
    for bit in bits.iter().take(Scalar::CAPACITY as usize) {
        num = num.add_bool_with_coeff(CS::one(), bit, coeff);

        coeff = coeff.double();
    }

    let alloc_num = AllocatedNum::alloc(cs.namespace(|| "input"), || {
        num.get_value().ok_or(SynthesisError::AssignmentMissing)
    })?;

    // num * 1 = input
    cs.enforce(
        || "packing constraint",
        |_| num.lc(Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + alloc_num.get_variable(),
    );

    Ok(alloc_num)
}

#[test]
fn test_multipacking() {
    use crate::ConstraintSystem;
    use blstrs::Scalar as Fr;
    use rand_core::{RngCore, SeedableRng};
    use rand_xorshift::XorShiftRng;

    use super::boolean::{AllocatedBit, Boolean};
    use crate::gadgets::test::*;

    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x3d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);

    for num_bits in 0..1500 {
        let mut cs = TestConstraintSystem::<Fr>::new();

        let bits: Vec<bool> = (0..num_bits)
            .map(|_| !rng.next_u32().is_multiple_of(2))
            .collect();

        let circuit_bits = bits
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                Boolean::from(
                    AllocatedBit::alloc(cs.namespace(|| format!("bit {}", i)), Some(b)).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let expected_inputs = compute_multipacking::<Fr>(&bits);

        pack_into_inputs(cs.namespace(|| "pack"), &circuit_bits).unwrap();

        assert!(cs.is_satisfied());
        assert!(cs.verify(&expected_inputs));
    }
}
