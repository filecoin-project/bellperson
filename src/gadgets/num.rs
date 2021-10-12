//! Gadgets representing numbers in the scalar field of the underlying curve.

use ff::{PrimeField, PrimeFieldBits};

use crate::{ConstraintSystem, LinearCombination, SynthesisError, Variable};

use super::Assignment;

use super::boolean::{self, AllocatedBit, Boolean};

pub struct AllocatedNum<Scalar: PrimeField> {
    value: Option<Scalar>,
    variable: Variable,
}

impl<Scalar: PrimeField> Clone for AllocatedNum<Scalar> {
    fn clone(&self) -> Self {
        AllocatedNum {
            value: self.value,
            variable: self.variable,
        }
    }
}

impl<Scalar: PrimeField> AllocatedNum<Scalar> {
    /// Allocate a `Variable(Aux)` in a `ConstraintSystem`.
    pub fn alloc<CS, F>(mut cs: CS, value: F) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
        F: FnOnce() -> Result<Scalar, SynthesisError>,
    {
        let mut new_value = None;
        let var = cs.alloc(
            || "num",
            || {
                let tmp = value()?;

                new_value = Some(tmp);

                Ok(tmp)
            },
        )?;

        Ok(AllocatedNum {
            value: new_value,
            variable: var,
        })
    }

    /// Allocate a `Variable(Input)` in a `ConstraintSystem`.
    pub fn alloc_input<CS, F>(mut cs: CS, value: F) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
        F: FnOnce() -> Result<Scalar, SynthesisError>,
    {
        let mut new_value = None;
        let var = cs.alloc_input(
            || "input num",
            || {
                let tmp = value()?;

                new_value = Some(tmp);

                Ok(tmp)
            },
        )?;

        Ok(AllocatedNum {
            value: new_value,
            variable: var,
        })
    }

    /// Allocate a `Variable` of either `Aux` or `Input` in a
    /// `ConstraintSystem`. The `Variable` is a an `Input` if `is_input` is
    /// true. This allows uniform creation of circuits containing components
    /// which may or may not be public inputs.
    pub fn alloc_maybe_input<CS, F>(
        cs: CS,
        is_input: bool,
        value: F,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
        F: FnOnce() -> Result<Scalar, SynthesisError>,
    {
        if is_input {
            Self::alloc_input(cs, value)
        } else {
            Self::alloc(cs, value)
        }
    }

    pub fn inputize<CS>(&self, mut cs: CS) -> Result<(), SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
    {
        let input = cs.alloc_input(|| "input variable", || Ok(*self.value.get()?))?;

        cs.enforce(
            || "enforce input is correct",
            |lc| lc + input,
            |lc| lc + CS::one(),
            |lc| lc + self.variable,
        );

        Ok(())
    }

    /// Deconstructs this allocated number into its
    /// boolean representation in little-endian bit
    /// order, requiring that the representation
    /// strictly exists "in the field" (i.e., a
    /// congruency is not allowed.)
    pub fn to_bits_le_strict<CS>(&self, mut cs: CS) -> Result<Vec<Boolean>, SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
        Scalar: PrimeFieldBits,
    {
        pub fn kary_and<Scalar, CS>(
            mut cs: CS,
            v: &[AllocatedBit],
        ) -> Result<AllocatedBit, SynthesisError>
        where
            Scalar: PrimeField,
            CS: ConstraintSystem<Scalar>,
        {
            assert!(!v.is_empty());

            // Let's keep this simple for now and just AND them all
            // manually
            let mut cur = None;

            for (i, v) in v.iter().enumerate() {
                if cur.is_none() {
                    cur = Some(v.clone());
                } else {
                    cur = Some(AllocatedBit::and(
                        cs.namespace(|| format!("and {}", i)),
                        cur.as_ref().unwrap(),
                        v,
                    )?);
                }
            }

            Ok(cur.expect("v.len() > 0"))
        }

        // We want to ensure that the bit representation of a is
        // less than or equal to r - 1.
        let a = self.value.map(|e| e.to_le_bits());
        let b = (-Scalar::one()).to_le_bits();

        // Get the bits of `a` in big-endian order.
        let mut a = a.as_ref().map(|e| e.into_iter().rev());

        let mut result = vec![];

        // Runs of ones in r
        let mut last_run = None;
        let mut current_run = vec![];

        let mut found_one = false;
        let mut i = 0;
        for b in b.into_iter().rev() {
            let a_bit: Option<bool> = a.as_mut().map(|e| *e.next().unwrap());

            // Skip over unset bits at the beginning
            found_one |= b;
            if !found_one {
                // a_bit should also be false
                if let Some(a_bit) = a_bit {
                    assert!(!a_bit);
                }
                continue;
            }

            if b {
                // This is part of a run of ones. Let's just
                // allocate the boolean with the expected value.
                let a_bit = AllocatedBit::alloc(cs.namespace(|| format!("bit {}", i)), a_bit)?;
                // ... and add it to the current run of ones.
                current_run.push(a_bit.clone());
                result.push(a_bit);
            } else {
                if !current_run.is_empty() {
                    // This is the start of a run of zeros, but we need
                    // to k-ary AND against `last_run` first.

                    if last_run.is_some() {
                        current_run.push(last_run.clone().unwrap());
                    }
                    last_run = Some(kary_and(
                        cs.namespace(|| format!("run ending at {}", i)),
                        &current_run,
                    )?);
                    current_run.truncate(0);
                }

                // If `last_run` is true, `a` must be false, or it would
                // not be in the field.
                //
                // If `last_run` is false, `a` can be true or false.

                let a_bit = AllocatedBit::alloc_conditionally(
                    cs.namespace(|| format!("bit {}", i)),
                    a_bit,
                    &last_run.as_ref().expect("char always starts with a one"),
                )?;
                result.push(a_bit);
            }

            i += 1;
        }

        // char is prime, so we'll always end on
        // a run of zeros.
        assert_eq!(current_run.len(), 0);

        // Now, we have `result` in big-endian order.
        // However, now we have to unpack self!

        let mut lc = LinearCombination::zero();
        let mut coeff = Scalar::one();

        for bit in result.iter().rev() {
            lc = lc + (coeff, bit.get_variable());

            coeff = coeff.double();
        }

        lc = lc - self.variable;

        cs.enforce(|| "unpacking constraint", |lc| lc, |lc| lc, |_| lc);

        // Convert into booleans, and reverse for little-endian bit order
        Ok(result.into_iter().map(Boolean::from).rev().collect())
    }

    /// Convert the allocated number into its little-endian representation.
    /// Note that this does not strongly enforce that the commitment is
    /// "in the field."
    pub fn to_bits_le<CS>(&self, mut cs: CS) -> Result<Vec<Boolean>, SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
        Scalar: PrimeFieldBits,
    {
        let bits = boolean::field_into_allocated_bits_le(&mut cs, self.value)?;

        let mut lc = LinearCombination::zero();
        let mut coeff = Scalar::one();

        for bit in bits.iter() {
            lc = lc + (coeff, bit.get_variable());

            coeff = coeff.double();
        }

        lc = lc - self.variable;

        cs.enforce(|| "unpacking constraint", |lc| lc, |lc| lc, |_| lc);

        Ok(bits.into_iter().map(Boolean::from).collect())
    }

    pub fn mul<CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
    {
        let mut value = None;

        let var = cs.alloc(
            || "product num",
            || {
                let mut tmp = *self.value.get()?;
                tmp.mul_assign(other.value.get()?);

                value = Some(tmp);

                Ok(tmp)
            },
        )?;

        // Constrain: a * b = ab
        cs.enforce(
            || "multiplication constraint",
            |lc| lc + self.variable,
            |lc| lc + other.variable,
            |lc| lc + var,
        );

        Ok(AllocatedNum {
            value,
            variable: var,
        })
    }

    pub fn square<CS>(&self, mut cs: CS) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
    {
        let mut value = None;

        let var = cs.alloc(
            || "squared num",
            || {
                let mut tmp = *self.value.get()?;
                tmp = tmp.square();

                value = Some(tmp);

                Ok(tmp)
            },
        )?;

        // Constrain: a * a = aa
        cs.enforce(
            || "squaring constraint",
            |lc| lc + self.variable,
            |lc| lc + self.variable,
            |lc| lc + var,
        );

        Ok(AllocatedNum {
            value,
            variable: var,
        })
    }

    pub fn assert_nonzero<CS>(&self, mut cs: CS) -> Result<(), SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
    {
        let inv = cs.alloc(
            || "ephemeral inverse",
            || {
                let tmp = *self.value.get()?;

                if tmp.is_zero().into() {
                    Err(SynthesisError::DivisionByZero)
                } else {
                    Ok(tmp.invert().unwrap())
                }
            },
        )?;

        // Constrain a * inv = 1, which is only valid
        // iff a has a multiplicative inverse, untrue
        // for zero.
        cs.enforce(
            || "nonzero assertion constraint",
            |lc| lc + self.variable,
            |lc| lc + inv,
            |lc| lc + CS::one(),
        );

        Ok(())
    }

    /// Takes two allocated numbers (a, b) and returns
    /// (b, a) if the condition is true, and (a, b)
    /// otherwise.
    pub fn conditionally_reverse<CS>(
        mut cs: CS,
        a: &Self,
        b: &Self,
        condition: &Boolean,
    ) -> Result<(Self, Self), SynthesisError>
    where
        CS: ConstraintSystem<Scalar>,
    {
        let c = Self::alloc(cs.namespace(|| "conditional reversal result 1"), || {
            if *condition.get_value().get()? {
                Ok(*b.value.get()?)
            } else {
                Ok(*a.value.get()?)
            }
        })?;

        cs.enforce(
            || "first conditional reversal",
            |lc| lc + a.variable - b.variable,
            |_| condition.lc(CS::one(), Scalar::one()),
            |lc| lc + a.variable - c.variable,
        );

        let d = Self::alloc(cs.namespace(|| "conditional reversal result 2"), || {
            if *condition.get_value().get()? {
                Ok(*a.value.get()?)
            } else {
                Ok(*b.value.get()?)
            }
        })?;

        cs.enforce(
            || "second conditional reversal",
            |lc| lc + b.variable - a.variable,
            |_| condition.lc(CS::one(), Scalar::one()),
            |lc| lc + b.variable - d.variable,
        );

        Ok((c, d))
    }

    pub fn get_value(&self) -> Option<Scalar> {
        self.value
    }

    pub fn get_variable(&self) -> Variable {
        self.variable
    }
}

#[derive(Clone)]
pub struct Num<Scalar: PrimeField> {
    value: Option<Scalar>,
    lc: LinearCombination<Scalar>,
}

impl<Scalar: PrimeField> From<AllocatedNum<Scalar>> for Num<Scalar> {
    fn from(num: AllocatedNum<Scalar>) -> Num<Scalar> {
        Num {
            value: num.value,
            lc: LinearCombination::<Scalar>::from_variable(num.variable),
        }
    }
}

impl<Scalar: PrimeField> Num<Scalar> {
    pub fn zero() -> Self {
        Num {
            value: Some(Scalar::zero()),
            lc: LinearCombination::zero(),
        }
    }

    pub fn get_value(&self) -> Option<Scalar> {
        self.value
    }

    pub fn lc(&self, coeff: Scalar) -> LinearCombination<Scalar> {
        LinearCombination::zero() + (coeff, &self.lc)
    }

    pub fn add_bool_with_coeff(self, one: Variable, bit: &Boolean, coeff: Scalar) -> Self {
        let newval = match (self.value, bit.get_value()) {
            (Some(mut curval), Some(bval)) => {
                if bval {
                    curval.add_assign(&coeff);
                }

                Some(curval)
            }
            _ => None,
        };

        Num {
            value: newval,
            lc: self.lc + &bit.lc(one, coeff),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: &Self) -> Self {
        let lc = self.lc + &other.lc;
        let value = match (self.value, other.value) {
            (Some(v1), Some(v2)) => {
                let mut tmp = v1;
                tmp.add_assign(&v2);
                Some(tmp)
            }
            (Some(v), None) | (None, Some(v)) => Some(v),
            (None, None) => None,
        };

        Num { value, lc }
    }

    pub fn scale(mut self, scalar: Scalar) -> Self {
        for (_variable, fr) in self.lc.iter_mut() {
            fr.mul_assign(&scalar);
        }

        if let Some(ref mut v) = self.value {
            v.mul_assign(&scalar);
        }

        self
    }
}

#[cfg(test)]
mod test {
    use std::ops::{AddAssign, MulAssign, SubAssign};

    use crate::ConstraintSystem;
    use blstrs::Scalar as Fr;
    use ff::{Field, PrimeField, PrimeFieldBits};
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    use super::{AllocatedNum, Boolean, Num};
    use crate::gadgets::test::*;

    #[test]
    fn test_allocated_num() {
        let mut cs = TestConstraintSystem::<Fr>::new();

        AllocatedNum::alloc(&mut cs, || Ok(Fr::one())).unwrap();

        assert!(cs.get("num") == Fr::one());
    }

    #[test]
    fn test_num_squaring() {
        let mut cs = TestConstraintSystem::<Fr>::new();

        let n = AllocatedNum::alloc(&mut cs, || Ok(Fr::from(3u64))).unwrap();
        let n2 = n.square(&mut cs).unwrap();

        assert!(cs.is_satisfied());
        assert!(cs.get("squared num") == Fr::from(9u64));
        assert!(n2.value.unwrap() == Fr::from(9u64));
        cs.set("squared num", Fr::from(10u64));
        assert!(!cs.is_satisfied());
    }

    #[test]
    fn test_num_multiplication() {
        let mut cs = TestConstraintSystem::<Fr>::new();

        let n = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::from(12u64))).unwrap();
        let n2 = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::from(10u64))).unwrap();
        let n3 = n.mul(&mut cs, &n2).unwrap();

        assert!(cs.is_satisfied());
        assert!(cs.get("product num") == Fr::from(120u64));
        assert!(n3.value.unwrap() == Fr::from(120u64));
        cs.set("product num", Fr::from(121u64));
        assert!(!cs.is_satisfied());
    }

    #[test]
    fn test_num_conditional_reversal() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x3d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);
        {
            let mut cs = TestConstraintSystem::<Fr>::new();

            let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::random(&mut rng))).unwrap();
            let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::random(&mut rng))).unwrap();
            let condition = Boolean::constant(false);
            let (c, d) = AllocatedNum::conditionally_reverse(&mut cs, &a, &b, &condition).unwrap();

            assert!(cs.is_satisfied());

            assert_eq!(a.value.unwrap(), c.value.unwrap());
            assert_eq!(b.value.unwrap(), d.value.unwrap());
        }

        {
            let mut cs = TestConstraintSystem::<Fr>::new();

            let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::random(&mut rng))).unwrap();
            let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::random(&mut rng))).unwrap();
            let condition = Boolean::constant(true);
            let (c, d) = AllocatedNum::conditionally_reverse(&mut cs, &a, &b, &condition).unwrap();

            assert!(cs.is_satisfied());

            assert_eq!(a.value.unwrap(), d.value.unwrap());
            assert_eq!(b.value.unwrap(), c.value.unwrap());
        }
    }

    #[test]
    fn test_num_nonzero() {
        {
            let mut cs = TestConstraintSystem::<Fr>::new();

            let n = AllocatedNum::alloc(&mut cs, || Ok(Fr::from(3u64))).unwrap();
            n.assert_nonzero(&mut cs).unwrap();

            assert!(cs.is_satisfied());
            cs.set("ephemeral inverse", Fr::from(3u64));
            assert!(cs.which_is_unsatisfied() == Some("nonzero assertion constraint"));
        }
        {
            let mut cs = TestConstraintSystem::<Fr>::new();

            let n = AllocatedNum::alloc(&mut cs, || Ok(Fr::zero())).unwrap();
            assert!(n.assert_nonzero(&mut cs).is_err());
        }
    }

    #[test]
    fn test_into_bits_strict() {
        let negone = -Fr::one();

        let mut cs = TestConstraintSystem::<Fr>::new();

        let n = AllocatedNum::alloc(&mut cs, || Ok(negone)).unwrap();
        n.to_bits_le_strict(&mut cs).unwrap();

        assert!(cs.is_satisfied());

        // make the bit representation the characteristic
        cs.set("bit 254/boolean", Fr::one());

        // this makes the conditional boolean constraint fail
        assert_eq!(
            cs.which_is_unsatisfied().unwrap(),
            "bit 254/boolean constraint"
        );
    }

    #[test]
    fn test_into_bits() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x3d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for i in 0..200 {
            let r = Fr::random(&mut rng);
            let mut cs = TestConstraintSystem::<Fr>::new();

            let n = AllocatedNum::alloc(&mut cs, || Ok(r)).unwrap();

            let bits = if i % 2 == 0 {
                n.to_bits_le(&mut cs).unwrap()
            } else {
                n.to_bits_le_strict(&mut cs).unwrap()
            };

            assert!(cs.is_satisfied());

            for (i, b) in r.to_le_bits().iter().enumerate() {
                // `r.to_le_bits()` contains every bit in a representation (including bits which
                // exceed the field size), whereas the length of `bits` does not exceed the field
                // size.
                match bits.get(i) {
                    Some(Boolean::Is(a)) => assert_eq!(b, a.get_value().unwrap()),
                    Some(_) => unreachable!(),
                    None => assert_eq!(b, false),
                };
            }

            cs.set("num", Fr::random(&mut rng));
            assert!(!cs.is_satisfied());
            cs.set("num", r);
            assert!(cs.is_satisfied());

            for i in 0..Fr::NUM_BITS {
                let name = format!("bit {}/boolean", i);
                let cur = cs.get(&name);
                let mut tmp = Fr::one();
                tmp.sub_assign(&cur);
                cs.set(&name, tmp);
                assert!(!cs.is_satisfied());
                cs.set(&name, cur);
                assert!(cs.is_satisfied());
            }
        }
    }

    #[test]
    fn test_num_scale() {
        use crate::{Index, LinearCombination, Variable};

        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x3d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        let n = 5;

        let mut lc = LinearCombination::<Fr>::zero();

        let mut expected_sums = vec![Fr::zero(); n];
        let mut value = Fr::zero();
        for (i, expected_sum) in expected_sums.iter_mut().enumerate() {
            let coeff = Fr::random(&mut rng);
            lc = lc + (coeff, Variable::new_unchecked(Index::Aux(i)));
            expected_sum.add_assign(&coeff);

            value.add_assign(&coeff);
        }

        let scalar = Fr::random(&mut rng);
        let num = Num {
            value: Some(value),
            lc,
        };

        let scaled_num = num.clone().scale(scalar);

        let mut scaled_value = num.value.unwrap();
        scaled_value.mul_assign(&scalar);

        assert_eq!(scaled_value, scaled_num.value.unwrap());

        // Each variable has the expected coefficient, the sume of those added by its Index.
        scaled_num.lc.iter().for_each(|(var, coeff)| match var.0 {
            Index::Aux(i) => {
                let mut tmp = expected_sums[i];
                tmp.mul_assign(&scalar);
                assert_eq!(tmp, *coeff)
            }
            _ => panic!("unexpected variable type"),
        });
    }
}
