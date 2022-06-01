#![allow(clippy::op_ref)]

use ff::{Field, PrimeField};
use group::{
    prime::{PrimeCurve, PrimeCurveAffine, PrimeGroup},
    Curve, Group, GroupEncoding, UncompressedEncoding, WnafGroup,
};
use pairing::{Engine, MillerLoopResult, MultiMillerLoop, PairingCurveAffine};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use rand_core::RngCore;
use std::convert::TryInto;
use std::fmt;
use std::iter::Sum;
use std::num::Wrapping;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

const MODULUS_R: Wrapping<u32> = Wrapping(64513);
const R: u32 = 1;
#[cfg(any(feature = "cuda", feature = "opencl"))]
const R2: u32 = 1;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fr(Wrapping<u32>);

impl fmt::Display for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", (self.0).0)
    }
}

impl From<u64> for Fr {
    fn from(n: u64) -> Fr {
        assert!(n < MODULUS_R.0 as u64, "not in field");
        Fr(Wrapping(n as u32))
    }
}

impl Neg for &Fr {
    type Output = Fr;

    fn neg(self) -> Fr {
        Fr(MODULUS_R - self.0)
    }
}

impl Neg for Fr {
    type Output = Fr;

    fn neg(self) -> Fr {
        -&self
    }
}

impl Add<&Fr> for &Fr {
    type Output = Fr;

    fn add(self, rhs: &Fr) -> Fr {
        Fr((self.0 + rhs.0) % MODULUS_R)
    }
}

impl Add<Fr> for &Fr {
    type Output = Fr;

    fn add(self, rhs: Fr) -> Fr {
        self + &rhs
    }
}

impl Add<&Fr> for Fr {
    type Output = Fr;

    fn add(self, rhs: &Fr) -> Fr {
        &self + rhs
    }
}

impl Add<Fr> for Fr {
    type Output = Fr;

    fn add(self, rhs: Fr) -> Fr {
        &self + &rhs
    }
}

impl Sub<&Fr> for &Fr {
    type Output = Fr;

    fn sub(self, rhs: &Fr) -> Fr {
        self + -rhs
    }
}

impl Sub<Fr> for &Fr {
    type Output = Fr;

    fn sub(self, rhs: Fr) -> Fr {
        self - &rhs
    }
}

impl Sub<&Fr> for Fr {
    type Output = Fr;

    fn sub(self, rhs: &Fr) -> Fr {
        &self - rhs
    }
}

impl Sub<Fr> for Fr {
    type Output = Fr;

    fn sub(self, rhs: Fr) -> Fr {
        &self - &rhs
    }
}

impl Mul<&Fr> for &Fr {
    type Output = Fr;

    fn mul(self, rhs: &Fr) -> Fr {
        Fr((self.0 * rhs.0) % MODULUS_R)
    }
}

impl Mul<Fr> for &Fr {
    type Output = Fr;

    fn mul(self, rhs: Fr) -> Fr {
        self * &rhs
    }
}

impl Mul<&Fr> for Fr {
    type Output = Fr;

    fn mul(self, rhs: &Fr) -> Fr {
        &self * rhs
    }
}

impl Mul<Fr> for Fr {
    type Output = Fr;

    fn mul(self, rhs: Fr) -> Fr {
        &self * &rhs
    }
}

impl AddAssign<&Fr> for Fr {
    fn add_assign(&mut self, rhs: &Fr) {
        *self = *self + rhs;
    }
}

impl AddAssign<Fr> for Fr {
    fn add_assign(&mut self, rhs: Fr) {
        *self = *self + rhs;
    }
}

impl SubAssign<&Fr> for Fr {
    fn sub_assign(&mut self, rhs: &Fr) {
        *self = *self - rhs;
    }
}

impl SubAssign<Fr> for Fr {
    fn sub_assign(&mut self, rhs: Fr) {
        *self = *self - rhs;
    }
}

impl MulAssign<&Fr> for Fr {
    fn mul_assign(&mut self, rhs: &Fr) {
        *self = *self * rhs;
    }
}

impl MulAssign<Fr> for Fr {
    fn mul_assign(&mut self, rhs: Fr) {
        *self = *self * rhs;
    }
}

impl ConditionallySelectable for Fr {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        if choice.unwrap_u8() == 0 {
            *a
        } else {
            *b
        }
    }
}

impl ConstantTimeEq for Fr {
    fn ct_eq(&self, other: &Self) -> Choice {
        (self.0).0.ct_eq(&(other.0).0)
    }
}

impl Field for Fr {
    fn random(mut rng: impl RngCore) -> Self {
        Fr(Wrapping(rng.next_u32()) % MODULUS_R)
    }

    fn zero() -> Self {
        Fr(Wrapping(0))
    }

    fn one() -> Self {
        Fr(Wrapping(R))
    }

    fn is_zero(&self) -> Choice {
        (self.0).0.ct_eq(&0)
    }

    fn square(&self) -> Self {
        Fr((self.0 * self.0) % MODULUS_R)
    }

    fn double(&self) -> Self {
        Fr((self.0 << 1) % MODULUS_R)
    }

    fn invert(&self) -> CtOption<Self> {
        if <Fr as Field>::is_zero(self).into() {
            CtOption::new(Self::default(), Choice::from(0))
        } else {
            let inv = self.pow_vartime(&[(MODULUS_R.0 as u64) - 2]);
            CtOption::new(inv, Choice::from(1))
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn sqrt(&self) -> CtOption<Self> {
        // Tonelli-Shank's algorithm for q mod 16 = 1
        // https://eprint.iacr.org/2012/685.pdf (page 12, algorithm 5)
        match self.legendre() {
            0 => CtOption::new(*self, Choice::from(1)),
            -1 => CtOption::new(Self::default(), Choice::from(0)),
            1 => {
                let mut c = Fr::root_of_unity();
                // r = self^((t + 1) // 2)
                let mut r = self.pow_vartime([32]);
                // t = self^t
                let mut t = self.pow_vartime([63]);
                let mut m = Fr::S;

                while t != <Fr as Field>::one() {
                    let mut i = 1;
                    {
                        let mut t2i = t;
                        t2i = t2i.square();
                        loop {
                            if t2i == <Fr as Field>::one() {
                                break;
                            }
                            t2i = t2i.square();
                            i += 1;
                        }
                    }

                    for _ in 0..(m - i - 1) {
                        c = c.square();
                    }
                    r.mul_assign(&c);
                    c = c.square();
                    t.mul_assign(&c);
                    m = i;
                }

                CtOption::new(r, Choice::from(1))
            }
            _ => unreachable!(),
        }
    }
}

impl Fr {
    fn legendre(&self) -> i8 {
        // s = self^((r - 1) // 2)
        let s = self.pow_vartime([32256]);
        if s == <Fr as Field>::zero() {
            0
        } else if s == <Fr as Field>::one() {
            1
        } else {
            -1
        }
    }
}

impl PrimeField for Fr {
    // `group::Wnaf` requires the field repr to be at least 64 bits despite `DummyEngine`'s scalars
    // being representable using two bytes (i.e. `Fr` has a 15-bit modulus).
    type Repr = [u8; 8];

    const NUM_BITS: u32 = 16;
    const CAPACITY: u32 = 15;
    const S: u32 = 10;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        // Only the first two bytes should be utilized.
        assert!(repr[2..].iter().all(|byte| *byte == 0));

        let repr: [u8; 2] = repr[2..].try_into().unwrap();
        let int = Wrapping(u16::from_le_bytes(repr) as u32);
        let is_valid = int < MODULUS_R;
        CtOption::new(Fr(int), Choice::from(is_valid as u8))
    }

    fn to_repr(&self) -> Self::Repr {
        let int: u16 = (self.0).0.try_into().unwrap();
        let mut repr = [0u8; 8];
        repr[..2].copy_from_slice(&int.to_le_bytes());
        repr
    }

    fn is_odd(&self) -> Choice {
        Choice::from(((self.0).0 & 1) as u8)
    }

    fn multiplicative_generator() -> Fr {
        Fr(Wrapping(5))
    }

    fn root_of_unity() -> Fr {
        Fr(Wrapping(57751))
    }
}

#[derive(Debug, Clone)]
pub struct DummyEngine;

impl blstrs::Compress for Fr {
    fn write_compressed<W: std::io::Write>(self, _out: W) -> std::io::Result<()> {
        unimplemented!()
    }

    fn read_compressed<R: std::io::Read>(_source: R) -> std::io::Result<Self> {
        unimplemented!()
    }
}

impl Engine for DummyEngine {
    type Fr = Fr;
    type G1 = Fr;
    type G1Affine = Fr;
    type G2 = Fr;
    type G2Affine = Fr;
    type Gt = Fr;

    fn pairing(p: &Self::G1Affine, q: &Self::G2Affine) -> Self::Gt {
        Self::multi_miller_loop(&[(p, q)]).final_exponentiation()
    }
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
impl ec_gpu::GpuName for Fr {
    fn name() -> String {
        "dummy_fr".to_string()
    }
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl crate::gpu::GpuName for Fr {}

impl MillerLoopResult for Fr {
    type Gt = Fr;

    fn final_exponentiation(&self) -> Self::Gt {
        *self
    }
}

impl MultiMillerLoop for DummyEngine {
    type G2Prepared = Fr;
    type Result = Fr;

    fn multi_miller_loop(i: &[(&Self::G1Affine, &Self::G2Prepared)]) -> Self::Result {
        let mut acc = <Fr as Field>::zero();
        for (&a, &b) in i {
            acc += a * b;
        }
        acc
    }
}

impl<'a> Sum<&'a Self> for Fr {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(<Self as Group>::identity(), |acc, item| acc + item)
    }
}

impl Sum for Fr {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.sum()
    }
}

impl Group for Fr {
    type Scalar = Fr;

    fn random(rng: impl RngCore) -> Self {
        <Fr as Field>::random(rng)
    }

    fn identity() -> Self {
        <Fr as Field>::zero()
    }

    fn generator() -> Self {
        <Fr as Field>::one()
    }

    fn is_identity(&self) -> Choice {
        <Fr as Field>::is_zero(self)
    }

    fn double(&self) -> Self {
        <Fr as Field>::double(self)
    }
}

impl GroupEncoding for Fr {
    type Repr = <Fr as PrimeField>::Repr;

    fn from_bytes(bytes: &Self::Repr) -> CtOption<Self> {
        <Fr as PrimeField>::from_repr(*bytes)
            .or_else(|| CtOption::new(Self::default(), Choice::from(1)))
    }

    fn from_bytes_unchecked(bytes: &Self::Repr) -> CtOption<Self> {
        Self::from_bytes(bytes)
    }

    fn to_bytes(&self) -> Self::Repr {
        self.to_repr()
    }
}

impl UncompressedEncoding for Fr {
    type Uncompressed = <Fr as GroupEncoding>::Repr;

    fn from_uncompressed(bytes: &Self::Uncompressed) -> CtOption<Self> {
        Self::from_bytes(bytes)
    }

    fn from_uncompressed_unchecked(bytes: &Self::Uncompressed) -> CtOption<Self> {
        Self::from_bytes_unchecked(bytes)
    }

    fn to_uncompressed(&self) -> Self::Uncompressed {
        self.to_bytes()
    }
}

impl WnafGroup for Fr {
    fn recommended_wnaf_for_num_scalars(_num_scalars: usize) -> usize {
        3
    }
}

impl PrimeGroup for Fr {}

impl Curve for Fr {
    type AffineRepr = Fr;

    fn to_affine(&self) -> Self::AffineRepr {
        *self
    }
}

impl PrimeCurve for Fr {
    type Affine = Fr;
}

impl PrimeCurveAffine for Fr {
    type Scalar = Fr;
    type Curve = Fr;

    fn identity() -> Self {
        <Fr as Field>::zero()
    }

    fn generator() -> Self {
        <Fr as Field>::one()
    }

    fn is_identity(&self) -> Choice {
        <Fr as Group>::is_identity(self)
    }

    fn to_curve(&self) -> Self::Curve {
        *self
    }
}

impl PairingCurveAffine for Fr {
    type Pair = Fr;
    type PairingResult = Fr;

    fn pairing_with(&self, other: &Self::Pair) -> Self::PairingResult {
        self.mul(*other)
    }
}
