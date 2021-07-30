/// This module implements two binding commitment schemes used in the Groth16
/// aggregation.
/// The first one is a commitment scheme that commits to a single vector $a$ of
/// length n in the second base group $G_1$ (for example):
/// * it requires a structured SRS $v_1$ of the form $(h,h^u,h^{u^2}, ...
/// ,g^{h^{n-1}})$ with $h \in G_2$ being a random generator of $G_2$ and $u$ a
/// random scalar (coming from a power of tau ceremony for example)
/// * it requires a second structured SRS $v_2$ of the form $(h,h^v,h^{v^2},
/// ...$ with $v$ being a random scalar different than u (coming from another
/// power of tau ceremony for example)
/// The Commitment is a tuple $(\prod_{i=0}^{n-1} e(a_i,v_{1,i}),
/// \prod_{i=0}^{n-1} e(a_i,v_{2,i}))$
///
/// The second one takes two vectors $a \in G_1^n$ and $b \in G_2^n$ and commits
/// to them using a similar approach as above. It requires an additional SRS
/// though:
/// * $v_1$ and $v_2$ stay the same
/// * An additional tuple $w_1 = (g^{u^n},g^{u^{n+1}},...g^{u^{2n-1}})$ and $w_2 =
/// (g^{v^n},g^{v^{n+1},...,g^{v^{2n-1}})$ where $g$ is a random generator of
/// $G_1$
/// The commitment scheme returns a tuple:
/// * $\prod_{i=0}^{n-1} e(a_i,v_{1,i})e(w_{1,i},b_i)$
/// * $\prod_{i=0}^{n-1} e(a_i,v_{2,i})e(w_{2,i},b_i)$
///
/// The second commitment scheme enables to save some KZG verification in the
/// verifier of the Groth16 verification protocol since we pack two vectors in
/// one commitment.
use std::ops::AddAssign;

use group::{prime::PrimeCurveAffine, Curve};
use rayon::prelude::*;

use crate::groth16::aggregate::inner_product;
use crate::SynthesisError;
use pairing::{Engine, MultiMillerLoop};

/// Key is a generic commitment key that is instanciated with g and h as basis,
/// and a and b as powers.
#[derive(Clone, Debug)]
pub struct Key<G: PrimeCurveAffine> {
    /// Exponent is a
    pub a: Vec<G>,
    /// Exponent is b
    pub b: Vec<G>,
}
/// Commitment key used by the "single" commitment on G1 values as
/// well as in the "pair" commtitment.
/// It contains $\{h^a^i\}_{i=1}^n$ and $\{h^b^i\}_{i=1}^n$
pub type VKey<E> = Key<<E as Engine>::G2Affine>;

/// Commitment key used by the "pair" commitment. Note the sequence of
/// powers starts at $n$ already.
/// It contains $\{g^{a^{n+i}}\}_{i=1}^n$ and $\{g^{b^{n+i}}\}_{i=1}^n$
pub type WKey<E> = Key<<E as Engine>::G1Affine>;

impl<G> Key<G>
where
    G: PrimeCurveAffine,
{
    /// Returns true if commitment keys have the exact required length.
    /// It is necessary for the IPP scheme to work that commitment
    /// key have the exact same number of arguments as the number of proofs to
    /// aggregate.
    pub fn has_correct_len(&self, n: usize) -> bool {
        self.a.len() == n && self.b.len() == n
    }

    /// Returns both vectors scaled by the given vector entrywise.
    /// In other words, it returns $\{v_i^{s_i}\}$
    pub fn scale(&self, s_vec: &[G::Scalar]) -> Result<Self, SynthesisError> {
        if self.a.len() != s_vec.len() {
            return Err(SynthesisError::IncompatibleLengthVector(
                "scaling commitment key".to_string(),
            ));
        }
        let (a, b) = self
            .a
            .par_iter()
            .zip(self.b.par_iter())
            .zip(s_vec.par_iter())
            .map(|((ap, bp), si)| {
                let v1s = (ap.to_curve() * si).to_affine();
                let v2s = (bp.to_curve() * si).to_affine();
                (v1s, v2s)
            })
            .unzip();

        Ok(Self { a, b })
    }

    /// Returns the left and right commitment key part. It makes copy.
    pub fn split(mut self, at: usize) -> (Self, Self) {
        let a_right = self.a.split_off(at);
        let b_right = self.b.split_off(at);
        (
            Self {
                a: self.a,
                b: self.b,
            },
            Self {
                a: a_right,
                b: b_right,
            },
        )
    }

    /// Takes a left and right commitment key and returns a commitment
    /// key $left \circ right^{scale} = (left_i*right_i^{scale} ...)$. This is
    /// required step during GIPA recursion.
    pub fn compress(&self, right: &Self, scale: &G::Scalar) -> Result<Self, SynthesisError> {
        let left = self;
        if left.a.len() != right.a.len() {
            return Err(SynthesisError::IncompatibleLengthVector(
                "compressing commitment key".to_string(),
            ));
        }
        let (a, b): (Vec<G>, Vec<G>) = left
            .a
            .par_iter()
            .zip(left.b.par_iter())
            .zip(right.a.par_iter())
            .zip(right.b.par_iter())
            .map(|(((left_a, left_b), right_a), right_b)| {
                let mut ra = right_a.to_curve() * scale;
                let mut rb = right_b.to_curve() * scale;
                ra.add_assign(left_a);
                rb.add_assign(left_b);
                (ra.to_affine(), rb.to_affine())
            })
            .unzip();

        Ok(Self { a, b })
    }

    /// Returns the first values in the vector of v1 and v2 (respectively
    /// w1 and w2). When commitment key is of size one, it's a proxy to get the
    /// final values.
    pub fn first(&self) -> (G, G) {
        (self.a[0], self.b[0])
    }
}

/// Both commitment outputs a pair of $F_q^k$ element.
pub type Output<E> = (<E as Engine>::Gt, <E as Engine>::Gt);

/// Commits to a single vector of G1 elements in the following way:
/// $T = \prod_{i=0}^n e(A_i, v_{1,i})$
/// $U = \prod_{i=0}^n e(A_i, v_{2,i})$
/// Output is $(T,U)$
pub fn single_g1<E>(vkey: &VKey<E>, a_vec: &[E::G1Affine]) -> Result<Output<E>, SynthesisError>
where
    E: MultiMillerLoop,
{
    try_par! {
        let a = inner_product::pairing::<E>(a_vec, &vkey.a),
        let b = inner_product::pairing::<E>(a_vec, &vkey.b)
    };
    Ok((a, b))
}

/// Commits to a tuple of G1 vector and G2 vector in the following way:
/// $T = \prod_{i=0}^n e(A_i, v_{1,i})e(B_i,w_{1,i})$
/// $U = \prod_{i=0}^n e(A_i, v_{2,i})e(B_i,w_{2,i})$
/// Output is $(T,U)$
pub fn pair<E>(
    vkey: &VKey<E>,
    wkey: &WKey<E>,
    a: &[E::G1Affine],
    b: &[E::G2Affine],
) -> Result<Output<E>, SynthesisError>
where
    E: MultiMillerLoop,
{
    try_par! {
        // (A * v)
        let t1 = inner_product::pairing::<E>(a, &vkey.a),
        // (w * B)
        let t2 = inner_product::pairing::<E>(&wkey.a, b),
        let u1 = inner_product::pairing::<E>(a, &vkey.b),
        let u2 = inner_product::pairing::<E>(&wkey.b, b)
    };

    // (A * v)(w * B)
    Ok((t1 + t2, u1 + u2))
}

#[allow(clippy::many_single_char_names)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::groth16::aggregate::structured_generators_scalar_power;
    use blstrs::{Bls12, G1Projective, G2Projective, Scalar as Fr};
    use ff::Field;
    use group::Group;
    use rand_core::SeedableRng;

    #[test]
    fn test_commit_single() {
        let n = 6;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let h = G2Projective::generator();
        let u = Fr::random(&mut rng);
        let v = Fr::random(&mut rng);
        let v1 = structured_generators_scalar_power(n, &h, &u);
        let v2 = structured_generators_scalar_power(n, &h, &v);
        let vkey = VKey::<Bls12> { a: v1, b: v2 };
        let a = (0..n)
            .map(|_| G1Projective::random(&mut rng).to_affine())
            .collect::<Vec<_>>();
        let c1 = single_g1::<Bls12>(&vkey, &a).unwrap();
        let c2 = single_g1::<Bls12>(&vkey, &a).unwrap();
        assert_eq!(c1, c2);
        let b = (0..n)
            .map(|_| G1Projective::random(&mut rng).to_affine())
            .collect::<Vec<_>>();
        let c3 = single_g1::<Bls12>(&vkey, &b).unwrap();
        assert!(c1 != c3);
    }

    #[test]
    fn test_commit_pair() {
        let n = 6;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let h = G2Projective::generator();
        let g = G1Projective::generator();
        let u = Fr::random(&mut rng);
        let v = Fr::random(&mut rng);
        let v1 = structured_generators_scalar_power(n, &h, &u);
        let v2 = structured_generators_scalar_power(n, &h, &v);
        let w1 = structured_generators_scalar_power(2 * n, &g, &u);
        let w2 = structured_generators_scalar_power(2 * n, &g, &v);

        let vkey = VKey::<Bls12> { a: v1, b: v2 };
        let wkey = WKey::<Bls12> {
            a: w1[n..].to_vec(),
            b: w2[n..].to_vec(),
        };
        let a = (0..n)
            .map(|_| G1Projective::random(&mut rng).to_affine())
            .collect::<Vec<_>>();
        let b = (0..n)
            .map(|_| G2Projective::random(&mut rng).to_affine())
            .collect::<Vec<_>>();
        let c1 = pair::<Bls12>(&vkey, &wkey, &a, &b).unwrap();
        let c2 = pair::<Bls12>(&vkey, &wkey, &a, &b).unwrap();
        assert_eq!(c1, c2);
        pair::<Bls12>(&vkey, &wkey, &a[1..2], &b).expect_err("this should have failed");
    }
}
