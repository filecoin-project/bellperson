use std::io::{Read, Write};

use blstrs::Compress;
use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Curve, GroupEncoding};
use pairing::{Engine, MultiMillerLoop};
use serde::{Deserialize, Serialize};

use crate::groth16::aggregate::{commit, srs};
use crate::SynthesisError;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AggregateProofAndInstance<E: Engine>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    pub num_inputs: u32,
    #[serde(bound(
        serialize = "AggregateProof<E>: Serialize",
        deserialize = "AggregateProof<E>: Deserialize<'de>",
    ))]
    pub pi_agg: AggregateProof<E>,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::G1: Serialize",
        deserialize = "<E as pairing::Engine>::G1: Deserialize<'de>",
    ))]
    pub com_f: Vec<E::G1>,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::G1: Serialize",
        deserialize = "<E as pairing::Engine>::G1: Deserialize<'de>",
    ))]
    pub com_w0: Vec<E::G1>,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::G1: Serialize",
        deserialize = "<E as pairing::Engine>::G1: Deserialize<'de>",
    ))]
    pub com_wd: Vec<E::G1>,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::Fr: Serialize",
        deserialize = "<E as pairing::Engine>::Fr: Deserialize<'de>",
    ))]
    pub f_eval: Vec<E::Fr>,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::G1: Serialize",
        deserialize = "<E as pairing::Engine>::G1: Deserialize<'de>",
    ))]
    pub f_eval_proof: Vec<E::G1>,
}

impl<E> PartialEq for AggregateProofAndInstance<E>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    fn eq(&self, other: &Self) -> bool {
        self.pi_agg == other.pi_agg
            && self.com_f == other.com_f
            && self.com_w0 == other.com_w0
            && self.com_wd == other.com_wd
            && self.f_eval == other.f_eval
            && self.f_eval_proof == other.f_eval_proof
    }
}

/// AggregateProof contains all elements to verify n aggregated Groth16 proofs
/// using inner pairing product arguments. This proof can be created by any
/// party in possession of valid Groth16 proofs.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AggregateProof<E>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    /// commitment to A and B using the pair commitment scheme needed to verify
    /// TIPP relation.
    #[serde(bound(
        serialize = "<E as pairing::Engine>::Gt: Serialize",
        deserialize = "<E as pairing::Engine>::Gt: Deserialize<'de>",
    ))]
    pub com_ab: commit::Output<E>,
    /// commit to C separate since we use it only in MIPP
    #[serde(bound(
        serialize = "<E as pairing::Engine>::Gt: Serialize",
        deserialize = "<E as pairing::Engine>::Gt: Deserialize<'de>",
    ))]
    pub com_c: commit::Output<E>,
    /// $A^r * B = Z$ is the left value on the aggregated Groth16 equation
    #[serde(bound(
        serialize = "<E as pairing::Engine>::Gt: Serialize",
        deserialize = "<E as pairing::Engine>::Gt: Deserialize<'de>",
    ))]
    pub ip_ab: <E as Engine>::Gt,
    /// $C^r$ is used on the right side of the aggregated Groth16 equation
    pub agg_c: E::G1,
    #[serde(bound(
        serialize = "TippMippProof<E>: Serialize",
        deserialize = "TippMippProof<E>: Deserialize<'de>",
    ))]
    pub tmipp: TippMippProof<E>,
}

impl<E> PartialEq for AggregateProof<E>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    fn eq(&self, other: &Self) -> bool {
        self.com_ab == other.com_ab
            && self.com_c == other.com_c
            && self.ip_ab == other.ip_ab
            && self.agg_c == other.agg_c
            && self.tmipp == other.tmipp
    }
}

impl<E> AggregateProof<E>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    /// Performs some high level checks on the length of vectors and others to
    /// make sure all items in the proofs are consistent with each other.
    pub fn parsing_check(&self) -> Result<(), SynthesisError> {
        let gipa = &self.tmipp.gipa;
        // 1. Check length of the proofs
        if gipa.nproofs < 2 || gipa.nproofs as usize > srs::MAX_SRS_SIZE {
            return Err(SynthesisError::MalformedProofs(
                "invalid nproofs field".to_string(),
            ));
        }
        // 2. Check if it's a power of two
        if !gipa.nproofs.is_power_of_two() {
            return Err(SynthesisError::MalformedProofs(
                "number of proofs is not a power of two".to_string(),
            ));
        }
        // 3. Check all vectors are of the same length and of the correct length
        let ref_len = gipa.comms_ab.len();
        let good_len = ref_len == (gipa.nproofs as f32).log2().ceil() as usize;
        if !good_len {
            return Err(SynthesisError::MalformedProofs(
                "proof vectors have not indicated size".to_string(),
            ));
        }

        let all_same = ref_len == gipa.comms_c.len()
            && ref_len == gipa.z_ab.len()
            && ref_len == gipa.z_c.len();
        if !all_same {
            return Err(SynthesisError::MalformedProofs(
                "proofs vectors don't have the same size".to_string(),
            ));
        }
        Ok(())
    }
    /// Writes the agggregated proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // com_ab
        self.com_ab.0.write_compressed(&mut out)?;
        self.com_ab.1.write_compressed(&mut out)?;

        // com_c
        self.com_c.0.write_compressed(&mut out)?;
        self.com_c.1.write_compressed(&mut out)?;

        // ip_ab
        self.ip_ab.write_compressed(&mut out)?;

        // agg_c
        let agg_c = self.agg_c.to_affine().to_bytes();
        out.write_all(agg_c.as_ref())?;

        // tmpip
        self.tmipp.write(&mut out)?;

        Ok(())
    }

    /// Returns the number of bytes this proof is serialized to.
    pub fn serialized_len(&self) -> usize {
        // TODO: calculate
        let mut out = Vec::new();
        self.write(&mut out).unwrap();

        out.len()
    }

    pub fn read(mut source: impl Read) -> std::io::Result<Self> {
        let com_ab = (
            <<E as Engine>::Gt as Compress>::read_compressed(&mut source)?,
            <<E as Engine>::Gt as Compress>::read_compressed(&mut source)?,
        );

        let com_c = (
            <<E as Engine>::Gt as Compress>::read_compressed(&mut source)?,
            <<E as Engine>::Gt as Compress>::read_compressed(&mut source)?,
        );

        let ip_ab = <<E as Engine>::Gt as Compress>::read_compressed(&mut source)?;
        let agg_c = read_affine::<E::G1Affine, _>(&mut source)?.to_curve();

        let tmipp = TippMippProof::read(&mut source)?;

        Ok(AggregateProof {
            com_ab,
            com_c,
            ip_ab,
            agg_c,
            tmipp,
        })
    }
}

/// It contains all elements derived in the GIPA loop for both TIPP and MIPP at
/// the same time.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GipaProof<E>
where
    E: MultiMillerLoop,
{
    pub nproofs: u32,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::Gt: Serialize",
        deserialize = "<E as pairing::Engine>::Gt: Deserialize<'de>",
    ))]
    pub comms_ab: Vec<(commit::Output<E>, commit::Output<E>)>,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::Gt: Serialize",
        deserialize = "<E as pairing::Engine>::Gt: Deserialize<'de>",
    ))]
    pub comms_c: Vec<(commit::Output<E>, commit::Output<E>)>,
    #[serde(bound(
        serialize = "<E as pairing::Engine>::Gt: Serialize",
        deserialize = "<E as pairing::Engine>::Gt: Deserialize<'de>",
    ))]
    pub z_ab: Vec<(<E as Engine>::Gt, <E as Engine>::Gt)>,
    #[serde(bound(
        serialize = "E::G1: Serialize",
        deserialize = "E::G1: Deserialize<'de>",
    ))]
    pub z_c: Vec<(E::G1, E::G1)>,
    #[serde(bound(
        serialize = "E::G1Affine: Serialize",
        deserialize = "E::G1Affine: Deserialize<'de>",
    ))]
    pub final_a: E::G1Affine,
    #[serde(bound(
        serialize = "E::G2Affine: Serialize",
        deserialize = "E::G2Affine: Deserialize<'de>",
    ))]
    pub final_b: E::G2Affine,
    #[serde(bound(
        serialize = "E::G1Affine: Serialize",
        deserialize = "E::G1Affine: Deserialize<'de>",
    ))]
    pub final_c: E::G1Affine,
    /// final commitment keys $v$ and $w$ - there is only one element at the
    /// end for v1 and v2 hence it's a tuple.
    #[serde(bound(
        serialize = "E::G2Affine: Serialize",
        deserialize = "E::G2Affine: Deserialize<'de>",
    ))]
    pub final_vkey: (E::G2Affine, E::G2Affine),
    #[serde(bound(
        serialize = "E::G1Affine: Serialize",
        deserialize = "E::G1Affine: Deserialize<'de>",
    ))]
    pub final_wkey: (E::G1Affine, E::G1Affine),
}

impl<E> PartialEq for GipaProof<E>
where
    E: MultiMillerLoop,
{
    fn eq(&self, other: &Self) -> bool {
        self.nproofs == other.nproofs
            && self.comms_ab == other.comms_ab
            && self.comms_c == other.comms_c
            && self.z_ab == other.z_ab
            && self.z_c == other.z_c
            && self.final_a == other.final_a
            && self.final_b == other.final_b
            && self.final_c == other.final_c
            && self.final_vkey == other.final_vkey
            && self.final_wkey == other.final_wkey
    }
}

fn log_proofs(nproofs: usize) -> usize {
    (nproofs as f32).log2().ceil() as usize
}

impl<E> GipaProof<E>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    /// Writes the  proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // number of proofs
        out.write_all(&self.nproofs.to_le_bytes()[..])?;

        let log_proofs = log_proofs(self.nproofs as usize);
        assert_eq!(self.comms_ab.len(), log_proofs);
        // comms_ab
        for (x, y) in &self.comms_ab {
            x.0.write_compressed(&mut out)?;
            x.1.write_compressed(&mut out)?;
            y.0.write_compressed(&mut out)?;
            y.1.write_compressed(&mut out)?;
        }

        assert_eq!(self.comms_c.len(), log_proofs);
        // comms_c
        for (x, y) in &self.comms_c {
            x.0.write_compressed(&mut out)?;
            x.1.write_compressed(&mut out)?;
            y.0.write_compressed(&mut out)?;
            y.1.write_compressed(&mut out)?;
        }

        assert_eq!(self.z_ab.len(), log_proofs);
        // z_ab
        for (x, y) in &self.z_ab {
            x.write_compressed(&mut out)?;
            y.write_compressed(&mut out)?;
        }

        assert_eq!(self.z_c.len(), log_proofs);
        // z_c
        for (x, y) in &self.z_c {
            out.write_all(x.to_affine().to_bytes().as_ref())?;
            out.write_all(y.to_affine().to_bytes().as_ref())?;
        }

        // final_a
        out.write_all(self.final_a.to_bytes().as_ref())?;

        // final_b
        out.write_all(self.final_b.to_bytes().as_ref())?;

        // final_c
        out.write_all(self.final_c.to_bytes().as_ref())?;

        // final_vkey
        out.write_all(self.final_vkey.0.to_bytes().as_ref())?;
        out.write_all(self.final_vkey.1.to_bytes().as_ref())?;

        // final_wkey
        out.write_all(self.final_wkey.0.to_bytes().as_ref())?;
        out.write_all(self.final_wkey.1.to_bytes().as_ref())?;

        Ok(())
    }

    fn read(mut source: impl Read) -> std::io::Result<Self> {
        let mut buffer = 0u32.to_le_bytes();
        source.read_exact(&mut buffer)?;
        let nproofs = u32::from_le_bytes(buffer);
        if nproofs < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "number of proofs is invalid",
            ));
        }

        let log_proofs = log_proofs(nproofs as usize);

        fn read_output<E, R>(mut source: R) -> std::io::Result<commit::Output<E>>
        where
            E: MultiMillerLoop,
            <E as Engine>::Gt: Compress,
            R: Read,
        {
            let a = <<E as Engine>::Gt as Compress>::read_compressed(&mut source)?;
            let b = <<E as Engine>::Gt as Compress>::read_compressed(&mut source)?;
            Ok((a, b))
        }

        let mut comms_ab = Vec::with_capacity(log_proofs);
        for _ in 0..log_proofs {
            comms_ab.push((
                read_output::<E, _>(&mut source)?,
                read_output::<E, _>(&mut source)?,
            ));
        }

        let mut comms_c = Vec::with_capacity(log_proofs);
        for _ in 0..log_proofs {
            comms_c.push((
                read_output::<E, _>(&mut source)?,
                read_output::<E, _>(&mut source)?,
            ));
        }

        let mut z_ab = Vec::with_capacity(log_proofs);
        for _ in 0..log_proofs {
            z_ab.push(read_output::<E, _>(&mut source)?);
        }

        let mut z_c = Vec::with_capacity(log_proofs);
        for _ in 0..log_proofs {
            z_c.push((
                read_affine::<E::G1Affine, _>(&mut source)?.to_curve(),
                read_affine::<E::G1Affine, _>(&mut source)?.to_curve(),
            ));
        }

        let final_a = read_affine(&mut source)?;
        let final_b = read_affine(&mut source)?;
        let final_c = read_affine(&mut source)?;

        let final_vkey = (read_affine(&mut source)?, read_affine(&mut source)?);
        let final_wkey = (read_affine(&mut source)?, read_affine(&mut source)?);

        Ok(GipaProof {
            nproofs,
            comms_ab,
            comms_c,
            z_ab,
            z_c,
            final_a,
            final_b,
            final_c,
            final_vkey,
            final_wkey,
        })
    }
}

/// It contains the GIPA recursive elements as well as the KZG openings for v
/// and w
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TippMippProof<E>
where
    E: MultiMillerLoop,
{
    #[serde(bound(
        serialize = "GipaProof<E>: Serialize",
        deserialize = "GipaProof<E>: Deserialize<'de>",
    ))]
    pub gipa: GipaProof<E>,
    #[serde(bound(
        serialize = "E::G2Affine: Serialize",
        deserialize = "E::G2Affine: Deserialize<'de>",
    ))]
    pub vkey_opening: KZGOpening<E::G2Affine>,
    #[serde(bound(
        serialize = "E::G1Affine: Serialize",
        deserialize = "E::G1Affine: Deserialize<'de>",
    ))]
    pub wkey_opening: KZGOpening<E::G1Affine>,
}

impl<E> PartialEq for TippMippProof<E>
where
    E: MultiMillerLoop,
{
    fn eq(&self, other: &Self) -> bool {
        self.gipa == other.gipa
            && self.vkey_opening == other.vkey_opening
            && self.wkey_opening == other.wkey_opening
    }
}

impl<E> TippMippProof<E>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    /// Writes the  proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // gipa
        self.gipa.write(&mut out)?;

        // vkey_opening
        let x0 = self.vkey_opening.0.to_bytes();
        let x1 = self.vkey_opening.1.to_bytes();

        out.write_all(x0.as_ref())?;
        out.write_all(x1.as_ref())?;

        // wkey_opening
        let x0 = self.wkey_opening.0.to_bytes();
        let x1 = self.wkey_opening.1.to_bytes();

        out.write_all(x0.as_ref())?;
        out.write_all(x1.as_ref())?;

        Ok(())
    }

    fn read(mut source: impl Read) -> std::io::Result<Self> {
        let gipa = GipaProof::read(&mut source)?;
        let vkey_opening = (read_affine(&mut source)?, read_affine(&mut source)?);

        let wkey_opening = (read_affine(&mut source)?, read_affine(&mut source)?);

        Ok(TippMippProof {
            gipa,
            vkey_opening,
            wkey_opening,
        })
    }
}

impl<E> AggregateProofAndInstance<E>
where
    E: MultiMillerLoop,
    <E as Engine>::Gt: Compress,
{
    pub fn parsing_check(&self) -> Result<(), SynthesisError> {
        self.pi_agg.parsing_check()?;
        let n = (self.num_inputs / 2) as usize;

        if n < 2 {
            return Err(SynthesisError::MalformedProofs(
                "num_inputs field".to_string(),
            ));
        }

        if self.com_f.len() != n {
            return Err(SynthesisError::MalformedProofs(format!(
                "com_f must be equal to {}",
                n,
            )));
        }

        if self.com_w0.len() != n {
            return Err(SynthesisError::MalformedProofs(format!(
                "com_w0 must be equal to {}",
                n
            )));
        }

        if self.com_wd.len() != n {
            return Err(SynthesisError::MalformedProofs(format!(
                "com_wd must be equal to {}",
                n
            )));
        }

        if self.f_eval.len() != n {
            return Err(SynthesisError::MalformedProofs(format!(
                "f_eval must be equal to {}",
                n
            )));
        }

        if self.f_eval_proof.len() != n {
            return Err(SynthesisError::MalformedProofs(format!(
                "f_eval_proof must be equal to {}",
                n
            )));
        }

        Ok(())
    }

    /// Writes the proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        out.write_all(&self.num_inputs.to_le_bytes())?;

        self.pi_agg.write(&mut out)?;

        for e in &self.com_f {
            out.write_all(e.to_affine().to_bytes().as_ref())?;
        }

        for e in &self.com_w0 {
            out.write_all(e.to_bytes().as_ref())?;
        }

        for e in &self.com_wd {
            out.write_all(e.to_affine().to_bytes().as_ref())?;
        }

        for e in &self.f_eval {
            out.write_all(e.to_repr().as_ref())?;
        }

        for e in &self.f_eval_proof {
            out.write_all(e.to_affine().to_bytes().as_ref())?;
        }

        Ok(())
    }

    pub fn read(mut source: impl Read) -> std::io::Result<Self> {
        let mut buffer = 0u32.to_le_bytes();
        source.read_exact(&mut buffer)?;
        let num_inputs = u32::from_le_bytes(buffer);

        let pi_agg = AggregateProof::read(&mut source)?;

        let n = (num_inputs / 2) as usize;
        let mut com_f = Vec::with_capacity(n);
        for _ in 0..n {
            com_f.push(read_affine::<E::G1Affine, _>(&mut source)?.to_curve());
        }
        let mut com_w0 = Vec::with_capacity(n);
        for _ in 0..n {
            com_w0.push(read_affine::<E::G1Affine, _>(&mut source)?.to_curve());
        }

        let mut com_wd = Vec::with_capacity(n);
        for _ in 0..n {
            com_wd.push(read_affine::<E::G1Affine, _>(&mut source)?.to_curve());
        }

        let mut f_eval = Vec::with_capacity(n);
        let mut buf = <E::Fr as PrimeField>::Repr::default();
        for _ in 0..n {
            source.read_exact(buf.as_mut())?;
            f_eval.push(<E::Fr as PrimeField>::from_repr_vartime(buf).unwrap());
        }

        let mut f_eval_proof = Vec::with_capacity(n);
        for _ in 0..n {
            f_eval_proof.push(read_affine::<E::G1Affine, _>(&mut source)?.to_curve());
        }

        Ok(AggregateProofAndInstance {
            num_inputs,
            pi_agg,
            com_f,
            com_w0,
            com_wd,
            f_eval,
            f_eval_proof,
        })
    }
}

/// KZGOpening represents the KZG opening of a commitment key (which is a tuple
/// given commitment keys are a tuple).
#[allow(clippy::upper_case_acronyms)]
pub type KZGOpening<G> = (G, G);

fn read_affine<G: PrimeCurveAffine, R: std::io::Read>(mut source: R) -> std::io::Result<G> {
    // Read as compressed affine point.
    let mut affine_compressed = <G as GroupEncoding>::Repr::default();
    source.read_exact(affine_compressed.as_mut())?;
    let opt: Option<_> = G::from_bytes(&affine_compressed).into();

    let affine =
        opt.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid point"))?;

    Ok(affine)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ff::Field;
    use group::Group;

    use blstrs::{Bls12, G1Affine, G1Projective, G2Affine, G2Projective, Scalar};

    fn fake_proof() -> AggregateProof<Bls12> {
        // create pairing, as pairing results can be compressed
        let p = G1Projective::generator().to_affine();
        let q = G2Projective::generator().to_affine();
        let a = Bls12::pairing(&p, &q);

        AggregateProof::<Bls12> {
            com_ab: (a, a),
            com_c: (a, a),
            ip_ab: a,
            agg_c: G1Projective::generator(),
            tmipp: TippMippProof::<Bls12> {
                gipa: GipaProof {
                    nproofs: 4,
                    comms_ab: vec![((a, a), (a, a)), ((a, a), (a, a))],
                    comms_c: vec![((a, a), (a, a)), ((a, a), (a, a))],
                    z_ab: vec![(a, a), (a, a)],
                    z_c: vec![
                        (G1Projective::generator(), G1Projective::generator()),
                        (G1Projective::generator(), G1Projective::generator()),
                    ],
                    final_a: G1Affine::generator(),
                    final_b: G2Affine::generator(),
                    final_c: G1Affine::generator(),
                    final_vkey: (G2Affine::generator(), G2Affine::generator()),
                    final_wkey: (G1Affine::generator(), G1Affine::generator()),
                },
                vkey_opening: (G2Affine::generator(), G2Affine::generator()),
                wkey_opening: (G1Affine::generator(), G1Affine::generator()),
            },
        }
    }

    #[test]
    fn test_proof_io() {
        let proof = fake_proof();
        let mut buffer = Vec::new();
        proof.write(&mut buffer).unwrap();
        assert_eq!(buffer.len(), 8_212);

        let out = AggregateProof::<Bls12>::read(std::io::Cursor::new(&buffer)).unwrap();
        assert_eq!(proof, out);

        let ser_proof = bincode::serialize(&proof).unwrap();
        let des_proof: AggregateProof<Bls12> = bincode::deserialize(&ser_proof).unwrap();
        assert_eq!(des_proof, proof);
    }

    #[test]
    fn test_proof_check() {
        let p = G1Projective::generator().to_affine();
        let q = G2Projective::generator().to_affine();
        let a = Bls12::pairing(&p, &q);

        let mut proof = fake_proof();
        proof.parsing_check().expect("proof should be valid");

        let oldn = proof.tmipp.gipa.nproofs;
        proof.tmipp.gipa.nproofs = 14;
        proof.parsing_check().expect_err("proof should be invalid");
        proof.tmipp.gipa.nproofs = oldn;

        proof
            .tmipp
            .gipa
            .comms_ab
            .append(&mut vec![((a, a), (a, a))]);
        proof.parsing_check().expect_err("Proof should be invalid");
    }

    fn fake_proof_instance() -> AggregateProofAndInstance<Bls12> {
        // create pairing, as pairing results can be compressed
        let a = G1Projective::generator();
        let b = G1Projective::generator();
        let c = Scalar::zero();

        AggregateProofAndInstance::<Bls12> {
            num_inputs: 8,
            pi_agg: fake_proof(),
            com_f: vec![a, a, a, a],
            com_w0: vec![a, a, a, a],
            com_wd: vec![a, a, a, a],
            f_eval: vec![c, c, c, c],
            f_eval_proof: vec![b, b, b, b],
        }
    }

    #[test]
    fn test_proof_io_instance() {
        let proof = fake_proof_instance();
        proof.parsing_check().unwrap();
        let mut buffer = Vec::new();
        proof.write(&mut buffer).unwrap();
        assert_eq!(buffer.len(), 9_112);

        let out = AggregateProofAndInstance::<Bls12>::read(std::io::Cursor::new(&buffer)).unwrap();
        assert_eq!(proof, out);

        let ser_proof = bincode::serialize(&proof).unwrap();
        let des_proof: AggregateProofAndInstance<Bls12> = bincode::deserialize(&ser_proof).unwrap();
        assert_eq!(des_proof, proof);
        des_proof.parsing_check().unwrap();
    }
}
