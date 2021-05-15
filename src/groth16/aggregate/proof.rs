use std::io::{Read, Write};

use groupy::{CurveAffine, CurveProjective, EncodedPoint};
use serde::{Deserialize, Serialize};

use crate::bls::{Compress, Engine};
use crate::groth16::aggregate::{commit, srs};
use crate::SynthesisError;

/// AggregateProof contains all elements to verify n aggregated Groth16 proofs
/// using inner pairing product arguments. This proof can be created by any
/// party in possession of valid Groth16 proofs.
#[derive(Serialize, Deserialize, Debug)]
pub struct AggregateProof<E: Engine> {
    /// commitment to A and B using the pair commitment scheme needed to verify
    /// TIPP relation.
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub com_ab: commit::Output<E>,
    /// commit to C separate since we use it only in MIPP
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub com_c: commit::Output<E>,
    /// $A^r * B = Z$ is the left value on the aggregated Groth16 equation
    pub ip_ab: E::Fqk,
    /// $C^r$ is used on the right side of the aggregated Groth16 equation
    pub agg_c: E::G1,
    #[serde(bound(
        serialize = "TippMippProof<E>: Serialize",
        deserialize = "TippMippProof<E>: Deserialize<'de>",
    ))]
    pub tmipp: TippMippProof<E>,
}

impl<E: Engine> PartialEq for AggregateProof<E> {
    fn eq(&self, other: &Self) -> bool {
        self.com_ab == other.com_ab
            && self.com_c == other.com_c
            && self.ip_ab == other.ip_ab
            && self.agg_c == other.agg_c
            && self.tmipp == other.tmipp
    }
}

impl<E: Engine> AggregateProof<E> {
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
        let agg_c = self.agg_c.into_affine().into_compressed();
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
            <E::Fqk as Compress>::read_compressed(&mut source)?,
            <E::Fqk as Compress>::read_compressed(&mut source)?,
        );

        let com_c = (
            <E::Fqk as Compress>::read_compressed(&mut source)?,
            <E::Fqk as Compress>::read_compressed(&mut source)?,
        );

        let ip_ab = <E::Fqk as Compress>::read_compressed(&mut source)?;
        let agg_c = read_affine::<E::G1Affine, _>(&mut source)?.into_projective();

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
#[derive(Serialize, Deserialize, Debug)]
pub struct GipaProof<E: Engine> {
    pub nproofs: u32,
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub comms_ab: Vec<(commit::Output<E>, commit::Output<E>)>,
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub comms_c: Vec<(commit::Output<E>, commit::Output<E>)>,
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub z_ab: Vec<(E::Fqk, E::Fqk)>,
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G1: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G1: Deserialize<'de>",
    ))]
    pub z_c: Vec<(E::G1, E::G1)>,
    pub final_a: E::G1Affine,
    pub final_b: E::G2Affine,
    pub final_c: E::G1Affine,
    /// final commitment keys $v$ and $w$ - there is only one element at the
    /// end for v1 and v2 hence it's a tuple.
    #[serde(bound(
        serialize = "E::G2Affine: Serialize, E::G2Affine: Serialize",
        deserialize = "E::G2Affine: Deserialize<'de>, E::G2Affine: Deserialize<'de>",
    ))]
    pub final_vkey: (E::G2Affine, E::G2Affine),
    #[serde(bound(
        serialize = "E::G1Affine: Serialize, E::G1Affine: Serialize",
        deserialize = "E::G1Affine: Deserialize<'de>, E::G1Affine: Deserialize<'de>",
    ))]
    pub final_wkey: (E::G1Affine, E::G1Affine),
}

impl<E: Engine> PartialEq for GipaProof<E> {
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

impl<E: Engine> GipaProof<E> {
    fn log_proofs(nproofs: usize) -> usize {
        (nproofs as f32).log2().ceil() as usize
    }

    /// Writes the  proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // number of proofs
        out.write_all(&self.nproofs.to_le_bytes()[..])?;

        let log_proofs = Self::log_proofs(self.nproofs as usize);
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
            out.write_all(x.into_affine().into_compressed().as_ref())?;
            out.write_all(y.into_affine().into_compressed().as_ref())?;
        }

        // final_a
        out.write_all(self.final_a.into_compressed().as_ref())?;

        // final_b
        out.write_all(self.final_b.into_compressed().as_ref())?;

        // final_c
        out.write_all(self.final_c.into_compressed().as_ref())?;

        // final_vkey
        out.write_all(self.final_vkey.0.into_compressed().as_ref())?;
        out.write_all(self.final_vkey.1.into_compressed().as_ref())?;

        // final_wkey
        out.write_all(self.final_wkey.0.into_compressed().as_ref())?;
        out.write_all(self.final_wkey.1.into_compressed().as_ref())?;

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

        let log_proofs = Self::log_proofs(nproofs as usize);

        fn read_output<E: Engine, R: Read>(mut source: R) -> std::io::Result<commit::Output<E>> {
            let a = <E::Fqk as Compress>::read_compressed(&mut source)?;
            let b = <E::Fqk as Compress>::read_compressed(&mut source)?;
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
                read_affine::<E::G1Affine, _>(&mut source)?.into_projective(),
                read_affine::<E::G1Affine, _>(&mut source)?.into_projective(),
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
#[derive(Serialize, Deserialize, Debug)]
pub struct TippMippProof<E: Engine> {
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

impl<E: Engine> PartialEq for TippMippProof<E> {
    fn eq(&self, other: &Self) -> bool {
        self.gipa == other.gipa
            && self.vkey_opening == other.vkey_opening
            && self.wkey_opening == other.wkey_opening
    }
}

impl<E: Engine> TippMippProof<E> {
    /// Writes the  proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // gipa
        self.gipa.write(&mut out)?;

        // vkey_opening
        let x0 = self.vkey_opening.0.into_compressed();
        let x1 = self.vkey_opening.1.into_compressed();

        out.write_all(x0.as_ref())?;
        out.write_all(x1.as_ref())?;

        // wkey_opening
        let x0 = self.wkey_opening.0.into_compressed();
        let x1 = self.wkey_opening.1.into_compressed();

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

/// KZGOpening represents the KZG opening of a commitment key (which is a tuple
/// given commitment keys are a tuple).
pub type KZGOpening<G> = (G, G);

fn read_affine<G: CurveAffine, R: std::io::Read>(mut source: R) -> std::io::Result<G> {
    let mut affine_compressed = <G as CurveAffine>::Compressed::empty();
    source.read_exact(affine_compressed.as_mut())?;
    let affine = affine_compressed
        .into_affine()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?;

    Ok(affine)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::bls::{Bls12, G1Affine, G1Projective, G2Affine, G2Projective};

    fn fake_proof() -> AggregateProof<Bls12> {
        // create pairing, as pairing results can be compressed
        let p = G1Projective::one().into_affine();
        let q = G2Projective::one().into_affine();
        let a = Bls12::pairing(p, q);

        let proof = AggregateProof::<Bls12> {
            com_ab: (a, a),
            com_c: (a, a),
            ip_ab: a,
            agg_c: G1Projective::one(),
            tmipp: TippMippProof::<Bls12> {
                gipa: GipaProof {
                    nproofs: 4,
                    comms_ab: vec![((a, a), (a, a)), ((a, a), (a, a))],
                    comms_c: vec![((a, a), (a, a)), ((a, a), (a, a))],
                    z_ab: vec![(a, a), (a, a)],
                    z_c: vec![
                        (G1Projective::one(), G1Projective::one()),
                        (G1Projective::one(), G1Projective::one()),
                    ],
                    final_a: G1Affine::one(),
                    final_b: G2Affine::one(),
                    final_c: G1Affine::one(),
                    final_vkey: (G2Affine::one(), G2Affine::one()),
                    final_wkey: (G1Affine::one(), G1Affine::one()),
                },
                vkey_opening: (G2Affine::one(), G2Affine::one()),
                wkey_opening: (G1Affine::one(), G1Affine::one()),
            },
        };
        proof
    }

    #[test]
    fn test_proof_io() {
        let proof = fake_proof();
        let mut buffer = Vec::new();
        proof.write(&mut buffer).unwrap();
        assert_eq!(buffer.len(), 8_212);

        let out = AggregateProof::<Bls12>::read(std::io::Cursor::new(&buffer)).unwrap();
        assert_eq!(proof, out);
    }

    #[test]
    fn test_proof_check() {
        let p = G1Projective::one().into_affine();
        let q = G2Projective::one().into_affine();
        let a = Bls12::pairing(p, q);

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
}
