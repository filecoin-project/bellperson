use super::msm;
use crate::bls::Engine;
use crate::groth16::aggregate::commit::*;
use crate::groth16::multiscalar::{precompute_fixed_window, MultiscalarPrecompOwned, WINDOW_SIZE};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective, EncodedPoint};
use memmap::Mmap;
use rayon::prelude::*;
use sha2::Sha256;
use std::convert::TryFrom;
use std::io::{self, Error, ErrorKind, Read, Write};
use std::mem::size_of;

/// It contains the maximum number of raw elements of the SRS needed to aggregate and verify
/// Groth16 proofs. One can derive specialized prover and verifier key for _specific_ size of
/// aggregations by calling `srs.specialize(n)`. The specialized prover key also contains
/// precomputed tables that drastically increase prover's performance.
/// This GenericSRS is usually formed from the transcript of two distinct power of taus ceremony
/// ,in other words from two distinct Groth16 CRS.
/// See [there](https://github.com/nikkolasg/taupipp) a way on how to generate this GenesisSRS.
#[derive(Clone, Debug)]
pub struct GenericSRS<E: Engine> {
    /// $\{g^a^i\}_{i=0}^{N}$ where N is the smallest size of the two Groth16 CRS.
    pub g_alpha_powers: Vec<E::G1Affine>,
    /// $\{h^a^i\}_{i=0}^{N}$ where N is the smallest size of the two Groth16 CRS.
    pub h_alpha_powers: Vec<E::G2Affine>,
    /// $\{g^b^i\}_{i=n}^{N}$ where N is the smallest size of the two Groth16 CRS.
    pub g_beta_powers: Vec<E::G1Affine>,
    /// $\{h^b^i\}_{i=0}^{N}$ where N is the smallest size of the two Groth16 CRS.
    pub h_beta_powers: Vec<E::G2Affine>,
}

/// ProverSRS is the specialized SRS version for the prover for a specific number of proofs to
/// aggregate. It contains as well the commitment keys for this specific size.
/// Note the size must be a power of two for the moment - if it is not, padding must be
/// applied.
#[derive(Clone, Debug)]
pub struct ProverSRS<E: Engine> {
    /// number of proofs to aggregate
    pub n: usize,
    /// $\{g^a^i\}_{i=n+1}^{2n}$ where n is the number of proofs to be aggregated table starts at
    /// i=n+1 since base is offset with commitment keys. Specially, during the KZG opening proof,
    /// we need the vector of the SRS for g to start at $g^{a^{n+1}}$ because the commitment key
    /// $w$ starts at the same power.
    pub g_alpha_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    /// $\{h^a^i\}_{i=1}^{n}$
    pub h_alpha_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
    /// $\{g^b^i\}_{i=n+1}^{2n}$
    pub g_beta_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    /// $\{h^b^i\}_{i=1}^{n}$
    pub h_beta_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
    /// commitment key using in MIPP and TIPP
    pub vkey: VKey<E>,
    /// commitment key using in TIPP
    pub wkey: WKey<E>,
}

/// Contains the necessary elements to verify an aggregated Groth16 proof; it is of fixed size
/// regardless of the number of proofs aggregated. However, a verifier SRS will be determined by
/// the number of proofs being aggregated.
#[derive(Clone, Debug)]
pub struct VerifierSRS<E: Engine> {
    pub n: usize,
    pub g: E::G1,
    pub h: E::G2,
    pub g_alpha: E::G1,
    pub g_beta: E::G1,
    pub h_alpha: E::G2,
    pub h_beta: E::G2,
    /// equals to $g^{alpha^{n}}$
    pub g_alpha_n1: E::G1,
    /// equals to $g^{beta^{n}}$
    pub g_beta_n1: E::G1,
}

impl<E: Engine> PartialEq for GenericSRS<E> {
    fn eq(&self, other: &Self) -> bool {
        self.g_alpha_powers == other.g_alpha_powers
            && self.g_beta_powers == other.g_beta_powers
            && self.h_alpha_powers == other.h_alpha_powers
            && self.h_beta_powers == other.h_beta_powers
    }
}

impl<E: Engine> PartialEq for VerifierSRS<E> {
    fn eq(&self, other: &Self) -> bool {
        self.g == other.g
            && self.h == other.h
            && self.g_alpha == other.g_alpha
            && self.g_beta == other.g_beta
            && self.h_alpha == other.h_alpha
            && self.h_beta == other.h_beta
            && self.g_alpha_n1 == other.g_alpha_n1
            && self.g_beta_n1 == other.g_beta_n1
    }
}

impl<E: Engine> ProverSRS<E> {
    /// Returns true if commitment keys have the exact required length.
    /// It is necessary for the IPP scheme to work that commitment
    /// key have the exact same number of arguments as the number of proofs to
    /// aggregate.
    pub fn has_correct_len(&self, n: usize) -> bool {
        self.vkey.has_correct_len(n) && self.wkey.has_correct_len(n)
    }
}

impl<E: Engine> GenericSRS<E> {
    /// specializes returns the prover and verifier SRS for a specific number of
    /// proofs to aggregate. The number of proofs MUST BE a power of two, it
    /// panics otherwise. The number of proofs must be inferior to half of the
    /// size of the generic srs otherwise it panics.
    pub fn specialize(&self, num_proofs: usize) -> (ProverSRS<E>, VerifierSRS<E>) {
        assert!(num_proofs.is_power_of_two());
        let tn = 2 * num_proofs; // size of the CRS we need
        assert!(self.g_alpha_powers.len() >= tn);
        assert!(self.h_alpha_powers.len() >= tn);
        assert!(self.g_beta_powers.len() >= tn);
        assert!(self.h_beta_powers.len() >= tn);
        let n = num_proofs;
        // g^n -> g^{n-1}
        let g_low = n;
        let g_up = g_low + n;
        let h_low = 0;
        let h_up = h_low + n;
        let g_alpha_powers_table =
            precompute_fixed_window(&self.g_alpha_powers[g_low..g_up], WINDOW_SIZE);
        let g_beta_powers_table =
            precompute_fixed_window(&self.g_beta_powers[g_low..g_up], WINDOW_SIZE);
        let h_alpha_powers_table =
            precompute_fixed_window(&self.h_alpha_powers[h_low..h_up], WINDOW_SIZE);
        let h_beta_powers_table =
            precompute_fixed_window(&self.h_beta_powers[h_low..h_up], WINDOW_SIZE);
        let v1 = self.h_alpha_powers[h_low..h_up].to_vec();
        let v2 = self.h_beta_powers[h_low..h_up].to_vec();
        let vkey = VKey::<E> { a: v1, b: v2 };
        assert!(vkey.has_correct_len(n));
        let w1 = self.g_alpha_powers[g_low..g_up].to_vec();
        let w2 = self.g_beta_powers[g_low..g_up].to_vec();
        // needed by the verifier to check KZG opening with a shifted base point for
        // the w commitment
        let g_alpha_n1 = w1[0].into_projective();
        let g_beta_n1 = w2[0].into_projective();
        let wkey = WKey::<E> { a: w1, b: w2 };
        assert!(wkey.has_correct_len(n));
        let pk = ProverSRS::<E> {
            g_alpha_powers_table,
            g_beta_powers_table,
            h_alpha_powers_table,
            h_beta_powers_table,
            vkey,
            wkey,
            n,
        };
        let vk = VerifierSRS::<E> {
            n: n,
            g: self.g_alpha_powers[0].into_projective(),
            h: self.h_alpha_powers[0].into_projective(),
            g_alpha: self.g_alpha_powers[1].into_projective(),
            g_beta: self.g_beta_powers[1].into_projective(),
            h_alpha: self.h_alpha_powers[1].into_projective(),
            h_beta: self.h_beta_powers[1].into_projective(),
            g_alpha_n1,
            g_beta_n1,
        };
        (pk, vk)
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_vec(writer, &self.g_alpha_powers)?;
        write_vec(writer, &self.g_beta_powers)?;
        write_vec(writer, &self.h_alpha_powers)?;
        write_vec(writer, &self.h_beta_powers)?;
        Ok(())
    }

    /// Returns the hash over all powers of this generic srs.
    pub fn hash(&self) -> Vec<u8> {
        let mut v = Vec::new();
        self.write(&mut v).expect("failed to compute hash");
        Sha256::digest(&v).to_vec()
    }

    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let g_alpha_powers = read_vec(reader)?;
        let g_beta_powers = read_vec(reader)?;
        let h_alpha_powers = read_vec(reader)?;
        let h_beta_powers = read_vec(reader)?;
        Ok(Self {
            g_alpha_powers,
            g_beta_powers,
            h_alpha_powers,
            h_beta_powers,
        })
    }

    pub fn read_mmap(reader: &Mmap) -> io::Result<Self> {
        fn read_length(mmap: &Mmap, offset: &mut usize) -> Result<usize, std::io::Error> {
            let u32_len = size_of::<u32>();
            let mut raw_len = &mmap[*offset..*offset + u32_len];
            *offset += u32_len;

            match raw_len.read_u32::<BigEndian>() {
                Ok(len) => Ok(len as usize),
                Err(err) => Err(err),
            }
        }

        fn mmap_read_vec<G: CurveAffine>(mmap: &Mmap, offset: &mut usize) -> io::Result<Vec<G>> {
            let point_len = size_of::<G::Compressed>();
            let vec_len = read_length(mmap, offset)?;
            let vec: Vec<G> = (0..vec_len)
                .into_par_iter()
                .map(|i| {
                    let data_start = *offset + (i * point_len);
                    let data_end = data_start + point_len;
                    let ptr = &mmap[data_start..data_end];

                    // Safety: this operation is safe because it's a read on
                    // a buffer that's already allocated and being iterated on.
                    let g1_repr: G::Compressed =
                        unsafe { *(ptr as *const [u8] as *const G::Compressed) };
                    g1_repr
                        .into_affine()
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                        .and_then(|s| Ok(s))
                })
                .collect::<io::Result<Vec<_>>>()?;
            *offset += vec_len * point_len;
            Ok(vec)
        }

        let mut offset: usize = 0;
        let g_alpha_powers = mmap_read_vec::<E::G1Affine>(&reader, &mut offset)?;
        let g_beta_powers = mmap_read_vec::<E::G1Affine>(&reader, &mut offset)?;
        let h_alpha_powers = mmap_read_vec::<E::G2Affine>(&reader, &mut offset)?;
        let h_beta_powers = mmap_read_vec::<E::G2Affine>(&reader, &mut offset)?;
        Ok(Self {
            g_alpha_powers,
            g_beta_powers,
            h_alpha_powers,
            h_beta_powers,
        })
    }
}

pub fn setup_fake_srs<E: Engine, R: rand::RngCore>(rng: &mut R, size: usize) -> GenericSRS<E> {
    let alpha = E::Fr::random(rng);
    let beta = E::Fr::random(rng);
    let g = E::G1::one();
    let h = E::G2::one();

    let mut g_alpha_powers = Vec::new();
    let mut g_beta_powers = Vec::new();
    let mut h_alpha_powers = Vec::new();
    let mut h_beta_powers = Vec::new();
    rayon::scope(|s| {
        let alpha = &alpha;
        let h = &h;
        let g = &g;
        let beta = &beta;
        let g_alpha_powers = &mut g_alpha_powers;
        s.spawn(move |_| {
            *g_alpha_powers = structured_generators_scalar_power(2 * size, g, alpha);
        });
        let g_beta_powers = &mut g_beta_powers;
        s.spawn(move |_| {
            *g_beta_powers = structured_generators_scalar_power(2 * size, g, beta);
        });

        let h_alpha_powers = &mut h_alpha_powers;
        s.spawn(move |_| {
            *h_alpha_powers = structured_generators_scalar_power(2 * size, h, alpha);
        });

        let h_beta_powers = &mut h_beta_powers;
        s.spawn(move |_| {
            *h_beta_powers = structured_generators_scalar_power(2 * size, h, beta);
        });
    });

    debug_assert!(h_alpha_powers[0] == E::G2::one().into_affine());
    debug_assert!(h_beta_powers[0] == E::G2::one().into_affine());
    debug_assert!(g_alpha_powers[0] == E::G1::one().into_affine());
    debug_assert!(g_beta_powers[0] == E::G1::one().into_affine());
    GenericSRS {
        g_alpha_powers,
        g_beta_powers,
        h_alpha_powers,
        h_beta_powers,
    }
}

pub(crate) fn structured_generators_scalar_power<G: CurveProjective>(
    num: usize,
    g: &G,
    s: &G::Scalar,
) -> Vec<G::Affine> {
    assert!(num > 0);
    let mut powers_of_scalar = Vec::with_capacity(num);
    let mut pow_s = G::Scalar::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s.mul_assign(s);
    }

    let window_size = msm::fixed_base::get_mul_window_size(num);
    let scalar_bits = G::Scalar::NUM_BITS as usize;
    let g_table = msm::fixed_base::get_window_table(scalar_bits, window_size, g.clone());
    let powers_of_g = msm::fixed_base::multi_scalar_mul::<G>(
        scalar_bits,
        window_size,
        &g_table,
        &powers_of_scalar[..],
    );
    powers_of_g.into_iter().map(|v| v.into_affine()).collect()
}

fn write_vec<G: CurveAffine, W: Write>(w: &mut W, v: &[G]) -> io::Result<()> {
    w.write_u32::<BigEndian>(u32::try_from(v.len()).map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("invalid vector length > u32: {}", v.len()),
        )
    })?)?;
    for p in v {
        write_point(w, p)?;
    }
    Ok(())
}

fn write_point<G: CurveAffine, W: Write>(w: &mut W, p: &G) -> io::Result<()> {
    w.write_all(p.into_compressed().as_ref())?;
    Ok(())
}

fn read_vec<G: CurveAffine, R: Read>(r: &mut R) -> io::Result<Vec<G>> {
    let vector_len = r.read_u32::<BigEndian>()? as usize;
    let mut data = vec![G::Compressed::empty(); vector_len];
    for encoded in &mut data {
        r.read_exact(encoded.as_mut())?;
    }
    Ok(data
        .par_iter()
        .map(|enc| {
            enc.into_affine()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                .and_then(|s| Ok(s))
        })
        .collect::<io::Result<Vec<_>>>()?)
}
