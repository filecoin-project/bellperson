use super::msm;
use crate::groth16::aggregate::commit::*;
use crate::groth16::multiscalar::{precompute_fixed_window, MultiscalarPrecompOwned, WINDOW_SIZE};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use digest::Digest;
use ff::{Field, PrimeField, PrimeFieldBits};
use group::{
    prime::{PrimeCurve, PrimeCurveAffine},
    Curve, Group, GroupEncoding,
};
use memmap::Mmap;
use pairing::Engine;
use rayon::prelude::*;
use sha2::Sha256;
use std::convert::TryFrom;
use std::io::{self, Error, ErrorKind, Read, Write};
use std::mem::size_of;
use std::ops::MulAssign;

/// Maximum size of the generic SRS constructed from Filecoin and Zcash power of
/// taus.
///
/// https://github.com/nikkolasg/taupipp/blob/baca1426266bf39416c45303e35c966d69f4f8b4/src/bin/assemble.rs#L12
pub const MAX_SRS_SIZE: usize = (2 << 19) + 1;

/// It contains the maximum number of raw elements of the SRS needed to aggregate and verify
/// Groth16 proofs. One can derive specialized prover and verifier key for _specific_ size of
/// aggregations by calling `srs.specialize(n)`. The specialized prover key also contains
/// precomputed tables that drastically increase prover's performance.
/// This GenericSRS is usually formed from the transcript of two distinct power of taus ceremony
/// ,in other words from two distinct Groth16 CRS.
/// See [there](https://github.com/nikkolasg/taupipp) a way on how to generate this GenesisSRS.
#[allow(clippy::upper_case_acronyms)]
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
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct ProverSRS<E: Engine> {
    /// number of proofs to aggregate
    pub n: usize,
    /// $\{g^a^i\}_{i=0}^{2n-1}$ where n is the number of proofs to be aggregated
    /// We take all powers instead of only ones from n -> 2n-1 (w commitment key
    /// is formed from these powers) since the prover will create a shifted
    /// polynomial of degree 2n-1 when doing the KZG opening proof.
    pub g_alpha_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    /// $\{h^a^i\}_{i=0}^{n-1}$ - here we don't need to go to 2n-1 since v
    /// commitment key only goes up to n-1 exponent.
    pub h_alpha_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
    /// $\{g^b^i\}_{i=0}^{2n-1}$
    pub g_beta_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    /// $\{h^b^i\}_{i=0}^{n-1}$
    pub h_beta_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
    /// commitment key using in MIPP and TIPP
    pub vkey: VKey<E>,
    /// commitment key using in TIPP
    pub wkey: WKey<E>,
}

/// Contains the necessary elements to verify an aggregated Groth16 proof; it is of fixed size
/// regardless of the number of proofs aggregated. However, a verifier SRS will be determined by
/// the number of proofs being aggregated.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct VerifierSRS<E: Engine> {
    pub n: usize,
    pub g: E::G1,
    pub h: E::G2,
    pub g_alpha: E::G1,
    pub g_beta: E::G1,
    pub h_alpha: E::G2,
    pub h_beta: E::G2,
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

impl<E> GenericSRS<E>
where
    E: Engine,
    <E::G1Affine as GroupEncoding>::Repr: Sync,
    <E::G2Affine as GroupEncoding>::Repr: Sync,
{
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
        // when doing the KZG opening we need _all_ coefficients from 0
        // to 2n-1 because the polynomial is of degree 2n-1.
        let g_low = 0;
        let g_up = tn;
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
        // however, here we only need the "right" shifted bases for the
        // commitment scheme.
        let w1 = self.g_alpha_powers[n..g_up].to_vec();
        let w2 = self.g_beta_powers[n..g_up].to_vec();
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
            n,
            g: self.g_alpha_powers[0].to_curve(),
            h: self.h_alpha_powers[0].to_curve(),
            g_alpha: self.g_alpha_powers[1].to_curve(),
            g_beta: self.g_beta_powers[1].to_curve(),
            h_alpha: self.h_alpha_powers[1].to_curve(),
            h_beta: self.h_beta_powers[1].to_curve(),
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

    pub fn read_mmap(reader: &Mmap, max_len: usize) -> io::Result<Self> {
        fn read_length(mmap: &Mmap, offset: &mut usize) -> Result<usize, std::io::Error> {
            let u32_len = size_of::<u32>();
            let mut raw_len = &mmap[*offset..*offset + u32_len];
            *offset += u32_len;

            match raw_len.read_u32::<BigEndian>() {
                Ok(len) => Ok(len as usize),
                Err(err) => Err(err),
            }
        }

        // The 'max_len' argument allows us to read up to that max
        // (e.g.. 2 << 14), rather then entire vec_len (i.e. 2 << 19)
        fn mmap_read_vec<G: PrimeCurveAffine>(
            mmap: &Mmap,
            offset: &mut usize,
            max_len: usize,
        ) -> io::Result<Vec<G>> {
            let point_len = size_of::<G::Repr>();
            let vec_len = read_length(mmap, offset)?;
            if vec_len > MAX_SRS_SIZE {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid SRS vector length {}", vec_len,),
                ));
            }

            let max_len = if max_len > vec_len { vec_len } else { max_len };

            let vec: Vec<G> = (0..max_len)
                .into_par_iter()
                .map(|i| {
                    let data_start = *offset + (i * point_len);
                    let data_end = data_start + point_len;
                    let ptr = &mmap[data_start..data_end];

                    // Safety: this operation is safe because it's a read on
                    // a buffer that's already allocated and being iterated on.
                    let g1_repr = unsafe { &*(ptr as *const [u8] as *const G::Repr) };
                    let opt: Option<G> = G::from_bytes(&g1_repr).into();
                    opt.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not on curve"))
                })
                .collect::<io::Result<Vec<_>>>()?;
            *offset += vec_len * point_len;
            Ok(vec)
        }

        let mut offset: usize = 0;
        let g_alpha_powers = mmap_read_vec::<E::G1Affine>(&reader, &mut offset, max_len)?;
        let g_beta_powers = mmap_read_vec::<E::G1Affine>(&reader, &mut offset, max_len)?;
        let h_alpha_powers = mmap_read_vec::<E::G2Affine>(&reader, &mut offset, max_len)?;
        let h_beta_powers = mmap_read_vec::<E::G2Affine>(&reader, &mut offset, max_len)?;
        Ok(Self {
            g_alpha_powers,
            g_beta_powers,
            h_alpha_powers,
            h_beta_powers,
        })
    }
}

pub fn setup_fake_srs<E, R>(rng: &mut R, size: usize) -> GenericSRS<E>
where
    E: Engine,
    E::Fr: PrimeFieldBits,
    R: rand_core::RngCore,
{
    let alpha = E::Fr::random(&mut *rng);
    let beta = E::Fr::random(&mut *rng);
    let g = E::G1::generator();
    let h = E::G2::generator();

    let alpha = &alpha;
    let h = &h;
    let g = &g;
    let beta = &beta;
    par! {
        let g_alpha_powers = structured_generators_scalar_power(2 * size, g, alpha),
        let g_beta_powers = structured_generators_scalar_power(2 * size, g, beta),
        let h_alpha_powers = structured_generators_scalar_power(2 * size, h, alpha),
        let h_beta_powers = structured_generators_scalar_power(2 * size, h, beta)
    };

    debug_assert!(h_alpha_powers[0] == E::G2::generator().to_affine());
    debug_assert!(h_beta_powers[0] == E::G2::generator().to_affine());
    debug_assert!(g_alpha_powers[0] == E::G1::generator().to_affine());
    debug_assert!(g_beta_powers[0] == E::G1::generator().to_affine());

    GenericSRS {
        g_alpha_powers,
        g_beta_powers,
        h_alpha_powers,
        h_beta_powers,
    }
}

pub(crate) fn structured_generators_scalar_power<G>(
    num: usize,
    g: &G,
    s: &G::Scalar,
) -> Vec<G::AffineRepr>
where
    G: PrimeCurve,
    G::Scalar: PrimeFieldBits,
    G::AffineRepr: Send,
{
    assert!(num > 0);
    let mut powers_of_scalar = Vec::with_capacity(num);
    let mut pow_s = G::Scalar::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s.mul_assign(s);
    }

    let window_size = msm::fixed_base::get_mul_window_size(num);
    let scalar_bits = G::Scalar::NUM_BITS as usize;
    let g_table = msm::fixed_base::get_window_table(scalar_bits, window_size, *g);
    let powers_of_g = msm::fixed_base::multi_scalar_mul::<G>(
        scalar_bits,
        window_size,
        &g_table,
        &powers_of_scalar[..],
    );
    powers_of_g.into_iter().map(|v| v.to_affine()).collect()
}

fn write_vec<G: PrimeCurveAffine, W: Write>(w: &mut W, v: &[G]) -> io::Result<()> {
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

fn write_point<G: PrimeCurveAffine, W: Write>(w: &mut W, p: &G) -> io::Result<()> {
    w.write_all(p.to_bytes().as_ref())?;
    Ok(())
}

fn read_vec<G, R>(r: &mut R) -> io::Result<Vec<G>>
where
    G: PrimeCurveAffine,
    G::Repr: Sync,
    R: Read,
{
    let vector_len = r.read_u32::<BigEndian>()? as usize;
    if vector_len > MAX_SRS_SIZE {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("invalid SRS vector length {}", vector_len),
        ));
    }

    let data: Vec<G::Repr> = (0..vector_len)
        .map(|_| {
            let mut el = G::Repr::default();
            r.read_exact(el.as_mut())?;
            Ok(el)
        })
        .collect::<Result<_, io::Error>>()?;

    data.par_iter()
        .map(|enc| {
            let opt: Option<G> = G::from_bytes(enc).into();
            opt.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not on curve"))
        })
        .collect::<io::Result<Vec<_>>>()
}

#[cfg(test)]
mod test {
    use super::*;
    use blstrs::Bls12;
    use rand_core::SeedableRng;
    use std::io::Cursor;

    #[test]
    fn test_srs_invalid_length() {
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let size = 8;
        let srs = setup_fake_srs::<Bls12, _>(&mut rng, size);
        let vec_len = srs.g_alpha_powers.len();
        let mut buffer = Vec::new();
        srs.write(&mut buffer).expect("writing to buffer failed");
        // tryingout normal operations
        GenericSRS::<Bls12>::read(&mut Cursor::new(&buffer)).expect("can't read the srs");

        // trying to read the first size
        let read_size = Cursor::new(&buffer).read_u32::<BigEndian>().unwrap() as usize;
        assert_eq!(vec_len, read_size);

        // remove the previous size from the bufer - u32 = 4 bytes
        // and replace the size by appending the rest
        let mut new_buffer = Vec::new();
        let invalid_size = MAX_SRS_SIZE + 1;
        new_buffer
            .write_u32::<BigEndian>(invalid_size as u32)
            .expect("failed to write invalid size");
        buffer.drain(0..4);
        new_buffer.append(&mut buffer);
        GenericSRS::<Bls12>::read(&mut Cursor::new(&new_buffer))
            .expect_err("this should have failed");
    }
}
