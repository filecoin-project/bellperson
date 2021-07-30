use group::{prime::PrimeCurveAffine, UncompressedEncoding};
use pairing::{Engine, MultiMillerLoop};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use memmap::Mmap;
use std::io::{self, Read, Write};
use std::mem;

use super::multiscalar;

#[derive(Debug, Clone)]
pub struct VerifyingKey<E: Engine + MultiMillerLoop> {
    // alpha in g1 for verifying and for creating A/C elements of
    // proof. Never the point at infinity.
    pub alpha_g1: E::G1Affine,

    // beta in g1 and g2 for verifying and for creating B/C elements
    // of proof. Never the point at infinity.
    pub beta_g1: E::G1Affine,
    pub beta_g2: E::G2Affine,

    // gamma in g2 for verifying. Never the point at infinity.
    pub gamma_g2: E::G2Affine,

    // delta in g1/g2 for verifying and proving, essentially the magic
    // trapdoor that forces the prover to evaluate the C element of the
    // proof with only components from the CRS. Never the point at
    // infinity.
    pub delta_g1: E::G1Affine,
    pub delta_g2: E::G2Affine,

    // Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / gamma
    // for all public inputs. Because all public inputs have a dummy constraint,
    // this is the same size as the number of inputs, and never contains points
    // at infinity.
    pub ic: Vec<E::G1Affine>,
}

impl<E: Engine + MultiMillerLoop> PartialEq for VerifyingKey<E> {
    fn eq(&self, other: &Self) -> bool {
        self.alpha_g1 == other.alpha_g1
            && self.beta_g1 == other.beta_g1
            && self.beta_g2 == other.beta_g2
            && self.gamma_g2 == other.gamma_g2
            && self.delta_g1 == other.delta_g1
            && self.delta_g2 == other.delta_g2
            && self.ic == other.ic
    }
}

fn read_uncompressed_point<C: UncompressedEncoding>(repr: &C::Uncompressed) -> io::Result<C> {
    let opt = C::from_uncompressed(repr);
    Option::from(opt).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not on curve"))
}

impl<E: Engine + MultiMillerLoop> VerifyingKey<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(self.alpha_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.beta_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.beta_g2.to_uncompressed().as_ref())?;
        writer.write_all(self.gamma_g2.to_uncompressed().as_ref())?;
        writer.write_all(self.delta_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.delta_g2.to_uncompressed().as_ref())?;
        writer.write_u32::<BigEndian>(self.ic.len() as u32)?;
        for ic in &self.ic {
            writer.write_all(ic.to_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut g1_repr = <E::G1Affine as UncompressedEncoding>::Uncompressed::default();
        let mut g2_repr = <E::G2Affine as UncompressedEncoding>::Uncompressed::default();

        reader.read_exact(g1_repr.as_mut())?;
        let alpha_g1 = read_uncompressed_point(&g1_repr)?;

        reader.read_exact(g1_repr.as_mut())?;
        let beta_g1 = read_uncompressed_point(&g1_repr)?;

        reader.read_exact(g2_repr.as_mut())?;
        let beta_g2 = read_uncompressed_point(&g2_repr)?;

        reader.read_exact(g2_repr.as_mut())?;
        let gamma_g2 = read_uncompressed_point(&g2_repr)?;

        reader.read_exact(g1_repr.as_mut())?;
        let delta_g1 = read_uncompressed_point(&g1_repr)?;

        reader.read_exact(g2_repr.as_mut())?;
        let delta_g2 = read_uncompressed_point(&g2_repr)?;

        let ic_len = reader.read_u32::<BigEndian>()? as usize;

        let mut ic = vec![];

        for _ in 0..ic_len {
            reader.read_exact(g1_repr.as_mut())?;
            let g1: E::G1Affine = read_uncompressed_point(&g1_repr)?;
            if g1.is_identity().into() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "point at infinity",
                ));
            }
            ic.push(g1);
        }

        Ok(VerifyingKey {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        })
    }

    pub fn read_mmap(mmap: &Mmap, offset: &mut usize) -> io::Result<Self> {
        let u32_len = mem::size_of::<u32>();
        let g1_len = mem::size_of::<<E::G1Affine as UncompressedEncoding>::Uncompressed>();
        let g2_len = mem::size_of::<<E::G2Affine as UncompressedEncoding>::Uncompressed>();

        let read_g1 = |mmap: &Mmap,
                       offset: &mut usize|
         -> Result<<E as Engine>::G1Affine, std::io::Error> {
            let ptr = &mmap[*offset..*offset + g1_len];
            // Safety: this operation is safe, because it's simply
            // casting to a known struct at the correct offset, given
            // the structure of the on-disk data.
            let g1_repr = unsafe {
                &*(ptr as *const [u8] as *const <E::G1Affine as UncompressedEncoding>::Uncompressed)
            };

            *offset += g1_len;
            read_uncompressed_point(g1_repr)
        };

        let read_g2 = |mmap: &Mmap,
                       offset: &mut usize|
         -> Result<<E as Engine>::G2Affine, std::io::Error> {
            let ptr = &mmap[*offset..*offset + g2_len];
            // Safety: this operation is safe, because it's simply
            // casting to a known struct at the correct offset, given
            // the structure of the on-disk data.
            let g2_repr = unsafe {
                &*(ptr as *const [u8] as *const <E::G2Affine as UncompressedEncoding>::Uncompressed)
            };

            *offset += g2_len;
            read_uncompressed_point(g2_repr)
        };

        let alpha_g1 = read_g1(&mmap, &mut *offset)?;
        let beta_g1 = read_g1(&mmap, &mut *offset)?;
        let beta_g2 = read_g2(&mmap, &mut *offset)?;
        let gamma_g2 = read_g2(&mmap, &mut *offset)?;
        let delta_g1 = read_g1(&mmap, &mut *offset)?;
        let delta_g2 = read_g2(&mmap, &mut *offset)?;

        let mut raw_ic_len = &mmap[*offset..*offset + u32_len];
        let ic_len = raw_ic_len.read_u32::<BigEndian>()? as usize;
        *offset += u32_len;

        let mut ic = vec![];

        for _ in 0..ic_len {
            let g1_repr = read_g1(&mmap, &mut *offset);
            let g1 = g1_repr.and_then(|e| {
                if e.is_identity().into() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

            ic.push(g1);
        }

        Ok(VerifyingKey {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        })
    }
}

pub struct PreparedVerifyingKey<E>
where
    E: MultiMillerLoop,
{
    /// Pairing result of alpha*beta
    pub(crate) alpha_g1_beta_g2: <E as Engine>::Gt,
    /// -gamma in G2 (used for single)
    pub(crate) neg_gamma_g2: <E as MultiMillerLoop>::G2Prepared,
    /// -delta in G2 (used for single)
    pub(crate) neg_delta_g2: <E as MultiMillerLoop>::G2Prepared,
    /// gamma in G2 (used for batch)
    pub(crate) gamma_g2: <E as MultiMillerLoop>::G2Prepared,
    /// delta in G2 (used for batch)
    pub(crate) delta_g2: <E as MultiMillerLoop>::G2Prepared,
    /// Copy of IC from `VerifiyingKey`.
    pub(crate) ic: Vec<E::G1Affine>,

    pub(crate) multiscalar: multiscalar::MultiscalarPrecompOwned<E::G1Affine>,

    // Aggregation specific prep
    pub(crate) alpha_g1: E::G1,
    pub(crate) beta_g2: <E as MultiMillerLoop>::G2Prepared,
    pub(crate) ic_projective: Vec<E::G1>,
}
