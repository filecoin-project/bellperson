use std::{io, marker::PhantomData, path::PathBuf, sync::Arc};

use bellpepper_core::SynthesisError;
use pairing::MultiMillerLoop;
use supraseal_c2::SRS;

use crate::groth16::{ParameterSource, VerifyingKey};

// The parameters for Supraseal live on the C++ side. We take this just as a wrapper so that their
// are properly initialized.
pub struct SuprasealParameters<E> {
    srs: SRS,
    _phantom: PhantomData<E>,
}

impl<E> SuprasealParameters<E> {
    pub fn new(param_file_path: PathBuf) -> io::Result<Self> {
        let srs = SRS::try_new(param_file_path, false)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;

        Ok(Self {
            srs,
            _phantom: PhantomData::<E>,
        })
    }
}

impl<E> ParameterSource<E> for &SuprasealParameters<E>
where
    E: MultiMillerLoop,
{
    type G1Builder = (Arc<Vec<E::G1Affine>>, usize);
    type G2Builder = (Arc<Vec<E::G2Affine>>, usize);

    fn get_vk(&self, _: usize) -> Result<&VerifyingKey<E>, SynthesisError> {
        unimplemented!()
    }

    fn get_h(&self, _num_h: usize) -> Result<Self::G1Builder, SynthesisError> {
        unimplemented!()
    }

    fn get_l(&self, _num_l: usize) -> Result<Self::G1Builder, SynthesisError> {
        unimplemented!()
    }

    fn get_a(
        &self,
        _num_inputs: usize,
        _num_a: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        unimplemented!()
    }

    fn get_b_g1(
        &self,
        _num_inputs: usize,
        _num_b_g1: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        unimplemented!()
    }

    fn get_b_g2(
        &self,
        _num_inputs: usize,
        _num_b_g2: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        unimplemented!()
    }
    fn get_supraseal_srs(&self) -> Option<&supraseal_c2::SRS> {
        Some(&self.srs)
    }
}
