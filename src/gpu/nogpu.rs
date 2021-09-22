use super::error::{GPUError, GPUResult};
use crate::multicore::Worker;
use ff::PrimeField;
use group::prime::PrimeCurveAffine;
use std::marker::PhantomData;
use std::sync::Arc;

// This module is compiled instead of `fft.rs` and `multiexp.rs` if `gpu` feature is disabled.
#[allow(clippy::upper_case_acronyms)]
pub struct FFTKernel<E>(PhantomData<E>)
where
    E: Engine;

impl<E> FFTKernel<E>
where
    E: Engine,
{
    pub fn create(_: bool) -> GPUResult<FFTKernel<E>> {
        Err(GPUError::GPUDisabled)
    }

    pub fn radix_fft(&mut self, _: &mut [E::Fr], _: &E::Fr, _: u32) -> GPUResult<()> {
        Err(GPUError::GPUDisabled)
    }
    pub fn radix_fft_many(
        &mut self,
        _: &mut [&mut [E::Fr]],
        _: &[E::Fr],
        _: &[u32],
    ) -> GPUResult<()> {
        Err(GPUError::GPUDisabled)
    }
}

pub struct MultiexpKernel<E>(PhantomData<E>)
where
    E: Engine;

impl<E> MultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(_: bool) -> GPUResult<MultiexpKernel<E>> {
        Err(GPUError::GPUDisabled)
    }

    pub fn multiexp<G>(
        &mut self,
        _: &Worker,
        _: Arc<Vec<G>>,
        _: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        _: usize,
        _: usize,
    ) -> GPUResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine,
    {
        Err(GPUError::GPUDisabled)
    }
}

use pairing::Engine;

macro_rules! locked_kernel {
    ($class:ident) => {
        #[allow(clippy::upper_case_acronyms)]
        pub struct $class<E>(PhantomData<E>);

        impl<E> $class<E>
        where
            E: Engine,
        {
            pub fn new(_: usize, _: bool) -> $class<E> {
                $class::<E>(PhantomData)
            }

            pub fn with<F, R, K>(&mut self, _: F) -> GPUResult<R>
            where
                F: FnMut(&mut K) -> GPUResult<R>,
            {
                return Err(GPUError::GPUDisabled);
            }
        }
    };
}

locked_kernel!(LockedFFTKernel);
locked_kernel!(LockedMultiexpKernel);
