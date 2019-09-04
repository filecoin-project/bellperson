use paired::{CurveAffine, CurveProjective};
use ff::{PrimeField};
use ocl::traits::OclPrm;

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct PrimeFieldStruct<T>(pub T);
impl<T> Default for PrimeFieldStruct<T> where T: PrimeField {
    fn default() -> Self { PrimeFieldStruct::<T>(T::zero()) }
}
unsafe impl<T> OclPrm for PrimeFieldStruct<T> where T: PrimeField { }

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct CurveAffineStruct<T>(pub T);
impl<T> Default for CurveAffineStruct<T> where T: CurveAffine {
    fn default() -> Self { CurveAffineStruct::<T>(T::zero()) }
}
unsafe impl<T> OclPrm for CurveAffineStruct<T> where T: CurveAffine { }

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct CurveProjectiveStruct<T>(pub T);
impl<T> Default for CurveProjectiveStruct<T> where T: CurveProjective {
    fn default() -> Self { CurveProjectiveStruct::<T>(T::zero()) }
}
unsafe impl<T> OclPrm for CurveProjectiveStruct<T> where T: CurveProjective { }
