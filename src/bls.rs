#[cfg(feature = "blst")]
pub use blstrs::{
    Bls12, Compress, Engine, Fp as Fq, Fp12 as Fq12, Fp2 as Fq2, FpRepr as FqRepr, G1Affine,
    G1Compressed, G1Projective, G1Uncompressed, G2Affine, G2Compressed, G2Prepared, G2Projective,
    G2Uncompressed, PairingCurveAffine, Scalar as Fr, ScalarRepr as FrRepr,
};

#[cfg(feature = "pairing")]
pub use paired::{
    bls12_381::{
        Bls12, Fq, Fq12, Fq2, FqRepr, Fr, FrRepr, G1Affine, G1Compressed, G1Uncompressed, G2Affine,
        G2Compressed, G2Prepared, G2Uncompressed, G1 as G1Projective, G2 as G2Projective,
    },
    Compress, Engine, PairingCurveAffine,
};
