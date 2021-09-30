//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

// The `DummyEngine` currently only works on the CPU as G1/G2 is using `Fr` and `Fr` isn't
// supported by the GPU kernels
#[cfg(all(test, not(any(feature = "cuda", feature = "opencl"))))]
mod tests;

pub mod aggregate;
mod ext;
mod generator;
mod mapped_params;
mod params;
mod proof;
mod prover;
mod prover_ext;
mod verifier;
mod verifying_key;

mod multiscalar;

pub use self::ext::*;
pub use self::generator::*;
pub use self::mapped_params::*;
pub use self::params::*;
pub use self::proof::*;
pub use self::prover::*;
pub(crate) use self::prover_ext::*;
pub use self::verifier::*;
pub use self::verifying_key::*;

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum BellTaskType {
    WinningPost,
    WindowPost,
    MerkleTree,
}
