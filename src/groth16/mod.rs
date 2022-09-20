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
#[cfg(not(target_arch = "wasm32"))]
mod mapped_params;
mod params;
mod proof;
mod prover;
mod verifier;
mod verifying_key;

mod multiscalar;

pub use self::ext::*;
pub use self::generator::*;
#[cfg(not(target_arch = "wasm32"))]
pub use self::mapped_params::*;
pub use self::params::*;
pub use self::proof::*;
pub use self::prover::*;
pub use self::verifier::*;
pub use self::verifying_key::*;
