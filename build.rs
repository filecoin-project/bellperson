fn main() {
    cfg_if_nightly();
    gpu_kernel();
}

#[rustversion::nightly]
fn cfg_if_nightly() {
    println!("cargo:rustc-cfg=nightly");
}

#[rustversion::not(nightly)]
fn cfg_if_nightly() {}

/// The build script is used to generate the CUDA kernel and OpenCL source at compile-time, if the
/// `cuda` and/or `opencl` feature is enabled.
#[cfg(any(feature = "cuda", feature = "opencl"))]
fn gpu_kernel() {
    use blstrs::{Fp, Fp2, G1Affine, G2Affine, Scalar};
    use ec_gpu_gen::SourceBuilder;

    let source_builder = SourceBuilder::new()
        .add_fft::<Scalar>()
        .add_multiexp::<G1Affine, Fp>()
        .add_multiexp::<G2Affine, Fp2>();
    ec_gpu_gen::generate(&source_builder);
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn gpu_kernel() {}
