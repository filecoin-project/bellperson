use super::GpuEngine;

// Instead of having a very large OpenCL program written for a specific curve, with a lot of
// rudandant codes (As OpenCL doesn't have generic types or templates), this module will dynamically
// generate OpenCL codes given different PrimeFields and curves.

static FFT_SRC: &str = include_str!("fft/fft.cl");
static MULTIEXP_SRC: &str = include_str!("multiexp/multiexp.cl");

fn fft(field: &str) -> String {
    String::from(FFT_SRC).replace("FIELD", field)
}

fn multiexp(point: &str, exp: &str) -> String {
    String::from(MULTIEXP_SRC)
        .replace("POINT", point)
        .replace("EXPONENT", exp)
}

// WARNING: This function works only with Short Weierstrass Jacobian curves with Fq2 extension field.
pub fn kernel<E>(limb64: bool) -> String
where
    E: GpuEngine,
{
    [
        ec_gpu_gen::gen_ec_source::<E>(limb64),
        fft("Fr"),
        multiexp("G1", "Fr"),
        multiexp("G2", "Fr"),
    ]
    .join("\n\n")
}
