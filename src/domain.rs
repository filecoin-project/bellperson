//! This module contains an [`EvaluationDomain`] abstraction for performing
//! various kinds of polynomial arithmetic on top of the scalar field.
//!
//! In pairing-based SNARKs like [Groth16], we need to calculate a quotient
//! polynomial over a target polynomial with roots at distinct points associated
//! with each constraint of the constraint system. In order to be efficient, we
//! choose these roots to be the powers of a 2<sup>n</sup> root of unity in the
//! field. This allows us to perform polynomial operations in O(n) by performing
//! an O(n log n) FFT over such a domain.
//!
//! [`EvaluationDomain`]: crate::domain::EvaluationDomain
//! [Groth16]: https://eprint.iacr.org/2016/260

use std::ops::{AddAssign, MulAssign, SubAssign};

use ff::{Field, PrimeField};
use group::Curve;
use pairing::Engine;

use super::multicore::Worker;
use super::SynthesisError;
use crate::gpu;

use log::{info, warn};

pub struct EvaluationDomain<E: Engine + gpu::GpuEngine, G: Group<E::Fr>> {
    coeffs: Vec<G>,
    exp: u32,
    omega: E::Fr,
    omegainv: E::Fr,
    geninv: E::Fr,
    minv: E::Fr,
}

impl<E: Engine + gpu::GpuEngine, G: Group<E::Fr>> AsRef<[G]> for EvaluationDomain<E, G> {
    fn as_ref(&self) -> &[G] {
        &self.coeffs
    }
}

impl<E: Engine + gpu::GpuEngine, G: Group<E::Fr>> AsMut<[G]> for EvaluationDomain<E, G> {
    fn as_mut(&mut self) -> &mut [G] {
        &mut self.coeffs
    }
}

impl<E: Engine + gpu::GpuEngine, G: Group<E::Fr>> EvaluationDomain<E, G> {
    pub fn into_coeffs(self) -> Vec<G> {
        self.coeffs
    }

    pub fn from_coeffs(mut coeffs: Vec<G>) -> Result<EvaluationDomain<E, G>, SynthesisError> {
        // Compute the size of our evaluation domain
        let mut m = 1;
        let mut exp = 0;
        while m < coeffs.len() {
            m *= 2;
            exp += 1;

            // The pairing-friendly curve may not be able to support
            // large enough (radix2) evaluation domains.
            if exp >= E::Fr::S {
                return Err(SynthesisError::PolynomialDegreeTooLarge);
            }
        }
        // Compute omega, the 2^exp primitive root of unity
        let mut omega = E::Fr::root_of_unity();
        for _ in exp..E::Fr::S {
            omega = omega.square();
        }

        // Extend the coeffs vector with zeroes if necessary
        coeffs.resize(m, G::group_zero());

        Ok(EvaluationDomain {
            coeffs,
            exp,
            omega,
            omegainv: omega.invert().unwrap(),
            geninv: E::Fr::multiplicative_generator().invert().unwrap(),
            minv: E::Fr::from(m as u64).invert().unwrap(),
        })
    }

    pub fn fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GPUResult<()> {
        best_fft(kern, &mut self.coeffs, worker, &self.omega, self.exp);
        Ok(())
    }

    pub fn ifft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GPUResult<()> {
        best_fft(kern, &mut self.coeffs, worker, &self.omegainv, self.exp);

        worker.scope(self.coeffs.len(), |scope, chunk| {
            let minv = self.minv;

            for v in self.coeffs.chunks_mut(chunk) {
                scope.execute(move || {
                    for v in v {
                        v.group_mul_assign(&minv);
                    }
                });
            }
        });

        Ok(())
    }

    pub fn distribute_powers(&mut self, worker: &Worker, g: E::Fr) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (i, v) in self.coeffs.chunks_mut(chunk).enumerate() {
                scope.execute(move || {
                    let mut u = g.pow_vartime(&[(i * chunk) as u64]);
                    for v in v.iter_mut() {
                        v.group_mul_assign(&u);
                        u.mul_assign(&g);
                    }
                });
            }
        });
    }

    pub fn coset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GPUResult<()> {
        self.distribute_powers(worker, E::Fr::multiplicative_generator());
        self.fft(worker, kern)?;
        Ok(())
    }

    pub fn icoset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GPUResult<()> {
        let geninv = self.geninv;
        self.ifft(worker, kern)?;
        self.distribute_powers(worker, geninv);
        Ok(())
    }

    /// This evaluates t(tau) for this domain, which is
    /// tau^m - 1 for these radix-2 domains.
    pub fn z(&self, tau: &E::Fr) -> E::Fr {
        let mut tmp = tau.pow_vartime(&[self.coeffs.len() as u64]);
        tmp.sub_assign(&E::Fr::one());

        tmp
    }

    /// The target polynomial is the zero polynomial in our
    /// evaluation domain, so we must perform division over
    /// a coset.
    pub fn divide_by_z_on_coset(&mut self, worker: &Worker) {
        let i = self.z(&E::Fr::multiplicative_generator()).invert().unwrap();

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.execute(move || {
                    for v in v {
                        v.group_mul_assign(&i);
                    }
                });
            }
        });
    }

    /// Perform O(n) multiplication of two polynomials in the domain.
    pub fn mul_assign(&mut self, worker: &Worker, other: &EvaluationDomain<E, Scalar<E>>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.execute(move || {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.group_mul_assign(&b.0);
                    }
                });
            }
        });
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, worker: &Worker, other: &EvaluationDomain<E, G>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.execute(move || {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        a.group_sub_assign(&b);
                    }
                });
            }
        });
    }
}

pub trait Group<S: PrimeField>: Sized + Copy + Clone + Send + Sync {
    fn group_zero() -> Self;
    fn group_mul_assign(&mut self, by: &S);
    fn group_add_assign(&mut self, other: &Self);
    fn group_sub_assign(&mut self, other: &Self);
}

pub struct Point<G: Curve>(pub G);

impl<G: Curve> PartialEq for Point<G> {
    fn eq(&self, other: &Point<G>) -> bool {
        self.0 == other.0
    }
}

impl<G: Curve> Copy for Point<G> {}

impl<G: Curve> Clone for Point<G> {
    fn clone(&self) -> Point<G> {
        *self
    }
}

impl<G: Curve> Group<G::Scalar> for Point<G> {
    fn group_zero() -> Self {
        Point(G::identity())
    }
    fn group_mul_assign(&mut self, by: &G::Scalar) {
        self.0.mul_assign(by);
    }
    fn group_add_assign(&mut self, other: &Self) {
        self.0.add_assign(&other.0);
    }
    fn group_sub_assign(&mut self, other: &Self) {
        self.0.sub_assign(&other.0);
    }
}

pub struct Scalar<E: Engine>(pub E::Fr);

impl<E: Engine> PartialEq for Scalar<E> {
    fn eq(&self, other: &Scalar<E>) -> bool {
        self.0 == other.0
    }
}

impl<E: Engine> Copy for Scalar<E> {}

impl<E: Engine> Clone for Scalar<E> {
    fn clone(&self) -> Scalar<E> {
        *self
    }
}

impl<E: Engine> Group<E::Fr> for Scalar<E> {
    fn group_zero() -> Self {
        Scalar(E::Fr::zero())
    }
    fn group_mul_assign(&mut self, by: &E::Fr) {
        self.0.mul_assign(by);
    }
    fn group_add_assign(&mut self, other: &Self) {
        self.0.add_assign(&other.0);
    }
    fn group_sub_assign(&mut self, other: &Self) {
        self.0.sub_assign(&other.0);
    }
}

fn best_fft<E: Engine + gpu::GpuEngine, T: Group<E::Fr>>(
    kern: &mut Option<gpu::LockedFFTKernel<E>>,
    a: &mut [T],
    worker: &Worker,
    omega: &E::Fr,
    log_n: u32,
) {
    if let Some(ref mut kern) = kern {
        if kern
            .with(|k: &mut gpu::FFTKernel<E>| gpu_fft(k, a, omega, log_n))
            .is_ok()
        {
            return;
        }
    }

    let log_cpus = worker.log_num_cpus();
    if log_n <= log_cpus {
        serial_fft::<E, T>(a, omega, log_n);
    } else {
        parallel_fft::<E, T>(a, worker, omega, log_n, log_cpus);
    }
}

pub fn gpu_fft<E: Engine + gpu::GpuEngine, T: Group<E::Fr>>(
    kern: &mut gpu::FFTKernel<E>,
    a: &mut [T],
    omega: &E::Fr,
    log_n: u32,
) -> gpu::GPUResult<()> {
    // EvaluationDomain module is supposed to work only with E::Fr elements, and not CurveProjective
    // points. The Bellman authors have implemented an unnecessarry abstraction called Group<E::Fr>
    // which is implemented for both PrimeField and CurveProjective elements. As nowhere in the code
    // is the CurveProjective version used, T and E::Fr are guaranteed to be equal and thus have same
    // size.
    // For compatibility/performance reasons we decided to transmute the array to the desired type
    // as it seems safe and needs less modifications in the current structure of Bellman library.
    let a = unsafe { &mut *(a as *mut [T] as *mut [E::Fr]) };
    kern.radix_fft(a, omega, log_n)?;
    Ok(())
}

#[allow(clippy::many_single_char_names)]
pub fn serial_fft<E: Engine, T: Group<E::Fr>>(a: &mut [T], omega: &E::Fr, log_n: u32) {
    fn bitreverse(mut n: u32, l: u32) -> u32 {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow_vartime(&[u64::from(n / (2 * m))]);

        let mut k = 0;
        while k < n {
            let mut w = E::Fr::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.group_mul_assign(&w);
                let mut tmp = a[(k + j) as usize];
                tmp.group_sub_assign(&t);
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize].group_add_assign(&t);
                w.mul_assign(&w_m);
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn parallel_fft<E: Engine, T: Group<E::Fr>>(
    a: &mut [T],
    worker: &Worker,
    omega: &E::Fr,
    log_n: u32,
    log_cpus: u32,
) {
    assert!(log_n >= log_cpus);

    let num_cpus = 1 << log_cpus;
    let log_new_n = log_n - log_cpus;
    let mut tmp = vec![vec![T::group_zero(); 1 << log_new_n]; num_cpus];
    let new_omega = omega.pow_vartime(&[num_cpus as u64]);

    worker.scope(0, |scope, _| {
        let a = &*a;

        for (j, tmp) in tmp.iter_mut().enumerate() {
            scope.execute(move || {
                // Shuffle into a sub-FFT
                let omega_j = omega.pow_vartime(&[j as u64]);
                let omega_step = omega.pow_vartime(&[(j as u64) << log_new_n]);

                let mut elt = E::Fr::one();
                for (i, tmp) in tmp.iter_mut().enumerate() {
                    for s in 0..num_cpus {
                        let idx = (i + (s << log_new_n)) % (1 << log_n);
                        let mut t = a[idx];
                        t.group_mul_assign(&elt);
                        tmp.group_add_assign(&t);
                        elt.mul_assign(&omega_step);
                    }
                    elt.mul_assign(&omega_j);
                }

                // Perform sub-FFT
                serial_fft::<E, T>(tmp, &new_omega, log_new_n);
            });
        }
    });

    // TODO: does this hurt or help?
    worker.scope(a.len(), |scope, chunk| {
        let tmp = &tmp;

        for (idx, a) in a.chunks_mut(chunk).enumerate() {
            scope.execute(move || {
                let mut idx = idx * chunk;
                let mask = (1 << log_cpus) - 1;
                for a in a {
                    *a = tmp[idx & mask][idx >> log_cpus];
                    idx += 1;
                }
            });
        }
    });
}

// Test multiplying various (low degree) polynomials together and
// comparing with naive evaluations.
#[test]
fn polynomial_arith() {
    use blstrs::Bls12;
    use rand_core::RngCore;

    fn test_mul<E: Engine + gpu::GpuEngine, R: RngCore>(rng: &mut R) {
        let worker = Worker::new();

        for coeffs_a in 0..70 {
            for coeffs_b in 0..70 {
                let mut a: Vec<_> = (0..coeffs_a)
                    .map(|_| Scalar::<E>(E::Fr::random(&mut *rng)))
                    .collect();
                let mut b: Vec<_> = (0..coeffs_b)
                    .map(|_| Scalar::<E>(E::Fr::random(&mut *rng)))
                    .collect();

                // naive evaluation
                let mut naive = vec![Scalar(E::Fr::zero()); coeffs_a + coeffs_b];
                for (i1, a) in a.iter().enumerate() {
                    for (i2, b) in b.iter().enumerate() {
                        let mut prod = *a;
                        prod.group_mul_assign(&b.0);
                        naive[i1 + i2].group_add_assign(&prod);
                    }
                }

                a.resize(coeffs_a + coeffs_b, Scalar(E::Fr::zero()));
                b.resize(coeffs_a + coeffs_b, Scalar(E::Fr::zero()));

                let mut a = EvaluationDomain::from_coeffs(a).unwrap();
                let mut b = EvaluationDomain::from_coeffs(b).unwrap();

                a.fft(&worker, &mut None).unwrap();
                b.fft(&worker, &mut None).unwrap();
                a.mul_assign(&worker, &b);
                a.ifft(&worker, &mut None).unwrap();

                for (naive, fft) in naive.iter().zip(a.coeffs.iter()) {
                    assert!(naive == fft);
                }
            }
        }
    }

    let rng = &mut rand::thread_rng();

    test_mul::<Bls12, _>(rng);
}

#[test]
fn fft_composition() {
    use blstrs::Bls12;
    use pairing::Engine;
    use rand_core::RngCore;

    fn test_comp<E: Engine + gpu::GpuEngine, R: RngCore>(rng: &mut R) {
        let worker = Worker::new();

        for coeffs in 0..10 {
            let coeffs = 1 << coeffs;

            let mut v = vec![];
            for _ in 0..coeffs {
                v.push(Scalar::<E>(E::Fr::random(&mut *rng)));
            }

            let mut domain = EvaluationDomain::<E, _>::from_coeffs(v.clone()).unwrap();
            domain.ifft(&worker, &mut None).unwrap();
            domain.fft(&worker, &mut None).unwrap();
            assert!(v == domain.coeffs);
            domain.fft(&worker, &mut None).unwrap();
            domain.ifft(&worker, &mut None).unwrap();
            assert!(v == domain.coeffs);
            domain.icoset_fft(&worker, &mut None).unwrap();
            domain.coset_fft(&worker, &mut None).unwrap();
            assert!(v == domain.coeffs);
            domain.coset_fft(&worker, &mut None).unwrap();
            domain.icoset_fft(&worker, &mut None).unwrap();
            assert!(v == domain.coeffs);
        }
    }

    let rng = &mut rand::thread_rng();

    test_comp::<Bls12, _>(rng);
}

#[test]
fn parallel_fft_consistency() {
    use blstrs::Bls12;
    use rand_core::RngCore;
    use std::cmp::min;

    fn test_consistency<E: Engine + gpu::GpuEngine, R: RngCore>(rng: &mut R) {
        let worker = Worker::new();

        for _ in 0..5 {
            for log_d in 0..10 {
                let d = 1 << log_d;

                let v1 = (0..d)
                    .map(|_| Scalar::<E>(E::Fr::random(&mut *rng)))
                    .collect::<Vec<_>>();
                let mut v1 = EvaluationDomain::<E, _>::from_coeffs(v1).unwrap();
                let mut v2 = EvaluationDomain::<E, _>::from_coeffs(v1.coeffs.clone()).unwrap();

                for log_cpus in log_d..min(log_d + 1, 3) {
                    parallel_fft::<E, _>(&mut v1.coeffs, &worker, &v1.omega, log_d, log_cpus);
                    serial_fft::<E, _>(&mut v2.coeffs, &v2.omega, log_d);

                    assert!(v1.coeffs == v2.coeffs);
                }
            }
        }
    }

    let rng = &mut rand::thread_rng();

    test_consistency::<Bls12, _>(rng);
}

pub fn create_fft_kernel<E>(_log_d: usize, priority: bool) -> Option<gpu::FFTKernel<E>>
where
    E: Engine + gpu::GpuEngine,
{
    match gpu::FFTKernel::create(priority) {
        Ok(k) => {
            info!("GPU FFT kernel instantiated!");
            Some(k)
        }
        Err(e) => {
            warn!("Cannot instantiate GPU FFT kernel! Error: {}", e);
            None
        }
    }
}

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use crate::domain::{gpu_fft, parallel_fft, serial_fft, EvaluationDomain, Scalar};
    use crate::gpu;
    use crate::multicore::Worker;
    use blstrs::{Bls12, Scalar as Fr};
    use ff::Field;
    use std::time::Instant;

    #[test]
    pub fn gpu_fft_consistency() {
        let _ = env_logger::try_init();
        gpu::dump_device_list();

        let mut rng = rand::thread_rng();

        let worker = Worker::new();
        let log_cpus = worker.log_num_cpus();
        let mut kern = gpu::FFTKernel::<Bls12>::create(false).expect("Cannot initialize kernel!");

        for log_d in 1..=20 {
            let d = 1 << log_d;

            let elems = (0..d)
                .map(|_| Scalar::<Bls12>(Fr::random(&mut rng)))
                .collect::<Vec<_>>();
            let mut v1 = EvaluationDomain::<Bls12, _>::from_coeffs(elems.clone()).unwrap();
            let mut v2 = EvaluationDomain::<Bls12, _>::from_coeffs(elems.clone()).unwrap();

            println!("Testing FFT for {} elements...", d);

            let mut now = Instant::now();
            gpu_fft(&mut kern, &mut v1.coeffs, &v1.omega, log_d).expect("GPU FFT failed!");
            let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("GPU took {}ms.", gpu_dur);

            now = Instant::now();
            if log_d <= log_cpus {
                serial_fft::<Bls12, _>(&mut v2.coeffs, &v2.omega, log_d);
            } else {
                parallel_fft::<Bls12, _>(&mut v2.coeffs, &worker, &v2.omega, log_d, log_cpus);
            }
            let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("CPU ({} cores) took {}ms.", 1 << log_cpus, cpu_dur);

            println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

            assert!(v1.coeffs == v2.coeffs);
            println!("============================");
        }
    }
}
