//! This module contains an `EvaluationDomain` abstraction for
//! performing various kinds of polynomial arithmetic on top of
//! the scalar field.
//!
//! In pairing-based SNARKs like Groth16, we need to calculate
//! a quotient polynomial over a target polynomial with roots
//! at distinct points associated with each constraint of the
//! constraint system. In order to be efficient, we choose these
//! roots to be the powers of a 2^n root of unity in the field.
//! This allows us to perform polynomial operations in O(n)
//! by performing an O(n log n) FFT over such a domain.

use ff::PrimeField;

use super::SynthesisError;

use super::multicore::Worker;

pub struct EvaluationDomain<F: PrimeField> {
    coeffs: Vec<F>,
    exp: u32,
    omega: F,
    omegainv: F,
    geninv: F,
    minv: F,
}

impl<F: PrimeField> EvaluationDomain<F> {
    pub fn as_ref(&self) -> &[F] {
        &self.coeffs
    }

    pub fn as_mut(&mut self) -> &mut [F] {
        &mut self.coeffs
    }

    pub fn into_coeffs(self) -> Vec<F> {
        self.coeffs
    }

    pub fn from_coeffs(mut coeffs: Vec<F>) -> Result<EvaluationDomain<F>, SynthesisError> {
        // Compute the size of our evaluation domain
        let mut m = 1;
        let mut exp = 0;
        while m < coeffs.len() {
            m *= 2;
            exp += 1;

            // The pairing-friendly curve may not be able to support
            // large enough (radix2) evaluation domains.
            if exp >= F::S {
                return Err(SynthesisError::PolynomialDegreeTooLarge);
            }
        }

        // Compute omega, the 2^exp primitive root of unity
        let mut omega = F::root_of_unity();
        for _ in exp..F::S {
            omega.square();
        }

        // Extend the coeffs vector with zeroes if necessary
        coeffs.resize(m, F::zero());

        Ok(EvaluationDomain {
            coeffs: coeffs,
            exp: exp,
            omega: omega,
            omegainv: omega.inverse().unwrap(),
            geninv: F::multiplicative_generator().inverse().unwrap(),
            minv: F::from_str(&format!("{}", m))
                .unwrap()
                .inverse()
                .unwrap(),
        })
    }

    pub fn fft(&mut self, worker: &Worker) {
        best_fft(&mut self.coeffs, worker, &self.omega, self.exp);
    }

    pub fn ifft(&mut self, worker: &Worker) {
        best_fft(&mut self.coeffs, worker, &self.omegainv, self.exp);

        worker
            .scope(self.coeffs.len(), |scope, chunk| {
                let minv = self.minv;

                for v in self.coeffs.chunks_mut(chunk) {
                    scope.spawn(move |_| {
                        for v in v {
                            v.mul_assign(&minv);
                        }
                    });
                }
            })
            .unwrap();
    }

    pub fn distribute_powers(&mut self, worker: &Worker, g: F) {
        worker
            .scope(self.coeffs.len(), |scope, chunk| {
                for (i, v) in self.coeffs.chunks_mut(chunk).enumerate() {
                    scope.spawn(move |_| {
                        let mut u = g.pow(&[(i * chunk) as u64]);
                        for v in v.iter_mut() {
                            v.mul_assign(&u);
                            u.mul_assign(&g);
                        }
                    });
                }
            })
            .unwrap();
    }

    pub fn coset_fft(&mut self, worker: &Worker) {
        self.distribute_powers(worker, F::multiplicative_generator());
        self.fft(worker);
    }

    pub fn icoset_fft(&mut self, worker: &Worker) {
        let geninv = self.geninv;

        self.ifft(worker);
        self.distribute_powers(worker, geninv);
    }

    /// This evaluates t(tau) for this domain, which is
    /// tau^m - 1 for these radix-2 domains.
    pub fn z(&self, tau: &F) -> F {
        let mut tmp = tau.pow(&[self.coeffs.len() as u64]);
        tmp.sub_assign(&F::one());

        tmp
    }

    /// The target polynomial is the zero polynomial in our
    /// evaluation domain, so we must perform division over
    /// a coset.
    pub fn divide_by_z_on_coset(&mut self, worker: &Worker) {
        let i = self
            .z(&F::multiplicative_generator())
            .inverse()
            .unwrap();

        worker
            .scope(self.coeffs.len(), |scope, chunk| {
                for v in self.coeffs.chunks_mut(chunk) {
                    scope.spawn(move |_| {
                        for v in v {
                            v.mul_assign(&i);
                        }
                    });
                }
            })
            .unwrap();
    }

    /// Perform O(n) multiplication of two polynomials in the domain.
    pub fn mul_assign(&mut self, worker: &Worker, other: &EvaluationDomain<F>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker
            .scope(self.coeffs.len(), |scope, chunk| {
                for (a, b) in self
                    .coeffs
                    .chunks_mut(chunk)
                    .zip(other.coeffs.chunks(chunk))
                {
                    scope.spawn(move |_| {
                        for (a, b) in a.iter_mut().zip(b.iter()) {
                            a.mul_assign(&b);
                        }
                    });
                }
            })
            .unwrap();
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, worker: &Worker, other: &EvaluationDomain<F>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker
            .scope(self.coeffs.len(), |scope, chunk| {
                for (a, b) in self
                    .coeffs
                    .chunks_mut(chunk)
                    .zip(other.coeffs.chunks(chunk))
                {
                    scope.spawn(move |_| {
                        for (a, b) in a.iter_mut().zip(b.iter()) {
                            a.sub_assign(&b);
                        }
                    });
                }
            })
            .unwrap();
    }
}

fn best_fft<F: PrimeField>(a: &mut [F], worker: &Worker, omega: &F, log_n: u32) {
    let log_cpus = worker.log_num_cpus();

    if log_n <= log_cpus {
        serial_fft(a, omega, log_n);
    } else {
        parallel_fft(a, worker, omega, log_n, log_cpus);
    }
}

fn serial_fft<F: PrimeField>(a: &mut [F], omega: &F, log_n: u32) {
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
        let w_m = omega.pow(&[(n / (2 * m)) as u64]);

        let mut k = 0;
        while k < n {
            let mut w = F::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.mul_assign(&w);
                let mut tmp = a[(k + j) as usize];
                tmp.sub_assign(&t);
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize].add_assign(&t);
                w.mul_assign(&w_m);
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn parallel_fft<F: PrimeField>(
    a: &mut [F],
    worker: &Worker,
    omega: &F,
    log_n: u32,
    log_cpus: u32,
) {
    assert!(log_n >= log_cpus);

    let num_cpus = 1 << log_cpus;
    let log_new_n = log_n - log_cpus;
    let mut tmp = vec![vec![F::zero(); 1 << log_new_n]; num_cpus];
    let new_omega = omega.pow(&[num_cpus as u64]);

    worker
        .scope(0, |scope, _| {
            let a = &*a;

            for (j, tmp) in tmp.iter_mut().enumerate() {
                scope.spawn(move |_| {
                    // Shuffle into a sub-FFT
                    let omega_j = omega.pow(&[j as u64]);
                    let omega_step = omega.pow(&[(j as u64) << log_new_n]);

                    let mut elt = F::one();
                    for i in 0..(1 << log_new_n) {
                        for s in 0..num_cpus {
                            let idx = (i + (s << log_new_n)) % (1 << log_n);
                            let mut t = a[idx];
                            t.mul_assign(&elt);
                            tmp[i].add_assign(&t);
                            elt.mul_assign(&omega_step);
                        }
                        elt.mul_assign(&omega_j);
                    }

                    // Perform sub-FFT
                    serial_fft(tmp, &new_omega, log_new_n);
                });
            }
        })
        .unwrap();

    // TODO: does this hurt or help?
    worker
        .scope(a.len(), |scope, chunk| {
            let tmp = &tmp;

            for (idx, a) in a.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut idx = idx * chunk;
                    let mask = (1 << log_cpus) - 1;
                    for a in a {
                        *a = tmp[idx & mask][idx >> log_cpus];
                        idx += 1;
                    }
                });
            }
        })
        .unwrap();
}

// Test multiplying various (low degree) polynomials together and
// comparing with naive evaluations.
#[test]
fn polynomial_arith() {
    use paired::bls12_381::Bls12;
    use paired::Engine;
    use ff::Field;
    use rand::{self, Rand};

    fn test_mul<E: Engine, R: rand::Rng>(rng: &mut R) {
        let worker = Worker::new();

        for coeffs_a in 0..70 {
            for coeffs_b in 0..70 {
                let mut a: Vec<_> = (0..coeffs_a)
                    .map(|_| E::Fr::rand(rng))
                    .collect();
                let mut b: Vec<_> = (0..coeffs_b)
                    .map(|_| E::Fr::rand(rng))
                    .collect();

                // naive evaluation
                let mut naive = vec![E::Fr::zero(); coeffs_a + coeffs_b];
                for (i1, a) in a.iter().enumerate() {
                    for (i2, b) in b.iter().enumerate() {
                        let mut prod = *a;
                        prod.mul_assign(&b);
                        naive[i1 + i2].add_assign(&prod);
                    }
                }

                a.resize(coeffs_a + coeffs_b, E::Fr::zero());
                b.resize(coeffs_a + coeffs_b, E::Fr::zero());

                let mut a = EvaluationDomain::from_coeffs(a).unwrap();
                let mut b = EvaluationDomain::from_coeffs(b).unwrap();

                a.fft(&worker);
                b.fft(&worker);
                a.mul_assign(&worker, &b);
                a.ifft(&worker);

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
    use paired::bls12_381::Bls12;
    use paired::Engine;
    use rand::{self, Rand};

    fn test_comp<E: Engine, R: rand::Rng>(rng: &mut R) {
        let worker = Worker::new();

        for coeffs in 0..10 {
            let coeffs = 1 << coeffs;

            let mut v = (0..coeffs)
                .map(|_| E::Fr::rand(rng))
                .collect::<Vec<_>>();

            let mut domain = EvaluationDomain::from_coeffs(v.clone()).unwrap();
            domain.ifft(&worker);
            domain.fft(&worker);
            assert!(v == domain.coeffs);
            domain.fft(&worker);
            domain.ifft(&worker);
            assert!(v == domain.coeffs);
            domain.icoset_fft(&worker);
            domain.coset_fft(&worker);
            assert!(v == domain.coeffs);
            domain.coset_fft(&worker);
            domain.icoset_fft(&worker);
            assert!(v == domain.coeffs);
        }
    }

    let rng = &mut rand::thread_rng();

    test_comp::<Bls12, _>(rng);
}

#[test]
fn parallel_fft_consistency() {
    use paired::bls12_381::Bls12;
    use paired::Engine;
    use rand::{self, Rand};
    use std::cmp::min;

    fn test_consistency<E: Engine, R: rand::Rng>(rng: &mut R) {
        let worker = Worker::new();

        for _ in 0..5 {
            for log_d in 0..10 {
                let d = 1 << log_d;

                let v1 = (0..d)
                    .map(|_| E::Fr::rand(rng))
                    .collect::<Vec<_>>();
                let mut v1 = EvaluationDomain::from_coeffs(v1).unwrap();
                let mut v2 = EvaluationDomain::from_coeffs(v1.coeffs.clone()).unwrap();

                for log_cpus in log_d..min(log_d + 1, 3) {
                    parallel_fft(&mut v1.coeffs, &worker, &v1.omega, log_d, log_cpus);
                    serial_fft(&mut v2.coeffs, &v2.omega, log_d);

                    assert!(v1.coeffs == v2.coeffs);
                }
            }
        }
    }

    let rng = &mut rand::thread_rng();

    test_consistency::<Bls12, _>(rng);
}
