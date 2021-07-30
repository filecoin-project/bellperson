use crossbeam_channel::{bounded, Receiver, Sender};
use ff::Field;
use group::{Curve, Group};
use pairing::{Engine, MillerLoopResult, MultiMillerLoop};
use rand_core::RngCore;
use rayon::prelude::*;

use crate::SynthesisError;
use std::ops::Mul;
use std::sync::{
    atomic::{AtomicBool, Ordering::SeqCst},
    Arc, Mutex,
};
use std::thread;

/// Holds the logic for merging multiple pairing checks of the form
///
/// $$
///  e(A,B)e(C,D)\dots = T
/// $$
///
/// Into a compressed form where only one final exponentiation is required. All
/// checks but up to one will be randomized.
#[derive(Debug)]
pub struct PairingChecks<E, R>
where
    E: MultiMillerLoop,
    R: RngCore + Send,
{
    /// Circuit breaker to allow canceling all checks and marking the whole check as failed.
    valid: Arc<AtomicBool>,
    merge_send: Sender<Result<PairingCheck<E>, SynthesisError>>,
    valid_recv: Receiver<Result<bool, SynthesisError>>,
    /// Random number generator used for generating the random coefficients.
    rng: Mutex<R>,
    /// Ensures that the non randomized check is only added exactly once.
    non_random_check_done: AtomicBool,
}

impl<E, R> PairingChecks<E, R>
where
    E: MultiMillerLoop,
    R: RngCore + Send,
{
    #[allow(clippy::type_complexity)]
    pub fn new(rng: R) -> Self {
        let (merge_send, merge_recv): (
            Sender<Result<PairingCheck<E>, SynthesisError>>,
            Receiver<Result<PairingCheck<E>, SynthesisError>>,
        ) = bounded(10);
        let (valid_send, valid_recv) = bounded(1);

        let valid = Arc::new(AtomicBool::new(true));
        let valid_copy = valid.clone();

        // Spawn this thread outside of the Rayon thread pool, so that it can always receive
        // messages, even if the thread pool is fully occupied.
        thread::spawn(move || {
            let mut acc = PairingCheck::new();
            while let Ok(tuple) = merge_recv.recv() {
                match tuple {
                    Ok(check) => {
                        // only do work as long as we know we are still valid
                        if valid_copy.load(SeqCst) {
                            acc.merge(&check);
                        } else {
                            return;
                        }
                    }
                    Err(e) => {
                        // we signal an invalid proof - malformed for example
                        valid_copy.store(false, SeqCst);
                        valid_send.send(Err(e)).expect("failed to send error");
                        return;
                    }
                }
            }
            if valid_copy.load(SeqCst) {
                valid_send.send(Ok(acc.verify())).expect("failed to send");
            }
        });

        PairingChecks {
            valid,
            merge_send,
            valid_recv,
            rng: Mutex::new(rng),
            non_random_check_done: AtomicBool::new(false),
        }
    }

    /// Fails the whole check.
    pub fn invalidate(&self) {
        self.valid.store(false, SeqCst);
    }

    pub fn report_err(&self, e: SynthesisError) {
        self.merge_send
            .send(Err(e))
            .expect("expect to send on channel");
    }

    fn merge_pair(
        &self,
        result: <E as MultiMillerLoop>::Result,
        exp: <E as Engine>::Gt,
        must_randomize: bool,
    ) {
        self.merge(PairingCheck::from_pair(result, exp), must_randomize);
    }

    fn merge_miller_one(&self, result: <E as MultiMillerLoop>::Result, must_randomize: bool) {
        self.merge(PairingCheck::from_miller_one(result), must_randomize);
    }

    /// takes a vector of elements  which are outputs of miller outputs and an
    /// right element which fulfills the following:
    ///
    /// $$
    /// \prod left_i = right
    /// $$
    ///
    /// It can only be called ONCE as this is the only non random check allowed.
    /// It panics if called more than once.
    pub fn merge_nonrandom(
        &self,
        left: Vec<<E as MultiMillerLoop>::Result>,
        right: <E as Engine>::Gt,
    ) {
        let randomize = self.non_random_check_done.load(SeqCst);
        self.merge_pair(left[0], right, randomize);
        for l in left[1..].iter() {
            self.merge_miller_one(*l, randomize);
        }
    }

    /// takes a vector of pairs elements to be passed down the miller loop and
    /// the expected right hand side of the equation
    ///
    /// $$
    /// FE ( \prod ML(A_i,B_i) ) = out
    /// $$
    /// where $FE$ is the final exponentiation and $ML$ is the miller loop.
    pub fn merge_miller_inputs<'a>(
        &self,
        it: &[(&'a E::G1Affine, &'a E::G2Affine)],
        out: &'a <E as Engine>::Gt,
    ) {
        let must_randomize = self.non_random_check_done.load(SeqCst);
        let coeff = {
            let rng: &mut R = &mut self.rng.lock().unwrap();
            derive_non_zero::<E, _>(rng)
        };
        self.merge(
            PairingCheck::new_random_from_miller_inputs(coeff, it, out),
            must_randomize,
        );
    }

    fn merge(&self, check: PairingCheck<E>, must_randomize: bool) {
        if !check.randomized {
            assert!(
                !must_randomize,
                "Cannot merge non-randomized check with must_randomize true."
            );
            self.non_random_check_done.store(true, SeqCst);
        };

        // This send is "best effort". If the verification in `verify_tipp_mipp()` identifies
        // an invalid aggregation, the `self.valid` is set to `false`. That terminates the thread
        // that receives those messages, hence also the receiving channel is closed.
        // This means that if the aggrigation is invalid, it is expected that the message cannot
        // be sent.
        let sent = self.merge_send.send(Ok(check));
        if sent.is_err() && self.valid.load(SeqCst) {
            panic!("Channel was closed although it is still valid.")
        }
    }

    pub fn verify(self) -> Result<bool, SynthesisError> {
        let Self {
            valid,
            merge_send,
            valid_recv,
            ..
        } = self;

        drop(merge_send); // stop the merge process

        if !valid.load(SeqCst) {
            return Ok(false);
        }
        valid_recv.recv().unwrap()
    }
}

/// PairingCheck represents a check of the form e(A,B)e(C,D)... = T. Checks can
/// be aggregated together using random linear combination. The efficiency comes
/// from keeping the results from the miller loop output before proceding to a final
/// exponentiation when verifying if all checks are verified.
/// It is a tuple:
/// - a miller loop result that is to be multiplied by other miller loop results
/// before going into a final exponentiation result
/// - a right side result which is already in the right subgroup Gt which is to
/// be compared to the left side when "final_exponentiatiat"-ed
#[derive(Debug)]
struct PairingCheck<E>
where
    E: MultiMillerLoop,
{
    left: <E as MultiMillerLoop>::Result,
    right: <E as Engine>::Gt,
    randomized: bool,
}

impl<E> PairingCheck<E>
where
    E: MultiMillerLoop,
{
    fn new() -> PairingCheck<E> {
        Self {
            left: <E as MultiMillerLoop>::Result::default(),
            right: <E as Engine>::Gt::generator(),
            randomized: false,
        }
    }

    /// Returns a pairing check from the output of the miller pairs and the expected
    /// right hand side such that the following must hold:
    /// $$
    /// \prod res = exp
    /// $$
    ///
    /// Note the check is NOT randomized and there must be only up to ONE check only that can not
    /// be randomized when merging.
    fn from_pair(
        result: <E as MultiMillerLoop>::Result,
        exp: <E as Engine>::Gt,
    ) -> PairingCheck<E> {
        Self {
            left: result,
            right: exp,
            randomized: false,
        }
    }

    fn from_miller_one(result: <E as MultiMillerLoop>::Result) -> PairingCheck<E> {
        Self {
            left: result,
            right: <E as Engine>::Gt::generator(),
            randomized: false,
        }
    }

    /// returns a pairing tuple that is scaled by a random element.
    /// When aggregating pairing checks, this creates a random linear
    /// combination of all checks so that it is secure. Specifically
    /// we have e(A,B)e(C,D)... = out <=> e(g,h)^{ab + cd} = out
    /// We rescale using a random element $r$ to give
    /// e(rA,B)e(rC,D) ... = out^r <=>
    /// e(A,B)^r e(C,D)^r = out^r <=> e(g,h)^{abr + cdr} = out^r
    /// (e(g,h)^{ab + cd})^r = out^r
    pub fn new_random_from_miller_inputs<'a>(
        coeff: E::Fr,
        it: &[(&'a E::G1Affine, &'a E::G2Affine)],
        out: &'a <E as Engine>::Gt,
    ) -> PairingCheck<E> {
        let miller_out = it
            .into_par_iter()
            .map(|(a, b)| {
                let na = a.mul(coeff).to_affine();
                (na, (**b).into())
            })
            .map(|(a, b)| E::multi_miller_loop(&[(&a, &b)]))
            .fold(<E as MultiMillerLoop>::Result::default, |acc, res| {
                acc + res
            })
            .reduce(<E as MultiMillerLoop>::Result::default, |acc, res| {
                acc + res
            });
        let right = if out != &<E as Engine>::Gt::generator() {
            // we only need to make this expensive operation is the output is
            // not one since 1^r = 1
            *out * coeff
        } else {
            *out
        };

        PairingCheck {
            left: miller_out,
            right,
            randomized: true,
        }
    }

    /// takes another pairing tuple and combine both sides together. Note the checks are not
    /// randomized when merged, the checks must have been randomized before.
    pub fn merge(&mut self, p2: &PairingCheck<E>) {
        self.left += &p2.left;
        add_if_not_one_gt::<E>(&mut self.right, &p2.right);

        // A merged PairingCheck is only randomized if both of its contributors are.
        self.randomized = self.randomized && p2.randomized;
    }

    fn verify(&self) -> bool {
        let left = self.left.final_exponentiation();
        left == self.right
    }
}

fn add_if_not_one_gt<E: Engine>(left: &mut E::Gt, right: &E::Gt) {
    let one = E::Gt::generator();
    if left == &one {
        *left = *right;
        return;
    } else if right == &one {
        // nothing to do here
        return;
    }
    *left += right
}

fn derive_non_zero<E: Engine, R: rand_core::RngCore>(rng: &mut R) -> E::Fr {
    loop {
        let coeff = E::Fr::random(&mut *rng);
        if coeff != E::Fr::zero() {
            return coeff;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use blstrs::{Bls12, G1Projective, G2Projective};
    use group::Group;
    use rand_core::RngCore;
    use rand_core::SeedableRng;

    fn gen_pairing_check<R: RngCore>(r: &mut R) -> PairingCheck<Bls12> {
        let g1r = G1Projective::random(&mut *r).to_affine();
        let g2r = G2Projective::random(&mut *r).to_affine();
        let exp = Bls12::pairing(&g1r, &g2r);
        let coeff = derive_non_zero::<Bls12, _>(r);
        let tuple =
            PairingCheck::<Bls12>::new_random_from_miller_inputs(coeff, &[(&g1r, &g2r)], &exp);
        assert!(tuple.verify());
        tuple
    }

    #[test]
    fn test_pairing_randomize() {
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let tuples = (0..3)
            .map(|_| gen_pairing_check(&mut rng))
            .collect::<Vec<_>>();
        let final_tuple = tuples
            .iter()
            .fold(PairingCheck::<Bls12>::new(), |mut acc, tu| {
                acc.merge(&tu);
                acc
            });
        assert!(final_tuple.verify());
    }
}
