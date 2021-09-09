//! An interface for dealing with the kinds of parallel computations involved in
//! `bellperson`.

use std::env;

use crossbeam_channel::{bounded, Receiver};
use lazy_static::lazy_static;
use yastl::Pool;

lazy_static! {
    static ref NUM_CPUS: usize = read_num_cpus();
    pub static ref THREAD_POOL: Pool = Pool::new(*NUM_CPUS);
}

fn read_num_cpus() -> usize {
    match env::var("BELLMAN_NUM_CPUS")
        .ok()
        .and_then(|num| num.parse::<usize>().ok())
    {
        Some(num) => {
            log::warn!("BELLMAN_NUM_CPUS is deprecated, please switch to RAYON_NUM_THREADS");
            // proxy to RAYON_NUM_THREAS for now
            env::set_var("RAYON_NUM_THREADS", num.to_string());

            num
        }
        None => {
            match env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|num| num.parse().ok())
            {
                Some(num) => {
                    // rayon defaults to the same value as num_cpus::get
                    num
                }
                None => num_cpus::get(),
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct Worker {}

impl Worker {
    pub fn new() -> Worker {
        Worker {}
    }

    pub fn log_num_cpus(&self) -> u32 {
        log2_floor(*NUM_CPUS)
    }

    pub fn compute<F, R>(&self, f: F) -> Waiter<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (sender, receiver) = bounded(1);

        THREAD_POOL.spawn(move || {
            let res = f();
            sender.send(res).unwrap();
        });

        Waiter { receiver }
    }

    pub fn scope<'a, F, R>(&self, elements: usize, f: F) -> R
    where
        F: FnOnce(&yastl::Scope<'a>, usize) -> R,
    {
        let chunk_size = if elements < *NUM_CPUS {
            1
        } else {
            elements / *NUM_CPUS
        };

        THREAD_POOL.scoped(|scope| f(scope, chunk_size))
    }

    /// Executes the passed in function, and returns the result once it is finished.
    pub fn scoped<'a, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&yastl::Scope<'a>) -> R,
    {
        let (sender, receiver) = bounded(1);
        THREAD_POOL.scoped(|s| {
            let res = f(s);
            sender.send(res).unwrap();
        });

        receiver.recv().unwrap()
    }
}

pub struct Waiter<T> {
    receiver: Receiver<T>,
}

impl<T> Waiter<T> {
    /// Wait for the result.
    pub fn wait(&self) -> T {
        self.receiver.recv().unwrap()
    }

    /// One off sending.
    pub fn done(val: T) -> Self {
        let (sender, receiver) = bounded(1);
        sender.send(val).unwrap();

        Waiter { receiver }
    }
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_log2_floor() {
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(3), 1);
        assert_eq!(log2_floor(4), 2);
        assert_eq!(log2_floor(5), 2);
        assert_eq!(log2_floor(6), 2);
        assert_eq!(log2_floor(7), 2);
        assert_eq!(log2_floor(8), 3);
    }
}
