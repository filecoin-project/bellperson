use std::convert::TryInto;
use std::ops::AddAssign;

use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Curve, Group};
use rayon::prelude::*;

pub const WINDOW_SIZE: usize = 8;

/// Abstraction over either a slice or a getter to produce a fixed number of scalars.
pub enum ScalarList<'a, G: PrimeCurveAffine, F: Fn(usize) -> <G::Scalar as PrimeField>::Repr> {
    Slice(&'a [<G::Scalar as PrimeField>::Repr]),
    Getter(F, usize),
}

impl<'a, G: PrimeCurveAffine, F: Fn(usize) -> <G::Scalar as PrimeField>::Repr>
    ScalarList<'a, G, F>
{
    pub fn len(&self) -> usize {
        match self {
            ScalarList::Slice(s) => s.len(),
            ScalarList::Getter(_, len) => *len,
        }
    }
}

pub type Getter<G> =
    dyn Fn(usize) -> <<G as PrimeCurveAffine>::Scalar as PrimeField>::Repr + Sync + Send;

/// Abstraction over owned and referenced multiscalar precomputations.
pub trait MultiscalarPrecomp<G: PrimeCurveAffine>: Send + Sync {
    fn window_size(&self) -> usize;
    fn window_mask(&self) -> u64;
    fn tables(&self) -> &[Vec<G>];
    fn at_point(&self, idx: usize) -> MultiscalarPrecompRef<'_, G>;
}

/// Owned variant of the multiscalar precomputations.
#[derive(Clone, Debug)]
pub struct MultiscalarPrecompOwned<G: PrimeCurveAffine> {
    num_points: usize,
    window_size: usize,
    window_mask: u64,
    table_entries: usize,
    tables: Vec<Vec<G>>,
}

impl<G: PrimeCurveAffine> PartialEq for MultiscalarPrecompOwned<G> {
    fn eq(&self, other: &Self) -> bool {
        self.num_points == other.num_points
            && self.window_size == other.window_size
            && self.window_mask == other.window_mask
            && self.table_entries == other.table_entries
            && self
                .tables
                .par_iter()
                .zip(other.tables.par_iter())
                .all(|(a, b)| a == b)
    }
}

impl<G: PrimeCurveAffine> MultiscalarPrecomp<G> for MultiscalarPrecompOwned<G> {
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn window_mask(&self) -> u64 {
        self.window_mask
    }

    fn tables(&self) -> &[Vec<G>] {
        &self.tables
    }

    fn at_point(&self, idx: usize) -> MultiscalarPrecompRef<'_, G> {
        MultiscalarPrecompRef {
            num_points: self.num_points - idx,
            window_size: self.window_size,
            window_mask: self.window_mask,
            table_entries: self.table_entries,
            tables: &self.tables[idx..],
        }
    }
}

/// Referenced version of the multiscalar precomputations.
#[derive(Debug)]
pub struct MultiscalarPrecompRef<'a, G: PrimeCurveAffine> {
    num_points: usize,
    window_size: usize,
    window_mask: u64,
    table_entries: usize,
    tables: &'a [Vec<G>],
}

impl<G: PrimeCurveAffine> MultiscalarPrecomp<G> for MultiscalarPrecompRef<'_, G> {
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn window_mask(&self) -> u64 {
        self.window_mask
    }

    fn tables(&self) -> &[Vec<G>] {
        self.tables
    }

    fn at_point(&self, idx: usize) -> MultiscalarPrecompRef<'_, G> {
        MultiscalarPrecompRef {
            num_points: self.num_points - idx,
            window_size: self.window_size,
            window_mask: self.window_mask,
            table_entries: self.table_entries,
            tables: &self.tables[idx..],
        }
    }
}

/// Precompute the tables for fixed bases.
pub fn precompute_fixed_window<G: PrimeCurveAffine>(
    points: &[G],
    window_size: usize,
) -> MultiscalarPrecompOwned<G> {
    let table_entries = (1 << window_size) - 1;
    let num_points = points.len();

    let tables = points
        .into_par_iter()
        .map(|point| {
            let mut table = Vec::with_capacity(table_entries);
            table.push(*point);

            let mut cur_precomp_point = point.to_curve();

            for _ in 1..table_entries {
                cur_precomp_point.add_assign(point);
                table.push(cur_precomp_point.to_affine());
            }

            table
        })
        .collect();

    MultiscalarPrecompOwned {
        num_points,
        window_size,
        window_mask: (1 << window_size) - 1,
        table_entries,
        tables,
    }
}

/// Multipoint scalar multiplication
/// Only supports window sizes that evenly divide a limb and nbits!!
pub fn multiscalar<G: PrimeCurveAffine>(
    k: &[<G::Scalar as ff::PrimeField>::Repr],
    precomp_table: &dyn MultiscalarPrecomp<G>,
    nbits: usize,
) -> G::Curve {
    // k is interpreted as having 64bit limbs
    debug_assert_eq!(
        std::mem::size_of::<<G::Scalar as ff::PrimeField>::Repr>() % 8,
        0
    );
    const BITS_PER_LIMB: usize = std::mem::size_of::<u64>() * 8;
    // TODO: support more bit sizes
    if nbits % precomp_table.window_size() != 0 || BITS_PER_LIMB % precomp_table.window_size() != 0
    {
        panic!("Unsupported multiscalar window size!");
    }

    let mut result = G::Curve::identity();

    // nbits must be evenly divided by window_size!
    let num_windows = (nbits + precomp_table.window_size() - 1) / precomp_table.window_size();
    let mut idx;

    // This version prefetches the next window and computes on the previous window.
    for i in (0..num_windows).rev() {
        let limb = (i * precomp_table.window_size()) / BITS_PER_LIMB;
        let window_in_limb = i % (BITS_PER_LIMB / precomp_table.window_size());

        for _ in 0..precomp_table.window_size() {
            result = result.double();
        }
        let mut prev_idx = 0;
        let mut prev_table: &Vec<G> = &precomp_table.tables()[0];
        let mut table: &Vec<G> = &precomp_table.tables()[0];

        for (m, point) in k.iter().enumerate() {
            let point_limb =
                u64::from_le_bytes(point.as_ref()[limb * 8..(limb + 1) * 8].try_into().unwrap());
            idx = point_limb >> (window_in_limb * precomp_table.window_size())
                & precomp_table.window_mask();
            if idx > 0 {
                table = &precomp_table.tables()[m];
                prefetch(&table[idx as usize - 1]);
            }
            if prev_idx > 0 && m > 0 {
                result.add_assign(&prev_table[prev_idx as usize - 1]);
            }
            prev_idx = idx;
            prev_table = table;
        }

        // Perform the final addition
        if prev_idx > 0 {
            result.add_assign(&prev_table[prev_idx as usize - 1]);
        }
    }

    result
}

/// Perform a threaded multiscalar multiplication and accumulation.
pub fn par_multiscalar<F, G: PrimeCurveAffine>(
    points: &ScalarList<'_, G, F>,
    precomp_table: &dyn MultiscalarPrecomp<G>,
    nbits: usize,
) -> G::Curve
where
    F: Fn(usize) -> <G::Scalar as PrimeField>::Repr + Sync,
{
    let num_points = points.len();

    // The granularity of work, in points. When a thread gets work it will
    // gather chunk_size points, perform muliscalar on them, and accumulate
    // the result. This is more efficient than evenly dividing the work among
    // threads because threads sometimes get preempted. When that happens
    // these long pole threads hold up progress across the board resulting in
    // occasional long delays.
    let mut chunk_size = 16; // TUNEABLE
    if num_points > 1024 {
        chunk_size = 256;
    }
    if chunk_size > num_points {
        chunk_size = 1; // fallback for tests and tiny inputs
    }

    let num_parts = (num_points + chunk_size - 1) / chunk_size;

    (0..num_parts)
        .into_par_iter()
        .map(|id| {
            // Temporary storage for scalars
            let mut scalar_storage: Vec<<G::Scalar as PrimeField>::Repr> = (0..chunk_size)
                .map(|_| <G::Scalar as PrimeField>::Repr::default())
                .collect();

            let start_idx = id * chunk_size;
            debug_assert!(start_idx < num_points);

            let mut end_idx = start_idx + chunk_size;
            if end_idx > num_points {
                end_idx = num_points;
            }

            let subset = precomp_table.at_point(start_idx);
            let scalars = match points {
                ScalarList::Slice(ref s) => &s[start_idx..end_idx],
                ScalarList::Getter(ref getter, _) => {
                    for i in start_idx..end_idx {
                        scalar_storage[i - start_idx] = getter(i);
                    }
                    &scalar_storage
                }
            };

            multiscalar(&scalars, &subset, nbits)
        }) // Accumulate results
        .reduce(G::Curve::identity, |mut acc, part| {
            acc.add_assign(&part);
            acc
        })
}

#[cfg(target_arch = "x86_64")]
fn prefetch<T>(p: *const T) {
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const _, core::arch::x86_64::_MM_HINT_T0);
    }
}

#[cfg(target_arch = "aarch64")]
fn prefetch<T>(p: *const T) {
    unsafe {
        use std::arch::aarch64::*;
        _prefetch(p as *const _, _PREFETCH_READ, _PREFETCH_LOCALITY3);
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn prefetch<T>(p: *const T) {}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ops::Mul;

    use blstrs::{G1Affine, G1Projective, Scalar as Fr};
    use ff::Field;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    fn multiscalar_naive(
        points: &[G1Affine],
        scalars: &[<Fr as PrimeField>::Repr],
    ) -> G1Projective {
        let mut acc = G1Projective::identity();
        for (scalar, point) in scalars.iter().zip(points.iter()) {
            let scalar = <Fr as PrimeField>::from_repr(*scalar).unwrap();
            acc.add_assign(&point.mul(scalar));
        }
        acc
    }

    #[test]
    fn test_multiscalar_single() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for _ in 0..50 {
            for (num_inputs, window_size) in &[(8, 4), (12, 1), (10, 1), (20, 2)] {
                let points: Vec<G1Affine> = (0..*num_inputs)
                    .map(|_| G1Projective::random(&mut rng).to_affine())
                    .collect();

                let scalars: Vec<<Fr as PrimeField>::Repr> = (0..*num_inputs)
                    .map(|_| Fr::random(&mut rng).to_repr())
                    .collect();

                let table = precompute_fixed_window::<G1Affine>(&points, *window_size);

                let naive_result = multiscalar_naive(&points, &scalars);
                let fast_result = multiscalar::<G1Affine>(
                    &scalars,
                    &table,
                    std::mem::size_of::<<Fr as PrimeField>::Repr>() * 8,
                );

                assert_eq!(naive_result, fast_result);
            }
        }
    }

    #[test]
    fn test_multiscalar_par() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for _ in 0..50 {
            for (num_inputs, window_size) in &[(8, 4), (12, 1), (10, 1), (20, 2)] {
                let points: Vec<G1Affine> = (0..*num_inputs)
                    .map(|_| G1Projective::random(&mut rng).to_affine())
                    .collect();

                let scalars: Vec<<Fr as PrimeField>::Repr> = (0..*num_inputs)
                    .map(|_| Fr::random(&mut rng).to_repr())
                    .collect();

                let table = precompute_fixed_window::<G1Affine>(&points, *window_size);

                let naive_result = multiscalar_naive(&points, &scalars);
                let fast_result = par_multiscalar::<&Getter<G1Affine>, G1Affine>(
                    &ScalarList::Slice(&scalars),
                    &table,
                    std::mem::size_of::<<Fr as PrimeField>::Repr>() * 8,
                );

                assert_eq!(naive_result, fast_result);
            }
        }
    }
}
