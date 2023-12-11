//! Prover implementation implemented using SupraSeal (C++).

use std::{cmp, collections::BTreeMap, io, ops, thread, time::Instant};

use bellpepper_core::{Circuit, ConstraintSystem, Index, SynthesisError, Variable};
use ec_gpu_gen::multiexp_cpu::DensityTracker;
use ff::{Field, PrimeField};
use log::info;
use pairing::MultiMillerLoop;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{ParameterSource, Proof, ProvingAssignment};
use crate::{gpu::GpuName, BELLMAN_VERSION};

/// The number of circuits that will synthesized in parallel.
///
/// Due to a memory optimized representation it's possible to synthesize circuits in bigger batches
/// than proving them. That optimized representation will then be transformed into the one the
/// prover expects in a separate step.
const SYNTHESIZE_BATCH_SIZE: usize = 20;

/// The number of synthesized circuits that are passed on to the prover. Those need a lot of memory
/// and the proving is mostly sequentially anyway, which means that bigger sized won't result in
/// much faster proving times. Lower memory usage is usally worth the trade-off.
const PROVER_BATCH_SIZE: usize = 5;

/// The number of scalars we pack into a single byte.
const SCALARS_PER_BYTE: usize = 4;

/// An enum to distinguish between common and other scalar values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarValue {
    Zero = 0,
    One = 1,
    Two = 2,
    Other = 3,
}

impl Default for ScalarValue {
    fn default() -> Self {
        Self::Zero
    }
}

/// Use a custom representation in order to use less memory. In Filecoin the synthesized exponents
/// are mostly zero, ones or twos. Those can be represented with 2 bits instead of their full field
/// representation of 256 bits. Other values have a slight overhead, but as there are so few, it
/// doesn't matter much.
#[derive(Debug, Eq, PartialEq)]
pub struct ScalarVec<Scalar> {
    /// The scalar representing zero. It's owned here so that it can be referenced later.
    zero: Scalar,
    /// The scalar representing one. It's owned here so that it can be referenced later.
    one: Scalar,
    /// The scalar representing two. It's owned here so that it can be referenced later.
    two: Scalar,
    /// This is the vector of all values. 4 values are packed into a single byte. If the value is
    /// [`ScalarValue::Other`], then there will be the actual value stored in the `other` field,
    /// keyed by the current position in the list of values (where the position is the one as if it
    /// wouldn't be packed).
    values: Vec<u8>,
    /// In case the value is [`ScalarValue::Other`], then the actual scalar is stored in this map,
    /// where the key the position within the list of values.
    other: BTreeMap<usize, Scalar>,
    /// Temporary buffer before the values are packed into a single byte.
    buffer: [ScalarValue; SCALARS_PER_BYTE],
    /// The offset where the next value within the buffer will be written to.
    buffer_pos: usize,
}

impl<Scalar: PrimeField> ScalarVec<Scalar> {
    pub fn new() -> Self {
        Self {
            zero: Scalar::ZERO,
            one: Scalar::ONE,
            two: Scalar::ONE.double(),
            values: Vec::new(),
            other: BTreeMap::new(),
            buffer: [ScalarValue::Zero; SCALARS_PER_BYTE],
            buffer_pos: 0,
        }
    }

    /// Tthe number of scalars stored.
    pub fn len(&self) -> usize {
        // The scalar values are 2 bit, we store 4 of them in a single byte.
        (self.values.len() * SCALARS_PER_BYTE) + self.buffer_pos
    }

    pub fn push(&mut self, scalar: Scalar) {
        let value = if scalar == Scalar::ZERO {
            ScalarValue::Zero
        } else if scalar == Scalar::ONE {
            ScalarValue::One
        } else if scalar == self.two {
            ScalarValue::Two
        } else {
            self.other.insert(self.len(), scalar);
            ScalarValue::Other
        };

        if self.buffer_pos < SCALARS_PER_BYTE {
            self.buffer[self.buffer_pos] = value;
            self.buffer_pos += 1;
        }

        // The buffer is full, flush the values into the actual data vector.
        if self.buffer_pos == SCALARS_PER_BYTE {
            self.buffer_pos = 0;
            self.flush_buffer();
        }
    }

    pub fn iter(&self) -> ScalarVecIterator<Scalar> {
        ScalarVecIterator {
            scalar_vec: self,
            pos: 0,
        }
    }

    /// Transform into arepresentation where all elements arranged in continuous memory.
    pub fn into_vec(self) -> Vec<Scalar> {
        // NOTE vmx 2023-12-13: A simple collect of the iterator is slower when micro-benchmarking.
        let mut output = Vec::with_capacity(self.len());
        for scalar in self.iter() {
            output.push(*scalar)
        }
        output
    }

    /// Flush the buffer into the actual vector of data.
    fn flush_buffer(&mut self) {
        let mut data_byte = 0;
        data_byte |= self.buffer[0] as u8;
        data_byte |= (self.buffer[1] as u8) << 2;
        data_byte |= (self.buffer[2] as u8) << 4;
        data_byte |= (self.buffer[3] as u8) << 6;
        self.values.push(data_byte);
    }

    fn get(&self, pos: usize) -> Option<&Scalar> {
        if pos < self.len() {
            // The position is within the stored values (not the buffer)
            if pos < self.values.len() * SCALARS_PER_BYTE {
                let value_byte = &self.values[pos / SCALARS_PER_BYTE];
                let within_buffer_pos = pos % SCALARS_PER_BYTE;
                // Determine where the bits we want to read. Each value is 2 bits => `* 2`.
                let bitmask = 0b11 << (within_buffer_pos * 2);
                // Read those bits and shift them back, so that it matches the enum values.
                let value = (value_byte & bitmask) >> (within_buffer_pos * 2);

                if value == ScalarValue::Zero as u8 {
                    Some(&self.zero)
                } else if value == ScalarValue::One as u8 {
                    Some(&self.one)
                } else if value == ScalarValue::Two as u8 {
                    Some(&self.two)
                } else if value == ScalarValue::Other as u8 {
                    self.other.get(&pos)
                } else {
                    unreachable!()
                }
            } else {
                let within_buffer_pos = pos - (self.values.len() * SCALARS_PER_BYTE);
                match self.buffer[within_buffer_pos] {
                    ScalarValue::Zero => Some(&self.zero),
                    ScalarValue::One => Some(&self.one),
                    ScalarValue::Two => Some(&self.two),
                    ScalarValue::Other => self.other.get(&pos),
                }
            }
        } else {
            None
        }
    }
}

impl<Scalar: PrimeField> Default for ScalarVec<Scalar> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, Scalar: PrimeField> Extend<&'a Scalar> for ScalarVec<Scalar> {
    fn extend<T: IntoIterator<Item = &'a Scalar>>(&mut self, iter: T) {
        for scalar in iter {
            self.push(*scalar);
        }
    }
}

impl<Scalar: PrimeField> ops::Index<usize> for ScalarVec<Scalar> {
    type Output = Scalar;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of range")
    }
}

pub struct ScalarVecIterator<'a, Scalar> {
    scalar_vec: &'a ScalarVec<Scalar>,
    pos: usize,
}

impl<'a, Scalar: PrimeField> Iterator for ScalarVecIterator<'a, Scalar> {
    type Item = &'a Scalar;

    fn next(&mut self) -> Option<Self::Item> {
        // Early return in case index is out of range.
        let value = self.scalar_vec.get(self.pos)?;
        self.pos += 1;
        Some(value)
    }
}

impl<'a, Scalar: PrimeField> IntoIterator for &'a ScalarVec<Scalar> {
    type Item = &'a Scalar;
    type IntoIter = ScalarVecIterator<'a, Scalar>;

    fn into_iter(self) -> Self::IntoIter {
        ScalarVecIterator {
            scalar_vec: self,
            pos: 0,
        }
    }
}

/// A copy of `[prover::ProvingAssignment` which has a lower memory footprint.
///
/// At the cost of the need to convert into the usual representation when it's passed into the
/// prover.
#[derive(Default)]
struct ProvingAssignmentCompact<Scalar: PrimeField> {
    // Density of queries
    a_aux_density: DensityTracker,
    b_input_density: DensityTracker,
    b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    a: ScalarVec<Scalar>,
    b: ScalarVec<Scalar>,
    c: ScalarVec<Scalar>,

    // Assignments of variables
    input_assignment: Vec<Scalar>,
    aux_assignment: ScalarVec<Scalar>,
}

super::proving_assignment_impls!(ProvingAssignmentCompact<Scalar>);

impl<Scalar: PrimeField> From<ProvingAssignmentCompact<Scalar>> for ProvingAssignment<Scalar> {
    fn from(assignment: ProvingAssignmentCompact<Scalar>) -> Self {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        let mut aux_assignment = Vec::new();
        rayon::scope(|s| {
            s.spawn(|_| a = assignment.a.into_vec());
            s.spawn(|_| b = assignment.b.into_vec());
            s.spawn(|_| c = assignment.c.into_vec());
            s.spawn(|_| aux_assignment = assignment.aux_assignment.into_vec());
        });

        Self {
            a_aux_density: assignment.a_aux_density,
            b_input_density: assignment.b_input_density,
            b_aux_density: assignment.b_aux_density,
            a,
            b,
            c,
            input_assignment: assignment.input_assignment,
            aux_assignment,
        }
    }
}

impl<Scalar> From<&ProvingAssignment<Scalar>> for supraseal_c2::Assignment<Scalar>
where
    Scalar: PrimeField,
{
    fn from(assignment: &ProvingAssignment<Scalar>) -> Self {
        assert_eq!(assignment.a.len(), assignment.b.len());
        assert_eq!(assignment.a.len(), assignment.c.len());

        Self {
            a_aux_density: assignment.a_aux_density.bv.as_raw_slice().as_ptr(),
            a_aux_bit_len: assignment.a_aux_density.bv.len(),
            a_aux_popcount: assignment.a_aux_density.get_total_density(),

            b_inp_density: assignment.b_input_density.bv.as_raw_slice().as_ptr(),
            b_inp_bit_len: assignment.b_input_density.bv.len(),
            b_inp_popcount: assignment.b_input_density.get_total_density(),

            b_aux_density: assignment.b_aux_density.bv.as_raw_slice().as_ptr(),
            b_aux_bit_len: assignment.b_aux_density.bv.len(),
            b_aux_popcount: assignment.b_aux_density.get_total_density(),

            a: assignment.a.as_ptr(),
            b: assignment.b.as_ptr(),
            c: assignment.c.as_ptr(),
            abc_size: assignment.a.len(),

            inp_assignment_data: assignment.input_assignment.as_ptr(),
            inp_assignment_size: assignment.input_assignment.len(),

            aux_assignment_data: assignment.aux_assignment.as_ptr(),
            aux_assignment_size: assignment.aux_assignment.len(),
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    randomization: Option<(Vec<E::Fr>, Vec<E::Fr>)>,
    _priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    info!(
        "Bellperson {} with SupraSeal is being used!",
        BELLMAN_VERSION
    );

    let (r_s, s_s) = randomization.unwrap_or((
        vec![E::Fr::ZERO; circuits.len()],
        vec![E::Fr::ZERO; circuits.len()],
    ));

    // The memory-optimized version, which is more CPU intensive only makes sense for larger batch
    // sizes. Hence use the normal synthesis for smaller batches.
    if circuits.len() <= 10 {
        let provers = super::synthesize_circuits_batch(circuits)?;
        proof_circuits_batch(provers, params, (r_s, s_s))
    } else {
        create_proof_batch_pipelined(circuits, params, (r_s, s_s))
    }
}

/// Create a custom [`SynthesisError`].
///
/// The closest to a custom error is the IO Error, hence use that.
fn custom_error(error: &str) -> SynthesisError {
    SynthesisError::IoError(io::Error::new(io::ErrorKind::Other, error))
}

/// The circuit synthesis is CPU intensive. Itself isn't parallelized, hence we parallelize with
/// running several synthesis at the same time. The proving isn't that CPU intensive.
/// Therefore we interleave the synthesis with the proving.
/// We create a large batch of synthesized circuits, and then proof in smaller batches as the
/// proving takes way more memory. Whenever the proving of synthesized batch starts, we kick of a
/// new batch for synthesis, while the proving is going on. We achieve that with having a bounded
/// message queue which blocks after a certain amount of batches.
///
/// The flow looks like that:
///
///   - Each uppercase letter corresponds to one proof.
///   - The total number of proofs is 18.
///   - The batch size for synthesis is 6.
///   - The batch size for proving is 2.
///   - The message queue size is the batch size of the synthesis divided bt the batch size of
///     the proving minus one, so that the queue blocks before the next synthesis starts.
///     => (6 / 2) - 1 = 2.
///
/// ```text
///  The downwards axis is time. The Synthesize and Prover thread run in parallel. If things
///  appeach on the same line it means that they start at the same time, but they might take
///  different amounts of time.
///
///  Description                             Synthesize thread    Message queue    Prover thread
///
///  The full set of proofs is:
///  A B C D E F G H I J K L M N O P Q R
///
///  Start with synthesizing a batch of         A B C D E F
///  circuits.
///
///  Once finished, put them into the                              (C D) (A B)
///  message queue. One item in the queue
///  consists is one batch for the prover.
///
///  Once the prover starts, the last item      G H I J K L        (E F) (C D)         A B
///  of the synthesis batch is pushed into                               (E F)         C D
///  queue, hence a new synthesis starts.                                              E F
///
///  The synthesis keeps pushing into the                          (I J) (G H)
///  queue whenever there's a free spot.
///
///  Keep repeating the previous two steps.     M N O P Q R        (K L) (I J)         G H
///                                                                      (K L)         I J
///
///                                                                (O P) (M N)
///
///  All sircuits were synthesized, hence                          (Q R) (O P)         M N
///  only the proving is to be done.                                     (Q R)         O P
///                                                                                    Q R
///  ```
fn create_proof_batch_pipelined<E, C, P>(
    circuits: Vec<C>,
    params: P,
    randomization: (Vec<E::Fr>, Vec<E::Fr>),
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
    P: ParameterSource<E>,
{
    let (r_s, s_s) = randomization;
    assert_eq!(circuits.len(), r_s.len());
    assert_eq!(circuits.len(), s_s.len());

    // This channel size makes sure that the next synthesizing batch starts as soon as the first
    // proving batch starts.
    let (sender, receiver) =
        crossbeam_channel::bounded((SYNTHESIZE_BATCH_SIZE / PROVER_BATCH_SIZE) - 1);

    let num_circuits = circuits.len();

    thread::scope(|s| {
        let synthesis = s.spawn(|| -> Result<(), SynthesisError> {
            let mut circuits_mut = circuits;
            // A vector of proofs is expected, hence drain it from the list of proofs, so that we
            // don't need to keep an extra copy around.
            while !circuits_mut.is_empty() {
                let size = cmp::min(SYNTHESIZE_BATCH_SIZE, circuits_mut.len());
                let batch = circuits_mut.drain(0..size).collect();
                let mut provers = synthesize_circuits_batch(batch)?;
                // Do not send all synthesized circuits at once, but only a subset as the memory
                // footprint will increase in the proving stage.
                while !provers.is_empty() {
                    let provers_size = cmp::min(PROVER_BATCH_SIZE, provers.len());
                    let provers_batch: Vec<_> = provers.drain(0..provers_size).collect();
                    sender
                        .send(provers_batch)
                        .map_err(|_| custom_error("cannot send circuits"))?;
                }
            }
            Ok(())
        });

        let prover = s.spawn(|| {
            let mut groth_proofs = Vec::with_capacity(num_circuits);
            // There is one randomnes element per circuit, hence we can use that as termination
            // condition for the loop.
            let mut r_s_mut = r_s;
            let mut s_s_mut = s_s;
            while !r_s_mut.is_empty() {
                let provers_compact = receiver
                    .recv()
                    .map_err(|_| custom_error("cannot receive circuits"))?;
                let r_s_batch = r_s_mut.drain(0..provers_compact.len()).collect();
                let s_s_batch = s_s_mut.drain(0..provers_compact.len()).collect();

                // Transform the provers from the memory efficient representation into one suitable
                // to be used with SupraSeal.
                log::trace!("converting representation of provers");
                let provers: Vec<ProvingAssignment<E::Fr>> =
                    provers_compact.into_par_iter().map(Into::into).collect();

                let proofs = proof_circuits_batch(provers, params.clone(), (r_s_batch, s_s_batch))?;
                groth_proofs.extend_from_slice(&proofs);
            }
            Ok(groth_proofs)
        });

        synthesis
            .join()
            .map_err(|_| custom_error("cannot prove circuits"))??;
        // The prover result is what we actually return.
        prover
            .join()
            .map_err(|_| custom_error("cannot prove circuits"))?
    })
}

fn proof_circuits_batch<E, P>(
    provers: Vec<ProvingAssignment<E::Fr>>,
    params: P,
    randomization: (Vec<E::Fr>, Vec<E::Fr>),
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
    P: ParameterSource<E>,
{
    // Start fft/multiexp prover timer
    let start = Instant::now();
    info!("starting proof timer");

    let num_circuits = provers.len();
    let (r_s, s_s) = randomization;

    // Make sure all circuits have the same input len.
    for prover in &provers {
        assert_eq!(
            prover.a.len(),
            provers[0].a.len(),
            "only equaly sized circuits are supported"
        );
    }

    let provers_c2: Vec<supraseal_c2::Assignment<E::Fr>> =
        provers.iter().map(|p| p.into()).collect();

    let mut proofs: Vec<Proof<E>> = Vec::with_capacity(num_circuits);
    // We call out to C++ code which is unsafe anyway, hence silence this warning.
    #[allow(clippy::uninit_vec)]
    unsafe {
        proofs.set_len(num_circuits);
    }

    let srs = params.get_supraseal_srs().ok_or_else(|| {
        log::error!("SupraSeal SRS wasn't allocated correctly");
        SynthesisError::MalformedSrs
    })?;
    supraseal_c2::generate_groth16_proofs(
        provers_c2.as_slice(),
        r_s.as_slice(),
        s_s.as_slice(),
        proofs.as_mut_slice(),
        srs,
    );

    let proof_time = start.elapsed();
    info!("prover time: {:?}", proof_time);

    Ok(proofs)
}

// The only difference to [`groth16::prover::synthesize_circuits-batch`] is, that it's using the
// memory optimized representation for the proving assignment.
fn synthesize_circuits_batch<Scalar, C>(
    circuits: Vec<C>,
) -> Result<Vec<ProvingAssignmentCompact<Scalar>>, SynthesisError>
where
    Scalar: PrimeField,
    C: Circuit<Scalar> + Send,
{
    let start = Instant::now();

    let provers = circuits
        .into_par_iter()
        .map(|circuit| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignmentCompact::new();

            prover.alloc_input(|| "", || Ok(Scalar::ONE))?;

            circuit.synthesize(&mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            Ok(prover)
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!("synthesis time: {:?}", start.elapsed());

    Ok(provers)
}
