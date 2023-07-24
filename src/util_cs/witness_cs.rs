//! Support for efficiently generating R1CS witness using bellperson.

use ff::PrimeField;

use crate::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

pub trait SizedWitness<Scalar: PrimeField> {
    fn num_constraints(&self) -> usize;
    fn num_inputs(&self) -> usize;
    fn num_aux(&self) -> usize;

    fn generate_witness_into(&mut self, aux: &mut [Scalar], inputs: &mut [Scalar]) -> Scalar;
    fn generate_witness(&mut self) -> (Vec<Scalar>, Vec<Scalar>, Scalar) {
        let aux_count = self.num_aux();
        let inputs_count = self.num_inputs();

        let mut aux = Vec::with_capacity(aux_count);
        let mut inputs = Vec::with_capacity(inputs_count);

        aux.resize(aux_count, Scalar::ZERO);
        inputs.resize(inputs_count, Scalar::ZERO);

        let result = self.generate_witness_into(&mut aux, &mut inputs);

        (aux, inputs, result)
    }

    fn generate_witness_into_cs<CS: ConstraintSystem<Scalar>>(&mut self, cs: &mut CS) -> Scalar {
        assert!(cs.is_witness_generator());

        let aux_count = self.num_aux();
        let inputs_count = self.num_inputs();

        let (aux, inputs) = cs.allocate_empty(aux_count, inputs_count);

        assert_eq!(aux.len(), aux_count);
        assert_eq!(inputs.len(), inputs_count);

        self.generate_witness_into(aux, inputs)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
pub struct WitnessCS<Scalar>
where
    Scalar: PrimeField,
{
    // Assignments of variables
    pub(crate) input_assignment: Vec<Scalar>,
    pub(crate) aux_assignment: Vec<Scalar>,
}

impl<Scalar> ConstraintSystem<Scalar> for WitnessCS<Scalar>
where
    Scalar: PrimeField,
{
    type Root = Self;

    fn new() -> Self {
        let input_assignment = vec![Scalar::ONE];

        Self {
            input_assignment,
            aux_assignment: vec![],
        }
    }

    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.aux_assignment.push(f()?);

        Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.input_assignment.push(f()?);

        Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, _a: LA, _b: LB, _c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        // Do nothing: we don't care about linear-combination evaluations in this context.
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Extensible
    fn is_extensible() -> bool {
        true
    }

    fn extend(&mut self, other: &Self) {
        self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
        self.aux_assignment.extend(&other.aux_assignment);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Witness generator
    fn is_witness_generator(&self) -> bool {
        true
    }

    fn extend_inputs(&mut self, new_inputs: &[Scalar]) {
        self.input_assignment.extend(new_inputs);
    }

    fn extend_aux(&mut self, new_aux: &[Scalar]) {
        self.aux_assignment.extend(new_aux);
    }

    fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
        let allocated_aux = {
            let i = self.aux_assignment.len();
            self.aux_assignment.resize(aux_n + i, Scalar::ZERO);
            &mut self.aux_assignment[i..]
        };

        let allocated_inputs = {
            let i = self.input_assignment.len();
            self.input_assignment.resize(inputs_n + i, Scalar::ZERO);
            &mut self.input_assignment[i..]
        };

        (allocated_aux, allocated_inputs)
    }

    fn inputs_slice(&self) -> &[Scalar] {
        &self.input_assignment
    }

    fn aux_slice(&self) -> &[Scalar] {
        &self.aux_assignment
    }
}

impl<Scalar: PrimeField> WitnessCS<Scalar> {
    pub fn scalar_inputs(&self) -> Vec<Scalar> {
        self.input_assignment.clone()
    }

    pub fn scalar_aux(&self) -> Vec<Scalar> {
        self.aux_assignment.clone()
    }
}
