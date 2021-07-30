use ff::Field;
use std::fmt;
use std::ops::{Div, Sub};

#[derive(Clone)]
pub struct DensePolynomial<F: Field> {
    coeffs: Vec<F>,
}

impl<F: Field> fmt::Debug for DensePolynomial<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for (i, coeff) in self
            .coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| !bool::from(c.is_zero()))
        {
            if i == 0 {
                write!(f, "\n{:?}", coeff)?;
            } else if i == 1 {
                write!(f, " + \n{:?} * x", coeff)?;
            } else {
                write!(f, " + \n{:?} * x^{}", coeff, i)?;
            }
        }
        Ok(())
    }
}

impl<F: Field> DensePolynomial<F> {
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        let mut result = Self { coeffs };
        // While there are zeros at the end of the coefficient vector, pop them off.
        result.truncate_leading_zeros();
        // Check that either the coefficients vec is empty or that the last coeff is
        // non-zero.
        assert!(result
            .coeffs
            .last()
            .map_or(true, |coeff| !bool::from(coeff.is_zero())));
        result
    }

    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    pub fn into_coeffs(self) -> Vec<F> {
        self.coeffs
    }

    /// Returns the total degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            assert!(self
                .coeffs
                .last()
                .map_or(false, |coeff| !bool::from(coeff.is_zero())));
            self.coeffs.len() - 1
        }
    }

    /// Checks if the given polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|coeff| coeff.is_zero().into())
    }

    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().map_or(false, |c| c.is_zero().into()) {
            self.coeffs.pop();
        }
    }
}

impl<'a, 'b, F: Field> Sub<&'a DensePolynomial<F>> for &'b DensePolynomial<F> {
    type Output = DensePolynomial<F>;

    fn sub(self, other: &'a DensePolynomial<F>) -> DensePolynomial<F> {
        let mut result = if self.is_zero() {
            let mut result = other.clone();
            for coeff in &mut result.coeffs {
                *coeff = -*coeff;
            }
            result
        } else if other.is_zero() {
            self.clone()
        } else if self.degree() >= other.degree() {
            let mut result = self.clone();
            for (a, b) in result.coeffs.iter_mut().zip(&other.coeffs) {
                a.sub_assign(b);
            }
            result
        } else {
            let mut result = self.clone();
            result.coeffs.resize(other.coeffs.len(), F::zero());
            for (a, b) in result.coeffs.iter_mut().zip(&other.coeffs) {
                a.sub_assign(b);
            }
            result
        };
        result.truncate_leading_zeros();
        result
    }
}

impl<'a, 'b, F: Field> Div<&'a DensePolynomial<F>> for &'b DensePolynomial<F> {
    type Output = DensePolynomial<F>;

    fn div(self, divisor: &'a DensePolynomial<F>) -> DensePolynomial<F> {
        if self.is_zero() {
            DensePolynomial::zero()
        } else if divisor.is_zero() {
            panic!("Dividing by zero polynomial")
        } else if self.degree() < divisor.degree() {
            DensePolynomial::zero()
        } else {
            // Now we know that self.degree() >= divisor.degree();
            let mut quotient = vec![F::zero(); self.degree() - divisor.degree() + 1];
            let mut remainder: DensePolynomial<F> = self.clone();
            // Can unwrap here because we know self is not zero.
            let divisor_leading_inv = divisor.coeffs.last().unwrap().invert().unwrap();
            while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
                let mut cur_q_coeff = *remainder.coeffs.last().unwrap();
                cur_q_coeff.mul_assign(&divisor_leading_inv);
                let cur_q_degree = remainder.degree() - divisor.degree();
                quotient[cur_q_degree] = cur_q_coeff;

                for (i, div_coeff) in divisor.coeffs.iter().enumerate() {
                    let mut x = cur_q_coeff;
                    x.mul_assign(div_coeff);
                    remainder.coeffs[cur_q_degree + i].sub_assign(&x);
                }
                while let Some(true) = remainder.coeffs.last().map(|c| c.is_zero().into()) {
                    remainder.coeffs.pop();
                }
            }
            DensePolynomial::from_coeffs(quotient)
        }
    }
}
