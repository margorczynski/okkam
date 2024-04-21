use std::fmt::{Debug, Display, Formatter};

use half::prelude::*;
use itertools::Itertools;
use rand_distr::num_traits::Pow;

use crate::{
    ga::chromosome::Chromosome,
    util::util::{bits_to_bit_vec_u16, bits_to_bit_vec_u8, bits_to_u16, bits_to_u8},
};

#[derive(Clone)]
pub struct Term {
    pub coefficient: f16,
    //TODO: Change to degrees
    pub degrees: Vec<u8>,
}

#[derive(Clone)]
pub struct Polynomial {
    pub terms: Vec<Term>,
    pub constant: f16,
}

impl Polynomial {
    pub fn get_bits_needed(term_num: usize, degree_bits_num: usize, variable_num: usize) -> usize {
        return term_num * (16 + degree_bits_num * variable_num) + 16;
    }

    #[allow(dead_code)]
    pub fn to_chromosome(&self, degree_bits_num: usize) -> Chromosome {
        let size = Self::get_bits_needed(self.terms.len(), degree_bits_num, self.terms.first().unwrap().degrees.len());
        let mut genes = vec![0u64; size/64 + 1];
        let mut bit_count = 0;

        // Encode the terms
        for term in &self.terms {
            // Encode the coefficient
            let coefficient_bits = term.coefficient.to_bits() as u64;
            for bit_pos in 0..16u64 {
                let coefficient_bit = (coefficient_bits >> bit_pos) & 1;

                genes[bit_count / 64] |= coefficient_bit  << (bit_count % 64);
                bit_count += 1;
            }

            // Encode the exponents
            for exponent in &term.degrees {
                for bit_pos in 0..degree_bits_num {
                    let exponent_bit = (exponent >> bit_pos) & 1;
    
                    genes[bit_count / 64] |= (exponent_bit as u64) << (bit_count % 64);
                    bit_count += 1;
                }
            }
        }

        let constant_bits = self.constant.to_bits() as u64;
        for bit_pos in 0..16u64 {
            let constant_bit = (constant_bits >> bit_pos) & 1;

            genes[bit_count / 64] |= constant_bit  << (bit_count % 64);
            bit_count += 1;
        }

        Chromosome { size, genes }
    }

    pub fn from_chromosome(
        term_num: usize,
        degree_bits_num: usize,
        variable_num: usize,
        chromosome: &Chromosome,
    ) -> Polynomial {
        let mut terms = Vec::new();
        let mut bit_count = 0;

        for _ in 0..term_num {
            let mut exponents: Vec<u8> = Vec::with_capacity(variable_num);
            // Extract the coefficient bits

            let mut coefficient_bits = 0u64;
            for bit_pos in 0..16u64 {
                let coefficient_bit = chromosome.genes[bit_count / 64] >> (bit_count % 64) & 1;

                coefficient_bits |= coefficient_bit  << bit_pos;
                bit_count += 1;
            }

            for _ in 0..variable_num {
                let mut exponent = 0u64;
                for bit_pos in 0..degree_bits_num {
                    let exponent_bit = chromosome.genes[bit_count / 64] >> (bit_count % 64) & 1;
    
                    exponent |= exponent_bit  << bit_pos;
                    bit_count += 1;
                }

                exponents.push(exponent as u8);
            }

            terms.push(Term {
                coefficient: f16::from_bits(coefficient_bits as u16),
                degrees: exponents,
            });
        }

        let mut constant_bits = 0u64;
        for bit_pos in 0..16u64 {
            let constant_bit = chromosome.genes[bit_count / 64] >> (bit_count % 64) & 1;

            constant_bits |= constant_bit  << bit_pos;
            bit_count += 1;
        }

        Polynomial { terms, constant: f16::from_bits(constant_bits as u16) }
    }

    #[allow(dead_code)]
    pub fn simplify(&self) -> Polynomial {
        let (constant_term, reduced_terms): (Vec<Term>, Vec<Term>) = self
            .terms
            .iter()
            .group_by(|term| term.degrees.clone())
            .into_iter()
            .map(|grp| {
                grp.1
                    .cloned()
                    .reduce(|reduced_term, term| Term {
                        coefficient: reduced_term.coefficient + term.coefficient,
                        degrees: reduced_term.degrees,
                    })
                    .unwrap()
            })
            .filter(|term| term.coefficient.is_normal())
            .partition(|term| term.degrees.iter().all(|degree| *degree != 1u8));

        Polynomial {
            terms: reduced_terms,
            constant: self.constant
                + constant_term
                    .first()
                    .map(|term| term.coefficient)
                    .unwrap_or(f16::from_f32(0.0f32)),
        }
    }

    pub fn evaluate(&self, inputs: &[f64]) -> f64 {
        //Assume inputs.len == every term.degrees len
        let mut output = self.constant.to_f64();

        output += self
            .terms
            .iter()
            .map(|term| {
                term.degrees
                    .iter()
                    .zip(inputs.iter())
                    .map(|(degree, input)| input.powi(*degree as i32))
                    .product::<f64>()
                    * term.coefficient.to_f64()
            })
            .sum::<f64>();

        output
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        if self.constant != other.constant {
            return false;
        }

        if self.terms.len() != other.terms.len() {
            return false;
        }

        for (self_term, other_term) in self.terms.iter().zip(other.terms.iter()) {
            if self_term.coefficient != other_term.coefficient {
                return false;
            }

            if self_term.degrees.len() != other_term.degrees.len() {
                return false;
            }

            for (left_degree, right_degree) in
                self_term.degrees.iter().zip(other_term.degrees.iter())
            {
                if left_degree != right_degree {
                    return false;
                }
            }
        }

        return true;
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut term_str = if self.coefficient.is_normal() {
            if self.coefficient == f16::from_f32(1.0) {
                String::new()
            } else {
                self.coefficient.to_string()
            }
        } else {
            return write!(f, "");
        };

        if self.degrees.is_empty() {
            return write!(f, "");
        }

        for (i, degree) in self.degrees.iter().enumerate() {
            if *degree == 0 {
                continue;
            }

            let mut var = format!("x{}", i);

            if *degree > 1 {
                var.push_str(&format!("^{}", *degree))
            }

            if !term_str.is_empty() {
                term_str.push_str("*")
            }

            term_str.push_str(&var);
        }

        write!(f, "{}", term_str)
    }
}

impl Display for Polynomial {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let _poly_str = String::new();

        let mut term_strings: Vec<String> =
            self.terms.iter().map(|term| term.to_string()).collect();

        if self.constant != f16::from_f32(0.0f32) {
            term_strings.push(self.constant.to_string());
        }

        write!(f, "{}", term_strings.join(" + "))
    }
}

impl Debug for Term {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Debug for Polynomial {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify() {
        //2*x0^2*x1 + 3*x0^2*x1 - 8*x0^2*x1 - x1 + x1 + 5 + 1
        let polynomial = Polynomial {
            terms: vec![
                Term {
                    coefficient: f16::from_f32(2.0),
                    degrees: vec![2, 1],
                },
                Term {
                    coefficient: f16::from_f32(3.0),
                    degrees: vec![2, 1],
                },
                Term {
                    coefficient: f16::from_f32(-8.0),
                    degrees: vec![2, 1],
                },
                Term {
                    coefficient: f16::from_f32(-1.0),
                    degrees: vec![0, 1],
                },
                Term {
                    coefficient: f16::from_f32(1.0),
                    degrees: vec![0, 1],
                },
                Term {
                    coefficient: f16::from_f32(0.0),
                    degrees: vec![1, 0],
                },
                Term {
                    coefficient: f16::from_f32(5.0),
                    degrees: vec![0, 0],
                },
            ],
            constant: f16::from_f32(1.0),
        };

        let simplified = polynomial.simplify();

        //5*x0^2*x1 - x1 + 6
        let expected_polynomial = Polynomial {
            terms: vec![Term {
                coefficient: f16::from_f32(-3.0),
                degrees: vec![2, 1],
            }],
            constant: f16::from_f32(6.0),
        };

        assert_eq!(simplified, expected_polynomial);
    }

    #[test]
    fn test_evaluate() {
        // Test case 1: Constant polynomial
        let poly = Polynomial {
            constant: f16::from_f32(5.0),
            terms: Vec::new(),
        };
        assert_eq!(poly.evaluate(&[]), 5.0);

        // Test case 2: Linear polynomial
        let poly = Polynomial {
            constant: f16::from_f32(2.0),
            terms: vec![Term {
                coefficient: f16::from_f32(3.0),
                degrees: vec![1],
            }],
        };
        assert_eq!(poly.evaluate(&[4.0]), 14.0);

        // Test case 3: Quadratic polynomial
        let poly = Polynomial {
            constant: f16::from_f32(1.0),
            terms: vec![
                Term {
                    coefficient: f16::from_f32(2.0),
                    degrees: vec![2],
                },
                Term {
                    coefficient: f16::from_f32(3.0),
                    degrees: vec![1],
                },
            ],
        };
        assert_eq!(poly.evaluate(&[2.0]), 15.0);

        // Test case 4: Multivariate polynomial
        let poly = Polynomial {
            constant: f16::from_f32(1.0),
            terms: vec![
                Term {
                    coefficient: f16::from_f32(2.0),
                    degrees: vec![1, 2],
                },
                Term {
                    coefficient: f16::from_f32(3.0),
                    degrees: vec![2, 1],
                },
            ],
        };
        //2*2*3^2 + 3*2^2*3 + 1 = 36 + 36 + 1 = 73
        assert_eq!(poly.evaluate(&[2.0, 3.0]), 73.0);
    }

    #[test]
    fn test_to_chromosome() {
        let p1 = Polynomial {
            terms: vec![
                Term {
                    coefficient: f16::from_f32(5.0),
                    degrees: vec![2, 1],
                },
                Term {
                    coefficient: f16::from_f32(-3.0),
                    degrees: vec![1, 0],
                },
            ],
            constant: f16::from_f32(7.0),
        };

        let p2 = Polynomial {
            terms: vec![
                Term {
                    coefficient: f16::from_f32(5.0),
                    degrees: vec![2, 1],
                },
                Term {
                    coefficient: f16::from_f32(-3.0),
                    degrees: vec![1, 0],
                },
                Term {
                    coefficient: f16::from_f32(20.0),
                    degrees: vec![2, 3],
                },
                Term {
                    coefficient: f16::from_f32(4.0),
                    degrees: vec![2, 0],
                },
            ],
            constant: f16::from_f32(7.0),
        };

        let chromosome1 = p1.to_chromosome(2);
        let chromosome2 = p2.to_chromosome(2);

        let expected_chromosome1 = Chromosome {
            size: 56,
            genes: vec![
                0b01000111000000000001110000100000000001100100010100000000,
            ],
        };

        let expected_chromosome2 = Chromosome {
            size: 96,
            genes: vec![
                0b0000111001001101000000000001110000100000000001100100010100000000,
                0b01000111000000000010010001000000
            ],
        };


        assert_eq!(chromosome1, expected_chromosome1);
        assert_eq!(chromosome2, expected_chromosome2);
    }

    #[test]
    fn test_to_from_chromosome() {
        // 3.0*x0^3*x1*x2^5 + 15.0*x0*x1^4 + x2 + 1.0

        let term1 = Term {
            coefficient: f16::from_f32(3.0),
            degrees: vec![3, 1, 5],
        };
        let term2 = Term {
            coefficient: f16::from_f32(15.0),
            degrees: vec![1, 4, 0],
        };
        let term3 = Term {
            coefficient: f16::from_f32(1.0),
            degrees: vec![0, 0, 1],
        };
        let poly = Polynomial {
            terms: vec![term1, term2, term3],
            constant: f16::from_f32(1.0),
        };

        let chromosome = poly.to_chromosome(4);

        let from_chromosome = Polynomial::from_chromosome(poly.terms.len(), 4, 3, &chromosome);

        //Each term is 16 + 4*3 bits = 28 bits
        //Constant is 16, total = 28 * 3 + 16 = 100 bits

        // Check that the chromosome has the correct length
        assert_eq!(100, Polynomial::get_bits_needed(3, 4, 3));
        assert_eq!(poly, from_chromosome);
    }
}
