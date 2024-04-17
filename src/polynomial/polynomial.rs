use std::fmt::{Debug, Display, Formatter};

use crate::ga::chromosome::Chromosome;

#[derive(Clone)]
pub struct Term {
    pub coefficient: f32,
    pub degrees: Vec<u32>
}

#[derive(Clone)]
pub struct Polynomial {
    pub terms: Vec<Term>,
    pub constant: f32,
}

impl Polynomial {
    // degree_bits_num: usize
    pub fn to_chromosome(&self, degree_bits_num: usize) -> Chromosome {
        let mut genes = Vec::new();

        // Encode the terms
        for term in &self.terms {
            // Encode the coefficient
            let coefficient_bits = term.coefficient.to_bits();
            genes.append(&mut Self::bits_to_bit_vec(&coefficient_bits));
            
            // Encode the degrees
            for degree in &term.degrees {
                let degree_bit_vec = Self::bits_to_bit_vec(degree);
                let degree_bit_vec_limited = degree_bit_vec.iter().rev().take(degree_bits_num).rev();
                genes.extend(degree_bit_vec_limited);
            }
        }
    
        // Encode the constant
        let constant_bits = self.constant.to_bits();
        genes.append(&mut Self::bits_to_bit_vec(&constant_bits));
    
        Chromosome { genes }
    }

    pub fn from_chromosome(term_num: usize, degree_bits_num: usize, degree_num: usize, chromosome: &Chromosome) -> Polynomial {
        let mut terms = Vec::new();
        let mut gene_index = 0;
    
        // Calculate the number of bits required for each term
        let coefficient_bits = 32; // 32 bits for f32
        let degree_bits = degree_bits_num; // 32 bits for u32
    
        for _ in 0..term_num {
            let mut degrees = Vec::with_capacity(degree_num);
            // Extract the coefficient bits
            let coefficient_bits_vec = chromosome.genes[gene_index..(gene_index + coefficient_bits)].to_vec();
            gene_index += coefficient_bits;
            let coefficient = f32::from_bits(Self::bits_to_u32(&coefficient_bits_vec));

            for _ in 0..degree_num {
                let degree_bits_vec = chromosome.genes[gene_index..(gene_index + degree_bits_num)].to_vec();
                gene_index += degree_bits_num;
                degrees.push(Self::bits_to_u32(&degree_bits_vec));
            }
    
            terms.push(Term { coefficient, degrees });
        }
    
        // Extract the constant bits
        let constant_bits_vec = chromosome.genes[gene_index..].to_vec();
        let constant = f32::from_bits(Self::bits_to_u32(&constant_bits_vec));
    
        Polynomial { terms, constant }
    }

    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        //Assume inputs.len == every term.degrees len
        let mut output = self.constant;

        for term in &self.terms {
            let mut term_value = term.coefficient;

            for (degree, input) in term.degrees.iter().zip(inputs.iter()) {
                term_value *= input.powi(*degree as i32);
            }

            output += term_value;
        }

        output
    }

    fn bits_to_bit_vec(bits: &u32) -> Vec<bool> {
        let mut result = Vec::new();
    
        for i in (0..32).rev() {
            result.push(((bits >> i) & 1) == 1);
        }
    
        result
    }

    fn bits_to_u32(bits: &[bool]) -> u32 {
        let mut result = 0;
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                result |= 1 << (bits.len() - 1 - i);
            }
        }
        result
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
            if(self_term.coefficient != other_term.coefficient) {
                return false;
            }

            if self_term.degrees.len() != other_term.degrees.len() {
                return false;
            }

            for (left_degree, right_degree) in self_term.degrees.iter().zip(other_term.degrees.iter()) {
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
            if self.coefficient == 1.0f32 {
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
        let mut poly_str = String::new();

        let mut term_strings: Vec<String> = self.terms
        .iter()
        .map(|term| term.to_string())
        .collect();

        if self.constant != 0.0 {
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
    fn test_evaluate() {
        // Test case 1: Constant polynomial
        let poly = Polynomial {
            constant: 5.0,
            terms: Vec::new(),
        };
        assert_eq!(poly.evaluate(&[]), 5.0);

        // Test case 2: Linear polynomial
        let poly = Polynomial {
            constant: 2.0,
            terms: vec![Term {
                coefficient: 3.0,
                degrees: vec![1],
            }],
        };
        assert_eq!(poly.evaluate(&[4.0]), 14.0);

        // Test case 3: Quadratic polynomial
        let poly = Polynomial {
            constant: 1.0,
            terms: vec![
                Term {
                    coefficient: 2.0,
                    degrees: vec![2],
                },
                Term {
                    coefficient: 3.0,
                    degrees: vec![1],
                },
            ],
        };
        assert_eq!(poly.evaluate(&[2.0]), 15.0);

        // Test case 4: Multivariate polynomial
        let poly = Polynomial {
            constant: 1.0,
            terms: vec![
                Term {
                    coefficient: 2.0,
                    degrees: vec![1, 2],
                },
                Term {
                    coefficient: 3.0,
                    degrees: vec![2, 1],
                },
            ],
        };
        //2*2*3^2 + 3*2^2*3 + 1 = 36 + 36 + 1 = 73
        assert_eq!(poly.evaluate(&[2.0, 3.0]), 73.0);
    }

    #[test]
    fn test_bits_to_bit_vec() {
        // Test case 1: All bits are 0
        let bits = 0u32;
        let expected = vec![false; 32];
        assert_eq!(Polynomial::bits_to_bit_vec(&bits), expected);

        // Test case 2: All bits are 1
        let bits = 0xFFFF_FFFFu32;
        let expected = vec![true; 32];
        assert_eq!(Polynomial::bits_to_bit_vec(&bits), expected);

        // Test case 3: Alternating bits
        let bits = 0x5555_5555u32;
        let expected = vec![
            false, true, false, true, false, true, false, true, false, true, false, true, false,
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, true, false, true, false, true,
        ];
        assert_eq!(Polynomial::bits_to_bit_vec(&bits), expected);
    }

    #[test]
    fn test_bits_to_u32() {
        // Test case 1: Empty vector
        let bits: Vec<bool> = vec![];
        assert_eq!(Polynomial::bits_to_u32(&bits), 0);

        // Test case 2: All bits are false
        let bits = vec![false; 32];
        assert_eq!(Polynomial::bits_to_u32(&bits), 0);

        // Test case 3: All bits are true
        let bits = vec![true; 32];
        assert_eq!(Polynomial::bits_to_u32(&bits), 0xFFFF_FFFF);

        // Test case 4: Alternating bits
        let bits = vec![
            false, true, false, true, false, true, false, true, false, true, false, true, false,
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, true, false, true, false, true,
        ];
        assert_eq!(Polynomial::bits_to_u32(&bits), 0x5555_5555);
    }

    #[test]
    fn test_to_from_chromosome() {

        // 3.0*x0^3*x1*x2^5 + 15.0*x0*x1^4 + x2 + 1.0

        let term1 = Term {
            coefficient: 3.0,
            degrees: vec![3, 1, 5],
        };
        let term2 = Term {
            coefficient: 15.0,
            degrees: vec![1,4,0],
        };
        let term3 = Term {
            coefficient: 1.0,
            degrees: vec![0,0,1],
        };
        let poly = Polynomial {
            terms: vec![term1, term2, term3],
            constant: 1.0,
        };

        let chromosome = poly.to_chromosome(4);

        let from_chromosome = Polynomial::from_chromosome(poly.terms.len(), 4, 3, &chromosome);

        //Each term is 32 + 4*3 bits = 44 bits
        //Constant is 32, total = 44 * 3 + 32 = 164 bits

        // Check that the chromosome has the correct length
        assert_eq!(chromosome.genes.len(), 164);
        assert_eq!(poly, from_chromosome);
    }
}