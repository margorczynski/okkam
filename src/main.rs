extern crate core;

mod ga;
mod polynomial;
mod common;

use crate::polynomial::polynomial::{Term, Polynomial};

fn main() {
    let data: Vec<(f32, f32)> = (0..=100)
        .map(|x| (x as f32, (x as f32).powi(2)))
        .collect();

    let terms_num = 5;
    let degree_bits_num = 3;

    for (input, result) in &data {
    }

    let poly = Polynomial {
        terms: vec![
            Term { coefficient: 4.0, degrees: vec![1] },
        ],
        constant: 3.0,
    };

    // 4*x^1 + 3

    let chromosome = poly.to_chromosome(4);

    let poly_from = Polynomial::from_chromosome(1, 4, 1, &chromosome);
    
    println!("Output: {}", chromosome);
}
