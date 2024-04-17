extern crate core;

mod ga;
mod polynomial;
mod common;

use std::io::{stdout, Result};
use std::collections::HashSet;
use std::cmp::Ordering;

use crossterm::{
    event::{self, KeyCode, KeyEventKind},
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
    ExecutableCommand,
};

use crate::polynomial::polynomial::Polynomial;
use crate::ga::chromosome::Chromosome;
use crate::ga::chromosome_with_fitness::ChromosomeWithFitness;
use crate::ga::ga::*;

fn main() -> Result<()> {
    let data: Vec<(Vec<f32>, f32)> = (1..=10)
        .map(|x| (vec![x as f32, x as f32 + 2.0f32], (x as f32).powi(2)*(x as f32 + 2.0f32 ).sqrt() + 4.0 * (x as f32).ln()))
        .collect();

    let terms_num = 8;
    let degree_bits_num = 5;
    let degree_num = 1;

    let epsilon = 0.05;

    let chromosome_bit_len = Polynomial::get_bits_needed(terms_num, degree_bits_num, degree_num);

    let mut population = generate_initial_population(512, chromosome_bit_len);

    let mut plot_iter = 0;

    let mut lowest_err = f32::INFINITY;

    //GA loop
    loop {
        //Use rank instead as f32 is not Eq + the GA algo doesn't care about the amount of error, just if it's better/worse than the other
        let mut chromosomes_with_error: Vec<(Chromosome, f32)> = Vec::new();
        for chromosome in &population {
            //TODO: Simplify makes it converge much slower and stop converging faster, why?
            let polynomial = Polynomial::from_chromosome(terms_num, degree_bits_num, degree_num, chromosome);//.simplify();
            let mut mean_squared_err = 0.0;
            for (inputs, output) in &data {
                let res = polynomial.evaluate(inputs);
                let diff = output - res;

                mean_squared_err += diff * diff;
            }

            mean_squared_err = mean_squared_err/ (data.len() as f32);

            if mean_squared_err < lowest_err {
                let rel_error = mean_squared_err / data.iter().map(|d| d.1).sum::<f32>();
                lowest_err = mean_squared_err;
                println!("ITER {}, Lowest: {}, rel_error: {:.2}%", plot_iter, lowest_err, rel_error * 100.0f32);
            }

            if mean_squared_err <= epsilon {
                println!("Found: {}", polynomial);
            }

            let pair = (chromosome.clone(), mean_squared_err);

            chromosomes_with_error.push(pair);
        }

        chromosomes_with_error.sort_by(|a, b| {
            let a_is_nan = a.1.is_nan();
            let b_is_nan = b.1.is_nan();
        
            if a_is_nan && b_is_nan {
                Ordering::Equal
            } else if a_is_nan {
                Ordering::Greater
            } else if b_is_nan {
                Ordering::Less
            } else {
                a.1.partial_cmp(&b.1).unwrap()
            }
        });

        let chromosomes_with_fitness: HashSet<ChromosomeWithFitness<u32>> =
        chromosomes_with_error
        .iter()
        .rev() //Reverse so the ones with the biggest error get the lowest index/rank
        .enumerate()
        .map(|(idx, (chromosome, _))| ChromosomeWithFitness::from_chromosome_and_fitness(chromosome.clone(), idx as u32))
        .collect::<HashSet<ChromosomeWithFitness<u32>>>();

        population = evolve(&chromosomes_with_fitness, SelectionStrategy::Tournament(5), 0.1f32, 0.1f32);

        plot_iter += 1;
    }
}