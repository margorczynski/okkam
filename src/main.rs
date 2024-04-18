extern crate core;

mod ga;
mod polynomial;
mod common;

use std::io::Result;
use std::collections::HashSet;
use std::cmp::Ordering;

use rayon::prelude::*;
use tokio::time::Instant;

use crate::polynomial::polynomial::Polynomial;
use crate::ga::chromosome::Chromosome;
use crate::ga::chromosome_with_fitness::ChromosomeWithFitness;
use crate::ga::ga::*;

fn main() -> Result<()> {

    let test_func = |x: f32, y: f32, z: f32| -> f32 {
        (x.powi(2) + x*y*z + y.sqrt()*z.ln() - z.powi(3)) * (x + y + z).sqrt()
    };

    let data: Vec<(Vec<f32>, f32)> = (1..=100)
        .map(|x| (vec![x as f32, x as f32, x as f32], test_func(x as f32, x as f32, x as f32)))
        .collect();

    let terms_num = 14;
    let degree_bits_num = 3;
    let degree_num = 3;

    let population_size = 1024;
    let tournament_size = 15;
    let mutation_rate = 0.1f32;
    let elite_factor = 0.1f32;

    let chromosome_bit_len = Polynomial::get_bits_needed(terms_num, degree_bits_num, degree_num);

    let mut population = generate_initial_population(population_size, chromosome_bit_len);

    let mut generation_idx = 0;

    let mut lowest_err: f32 = f32::INFINITY;

    let loop_start = Instant::now();

    println!("terms_num={}, degree_bits_num={}, degree_num={}, chromosome_bit_len={}", terms_num, degree_bits_num, degree_num, chromosome_bit_len);

    //GA loop
    loop {
        //Use rank instead as f32 is not Eq + the GA algo doesn't care about the amount of error, just if it's better/worse than the other
        let mut chromosomes_with_error: Vec<(&Chromosome, f32)> = population
        .par_iter()
        .map(|chromosome| {
            let polynomial = Polynomial::from_chromosome(terms_num, degree_bits_num, degree_num, chromosome);
            (chromosome, data.iter().map(|(inputs, output)| (polynomial.evaluate(inputs) - output).powi(2)).sum::<f32>().sqrt() / data.len() as f32)
        })
        .collect();

        chromosomes_with_error.par_sort_by(|a, b| {
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

        let curr_lower_err = chromosomes_with_error.first().unwrap().1;

        if curr_lower_err < lowest_err {
            let rel_error = curr_lower_err / data.iter().map(|d| d.1.powi(2)).sum::<f32>().sqrt();
            lowest_err = curr_lower_err;
            let avg_time = loop_start.elapsed() / (generation_idx + 1);
            println!("Generation: {}, Lowest Error: {}, Error (%): {:.3}%, Avg Time per loop: {:?}", generation_idx, lowest_err, rel_error * 100.0f32, avg_time);
        }

        let chromosomes_with_fitness: HashSet<ChromosomeWithFitness<u32>> =
        chromosomes_with_error
        .iter()
        .rev() //Reverse so the ones with the biggest error get the lowest index/rank
        .enumerate()
        .map(|(idx, (chromosome, _))| ChromosomeWithFitness::from_chromosome_and_fitness((*chromosome).clone(), idx as u32))
        .collect::<HashSet<ChromosomeWithFitness<u32>>>();

        population = evolve(&chromosomes_with_fitness, SelectionStrategy::Tournament(tournament_size), mutation_rate, elite_factor);

        generation_idx += 1;
    }
}