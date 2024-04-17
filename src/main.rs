extern crate core;

mod ga;
mod polynomial;
mod common;

use std::collections::HashSet;
use std::cmp::Ordering;

use plotters::prelude::*;

use crate::polynomial::polynomial::Polynomial;
use crate::ga::chromosome::Chromosome;
use crate::ga::chromosome_with_fitness::ChromosomeWithFitness;
use crate::ga::ga::*;

fn main() {
    let data: Vec<(Vec<f32>, f32)> = (1..=100)
        .map(|x| (vec![x as f32], (x as f32).ln()))
        .collect();

    let terms_num = 5;
    let degree_bits_num = 3;
    let degree_num = 1;

    let epsilon = 0.1;

    let chromosome_bit_len = Polynomial::get_bits_needed(terms_num, degree_bits_num, degree_num);

    let mut population = generate_initial_population(128, chromosome_bit_len);

    let mut plot_iter = 0;

    //GA loop
    loop {
        //Use rank instead as f32 is not Eq + the GA algo doesn't care about the amount of error, just if it's better/worse than the other
        let mut chromosomes_with_error: Vec<(Chromosome, f32)> = Vec::new();
        let mut err_accum = 0.0f32;
        for chromosome in &population {
            let polynomial = Polynomial::from_chromosome(terms_num, degree_bits_num, degree_num, chromosome);
            let mut mean_squared_err = 0.0;
            for (inputs, output) in &data {
                let res = polynomial.evaluate(inputs);
                let diff = output - res;

                mean_squared_err += diff * diff;
            }

            mean_squared_err = mean_squared_err/ (data.len() as f32);

            if(mean_squared_err <= epsilon) {
                println!("Found: {}", polynomial);
                return;
            }

            err_accum += mean_squared_err;

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

        population = evolve(&chromosomes_with_fitness, SelectionStrategy::Tournament(5), 0.1f32);

        // Create a new plot for this iteration
        let root_area = BitMapBackend::new("plot_iter_{}.png", (640, 480)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root_area)
            .caption(format!("Iteration {}", plot_iter), ("sans-serif", 20).into_font())
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(0f32..100f32, 0f32..10000f32)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Input")
            .y_desc("Output")
            .draw()
            .unwrap();

        // Plot the data points
        chart
            .draw_series(
                data
                    .iter()
                    .map(|(x, y)| Circle::new((*x.first().unwrap(), *y), 2, &RED)),
            )
            .unwrap();

        // Plot the best polynomial found so far
        let best_chromosome = &chromosomes_with_error.first().unwrap().0;
        let best_polynomial = Polynomial::from_chromosome(terms_num, degree_bits_num, degree_num, best_chromosome);

        chart
            .draw_series(
                data
                    .iter()
                    .map(|(x, _)| {
                        let y = best_polynomial.evaluate(x);
                        Circle::new((*x.first().unwrap(), y), 2, &BLUE)
                    }),
            )
            .unwrap();

        // Save the plot to a file
        root_area.present().unwrap();

        plot_iter += 1;
    }
}
