extern crate core;

mod config;
mod ga;
mod polynomial;
mod util;
mod ui;

use std::fs::File;
use std::io::Result;
use std::collections::HashSet;
use std::cmp::Ordering;
use std::sync::mpsc::{Receiver, Sender};

use clap::Parser;
use log::info;
use rayon::prelude::*;
use tokio::time::Instant;
use ui::ui::{App, Message};
use util::util::Dataset;

use crate::config::okkam_config::OkkamConfig;
use crate::ui::ui::run_ui;
use crate::util::util::{dataset_from_csv, setup};
use crate::util::args::Args;
use crate::polynomial::polynomial::Polynomial;
use crate::ga::chromosome::Chromosome;
use crate::ga::chromosome_with_fitness::ChromosomeWithFitness;
use crate::ga::ga::*;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    //Load config
    let config = OkkamConfig::new(&args.config_path).unwrap();

    //Setup logging
    setup(config.log_level.0, args.headless);

    //Load dataset from CSV file
    let csv_file = File::open(config.dataset_path.as_ref())?;
    let dataset = dataset_from_csv(csv_file, false, ',').unwrap();

    info!("Starting Okkam with the following configuration:");
    info!("{:?}", config);

    if args.headless {
        info!("Strating in headless mode...");
        search_loop(&config, &dataset, None, None)
    } else {
        info!("Starting terminal UI...");
        let computation = move |tx, rx| search_loop(&config, &dataset, Some(tx), Some(rx)); 
        run_ui(computation).unwrap();
    }

    Ok(())
}

fn search_loop(okkam_config: &OkkamConfig, dataset: &Dataset, tx_o: Option<Sender<Message>>, rx_o: Option<Receiver<Message>>) {
    //Get the number of variables and calculate chromosome bit length
    let variable_num = dataset.first().unwrap().0.len();
    let chromosome_bit_len = Polynomial::get_bits_needed(okkam_config.polynomial.terms_num, okkam_config.polynomial.degree_bits_num, variable_num);

    //Generate initial population and initialize basic values
    let mut population = generate_initial_population(okkam_config.ga.population_size, chromosome_bit_len);
    let mut iteration = 0;
    let mut lowest_err: f32 = f32::INFINITY;
    let loop_start = Instant::now();

    info!("Starting main GA search loop");

    loop {
        match &rx_o {
            Some(rx) => {
                match rx.try_recv() {
                    Ok(Message::Quit) => break,
                    _ => ()
                }
            },
            None => ()
        }

        //Use rank instead as f32 is not Eq + the GA algo doesn't care about the amount of error, just if it's better/worse than the other
        let mut chromosomes_with_diffs: Vec<(&Chromosome, Vec<f32>, f32)> = population
        .par_iter()
        .map(|chromosome| {
            let polynomial = Polynomial::from_chromosome(okkam_config.polynomial.terms_num, okkam_config.polynomial.degree_bits_num, variable_num, chromosome);
            let diffs: Vec<f32> = dataset.iter().map(|(inputs, output)| (polynomial.evaluate(inputs) - output).abs()).collect();
            let sum = diffs.iter().sum();
            (chromosome, diffs, sum)
        })
        .collect();

        chromosomes_with_diffs
        .par_sort_by(|a, b| {
            let a_diff_sum: f32 = a.2;
            let b_diff_sum: f32 = b.2;

            let a_is_nan = a_diff_sum.is_nan();
            let b_is_nan = b_diff_sum.is_nan();
        
            if a_is_nan && b_is_nan {
                Ordering::Equal
            } else if a_is_nan {
                Ordering::Greater
            } else if b_is_nan {
                Ordering::Less
            } else {
                a_diff_sum.partial_cmp(&b_diff_sum).unwrap()
            }
        });

        let lowest_err_chromosome = chromosomes_with_diffs.first().unwrap();
        let total_abs_error = lowest_err_chromosome.2;

        if total_abs_error < lowest_err {
            let n = dataset.len() as f32;
            let new_state = App {
                iteration: iteration,
                avg_duration_per_iteration: loop_start.elapsed() / (iteration + 1) as u32,
                best_mae: total_abs_error / n,
                best_mape: (100.0f32 * total_abs_error) / (n * dataset.iter().map(|(_, expected)| expected).sum::<f32>()),
                best_rmse: (lowest_err_chromosome.1.iter().map(|diff| diff.powi(2)).sum::<f32>() / n).sqrt(),
            };

            info!("{:?}", new_state);

            match &tx_o {
                Some(tx) =>
                    tx.send(Message::UpdateState(new_state)).unwrap(),
                None => ()
            }

            lowest_err = total_abs_error;
        }

        let chromosomes_with_fitness: HashSet<ChromosomeWithFitness<u32>> =
        chromosomes_with_diffs
        .iter()
        .rev() //Reverse so the ones with the biggest error get the lowest index/rank
        .enumerate()
        .map(|(idx, (chromosome, _, _))| ChromosomeWithFitness::from_chromosome_and_fitness((*chromosome).clone(), idx as u32))
        .collect::<HashSet<ChromosomeWithFitness<u32>>>();

        population = evolve(&chromosomes_with_fitness, SelectionStrategy::Tournament(okkam_config.ga.tournament_size), okkam_config.ga.mutation_rate, okkam_config.ga.elite_factor);

        iteration += 1;
    }
}