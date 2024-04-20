extern crate core;

mod config;
mod ga;
mod polynomial;
mod ui;
mod util;

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Result;
use std::sync::mpsc::{Receiver, Sender};

use clap::Parser;
use csv::Writer;

use log::info;
use rayon::prelude::*;
use tokio::time::Instant;
use ui::ui::{App, Message};
use util::util::Dataset;

use crate::config::okkam_config::OkkamConfig;
use crate::ga::chromosome::Chromosome;
use crate::ga::chromosome_with_fitness::ChromosomeWithFitness;
use crate::ga::ga::*;
use crate::polynomial::polynomial::Polynomial;
use crate::ui::ui::run_ui;
use crate::util::args::Args;
use crate::util::util::{dataset_from_csv, setup};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    //Load config
    let config = OkkamConfig::new(&args.config_path).unwrap();

    //Setup logging
    setup(
        &config.log_level.0,
        Some(&config.log_directory),
        args.headless,
    );

    //Load dataset from CSV file
    let csv_file = File::open(config.dataset_path.as_ref())?;
    let dataset = dataset_from_csv(csv_file, false, ',').unwrap();

    //Prepare output file with the best found polynomials
    let result_file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(config.result_path.as_ref())
        .unwrap();
    let mut result_writer = Writer::from_writer(result_file);

    info!("Starting Okkam with the following configuration:");
    info!("{:?}", config);

    if args.headless {
        info!("Strating in headless mode...");
        search_loop(&config, &dataset, &mut result_writer, None, None)
    } else {
        info!("Starting terminal UI...");
        let computation =
            move |tx, rx| search_loop(&config, &dataset, &mut result_writer, Some(tx), Some(rx));
        run_ui(computation).unwrap();
    }

    Ok(())
}

fn search_loop(
    okkam_config: &OkkamConfig,
    dataset: &Dataset,
    result_writer: &mut Writer<File>,
    tx_o: Option<Sender<Message>>,
    rx_o: Option<Receiver<Message>>,
) {
    //Get the number of variables and calculate chromosome bit length
    let variable_num = dataset.first().unwrap().0.len();
    let chromosome_bit_len = Polynomial::get_bits_needed(
        okkam_config.polynomial.terms_num,
        okkam_config.polynomial.degree_bits_num,
        variable_num,
    );

    //Generate initial population and initialize basic values
    let mut population =
        generate_initial_population(okkam_config.ga.population_size, chromosome_bit_len);
    let mut iteration = 0;
    let mut lowest_err: f32 = f32::INFINITY;
    let loop_start = Instant::now();

    let header = get_header_record(okkam_config.polynomial.terms_num, variable_num);
    result_writer.write_record(header).unwrap();
    result_writer.flush().unwrap();

    info!("Starting main GA search loop");

    loop {
        match &rx_o {
            Some(rx) => match rx.try_recv() {
                Ok(Message::Quit) => break,
                _ => (),
            },
            None => (),
        }

        //Use rank instead as f32 is not Eq + the GA algo doesn't care about the amount of error, just if it's better/worse than the other
        let mut chromosomes_with_diffs: Vec<(&Chromosome, Polynomial, Vec<f32>, f32)> = population
            .par_iter()
            .map(|chromosome| {
                let polynomial = Polynomial::from_chromosome(
                    okkam_config.polynomial.terms_num,
                    okkam_config.polynomial.degree_bits_num,
                    variable_num,
                    chromosome,
                );
                let diffs: Vec<f32> = dataset
                    .iter()
                    .map(|(inputs, output)| (polynomial.evaluate(inputs) - output).abs())
                    .collect();
                let sum = diffs.iter().sum();
                (chromosome, polynomial.clone(), diffs, sum)
            })
            .collect();

        chromosomes_with_diffs.par_sort_by(|a, b| {
            let a_diff_sum: f32 = a.3;
            let b_diff_sum: f32 = b.3;

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
        let total_abs_error = lowest_err_chromosome.3;

        if total_abs_error < lowest_err {
            let n = dataset.len() as f32;
            let mae = total_abs_error / n;
            let mape = (100.0f32 * total_abs_error)
                / (n * dataset.iter().map(|(_, expected)| expected).sum::<f32>());
            let rmse = (lowest_err_chromosome
                .2
                .iter()
                .map(|diff| diff.powi(2))
                .sum::<f32>()
                / n)
                .sqrt();
            let new_state = App {
                iteration: iteration,
                avg_duration_per_iteration: loop_start.elapsed() / (iteration + 1) as u32,
                best_mae: mae,
                best_mape: mape,
                best_rmse: rmse,
            };

            info!("{:?}", new_state);

            match &tx_o {
                Some(tx) => tx.send(Message::UpdateState(new_state)).unwrap(),
                None => (),
            }

            //Write the record with the polynomial information to the CSV file
            let record = get_polynomial_record(&lowest_err_chromosome.1, mae, mape, rmse);
            result_writer.write_record(record).unwrap();
            result_writer.flush().unwrap();

            lowest_err = total_abs_error;
        }

        let chromosomes_with_fitness: HashSet<ChromosomeWithFitness<u32>> = chromosomes_with_diffs
            .iter()
            .rev() //Reverse so the ones with the biggest error get the lowest index/rank
            .enumerate()
            .map(|(idx, (chromosome, _, _, _))| {
                ChromosomeWithFitness::from_chromosome_and_fitness(
                    (*chromosome).clone(),
                    idx as u32,
                )
            })
            .collect::<HashSet<ChromosomeWithFitness<u32>>>();

        population = evolve(
            &chromosomes_with_fitness,
            SelectionStrategy::Tournament(okkam_config.ga.tournament_size),
            okkam_config.ga.mutation_rate,
            okkam_config.ga.elite_factor,
        );

        iteration += 1;
    }
}

fn get_header_record(terms_num: usize, variable_num: usize) -> Vec<String> {
    let mut polynomial_header_record: Vec<String> = Vec::new();

    for term_idx in 0..terms_num {
        polynomial_header_record.push(format!("coeff_{}", term_idx));
        for var_idx in 0..variable_num {
            polynomial_header_record.push(format!("exponent_{}_{}", term_idx, var_idx));
        }
    }

    polynomial_header_record.extend(vec![
        "constant".to_string(),
        "mae".to_string(),
        "mape".to_string(),
        "rmse".to_string(),
    ]);

    polynomial_header_record
}

fn get_polynomial_record(polynomial: &Polynomial, mae: f32, mape: f32, rmse: f32) -> Vec<String> {
    let mut polynomial_record: Vec<String> = Vec::new();

    for term in &polynomial.terms {
        polynomial_record.push(term.coefficient.to_string());
        polynomial_record.extend(term.degrees.iter().map(|degree| degree.to_string()));
    }

    polynomial_record.extend(vec![
        polynomial.constant.to_string(),
        mae.to_string(),
        mape.to_string(),
        rmse.to_string(),
    ]);

    polynomial_record
}
