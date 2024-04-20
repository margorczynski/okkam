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
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};

use clap::Parser;
use csv::Writer;

use log::{debug, info};
use rayon::prelude::*;
use tokio::time::Instant;
use ui::ui::{App, Message};
use util::util::Dataset;

use crate::config::error_measure::ErrorMeasure::*;
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
    info!("✓ Started Okkam with the following arguments: {:?}", args);
    info!("✓ Loaded config: {:?}", config);
    info!("✓ Initialized logging");

    //Load dataset from CSV file
    let csv_file = File::open(config.dataset_path.as_ref())?;
    let dataset = dataset_from_csv(csv_file, false, ',').unwrap();
    info!(
        "✓ Loaded {} rows from the input dataset, number of variables is {}",
        dataset.len(),
        dataset.first().unwrap().0.len()
    );

    //Prepare output file with the best found polynomials
    let result_path = Path::new(config.result_path.as_ref());
    let result_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(result_path)
        .unwrap();
    let mut result_writer = Writer::from_writer(result_file);
    info!(
        "✓ Result CSV file writer created, will persist result under: {}",
        result_path.canonicalize().unwrap().to_str().unwrap()
    );

    //Either start the UI or continue with just the CLI raw mode
    if args.headless {
        info!("✓ Starting in headless mode...");
        search_loop(&config, &dataset, &mut result_writer, None, None)
    } else {
        info!("✓ Starting terminal UI...");
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
    info!("✓ Starting the main search loop");
    //Get the number of variables and calculate chromosome bit length
    let variable_num = dataset.first().unwrap().0.len();
    let chromosome_bit_len = Polynomial::get_bits_needed(
        okkam_config.polynomial.terms_num,
        okkam_config.polynomial.degree_bits_num,
        variable_num,
    );
    debug!(
        "variable_num={}, chromosome_bit_len={}",
        variable_num, chromosome_bit_len
    );

    //Generate initial population and initialize basic values
    let mut population =
        generate_initial_population(okkam_config.ga.population_size, chromosome_bit_len);
    let mut iteration = 0;
    let mut lowest_err: f64 = f64::INFINITY;
    let loop_start = Instant::now();
    debug!(
        "Generated initial population with the size: {}",
        population.len()
    );

    let header = get_header_record(okkam_config.polynomial.terms_num, variable_num);
    result_writer.write_record(&header).unwrap();
    result_writer.flush().unwrap();
    debug!("Written header to CSV result file: {:?}", &header);

    loop {
        debug!("Search loop start. Iteration: {}", iteration);
        match &rx_o {
            Some(rx) => match rx.try_recv() {
                Ok(Message::Quit) => {
                    debug!("Loop received Quit message, quitting...");
                    break;
                }
                _ => (),
            },
            None => (),
        }

        //Extract the polynomials from the chromosomes and attach the calculated MAE, MAPE and RMSE
        let mut chromosomes_with_errors: Vec<(&Chromosome, Polynomial, f64, f64, f64)> = population
            .par_iter()
            .map(|chromosome| {
                let polynomial = Polynomial::from_chromosome(
                    okkam_config.polynomial.terms_num,
                    okkam_config.polynomial.degree_bits_num,
                    variable_num,
                    chromosome,
                );
                let abs_diffs: Vec<f64> = dataset
                    .iter()
                    .map(|(inputs, output)| (polynomial.evaluate(inputs) - output).abs())
                    .collect();

                let n = dataset.len() as f64;
                let mae = abs_diffs.iter().sum::<f64>() / n;
                let mape = (abs_diffs
                    .iter()
                    .zip(dataset.iter().map(|ds| ds.1))
                    .map(|(abs_diff, expected)| abs_diff / expected.abs())
                    .sum::<f64>()
                    * 100.0f64)
                    / n;
                let rmse = (abs_diffs.iter().map(|diff| diff.powi(2)).sum::<f64>() / n).sqrt();
                (chromosome, polynomial.clone(), mae, mape, rmse)
            })
            .collect();
        debug!("chromosomes_with_errors calculated");

        chromosomes_with_errors.par_sort_by(|a, b| {
            let a_measure: f64;
            let b_measure: f64;

            match &okkam_config.minimized_error_measure {
                MAE => {
                    a_measure = a.2;
                    b_measure = b.2;
                }
                MAPE => {
                    a_measure = a.3;
                    b_measure = b.3;
                }
                RMSE => {
                    a_measure = a.4;
                    b_measure = b.4;
                }
            };
            debug!(
                "chromosomes_with_errors sorted with measure: {:?}",
                &okkam_config.minimized_error_measure
            );

            let a_is_nan = a_measure.is_nan();
            let b_is_nan = b_measure.is_nan();

            if a_is_nan && b_is_nan {
                Ordering::Equal
            } else if a_is_nan {
                Ordering::Greater
            } else if b_is_nan {
                Ordering::Less
            } else {
                a_measure.partial_cmp(&b_measure).unwrap()
            }
        });
        debug!("chromosomes_with_errors sorted");

        let lowest_err_chromosome = chromosomes_with_errors.first().unwrap();
        let best_mae = lowest_err_chromosome.2;
        let best_mape = lowest_err_chromosome.3;
        let best_rmse = lowest_err_chromosome.4;

        if best_mae < lowest_err {
            debug!(
                "New lowest MAE found. lowest_err={}, best_mae={}",
                lowest_err, best_mae
            );

            let new_state = App {
                iteration: iteration,
                avg_duration_per_iteration: loop_start.elapsed() / (iteration + 1) as u32,
                best_mae: best_mae,
                best_mape: best_mape,
                best_rmse: best_rmse,
            };

            info!("{:?}", new_state);

            match &tx_o {
                Some(tx) => tx.send(Message::UpdateState(new_state)).unwrap(),
                None => (),
            }

            //Write the record with the polynomial information to the CSV file
            let record =
                get_polynomial_record(&lowest_err_chromosome.1, best_mae, best_mape, best_rmse);
            result_writer.write_record(record).unwrap();
            result_writer.flush().unwrap();
            debug!("Saved new best result to CSV");

            lowest_err = best_mae;
        }

        let chromosomes_with_fitness: HashSet<ChromosomeWithFitness<u32>> = chromosomes_with_errors
            .iter()
            .rev() //Reverse so the ones with the biggest error get the lowest index/rank
            .enumerate()
            .map(|(idx, (chromosome, _, _, _, _))| {
                ChromosomeWithFitness::from_chromosome_and_fitness(
                    (*chromosome).clone(),
                    idx as u32,
                )
            })
            .collect::<HashSet<ChromosomeWithFitness<u32>>>();
        debug!("chromosomes_with_fitness calculated");

        population = evolve(
            &chromosomes_with_fitness,
            SelectionStrategy::Tournament(okkam_config.ga.tournament_size),
            okkam_config.ga.mutation_rate,
            okkam_config.ga.elite_factor,
        );
        debug!("Population set to new evolution result");

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

fn get_polynomial_record(polynomial: &Polynomial, mae: f64, mape: f64, rmse: f64) -> Vec<String> {
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
