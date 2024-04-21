use std::collections::HashSet;

use itertools::Itertools;
use log::debug;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::Binomial;
use rayon::prelude::*;

use crate::ga::chromosome::Chromosome;
use crate::ga::chromosome_with_fitness::ChromosomeWithFitness;

#[derive(Debug)]
pub enum SelectionStrategy {
    Tournament(usize),
}

pub fn generate_initial_population(
    initial_population_count: usize,
    chromosome_size: usize,
) -> HashSet<Chromosome> {
    debug!(
        "Generating initial population - initial_population_count: {}, chromosome_size: {}",
        initial_population_count, chromosome_size
    );

    let mut rng = StdRng::from_entropy();
    let mut population: HashSet<Chromosome> = HashSet::new();

    let gene_blocks_needed = (chromosome_size / 64) + 1;

    println!("Gene block needed {}", gene_blocks_needed);

    let res = (0..initial_population_count).into_par_iter().map(|_| {
        let mut rng_clone = rng.clone();
        
        let random_genes = (0..gene_blocks_needed)
            .map(|_| rng_clone.gen::<u64>())
            .collect();

        Chromosome::from_genes(chromosome_size, random_genes)
    });

    population.par_extend(res);

    while population.len() < initial_population_count {
        let random_genes = (0..gene_blocks_needed).map(|_| rng.gen::<u64>()).collect();

        let chromosome = Chromosome::from_genes(chromosome_size, random_genes);

        population.insert(chromosome);
    }

    population
}

// TODO: Change to be done in-place? Use Vec and take fitness as separate param
pub fn evolve<T: PartialEq + PartialOrd + Clone + Eq + Send>(
    chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>,
    selection_strategy: SelectionStrategy,
    mutation_rate: f32,
    elite_factor: f32,
) -> HashSet<Chromosome> {
    debug!("Evolve new generation - chromosomes_with_fitness.len(): {}, selection_strategy: {:?}, mutation_rate: {}, elite_factor: {}", chromosomes_with_fitness.len(), selection_strategy, mutation_rate, elite_factor);
    let mut new_generation: HashSet<Chromosome> = HashSet::new();

    let elite_amount = ((chromosomes_with_fitness.len() as f32) * elite_factor).floor() as usize;

    debug!("Elite amount: {}", elite_amount);

    let chromosomes_with_fitness_ordered: Vec<ChromosomeWithFitness<T>> =
        chromosomes_with_fitness.iter().cloned().sorted().collect();

    let elite = chromosomes_with_fitness_ordered
        .par_iter()
        .rev()
        .take(elite_amount)
        .cloned()
        .map(|cwf| cwf.chromosome);

    new_generation.par_extend(elite);

    let offspring = (0..((chromosomes_with_fitness.len() - new_generation.len()) / 2))
        .into_par_iter()
        .map(|_| {
            let parents = select(chromosomes_with_fitness, &selection_strategy);
            let (offspring_1, offspring_2) = crossover(parents, 1.0f32, mutation_rate);
            vec![offspring_1, offspring_2]
        })
        .flatten();

    new_generation.par_extend(offspring);

    //Below is a fallback if the above would generate duplicates of already existing chromosomes
    //TODO: Use Vec instead and allow duplicates?
    while new_generation.len() != chromosomes_with_fitness.len() {
        let parents = select(chromosomes_with_fitness, &selection_strategy);
        let offspring = crossover(parents, 1.0f32, mutation_rate);

        new_generation.insert(offspring.0);
        if new_generation.len() == chromosomes_with_fitness.len() {
            break;
        }
        new_generation.insert(offspring.1);
    }

    debug!(
        "Total number of chromosomes after crossovers (+ elites retained): {}",
        new_generation.len()
    );

    new_generation
}

fn select<T: PartialEq + PartialOrd + Clone + Eq + Send>(
    chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>,
    selection_strategy: &SelectionStrategy,
) -> (Chromosome, Chromosome) {
    match *selection_strategy {
        SelectionStrategy::Tournament(tournament_size) => {
            let mut rng = SmallRng::from_entropy();
            let mut get_winner = |cwf: &HashSet<ChromosomeWithFitness<T>>| {
                cwf.iter()
                    .choose_multiple(&mut rng, tournament_size)
                    .into_iter()
                    .max()
                    .unwrap()
                    .clone()
            };

            let first = get_winner(&chromosomes_with_fitness);
            let second = get_winner(&chromosomes_with_fitness);

            (first.chromosome, second.chromosome)
        }
    }
}

fn crossover(
    parents: (Chromosome, Chromosome),
    _crossover_rate: f32,
    mutation_rate: f32,
) -> (Chromosome, Chromosome) {
    let mut rng = SmallRng::from_entropy();
    let chromosome_len = parents.0.size;

    let crossover_point = rng.gen_range(1..(chromosome_len - 1));

    //Calculate which blocks will go to the left and right children
    let crossover_gene_block_index = crossover_point / 64;

    //Important - for the right ones we need to take the tail as we'll need to manually split it using masks/bit shifts
    let (fst_left, fst_right_with_extra) = parents.0.genes.split_at(crossover_gene_block_index);
    let (snd_left, snd_right_with_extra) = parents.1.genes.split_at(crossover_gene_block_index);

    let mut fst_child_genes: Vec<u64> = Vec::new();
    let mut snd_child_genes: Vec<u64> = Vec::new();

    let bit_split_position_in_block = crossover_point % 64;
    let left_mask: u64 = u64::MAX >> bit_split_position_in_block;
    let right_mask = !left_mask;

    let fst_parent_block_after_split = parents.0.genes[crossover_gene_block_index] & left_mask;
    let snd_parent_block_after_split = parents.1.genes[crossover_gene_block_index] & right_mask;

    fst_child_genes.extend(fst_left);
    fst_child_genes.push(fst_parent_block_after_split);
    fst_child_genes.extend(snd_right_with_extra.iter().skip(1));

    snd_child_genes.extend(fst_right_with_extra.iter().skip(1));
    snd_child_genes.push(snd_parent_block_after_split);
    snd_child_genes.extend(snd_left);

    let binomial = Binomial::new(chromosome_len as u64, mutation_rate as f64).unwrap();
    let uniform = Uniform::new(0, chromosome_len);

    let mutated_genes_count_1 = binomial.sample(&mut rng) as usize;
    let mutated_genes_count_2 = binomial.sample(&mut rng) as usize;

    let mut fst_mutation_indices: HashSet<usize> = HashSet::new();
    let mut snd_mutation_indices: HashSet<usize> = HashSet::new();

    while fst_mutation_indices.len() != mutated_genes_count_1 {
        fst_mutation_indices.insert(uniform.sample(&mut rng));
    }
    while snd_mutation_indices.len() != mutated_genes_count_2 {
        snd_mutation_indices.insert(uniform.sample(&mut rng));
    }

    for mutated_idx in fst_mutation_indices {
        let gene_idx = mutated_idx / 64;
        let bit_idx = mutated_idx % 64;
        fst_child_genes[gene_idx] ^= 1 << bit_idx;
    }
    for mutated_idx in snd_mutation_indices {
        let gene_idx = mutated_idx / 64;
        let bit_idx = mutated_idx % 64;
        snd_child_genes[gene_idx] ^= 1 << bit_idx;
    }

    (
        Chromosome::from_genes(chromosome_len, fst_child_genes),
        Chromosome::from_genes(chromosome_len, snd_child_genes),
    )
}

#[cfg(test)]
mod evolution_tests {

    use crate::util::util::setup;

    use super::*;

    #[test]
    fn generate_initial_population_test() {
        setup(&log::LevelFilter::Debug, None, true);
        let result = generate_initial_population(100, 50);

        assert_eq!(result.len(), 100);
        assert_eq!(result.iter().all(|c| c.size == 50), true);
        assert_eq!(result.iter().all(|c| c.genes.len() <= 1), true)
    }

    #[test]
    fn evolve_test() {
        setup(&log::LevelFilter::Debug, None, true);
        let selection_strategy = SelectionStrategy::Tournament(4);
        let chromosomes_with_fitness = HashSet::from_iter(vec![
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //1110
                Chromosome::from_genes(4, vec![14]),
                0,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //1000
                Chromosome::from_genes(4, vec![8]),
                10,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //0100
                Chromosome::from_genes(4, vec![4]),
                15,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //0010
                Chromosome::from_genes(4, vec![2]),
                20,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //0001
                Chromosome::from_genes(4, vec![1]),
                25,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //1111
                Chromosome::from_genes(4, vec![15]),
                30,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //0000
                Chromosome::from_genes(4, vec![0]),
                40,
            ),
        ]);

        let result = evolve(&chromosomes_with_fitness, selection_strategy, 0.5, 0.5);

        debug!("Evo test result: {:?}", result);

        assert_eq!(result.len(), 7);
        assert!(result.contains(&Chromosome::from_genes(4, vec![15])));
        assert!(result.contains(&Chromosome::from_genes(4, vec![0])));
    }

    #[test]
    fn select_test() {
        setup(&log::LevelFilter::Debug, None, true);
        let selection_strategy = SelectionStrategy::Tournament(5);
        let chromosomes_with_fitness = HashSet::from_iter(vec![
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //1000
                Chromosome::from_genes(4, vec![8]),
                10,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //0100
                Chromosome::from_genes(4, vec![4]),
                15,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //0010
                Chromosome::from_genes(4, vec![2]),
                20,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //0001
                Chromosome::from_genes(4, vec![1]),
                25,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                //1111
                Chromosome::from_genes(4, vec![15]),
                30,
            ),
        ]);

        let result = select(&chromosomes_with_fitness, &selection_strategy);

        let results_set: HashSet<Chromosome> = HashSet::from_iter(vec![result.0, result.1]);

        assert!(results_set.contains(&Chromosome::from_genes(4, vec![15])));
        assert!(results_set.contains(&Chromosome::from_genes(4, vec![15])));
    }
}
