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
    chromosome_to_fitness_map: &HashSet<ChromosomeWithFitness<T>>,
    selection_strategy: &SelectionStrategy,
) -> (Chromosome, Chromosome) {
    match selection_strategy {
        SelectionStrategy::Tournament(tournament_size) => {
            let mut rng = SmallRng::from_entropy();
            let mut tournament_participants_double =
                //TODO If the order isn't random then we can just use chromosomes + select head to get the max
                chromosome_to_fitness_map
                .into_iter()
                .choose_multiple(&mut rng, tournament_size * 2);

            let fst_winner = tournament_participants_double
                .drain(0..*tournament_size)
                .max()
                .unwrap();

            let snd_winner = tournament_participants_double.into_iter().max().unwrap();

            (fst_winner.chromosome.clone(), snd_winner.chromosome.clone())
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

    let mut fst_child_genes =
        merge_gene_blocks(&parents.0.genes, &parents.1.genes, crossover_point);
    let mut snd_child_genes =
        merge_gene_blocks(&parents.1.genes, &parents.0.genes, crossover_point);

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

fn merge_gene_blocks(
    gene_blocks_fst: &[u64],
    gene_blocks_snd: &[u64],
    crossover_point: usize,
) -> Vec<u64> {
    let mut new_genes = Vec::new();
    let crossover_block_idx = crossover_point / 64;

    //We split the fst blocks at the crossover point and take all the blocks before it
    let (left, _) = gene_blocks_fst.split_at(crossover_block_idx);
    let (_, right) = gene_blocks_snd.split_at(crossover_block_idx);

    new_genes.extend(left);

    if crossover_point % 64 == 0 {
        new_genes.extend(right.iter());
    } else {
        //Next we construct the block where the crossover point falls into using a left and right-side mask
        let left_mask = !0u64 << (64 - (crossover_point % 64));
        let right_mask = !left_mask;

        new_genes.push(0u64);

        new_genes[crossover_block_idx] |= gene_blocks_fst[crossover_block_idx] & left_mask;
        new_genes[crossover_block_idx] |= gene_blocks_snd[crossover_block_idx] & right_mask;

        //Finally we add up all the right-side blocks (skipping the first one which is the one we've constructed above)
        new_genes.extend(right.iter().skip(1));
    }

    new_genes
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
        let selection_strategy = SelectionStrategy::Tournament(3);
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

    #[test]
    fn test_merge_gene_blocks() {
        let gene_blocks_fst = [0x00FFFFFFFFFFFFFF];
        let gene_blocks_snd = [0xFFFFFFFFFFFFF000];
        let crossover_point = 32;

        let expected_result = [0x00FFFFFFFFFFF000];

        let result = merge_gene_blocks(&gene_blocks_fst, &gene_blocks_snd, crossover_point);

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_merge_gene_blocks_multiple_blocks() {
        let gene_blocks_fst = [
            0x1111111111111111,
            0b0001000100010001000100010001000100010001000100010001000100010001,
            0x1111111111111111,
        ];
        let gene_blocks_snd = [
            0xAAAAAAAAAAAAAAAA,
            0b1010101010101010101010101010101010101010101010101010101010101010,
            0xAAAAAAAAAAAAAAAA,
        ];

        let expected_result_77 = [
            0x1111111111111111,
            0b0001000100010010101010101010101010101010101010101010101010101010,
            0xAAAAAAAAAAAAAAAA,
        ];
        let expected_result_65 = [
            0x1111111111111111,
            0b0010101010101010101010101010101010101010101010101010101010101010,
            0xAAAAAAAAAAAAAAAA,
        ];
        let expected_result_64 = [
            0x1111111111111111,
            0b1010101010101010101010101010101010101010101010101010101010101010,
            0xAAAAAAAAAAAAAAAA,
        ];

        let result_77 = merge_gene_blocks(&gene_blocks_fst, &gene_blocks_snd, 77);
        let result_65 = merge_gene_blocks(&gene_blocks_fst, &gene_blocks_snd, 65);
        let result_64 = merge_gene_blocks(&gene_blocks_fst, &gene_blocks_snd, 64);

        assert_eq!(result_77, expected_result_77);
        assert_eq!(result_65, expected_result_65);
        assert_eq!(result_64, expected_result_64);
    }
}
