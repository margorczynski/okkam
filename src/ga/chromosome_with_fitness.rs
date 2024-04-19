use std::cmp::Ordering;

use crate::ga::chromosome::Chromosome;

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct ChromosomeWithFitness<T: Clone + Eq + Send> {
    pub chromosome: Chromosome,
    pub fitness: T,
}

impl<T: Clone + Eq + Send> ChromosomeWithFitness<T> {
    pub fn from_chromosome_and_fitness(
        chromosome: Chromosome,
        fitness: T,
    ) -> ChromosomeWithFitness<T> {
        ChromosomeWithFitness {
            chromosome,
            fitness,
        }
    }
}

unsafe impl<T: Clone + Eq + Send> Send for ChromosomeWithFitness<T> {}
unsafe impl<T: Clone + Eq + Send> Sync for ChromosomeWithFitness<T> {}

impl<T: PartialOrd + Clone + Eq + Send> PartialOrd for ChromosomeWithFitness<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        T::partial_cmp(&self.fitness, &other.fitness)
    }
}

impl<T: PartialOrd + Clone + Eq + Send> Ord for ChromosomeWithFitness<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Equal)
    }
}
