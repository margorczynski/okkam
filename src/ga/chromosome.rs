use std::fmt::{Debug, Display, Formatter};

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Chromosome {
    pub size: usize,
    pub genes: Vec<u64>,
}

impl Chromosome {
    pub fn from_genes(size: usize, genes: Vec<u64>) -> Chromosome {
        Chromosome { size, genes }
    }
}

//TODO: Take size into account
impl Display for Chromosome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result: String = self.genes.iter().map(|&g| format!("[{g:b}]")).collect();

        write!(f, "{}", &result)
    }
}

impl Debug for Chromosome {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
