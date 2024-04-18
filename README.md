# Okkam
This application uses a Genetic Algorithm (GA) to perform symbolic regression on a given dataset. The goal is to find a polynomial function that best fits the data, minimizing the error between the predicted and actual output values.

## Features
- Configurable polynomial representation (number of terms, degree bits)
- Parallelized fitness evaluation and population evolution using Rayon
- Tournament selection strategy for parent selection
- Elitism to preserve the fittest individuals across generations
- Mutation and crossover operators for genetic variation
- Real-time progress reporting with generation number, lowest error, and average time per loop

## Prerequisites
- Git
- Rust (stable version)
- Cargo

## Installation
1. Clone the repository:

`git clone https://github.com/margorczynski/okkam.git`

2. Navigate to the project directory:

`cd okkam`

3. Build the project:

`cargo build --release`

The compiled binary will be located in the target/release directory.

## Usage

`okkam [OPTIONS]`

### Options
- `--config-path <CONFIG_PATH>`: Path to the configuration file
- `--help`: Display help information

### Configuration
The application is configured using a TOML, JSON, YAML, INI, RON, JSON5 file. Here's an example configuration using TOML:

```

# The log level (Off, Error, Warn, Info, Debug, Trace)
log_level = "INFO"
# Path to the dataset file (CSV format)
dataset_path = "data/dataset.csv"

[ga]
# Population size
population_size = 100
# Tournament size for selection
tournament_size = 3
# Mutation rate (here 10%)
mutation_rate = 0.1
# Elite factor (percentage of top individuals to preserve, here 20%)
elite_factor = 0.2

[polynomial]
# Number of terms in the polynomial
terms_num = 5
# Number of bits to represent the degree of each term (2^4 = 16 so the degree can be a maximum of 15 (2^4 - 1))
degree_bits_num = 4
```

Or alternatively by using environment variables with the prefix `OKKAM_`, e.g. `OKKAM_GA_POPULATION_SIZE=100`, `OKKAM_POLYNOMIAL_TERMS_NUM=5`

### Dataset
The dataset should be provided in CSV format (no header), with each row representing a data point. The first columns should contain the input features, and the last column should be the target output value.

## Example
To run the application with the test config and dataset:

`./target/release/okkam --config-path examples/test_config.toml`

This will start the GA process and print the progress to the console, including the generation number, lowest error, relative error percentage, and average time per loop.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the Apache 2.0 License.
