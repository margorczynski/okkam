[![CI](https://github.com/margorczynski/okkam/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/margorczynski/okkam/actions/workflows/ci.yml)
# Okkam
This application uses a Genetic Algorithm (GA) to perform symbolic regression on a given dataset. The goal is to find a polynomial function that best fits the data, minimizing the error between the predicted and actual output values with many options to configure the GA and polynomial parameters.

## Features
- Configurable polynomial representation (number of terms, exponent of variables bits)
- Genetic Algorithm search with easily configurable parameters (population size, mutation rate, etc.)
- Choose which error measure to minimize (MAE, MAPE and RSME supported)
- Optimized for multithreaded usage
- Real-time progress reporting and visuals using a TUI (terminal UI) with a headless mode as an option
- CSV for reading input dataset and persisting output

## Prerequisites
- Git
- Rust (stable version)
- Cargo

## Installation
1. Clone the repository:

`git clone https://github.com/margorczynski/okkam.git`

2. Navigate to the project directory:

`cd okkam`

3. Build the project (optimized version):

`cargo build --release`

The compiled binary will be located in the target/release directory.

## Usage

`okkam [OPTIONS]`

### Options
- `--config-path <CONFIG_PATH>`: Path to the configuration file
- `--headless`: Flag for running in headless mode without the UI
- `--help`: Display help information

### Configuration
The application is configured using a TOML, JSON, YAML, INI, RON, JSON5 file. Here's an example configuration using TOML:


```toml
# The log level (Off, Error, Warn, Info, Debug, Trace)
log_level = "INFO"
# The directory that will be used to store the logfile
log_directory = "./logs"
# Path to the dataset file (CSV format)
dataset_path = "examples/test_dataset.csv"
# Path to the file that will be created to store the results for the best polynomials found
result_path = "okkam_result.csv"
# The measure which the GA will try to minimize
minimized_error_measure = "MAE"

[ga]
# Population size
population_size = 512
# Tournament size for selection
tournament_size = 8
# Mutation rate (here 10%)
mutation_rate = 0.1
# Elite factor (percentage of top individuals to preserve, here 10%)
elite_factor = 0.1

[polynomial]
# Number of terms in the polynomial
terms_num = 12
# Number of bits to represent the degree of each variable (2^4 = 16 so the degree is in the range of 0..(2^4-1))
degree_bits_num = 4
```

Or alternatively by using environment variables with the prefix `OKKAM_`, e.g. `OKKAM_GA_POPULATION_SIZE=100`, `OKKAM_POLYNOMIAL_TERMS_NUM=5`

### Dataset
The dataset should be provided in CSV format (no header), with each row representing a data point. The first columns should contain the input features, and the last column should be the target output value.

### UI
The UI consists of 3 main areas:
- Upper-left that contains the logo and the configuration that is used
- Upper-right with a table that will show the details of the best 25 (considering the chosen measure) results
- Bottom half with three charts for each of the available measures

## Example
To run the application with the test config and dataset:

`./target/release/okkam --config-path examples/test_config.toml`

Or directly using Cargo (here with the short version of the config flag):

`cargo run -- -c examples/test_config.toml`

This will start the application and the UI which should show you the progress on the search.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the Apache 2.0 License.