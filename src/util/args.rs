use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "Uses a Genetic Algorithm (GA) to perform symbolic regression on a given dataset"
)]
pub struct Args {
    #[arg(short, long)]
    pub config_path: String,

    #[arg(short = 'H', long)]
    pub headless: bool,
}
