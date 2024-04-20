use std::{
    io::{self, Stdout},
    sync::mpsc::{channel, Receiver, Sender},
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use indoc::indoc;
use itertools::izip;
use ratatui::{
    prelude::*,
    widgets::{
        Axis, Block, Borders, Cell, Chart, Dataset, GraphType, Padding, Paragraph, Row, Table, Wrap,
    },
};

use crate::config::okkam_config::OkkamConfig;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct App {
    pub iteration: usize,
    pub avg_duration_per_iteration: Duration,
    pub best_mae: f64,
    pub best_mape: f64,
    pub best_rmse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Message {
    UpdateState(App),
    Quit,
}

pub fn run_ui<F>(okkam_config: &OkkamConfig, computation: F) -> Result<()>
where
    F: FnOnce(Sender<Message>, Receiver<Message>) + Send + 'static,
{
    let mut terminal = setup_terminal().context("setup failed")?;

    let (tx_ui, rx_ui): (Sender<Message>, Receiver<Message>) = channel();
    let (tx_computation, rx_computation): (Sender<Message>, Receiver<Message>) = channel();

    let computation_handle = thread::spawn(move || {
        computation(tx_computation, rx_ui);
    });

    let mut app_history = vec![];

    loop {
        terminal.draw(|f| {
            render_app(f, &app_history, okkam_config);
        })?;

        if should_quit()? {
            tx_ui.send(Message::Quit).unwrap();
            break;
        }

        match rx_computation.try_recv() {
            Ok(Message::UpdateState(new_state)) => {
                app_history.push(new_state);
            }
            Ok(Message::Quit) => {
                break;
            }
            Err(_) => {}
        }
    }

    computation_handle.join().unwrap();

    restore_terminal(&mut terminal)
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    let mut stdout = io::stdout();
    enable_raw_mode().context("Failed to enable raw mode")?;
    execute!(stdout, EnterAlternateScreen).context("Unable to enter alternate screen")?;
    Terminal::new(CrosstermBackend::new(stdout)).context("Creating terminal failed")
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("Failed to disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("Unable to switch to main screen")?;
    terminal.show_cursor().context("Unable to show cursor")
}

fn should_quit() -> Result<bool> {
    if event::poll(Duration::from_millis(250)).context("event poll failed")? {
        if let Event::Key(key) = event::read().context("event read failed")? {
            return Ok(KeyCode::Char('q') == key.code);
        }
    }
    Ok(false)
}

fn render_app(frame: &mut Frame, app_history: &[App], okkam_config: &OkkamConfig) {
    let _latest_app = app_history.last();

    let size = frame.size();

    let outer_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(frame.size());

    let info_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(vec![Constraint::Percentage(25), Constraint::Percentage(75)])
        .split(outer_layout[0]);

    let config_info_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(info_layout[0]);

    let logo_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![
            Constraint::Percentage(25),
            Constraint::Percentage(50),
            Constraint::Percentage(25),
        ])
        .split(config_info_layout[0]);

    let chart_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(vec![
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(outer_layout[1]);

    let block = Block::default();
    frame.render_widget(block, size);

    if !app_history.is_empty() {
        let mae_data: Vec<(f64, f64)> = app_history
            .iter()
            .filter(|app| app.best_mae.is_finite())
            .map(|app| app.best_mae.log10())
            .enumerate()
            .map(|(idx, mae)| (idx as f64, mae))
            .collect();

        let mape_data: Vec<(f64, f64)> = app_history
            .iter()
            .filter(|app| app.best_mape.is_finite())
            .map(|app| app.best_mape.log10())
            .enumerate()
            .map(|(idx, mape)| (idx as f64, mape))
            .collect();

        let rmse_data: Vec<(f64, f64)> = app_history
            .iter()
            .filter(|app| app.best_rmse.is_finite())
            .map(|app| app.best_rmse.log10())
            .enumerate()
            .map(|(idx, mape)| (idx as f64, mape))
            .collect();

        frame.render_widget(
            Paragraph::new(logo())
                .style(Style::new().white())
                .alignment(Alignment::Center),
            logo_layout[1],
        );

        let (general_config, ga_config, polynomial_config) = create_config_info(&okkam_config);
        frame.render_widget(general_config, config_info_layout[1]);
        frame.render_widget(ga_config, config_info_layout[2]);
        frame.render_widget(polynomial_config, config_info_layout[3]);

        let table = create_table(app_history).block(create_block("Best 25 Polynomial Details"));
        frame.render_widget(table, info_layout[1]);

        let x_title = format!("N = {}", mae_data.len());
        let mae_y_title = format!("Max MAE = 10^{:.4}", mae_data.first().unwrap().1);
        let mae_chart = create_chart(
            &mae_data,
            "MAE (log10)",
            Color::LightRed,
            &x_title,
            &mae_y_title,
        )
        .block(create_block("Mean Absolute Error"));
        frame.render_widget(mae_chart, chart_layout[0]);

        let mape_y_title = format!("Max MAPE = 10^{:.4}%", mape_data.first().unwrap().1);
        let mape_chart = create_chart(
            &mape_data,
            "MAPE (log10, %)",
            Color::LightMagenta,
            &x_title,
            &mape_y_title,
        )
        .block(create_block("Mean Absolute Percentage Error"));
        frame.render_widget(mape_chart, chart_layout[1]);

        let rmse_y_title = format!("Max RMSE = 10^{:.4}", rmse_data.first().unwrap().1);
        if !rmse_data.is_empty() {
            let rmse_chart = create_chart(
                &rmse_data,
                "RMSE (log10)",
                Color::LightYellow,
                &x_title,
                &rmse_y_title,
            )
            .block(create_block("Root Mean Squared Error"));
            frame.render_widget(rmse_chart, chart_layout[2]);
        }
    }
}

fn create_block(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Gray))
        .padding(Padding::new(2, 2, 1, 1))
        .title(Span::styled(
            title,
            Style::default().add_modifier(Modifier::BOLD),
        ))
}

fn logo() -> String {
    let o = indoc! {"
            ▄▄▄▄
            █  █
            █▄▄█
        "};
    let k = indoc! {"
            ▄  ▄
            █▄▀
            █ ▀▄
        "};
    let a = indoc! {"
             ▄▄
            █▄▄█
            █  █
        "};
    let m = indoc! {"
            ▄   ▄
            █▀▄▀█
            █   █
        "};
    izip!(o.lines(), k.lines(), a.lines(), m.lines())
        .map(|(o, k, a, m)| format!("{o:5}{k:5}{k:5}{a:5}{m:5}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn create_config_info(okkam_config: &OkkamConfig) -> (Paragraph<'_>, Paragraph<'_>, Paragraph<'_>) {
    let okkam_general_config_text = vec![
        Line::from(Span::styled(
            format!(
                "log_level               = {}",
                okkam_config.log_level.0.as_str()
            ),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!("log_directory           = {}", okkam_config.log_directory),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!("dataset_path            = {}", okkam_config.dataset_path),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!("result_path             = {}", okkam_config.result_path),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!(
                "minimized_error_measure = {:?}",
                okkam_config.minimized_error_measure
            ),
            Style::new().bold(),
        )),
    ];
    let okkam_general_config_paragraph = Paragraph::new(okkam_general_config_text)
        .block(create_block("General Settings"))
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    let ga_config_text = vec![
        Line::from(Span::styled(
            format!("population_size = {}", okkam_config.ga.population_size),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!("tournament_size = {}", okkam_config.ga.tournament_size),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!("mutation_rate   = {}", okkam_config.ga.mutation_rate),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!("elite_factor    = {}", okkam_config.ga.elite_factor),
            Style::new().bold(),
        )),
    ];
    let ga_config_paragraph = Paragraph::new(ga_config_text)
        .block(create_block("Genetic Algorithm"))
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    let polynomial_config_text = vec![
        Line::from(Span::styled(
            format!("terms_num       = {}", okkam_config.polynomial.terms_num),
            Style::new().bold(),
        )),
        Line::from(Span::styled(
            format!(
                "degree_bits_num = {}",
                okkam_config.polynomial.degree_bits_num
            ),
            Style::new().bold(),
        )),
    ];
    let polynomial_config_paragraph = Paragraph::new(polynomial_config_text)
        .block(create_block("Polynomial"))
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    (
        okkam_general_config_paragraph,
        ga_config_paragraph,
        polynomial_config_paragraph,
    )
}

fn create_table(app_history: &[App]) -> Table {
    let header_style = Style::default().fg(Color::White).bg(Color::Cyan);
    let header = [
        Span::styled("Iteration", Style::new().bold()),
        Span::styled("Average time per iteration", Style::new().bold()),
        Span::styled("Mean Absolute Error", Style::new().bold()),
        Span::styled("Mean Absolute Percentage Error (%)", Style::new().bold()),
        Span::styled("Root Mean Squared Error", Style::new().bold()),
    ]
    .into_iter()
    .map(Cell::from)
    .collect::<Row>()
    .style(header_style);

    app_history
        .iter()
        .rev()
        .take(30)
        .rev()
        .map(|app| {
            Row::new(vec![
                Cell::from(app.iteration.to_string()),
                Cell::from(format!("{:?}", app.avg_duration_per_iteration)),
                Cell::from(app.best_mae.to_string()),
                Cell::from(app.best_mape.to_string()),
                Cell::from(app.best_rmse.to_string()),
            ])
        })
        .collect::<Table>()
        .header(header)
}

fn create_chart<'a>(
    data: &'a [(f64, f64)],
    name: &'a str,
    color: Color,
    x_axis_title: &'a str,
    y_axis_title: &'a str,
) -> Chart<'a> {
    let datasets = vec![Dataset::default()
        .name(name)
        .marker(symbols::Marker::Dot)
        .graph_type(GraphType::Scatter)
        .fg(color)
        //.style(Style::default().color(AnsiColor::Green))
        .data(data)];

    let x_axis = Axis::default()
        .title(x_axis_title.blue())
        .bounds([0.0, data.len() as f64])
        .style(Style::default().white());
    //.labels(vec!["1".into(), data.len().to_string().into()]);

    let max_val_rounded = u64::div_ceil(data.first().unwrap().1 as u64, 5) * 5;

    // Create the Y axis and define its properties
    let y_axis = Axis::default()
        .title(y_axis_title.blue())
        .bounds([0.0, max_val_rounded as f64])
        .style(Style::default().white());
    //.labels(vec!["0.0".into(), max_val_rounded.to_string().into()]);

    // Create the chart and link all the parts together
    Chart::new(datasets).x_axis(x_axis).y_axis(y_axis)
}
