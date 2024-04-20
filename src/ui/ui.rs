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
use ratatui::{
    prelude::*,
    widgets::{
        Axis, Block, Borders, Cell, Chart, Dataset, GraphType, Padding, Paragraph, Row, Table,
    },
};

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

pub fn run_ui<F>(computation: F) -> Result<()>
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
            render_app(f, &app_history);
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

fn render_app(frame: &mut Frame, app_history: &[App]) {
    let create_block = |title| {
        Block::default()
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Gray))
            .padding(Padding::new(2, 2, 1, 1))
            .title(Span::styled(
                title,
                Style::default().add_modifier(Modifier::BOLD),
            ))
    };

    let latest_app = app_history.last();

    let size = frame.size();

    let outer_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(frame.size());

    let info_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(vec![Constraint::Percentage(25), Constraint::Percentage(75)])
        .split(outer_layout[0]);

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

    let text = vec![
        Line::from(format!(
            "Found in iteration: {}",
            latest_app.map(|app| app.iteration).unwrap_or(0)
        )),
        Line::from(format!(
            "Average time per iteration: {:?}",
            latest_app
                .map(|app| app.avg_duration_per_iteration)
                .unwrap_or(Duration::ZERO)
        )),
        Line::from(format!(
            "MAE:  {}",
            latest_app.map(|app| app.best_mae).unwrap_or(0.0)
        )),
        Line::from(format!(
            "MAPE: {}%",
            latest_app.map(|app| app.best_mape).unwrap_or(0.0)
        )),
        Line::from(format!(
            "RMSE: {}",
            latest_app.map(|app| app.best_rmse).unwrap_or(0.0)
        )),
    ];

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

        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(Color::White))
            .block(create_block("Summary"));
        frame.render_widget(paragraph, info_layout[0]);

        let table = create_table(app_history).block(create_block("Best 25 Polynomial Details"));
        frame.render_widget(table, info_layout[1]);

        let mae_x_title = format!("N = {}", mae_data.len());
        let mae_y_title = format!("Max MAE = 10^{:.4}", mae_data.first().unwrap().1);
        let mae_chart = create_chart(
            &mae_data,
            "MAE (log10)",
            Color::LightRed,
            &mae_x_title,
            &mae_y_title,
        )
        .block(create_block("Mean Absolute Error"));
        frame.render_widget(mae_chart, chart_layout[0]);

        let mape_chart = create_chart(
            &mape_data,
            "MAPE (log10, %)",
            Color::LightMagenta,
            "N",
            "MAPE (log10, %)",
        )
        .block(create_block("Mean Absolute Percentage Error"));
        frame.render_widget(mape_chart, chart_layout[1]);

        if !rmse_data.is_empty() {
            let rmse_chart = create_chart(
                &rmse_data,
                "RMSE (log10)",
                Color::LightYellow,
                "N",
                "RMSE (log10)",
            )
            .block(create_block("Root Mean Squared Error"));
            frame.render_widget(rmse_chart, chart_layout[2]);
        }
    }
}

fn create_table(app_history: &[App]) -> Table {
    let header_style = Style::default().fg(Color::White).bg(Color::Cyan);
    let header = [
        Span::styled("Iteration", Style::new().bold()),
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
