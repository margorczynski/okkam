
use std::{
    io::{self, Stdout}, sync::Arc, time::Duration
};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode}, execute, style::Stylize, terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen}
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Wrap},
};

use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;


#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct App {
    pub iteration: usize,
    pub avg_duration_per_iteration: Duration,
    pub best_mae: f32,
    pub best_mape: f32,
    pub best_rmse: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Message {
    UpdateState(App),
    Quit,
}

pub fn run_ui<F>(computation: F) -> Result<()>
where F: FnOnce(Sender<Message>, Receiver<Message>) + Send + 'static,
{
    let mut terminal = setup_terminal().context("setup failed")?;

    let (tx_ui, rx_ui): (Sender<Message>, Receiver<Message>) = channel();
    let (tx_computation, rx_computation): (Sender<Message>, Receiver<Message>) = channel();

    let computation_handle = thread::spawn(move || {
        computation(tx_computation, rx_ui);
    });

    let mut app = App { iteration: 0, avg_duration_per_iteration: Duration::ZERO, best_mae: 0.0f32, best_mape: 0.0f32, best_rmse: 0.0f32 };

    loop {
        terminal.draw(|f| {
            render_app(f, &app);
          })?;

        if should_quit()? {
            tx_ui.send(Message::Quit).unwrap();
            break;
        }

        match rx_computation.try_recv() {
            Ok(Message::UpdateState(new_state)) => {
                app = new_state;
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
    enable_raw_mode().context("failed to enable raw mode")?;
    execute!(stdout, EnterAlternateScreen).context("unable to enter alternate screen")?;
    Terminal::new(CrosstermBackend::new(stdout)).context("creating terminal failed")
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("failed to disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("unable to switch to main screen")?;
    terminal.show_cursor().context("unable to show cursor")
}

fn should_quit() -> Result<bool> {
    if event::poll(Duration::from_millis(250)).context("event poll failed")? {
        if let Event::Key(key) = event::read().context("event read failed")? {
            return Ok(KeyCode::Char('q') == key.code);
        }
    }
    Ok(false)
}

fn render_app(frame: &mut Frame, app: &App) {
    let create_block = |title| {
        Block::default()
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Gray))
            .title(Span::styled(
                title,
                Style::default().add_modifier(Modifier::BOLD),
            ))
    };

    let size = frame.size();
    let layout = Layout::vertical([Constraint::Ratio(1, 8); 4]).split(size);

    let block = Block::default();
    frame.render_widget(block, size);

    let text = vec![
        Line::from(format!("Found in iteration: {}", app.iteration)),
        Line::from(format!("Average time per iteration: {:?}", app.avg_duration_per_iteration)),
        Line::from(format!("MAE:  {}", app.best_mae)),
        Line::from(format!("MAPE: {}%", app.best_mape)),
        Line::from(format!("RMSE: {}", app.best_rmse)),
    ];

    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(Color::White))
        .block(create_block("Top candidate information"));
    frame.render_widget(paragraph, layout[0]);

}