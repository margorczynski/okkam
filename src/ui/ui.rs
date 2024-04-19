
use std::{
    io::{self, Stdout}, sync::Arc, time::Duration
};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use log::info;
use ratatui::{prelude::*, widgets::Paragraph};

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

fn render_app(frame: &mut Frame, app: &App) {
    let greeting = Paragraph::new(format!("Hello World! (press 'q' to quit), iteration: {}, avg_time: {:?}", app.iteration, app.avg_duration_per_iteration));
    frame.render_widget(greeting, frame.size());
}

fn should_quit() -> Result<bool> {
    if event::poll(Duration::from_millis(250)).context("event poll failed")? {
        if let Event::Key(key) = event::read().context("event read failed")? {
            return Ok(KeyCode::Char('q') == key.code);
        }
    }
    Ok(false)
}