use clap::Parser;
use ask::OllamaClient;
use std::process::{self, Command};
use colored::*;
use serde::{Serialize, Deserialize};
use std::io::{self, Read};
use dialoguer::Confirm;

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    model: String,
    host: String,
    auto_copy: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "gemma4:e4b".to_string(),
            host: "http://localhost:11434".to_string(),
            auto_copy: true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct State {
    last_command: String,
}

/// A simple command line helper for remembering command line commands.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The question to ask (e.g., "How to find new files that start with S in this dir")
    question: Option<String>,

    /// Ollama model to use
    #[arg(short, long)]
    model: Option<String>,

    /// Ollama host URL
    #[arg(short = 'o', long)]
    host: Option<String>,

    /// Disable automatic copy to clipboard
    #[arg(long)]
    no_copy: bool,

    /// Explain the previous command
    #[arg(short, long)]
    explain_previous: bool,

    /// Refine the previous command with additional instructions
    #[arg(short, long)]
    refine: Option<String>,

    /// Fix a command based on error output from stdin (e.g., 'ls --wrong 2>&1 | ask --fix')
    #[arg(short, long)]
    fix: bool,

    /// Execute the suggested command after confirmation
    #[arg(short = 'x', long)]
    execute: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    
    // Load config
    let cfg: Config = confy::load("ask", None).unwrap_or_default();
    
    // Resolve values: CLI arg > Config file > Default
    let model = args.model.unwrap_or(cfg.model);
    let host = args.host.unwrap_or(cfg.host);
    let should_copy = if args.no_copy { false } else { cfg.auto_copy };

    let client = OllamaClient::new(host, model);

    if args.explain_previous {
        let state: State = confy::load("ask", "state").unwrap_or_default();
        if state.last_command.is_empty() {
            eprintln!("No previous command found to explain.");
            process::exit(1);
        }
        println!("{}", format!("Explaining: {}", state.last_command).yellow().bold());
        if let Err(e) = client.explain_command(&state.last_command).await {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    } else if args.fix {
        let mut buffer = String::new();
        if !atty::is(atty::Stream::Stdin) {
            io::stdin().read_to_string(&mut buffer).unwrap();
        }
        if buffer.trim().is_empty() {
            eprintln!("Error: --fix requires error output from stdin (e.g., 'cmd 2>&1 | ask --fix')");
            process::exit(1);
        }

        println!("{}", "Analyzing error...".yellow().bold());
        match client.fix_command(&buffer).await {
            Ok(full_response) => {
                let command = extract_command(&full_response);
                handle_new_command(command, should_copy, args.execute).await;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    } else if let Some(refinement) = args.refine {
        let state: State = confy::load("ask", "state").unwrap_or_default();
        if state.last_command.is_empty() {
            eprintln!("No previous command found to refine.");
            process::exit(1);
        }
        println!("{}", format!("Refining: {}", state.last_command).yellow().bold());
        match client.refine_command(&state.last_command, &refinement).await {
            Ok(command) => {
                handle_new_command(command, should_copy, args.execute).await;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    } else if let Some(question) = args.question {
        match client.stream_command(&question).await {
            Ok(command) => {
                handle_new_command(command, should_copy, args.execute).await;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    } else {
        eprintln!("Error: No question provided. Use 'ask --help' for usage.");
        process::exit(1);
    }
}

async fn handle_new_command(command: String, should_copy: bool, should_execute: bool) {
    if !command.is_empty() {
        // Save state
        let state = State { last_command: command.clone() };
        let _ = confy::store("ask", "state", state);

        if should_copy {
            let mut clipboard = arboard::Clipboard::new().unwrap();
            if let Err(e) = clipboard.set_text(command.clone()) {
                eprintln!("Warning: Failed to copy to clipboard: {}", e);
            } else {
                println!("{}", "✔ Command copied to clipboard".green().italic());
            }
        }

        if should_execute {
            if Confirm::new()
                .with_prompt("Run this command?")
                .default(false)
                .interact()
                .unwrap_or(false)
            {
                let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
                let status = Command::new(shell)
                    .arg("-c")
                    .arg(&command)
                    .status();

                match status {
                    Ok(s) if s.success() => (),
                    Ok(s) => eprintln!("Command exited with status: {}", s),
                    Err(e) => eprintln!("Failed to execute command: {}", e),
                }
            }
        }
    }
}

fn extract_command(response: &str) -> String {
    for line in response.lines() {
        if line.to_lowercase().starts_with("command:") {
            return line[8..].trim().to_string();
        }
    }
    // Fallback if formatting was missed
    response.lines().last().unwrap_or("").trim().to_string()
}
