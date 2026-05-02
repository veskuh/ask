use clap::Parser;
use ask::OllamaClient;
use std::process;
use colored::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    model: String,
    host: String,
    auto_copy: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "gemma4:latest".to_string(),
            host: "http://localhost:11434".to_string(),
            auto_copy: true,
        }
    }
}

/// A simple command line helper for remembering command line commands.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The question to ask (e.g., "How to find new files that start with S in this dir")
    question: String,

    /// Ollama model to use
    #[arg(short, long)]
    model: Option<String>,

    /// Ollama host URL
    #[arg(short = 'o', long)]
    host: Option<String>,

    /// Disable automatic copy to clipboard
    #[arg(long)]
    no_copy: bool,
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

    match client.stream_command(&args.question).await {
        Ok(command) => {
            if should_copy && !command.is_empty() {
                let mut clipboard = arboard::Clipboard::new().unwrap();
                if let Err(e) = clipboard.set_text(command) {
                    eprintln!("Warning: Failed to copy to clipboard: {}", e);
                } else {
                    println!("{}", "✔ Command copied to clipboard".green().italic());
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    }
}
