use clap::Parser;
use ask::OllamaClient;
use std::process;
use colored::*;

/// A simple command line helper for remembering command line commands.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The question to ask (e.g., "How to find new files that start with S in this dir")
    question: String,

    /// Ollama model to use
    #[arg(short, long, default_value = "qwen3:8b")]
    model: String,

    /// Ollama host URL
    #[arg(short = 'o', long, default_value = "http://localhost:11434")]
    host: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let client = OllamaClient::new(args.host, args.model);

    match client.stream_command(&args.question).await {
        Ok(command) => {
            if !command.is_empty() {
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
