use clap::Parser;
use ask::client::{OllamaClient, OllamaError};
use ask::config::{Config, State};
use ask::cache::{Cache, CacheEntry, cosine_similarity};
use std::process::Command;
use colored::*;
use std::io::{self, Read};
use dialoguer::{Confirm, Select};
use anyhow::{Context, Result, bail};

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

    /// Disable semantic cache for this run
    #[arg(long)]
    no_cache: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Load config
    let cfg: Config = confy::load("ask", None).context("Failed to load configuration")?;
    cfg.validate().context("Configuration validation failed")?;
    
    // Resolve values
    let mut cfg = cfg;
    if let Some(m) = args.model { cfg.model = m; }
    if let Some(h) = args.host { cfg.host = h; }
    let should_copy = if args.no_copy { false } else { cfg.auto_copy };

    cfg.validate().context("Configuration validation failed")?;

    let client = OllamaClient::new(cfg.host.clone(), cfg.model.clone());

    if args.explain_previous {
        let state: State = confy::load("ask", "state").context("Failed to load state")?;
        if state.last_command.is_empty() {
            bail!("No previous command found to explain.");
        }
        println!("{}", format!("Explaining: {}", state.last_command).yellow().bold());
        client.explain_command(&state.last_command).await?;
    } else if args.fix {
        let mut buffer = String::new();
        if !atty::is(atty::Stream::Stdin) {
            io::stdin().read_to_string(&mut buffer).context("Failed to read from stdin")?;
        }
        if buffer.trim().is_empty() {
            bail!("--fix requires error output from stdin (e.g., 'cmd 2>&1 | ask --fix')");
        }

        println!("{}", "Analyzing error...".yellow().bold());
        match client.fix_command(&buffer).await {
            Ok(full_response) => {
                let commands = extract_commands(&full_response);
                handle_new_commands(commands, should_copy, args.execute).await?;
            }
            Err(e) => {
                bail!("Error: {}", e);
            }
        }
    } else if let Some(refinement) = args.refine {
        let state: State = confy::load("ask", "state").context("Failed to load state")?;
        if state.last_command.is_empty() {
            bail!("No previous command found to refine.");
        }
        println!("{}", format!("Refining: {}", state.last_command).yellow().bold());
        match client.refine_command(&state.last_command, &refinement).await {
            Ok(full_response) => {
                let commands = extract_commands(&full_response);
                handle_new_commands(commands, should_copy, args.execute).await?;
            }
            Err(e) => {
                bail!("Error: {}", e);
            }
        }
    } else if let Some(question) = args.question {
        let os = std::env::consts::OS.to_string();
        let mut cached_command: Option<String> = None;
        let mut current_embedding: Option<Vec<f32>> = None;
        let mut cache_match_idx: Option<usize> = None;

        match client.get_embeddings(&cfg.embedding_model, &question).await {
            Ok(emb) => {
                current_embedding = Some(emb.clone());
                let cache: Cache = confy::load("ask", "cache").unwrap_or_default();
                for (i, entry) in cache.entries.iter().enumerate() {
                    if entry.os == os && cosine_similarity(&emb, &entry.embedding) >= cfg.cache_threshold {
                        cached_command = Some(entry.command.clone());
                        cache_match_idx = Some(i);
                        break;
                    }
                }
            }
            Err(e) => {
                if let OllamaError::NotRunning { .. } = e {
                    bail!("Error: {}", e);
                }
                eprintln!("{}: Cache optimization disabled ({}).", "Note".yellow().bold(), e);
            }
        }

        if let (Some(cmd), false) = (cached_command, args.no_cache) {
            println!("{}", "⚡ Cache hit! Serving instantly...".cyan().italic());
            println!("{}", cmd);
            handle_new_commands(vec![cmd], should_copy, args.execute).await?;
        } else {
            match client.stream_command(&question).await {
                Ok(full_response) => {
                    let commands = extract_commands(&full_response);
                    if let Some(first_cmd) = commands.first() {
                        // Save or Update cache if we have an embedding
                        if let Some(emb) = current_embedding {
                            let mut cache: Cache = confy::load("ask", "cache").unwrap_or_default();
                            if let Some(idx) = cache_match_idx {
                                // Update existing entry if user forced no-cache
                                println!("{}", "🔄 Updating cache with fresh result...".yellow().italic());
                                cache.entries[idx].command = first_cmd.clone();
                                cache.entries[idx].question = question.clone();
                                cache.entries[idx].embedding = emb;
                            } else {
                                // Add new entry
                                cache.entries.push(CacheEntry {
                                    question: question.clone(),
                                    embedding: emb,
                                    command: first_cmd.clone(),
                                    os,
                                });
                            }
                            let _ = confy::store("ask", "cache", cache);
                        }
                    }
                    handle_new_commands(commands, should_copy, args.execute).await?;
                }
                Err(e) => {
                    bail!("Error: {}", e);
                }
            }
        }
    } else {
        bail!("No question provided. Use 'ask --help' for usage.");
    }

    Ok(())
}

async fn handle_new_commands(commands: Vec<String>, should_copy: bool, should_execute: bool) -> Result<()> {
    if commands.is_empty() { return Ok(()); }

    let selected_command = if commands.len() > 1 {
        let selection = Select::new()
            .with_prompt("Multiple options found. Select one")
            .items(&commands)
            .default(0)
            .interact()
            .context("Selection interrupted")?;
        commands[selection].clone()
    } else {
        commands[0].clone()
    };

    if !selected_command.is_empty() {
        // Save state
        let state = State { last_command: selected_command.clone() };
        let _ = confy::store("ask", "state", state);

        if should_copy {
            copy_to_clipboard(&selected_command);
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
                    .arg(&selected_command)
                    .status();

                match status {
                    Ok(s) if s.success() => (),
                    Ok(s) => eprintln!("Command exited with status: {}", s),
                    Err(e) => eprintln!("Failed to execute command: {}", e),
                }
            }
        }
    }
    Ok(())
}

fn copy_to_clipboard(text: &str) {
    #[cfg(target_os = "linux")]
    {
        // On Linux, arboard/clipboard often fails to persist after process exit.
        // Try wl-copy (Wayland) or xclip (X11) first as they handle persistence better.
        let has_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
        
        if has_wayland {
            if let Ok(mut child) = Command::new("wl-copy").stdin(std::process::Stdio::piped()).spawn() {
                if let Some(mut stdin) = child.stdin.take() {
                    use std::io::Write;
                    if stdin.write_all(text.as_bytes()).is_ok() {
                        println!("{}", "✔ Command copied to clipboard (via wl-copy)".green().italic());
                        return;
                    }
                }
            }
        }

        if let Ok(mut child) = Command::new("xclip").args(["-selection", "clipboard"]).stdin(std::process::Stdio::piped()).spawn() {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                if stdin.write_all(text.as_bytes()).is_ok() {
                    println!("{}", "✔ Command copied to clipboard (via xclip)".green().italic());
                    return;
                }
            }
        }
    }

    // Fallback to arboard for macOS/Windows or if Linux tools are missing
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => {
            if let Err(e) = clipboard.set_text(text.to_string()) {
                eprintln!("{}: Failed to copy to clipboard: {}", "Warning".yellow().bold(), e);
            } else {
                #[cfg(target_os = "linux")]
                {
                    println!("{}", "✔ Command copied to clipboard (arboard)".green().italic());
                    println!("{}: On some Linux setups, clipboard may clear when the app exits.", "Note".yellow());
                }
                #[cfg(not(target_os = "linux"))]
                println!("{}", "✔ Command copied to clipboard".green().italic());
            }
        }
        Err(e) => {
            eprintln!("{}: Clipboard unavailable: {}", "Warning".yellow().bold(), e);
        }
    }
}

fn extract_commands(response: &str) -> Vec<String> {
    let mut commands = Vec::new();
    for line in response.lines() {
        if line.to_lowercase().starts_with("command:") {
            let cmd = line[8..].trim().to_string();
            if !cmd.is_empty() {
                commands.push(cmd);
            }
        }
    }
    
    if commands.is_empty() {
        // Fallback: treat the non-empty last line as the command if no "Command:" prefix found
        if let Some(last_line) = response.lines().rev().find(|l| !l.trim().is_empty()) {
            commands.push(last_line.trim().to_string());
        }
    }
    commands
}
