use anyhow::{Context, Result, bail};
use ask::cache::{Cache, CacheEntry, cosine_similarity};
use ask::client::{ClientError, LlmClient, OllamaClient, OpenAiClient};
use ask::config::{Config, State};
use ask::ui::{copy_to_clipboard, extract_commands, print_config};
use clap::{Parser, Subcommand};
use colored::*;
use dialoguer::{Confirm, Select};
use std::io::{self, Read};
use std::process::Command;

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inspect or modify configuration
    Config {
        /// Target provider to configure (ollama, openrouter, openai)
        #[arg(short = 'p', long)]
        provider: Option<String>,

        /// Show current configuration
        #[arg(long)]
        show: bool,

        /// Set default provider (ollama, openrouter, openai)
        #[arg(long)]
        set_provider: Option<String>,

        /// Set API key for provider (OpenRouter/OpenAI)
        #[arg(long)]
        set_api_key: Option<String>,

        /// Set model for the active or targeted provider
        #[arg(long)]
        set_model: Option<String>,

        /// Set host or base URL for the active or targeted provider
        #[arg(long)]
        set_host: Option<String>,
    },
}

/// A simple command line helper for remembering command line commands.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The question to ask (e.g., "How to find new files that start with S in this dir")
    question: Option<String>,

    /// LLM provider to use (ollama, openrouter, openai)
    #[arg(short = 'p', long)]
    provider: Option<String>,

    /// API key for cloud providers (overrides config and environment)
    #[arg(long)]
    api_key: Option<String>,

    /// Model to use (overrides config)
    #[arg(short, long)]
    model: Option<String>,

    /// Host or Base URL (overrides config)
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

    // Handle 'ask config' subcommand if invoked
    if let Some(Commands::Config {
        provider,
        show,
        set_provider,
        set_api_key,
        set_model,
        set_host,
    }) = args.command
    {
        return handle_config_command(
            cfg,
            provider,
            show,
            set_provider,
            set_api_key,
            set_model,
            set_host,
        );
    }

    let should_copy = if args.no_copy { false } else { cfg.auto_copy };

    // Resolve client
    let client = resolve_client(
        &cfg,
        args.provider.as_deref(),
        args.host.as_deref(),
        args.model.as_deref(),
        args.api_key.as_deref(),
    )?;

    if args.explain_previous {
        let state: State = confy::load("ask", "state").context("Failed to load state")?;
        if state.last_command.is_empty() {
            bail!("No previous command found to explain.");
        }
        println!(
            "{}",
            format!("Explaining: {}", state.last_command)
                .yellow()
                .bold()
        );
        match client.explain_command(&state.last_command).await {
            Ok(()) => {}
            Err(e) => {
                if let Some(fallback_client) =
                    try_get_fallback(&client, &cfg, args.api_key.as_deref(), &e)
                {
                    fallback_client.explain_command(&state.last_command).await?;
                } else {
                    bail!("Error: {}", e);
                }
            }
        }
    } else if args.fix {
        let mut buffer = String::new();
        if !atty::is(atty::Stream::Stdin) {
            io::stdin()
                .read_to_string(&mut buffer)
                .context("Failed to read from stdin")?;
        }
        if buffer.trim().is_empty() {
            bail!("--fix requires error output from stdin (e.g., 'cmd 2>&1 | ask --fix')");
        }

        println!("{}", "Analyzing error...".yellow().bold());
        let full_response = match client.fix_command(&buffer).await {
            Ok(res) => res,
            Err(e) => {
                if let Some(fallback_client) =
                    try_get_fallback(&client, &cfg, args.api_key.as_deref(), &e)
                {
                    fallback_client.fix_command(&buffer).await?
                } else {
                    bail!("Error: {}", e);
                }
            }
        };
        let commands = extract_commands(&full_response);
        handle_new_commands(commands, should_copy, args.execute).await?;
    } else if let Some(refinement) = args.refine {
        let state: State = confy::load("ask", "state").context("Failed to load state")?;
        if state.last_command.is_empty() {
            bail!("No previous command found to refine.");
        }
        println!(
            "{}",
            format!("Refining: {}", state.last_command).yellow().bold()
        );
        let full_response = match client
            .refine_command(&state.last_command, &refinement)
            .await
        {
            Ok(res) => res,
            Err(e) => {
                if let Some(fallback_client) =
                    try_get_fallback(&client, &cfg, args.api_key.as_deref(), &e)
                {
                    fallback_client
                        .refine_command(&state.last_command, &refinement)
                        .await?
                } else {
                    bail!("Error: {}", e);
                }
            }
        };
        let commands = extract_commands(&full_response);
        handle_new_commands(commands, should_copy, args.execute).await?;
    } else if let Some(question) = args.question {
        let os = std::env::consts::OS.to_string();
        let mut cached_command: Option<String> = None;
        let mut current_embedding: Option<Vec<f32>> = None;
        let mut cache_match_idx: Option<usize> = None;

        // Try local semantic caching (Ollama) if cache is not disabled and active client is Ollama
        if !args.no_cache && client.provider_name() == "Ollama" {
            let ollama_embed = OllamaClient::new(cfg.host.clone(), cfg.model.clone());
            match ollama_embed
                .get_embeddings(&cfg.embedding_model, &question)
                .await
            {
                Ok(emb) => {
                    current_embedding = Some(emb.clone());
                    let cache: Cache = confy::load("ask", "cache").unwrap_or_default();
                    for (i, entry) in cache.entries.iter().enumerate() {
                        if entry.os == os
                            && cosine_similarity(&emb, &entry.embedding) >= cfg.cache_threshold
                        {
                            cached_command = Some(entry.command.clone());
                            cache_match_idx = Some(i);
                            break;
                        }
                    }
                }
                Err(e) => {
                    // Only print warning if using Ollama and fallback is not available
                    if client.provider_name() == "Ollama" {
                        let has_fallback = cfg
                            .resolve_api_key("openrouter", args.api_key.as_deref())
                            .is_some();
                        if !has_fallback {
                            if let ask::client::OllamaError::NotRunning { .. } = e {
                                bail!(
                                    "Error: {}\nTip: You can use OpenRouter cloud models by setting the OPENROUTER_API_KEY environment variable.",
                                    e
                                );
                            }
                            eprintln!(
                                "{}: Cache optimization disabled ({}).",
                                "Note".yellow().bold(),
                                e
                            );
                        }
                    }
                }
            }
        }

        if let (Some(cmd), false) = (cached_command, args.no_cache) {
            println!("{}", "⚡ Cache hit! Serving instantly...".cyan().italic());
            println!("{}", cmd);
            handle_new_commands(vec![cmd], should_copy, args.execute).await?;
        } else {
            let full_response = match client.stream_command(&question).await {
                Ok(res) => res,
                Err(e) => {
                    if let Some(fallback_client) =
                        try_get_fallback(&client, &cfg, args.api_key.as_deref(), &e)
                    {
                        fallback_client.stream_command(&question).await?
                    } else {
                        bail!("Error: {}", e);
                    }
                }
            };

            let commands = extract_commands(&full_response);
            if let Some(first_cmd) = commands.first() {
                // Save or Update cache if we have an embedding
                if let Some(emb) = current_embedding {
                    let mut cache: Cache = confy::load("ask", "cache").unwrap_or_default();
                    if let Some(idx) = cache_match_idx {
                        // Update existing entry if user forced no-cache
                        println!(
                            "{}",
                            "🔄 Updating cache with fresh result...".yellow().italic()
                        );
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
    } else {
        bail!("No question provided. Use 'ask --help' for usage.");
    }

    Ok(())
}

fn resolve_client(
    cfg: &Config,
    cli_provider: Option<&str>,
    cli_host: Option<&str>,
    cli_model: Option<&str>,
    cli_api_key: Option<&str>,
) -> Result<LlmClient> {
    let provider = cli_provider
        .map(str::to_lowercase)
        .unwrap_or_else(|| cfg.provider.to_lowercase());

    match provider.as_str() {
        "ollama" => {
            let host = cli_host
                .map(String::from)
                .unwrap_or_else(|| cfg.host.clone());
            let model = cli_model
                .map(String::from)
                .unwrap_or_else(|| cfg.model.clone());
            Ok(LlmClient::Ollama(OllamaClient::new(host, model)))
        }
        "openrouter" => {
            let base_url = cli_host
                .map(String::from)
                .unwrap_or_else(|| cfg.openrouter.base_url.clone());
            let model = cli_model
                .map(String::from)
                .unwrap_or_else(|| cfg.openrouter.model.clone());
            let api_key = cfg.resolve_api_key("openrouter", cli_api_key);
            if api_key.is_none() {
                bail!(
                    "OpenRouter requires an API key. Set OPENROUTER_API_KEY environment variable or run 'ask config --set-api-key <KEY>'."
                );
            }
            Ok(LlmClient::OpenAi(OpenAiClient::new(
                "OpenRouter".to_string(),
                base_url,
                model,
                api_key,
            )))
        }
        "openai" => {
            let base_url = cli_host
                .map(String::from)
                .unwrap_or_else(|| cfg.openai.base_url.clone());
            let model = cli_model
                .map(String::from)
                .unwrap_or_else(|| cfg.openai.model.clone());
            let api_key = cfg.resolve_api_key("openai", cli_api_key);
            Ok(LlmClient::OpenAi(OpenAiClient::new(
                "OpenAI".to_string(),
                base_url,
                model,
                api_key,
            )))
        }
        other => {
            bail!(
                "Unknown provider '{}'. Supported providers: ollama, openrouter, openai",
                other
            );
        }
    }
}

fn try_get_fallback(
    client: &LlmClient,
    cfg: &Config,
    cli_api_key: Option<&str>,
    error: &ClientError,
) -> Option<LlmClient> {
    if client.provider_name() == "Ollama"
        && matches!(error, ClientError::Unreachable { .. })
        && let Some(openrouter_key) = cfg.resolve_api_key("openrouter", cli_api_key)
    {
        eprintln!(
            "{}: Ollama unreachable at {}. Falling back to OpenRouter ({})...",
            "Notice".yellow().bold(),
            cfg.host,
            cfg.openrouter.model
        );
        return Some(LlmClient::OpenAi(OpenAiClient::new(
            "OpenRouter".to_string(),
            cfg.openrouter.base_url.clone(),
            cfg.openrouter.model.clone(),
            Some(openrouter_key),
        )));
    }
    None
}

fn handle_config_command(
    mut cfg: Config,
    target_provider: Option<String>,
    _show: bool,
    set_provider: Option<String>,
    set_api_key: Option<String>,
    set_model: Option<String>,
    set_host: Option<String>,
) -> Result<()> {
    let mut modified = false;

    if let Some(p) = set_provider {
        let p_lower = p.to_lowercase();
        if !["ollama", "openrouter", "openai"].contains(&p_lower.as_str()) {
            bail!(
                "Invalid provider '{}'. Supported providers: ollama, openrouter, openai",
                p
            );
        }
        cfg.provider = p_lower;
        modified = true;
    }

    let target = if let Some(p) = target_provider {
        let p_lower = p.to_lowercase();
        if !["ollama", "openrouter", "openai"].contains(&p_lower.as_str()) {
            bail!(
                "Invalid target provider '{}'. Supported providers: ollama, openrouter, openai",
                p
            );
        }
        Some(p_lower)
    } else {
        None
    };

    let effective_target = target.as_deref().unwrap_or(cfg.provider.as_str());

    if let Some(key) = set_api_key {
        match effective_target {
            "openai" => cfg.openai.api_key = key,
            "openrouter" => cfg.openrouter.api_key = key,
            "ollama" => {
                if target.is_some() {
                    bail!(
                        "Ollama does not use an API key. Target 'openrouter' or 'openai' with -p/--provider."
                    );
                } else {
                    cfg.openrouter.api_key = key;
                }
            }
            _ => unreachable!(),
        }
        modified = true;
    }

    if let Some(m) = set_model {
        match effective_target {
            "openrouter" => cfg.openrouter.model = m,
            "openai" => cfg.openai.model = m,
            _ => cfg.model = m,
        }
        modified = true;
    }

    if let Some(h) = set_host {
        match effective_target {
            "openrouter" => cfg.openrouter.base_url = h,
            "openai" => cfg.openai.base_url = h,
            _ => cfg.host = h,
        }
        modified = true;
    }

    if modified {
        cfg.validate()
            .context("Failed to validate new configuration")?;
        confy::store("ask", None, &cfg).context("Failed to save configuration")?;
        println!("{}", "✔ Configuration updated successfully.".green().bold());
        println!();
    }

    print_config(&cfg);
    Ok(())
}

async fn handle_new_commands(
    commands: Vec<String>,
    should_copy: bool,
    should_execute: bool,
) -> Result<()> {
    if commands.is_empty() {
        return Ok(());
    }

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
        let state = State {
            last_command: selected_command.clone(),
        };
        let _ = confy::store("ask", "state", state);

        if should_copy {
            copy_to_clipboard(&selected_command);
        }

        if should_execute
            && Confirm::new()
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
    Ok(())
}
