//! Terminal presentation, clipboard management, and response parsing utilities.

use crate::config::Config;
use colored::*;
#[cfg(target_os = "linux")]
use std::process::Command;

/// Formatted terminal configuration status display without leaking plaintext API keys.
pub fn print_config(cfg: &Config) {
    println!("{}", "=== ask configuration ===".bold().cyan());
    println!(
        "{}: {}",
        "Active Provider".bold(),
        cfg.provider.yellow().bold()
    );
    println!("{}: {}", "Auto-copy to Clipboard".bold(), cfg.auto_copy);
    println!(
        "{}: {}",
        "Semantic Cache Threshold".bold(),
        cfg.cache_threshold
    );
    println!();

    println!("{}", "[Ollama (Local)]".bold().underline());
    println!("  Host: {}", cfg.host);
    println!("  Model: {}", cfg.model);
    println!("  Embedding Model: {}", cfg.embedding_model);
    println!();

    println!("{}", "[OpenRouter (Cloud)]".bold().underline());
    println!("  Base URL: {}", cfg.openrouter.base_url);
    println!("  Model: {}", cfg.openrouter.model);
    let or_key_source = if std::env::var("OPENROUTER_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
    {
        "Configured via $OPENROUTER_API_KEY".green()
    } else if !cfg.openrouter.api_key.is_empty() {
        "Configured in config file".green()
    } else {
        "Not set (export OPENROUTER_API_KEY or use 'ask config --set-api-key <KEY>')".yellow()
    };
    println!("  API Key: {}", or_key_source);
    println!();

    println!("{}", "[OpenAI-Compatible]".bold().underline());
    println!("  Base URL: {}", cfg.openai.base_url);
    println!("  Model: {}", cfg.openai.model);
    let oa_key_source = if std::env::var("OPENAI_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
    {
        "Configured via $OPENAI_API_KEY".green()
    } else if !cfg.openai.api_key.is_empty() {
        "Configured in config file".green()
    } else {
        "Not set".dimmed()
    };
    println!("  API Key: {}", oa_key_source);
}

/// Multi-platform clipboard copy supporting Wayland (`wl-copy`), X11 (`xclip`), and macOS/Windows (`arboard`).
pub fn copy_to_clipboard(text: &str) {
    #[cfg(target_os = "linux")]
    {
        // On Linux, arboard/clipboard often fails to persist after process exit.
        // Try wl-copy (Wayland) or xclip (X11) first as they handle persistence better.
        let has_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();

        if has_wayland {
            if let Ok(mut child) = Command::new("wl-copy")
                .stdin(std::process::Stdio::piped())
                .spawn()
            {
                if let Some(mut stdin) = child.stdin.take() {
                    use std::io::Write;
                    if stdin.write_all(text.as_bytes()).is_ok() {
                        println!(
                            "{}",
                            "✔ Command copied to clipboard (via wl-copy)"
                                .green()
                                .italic()
                        );
                        return;
                    }
                }
            }
        }

        if let Ok(mut child) = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                if stdin.write_all(text.as_bytes()).is_ok() {
                    println!(
                        "{}",
                        "✔ Command copied to clipboard (via xclip)".green().italic()
                    );
                    return;
                }
            }
        }
    }

    // Fallback to arboard for macOS/Windows or if Linux tools are missing
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => {
            if let Err(e) = clipboard.set_text(text.to_string()) {
                eprintln!(
                    "{}: Failed to copy to clipboard: {}",
                    "Warning".yellow().bold(),
                    e
                );
            } else {
                #[cfg(target_os = "linux")]
                {
                    println!(
                        "{}",
                        "✔ Command copied to clipboard (arboard)".green().italic()
                    );
                    println!(
                        "{}: On some Linux setups, clipboard may clear when the app exits.",
                        "Note".yellow()
                    );
                }
                #[cfg(not(target_os = "linux"))]
                println!("{}", "✔ Command copied to clipboard".green().italic());
            }
        }
        Err(e) => {
            eprintln!(
                "{}: Clipboard unavailable: {}",
                "Warning".yellow().bold(),
                e
            );
        }
    }
}

/// Parses commands from an LLM response, looking for `Command: <cmd>` lines or falling back to the last non-empty line.
pub fn extract_commands(response: &str) -> Vec<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_commands_with_command_prefix() {
        let response = "Here are options:\nCommand: git status\nCommand: git status -s";
        let commands = extract_commands(response);
        assert_eq!(commands, vec!["git status", "git status -s"]);
    }

    #[test]
    fn test_extract_commands_case_insensitive_prefix() {
        let response = "COMMAND: docker ps -a\ncommand:   podman ps  ";
        let commands = extract_commands(response);
        assert_eq!(commands, vec!["docker ps -a", "podman ps"]);
    }

    #[test]
    fn test_extract_commands_fallback_to_last_line() {
        let response = "I suggest using this command:\n\nls -la /tmp\n\n";
        let commands = extract_commands(response);
        assert_eq!(commands, vec!["ls -la /tmp"]);
    }

    #[test]
    fn test_extract_commands_empty_response() {
        assert_eq!(extract_commands(""), Vec::<String>::new());
        assert_eq!(extract_commands("   \n\n  \t  "), Vec::<String>::new());
    }

    #[test]
    fn test_extract_commands_ignores_empty_command_line() {
        let response = "Command:\nCommand:  \nCommand: echo hello";
        let commands = extract_commands(response);
        assert_eq!(commands, vec!["echo hello"]);
    }

    #[test]
    fn test_print_config_does_not_panic() {
        let cfg = Config::default();
        print_config(&cfg);
    }
}
