# ask

A simple Rust-based command-line helper for remembering terminal commands. It uses a locally hosted Ollama model to translate natural language questions into raw shell commands.

## Features

- **Instant Answers:** Get the command you need without leaving the terminal.
- **Semantic Caching:** Uses embeddings (`nomic-embed-text`) to understand intent. Similar questions return instant results.
- **Self-Correcting Cache:** Running with `--no-cache` automatically updates the cache with the new, preferred result.
- **Streaming Output:** See the model's response in real-time.
- **Thought Separation:** Automatically detects and colorizes reasoning/thought blocks in dimmed cyan.
- **Automatic Clipboard Copy:** The generated command is automatically copied to your system clipboard.
- **Context Awareness:** Automatically detects your OS, Linux Distro, Shell, and Current Working Directory.
- **Command Fixer:** Pipe error output to `ask --fix` to get an explanation and a corrected command.
- **Interactive Selection:** If the model suggests multiple ways to do something, you can choose the best one from a menu.
- **Prompted Execution:** Use the `-x` flag to execute a command after a confirmation prompt.
- **Command Refinement:** Iteratively adjust commands with the `--refine` flag.
- **Command Explanation:** Get a concise technical breakdown with the `--explain-previous` flag.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (2024 edition)
- [Ollama](https://ollama.com/) installed and running.
- **Models:** 
    - `gemma4:e4b` (default for command generation)
    - `nomic-embed-text` (required for semantic caching)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ask.git
cd ask

# Build and install
cargo install --path .
```

Ensure `$HOME/.cargo/bin` is in your `PATH`.

## Usage

Basic usage:
```bash
ask "How to find new files that start with S in this dir"
```

### Command Line Options

| Option | Short | Description |
| :--- | :--- | :--- |
| `--model <MODEL>` | `-m` | Specify the Ollama model to use (overrides config). |
| `--host <HOST>` | `-o` | Specify the Ollama host URL (overrides config). |
| `--execute` | `-x` | Execute the suggested command after a confirmation prompt. |
| `--fix` | `-f` | Fix a command based on error output from stdin (e.g., `ls --wrong 2>&1 \| ask --fix`). |
| `--refine <TEXT>` | `-r` | Refine the previous command with additional instructions. |
| `--explain-previous`| `-e` | Provide a concise technical breakdown of the last generated command. |
| `--no-cache` | | Bypass the semantic cache and force a new generation (updates the cache). |
| `--no-copy` | | Disable automatic copy to clipboard for the current run. |
| `--help` | `-h` | Show help information. |
| `--version` | `-V` | Show version information. |

### Examples

**Execute immediately:**
```bash
ask "list all docker containers" -x
```

**Fix a failing command:**
```bash
# Pipe error output to ask
ls --invalid-flag 2>&1 | ask --fix
```

**Refine the last command:**
```bash
ask --refine "actually make it recursive"
```

## Configuration

- **macOS:** `~/Library/Application Support/rs.ask/default-config.toml`
- **Linux:** `~/.config/ask/default-config.toml`

### Default Settings:
```toml
model = "gemma4:e4b"
host = "http://localhost:11434"
auto_copy = true
embedding_model = "nomic-embed-text"
cache_threshold = 0.92
```

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
