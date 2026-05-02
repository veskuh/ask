# ask

A simple Rust-based command-line helper for remembering terminal commands. It uses a locally hosted Ollama model to translate natural language questions into raw shell commands.

## Features

- **Instant Answers:** Get the command you need without leaving the terminal.
- **Streaming Output:** See the model's response in real-time.
- **Thought Separation:** Automatically detects and colorizes reasoning/thought blocks (e.g., from models like DeepSeek-R1 or Qwen2.5-Coder) in dimmed cyan.
- **Automatic Clipboard Copy:** The generated command is automatically copied to your system clipboard for instant use.
- **Context Awareness:** Automatically detects your OS, Linux Distro, Shell, and Current Working Directory to ensure commands are compatible and relevant.
- **Persistent Configuration:** Save your preferred model and host settings in a config file.
- **Command Fixer:** Pipe error output to `ask --fix` to get an explanation and a corrected command.
- **Command Refinement:** Use the `--refine` flag to iteratively adjust the last generated command.
- **Command Explanation:** Use the `--explain-previous` flag to get a detailed breakdown of the last generated command.
- **Prompted Execution:** Use the `-x` flag to immediately execute the suggested command after a confirmation prompt.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (2024 edition)
- [Ollama](https://ollama.com/) installed and running.
- A model downloaded (default is `gemma4:e4b`, but works great with `codellama`, `mistral`, `deepseek-r1`, etc.).

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

Execute a command immediately (with confirmation):
```bash
ask "list all docker containers" -x
```

Fix a failing command:
```bash
# Pipe error output to ask
ls --invalid-flag 2>&1 | ask --fix
# or
ask -f < error.log
```

Refine the last command:
```bash
ask --refine "actually make it recursive"
```

Explain the last command:
```bash
ask --explain-previous
```

### Options
- `-x, --execute`: Execute the suggested command after confirmation.
- `-f, --fix`: Fix a command based on error output from stdin.
- `-r, --refine <INSTRUCTION>`: Refine the previous command.
- `-e, --explain-previous`: Explain the previous command.
- `-m, --model <MODEL>`: Ollama model to use (overrides config).
- `-o, --host <HOST>`: Ollama host URL (overrides config).
- `--no-copy`: Disable automatic copy to clipboard for this run.
- `-h, --help`: Show help information.

## Configuration & State

- **Config File:** `~/Library/Application Support/rs.ask/config.toml` (macOS) or `~/.config/ask/config.toml` (Linux).
- **State File:** Stores the last command for the refine and explain features.

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
