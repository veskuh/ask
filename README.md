# ask

A simple Rust-based command-line helper for remembering terminal commands. It uses a locally hosted Ollama model to translate natural language questions into raw shell commands.

## Features

- **Instant Answers:** Get the command you need without leaving the terminal.
- **Streaming Output:** See the model's response in real-time.
- **Thought Separation:** Automatically detects and colorizes reasoning/thought blocks (e.g., from models like DeepSeek-R1 or Qwen2.5-Coder) and separates them from the final command.
- **Customizable:** Easily change the Ollama model or host via CLI flags.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (2024 edition)
- [Ollama](https://ollama.com/) installed and running.
- A model downloaded (default is `qwen3:8b`, but works great with `codellama`, `mistral`, etc.).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ask.git
cd ask

# Build and install
cargo install --path .
```

## Usage

Basic usage:
```bash
ask "How to find new files that start with S in this dir"
```

Specifying a different model or host:
```bash
ask "list all docker containers" --model codellama --host http://192.168.1.10:11434
```

### Options
- `-m, --model <MODEL>`: Ollama model to use (default: `qwen3:8b`).
- `-o, --host <HOST>`: Ollama host URL (default: `http://localhost:11434`).
- `-h, --help`: Show help information.

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
