use serde::{Deserialize, Serialize};
use reqwest::{Client, StatusCode};
use futures_util::StreamExt;
use std::io::{self, Write};
use colored::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OllamaError {
    #[error("Ollama is not running or unreachable at {host}. Please make sure Ollama is installed and running (https://ollama.com).")]
    NotRunning { host: String, source: reqwest::Error },
    
    #[error("Ollama API error: {0}")]
    ApiError(String),
    
    #[error("Failed to parse response from Ollama: {0}")]
    ParseError(#[from] serde_json::Error),
    
    #[error("Network error communicating with Ollama: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    
    #[error("Encoding error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),

    #[error("Data conversion error: {0}")]
    DataError(#[from] std::str::Utf8Error),
}

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("{provider} is not running or unreachable at {host}. {hint}")]
    Unreachable {
        provider: String,
        host: String,
        hint: String,
        source: Option<reqwest::Error>,
    },

    #[error("Authentication failed for {provider}: {message}")]
    AuthError {
        provider: String,
        message: String,
    },

    #[error("{provider} API error ({status}): {message}")]
    ApiError {
        provider: String,
        status: StatusCode,
        message: String,
    },

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("Encoding error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),

    #[error("Data conversion error: {0}")]
    DataError(#[from] std::str::Utf8Error),
}

impl From<OllamaError> for ClientError {
    fn from(err: OllamaError) -> Self {
        match err {
            OllamaError::NotRunning { host, source } => ClientError::Unreachable {
                provider: "Ollama".to_string(),
                host,
                hint: "Please make sure Ollama is installed and running (https://ollama.com).".to_string(),
                source: Some(source),
            },
            OllamaError::ApiError(msg) => ClientError::ApiError {
                provider: "Ollama".to_string(),
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: msg,
            },
            OllamaError::ParseError(e) => ClientError::ParseError(e.to_string()),
            OllamaError::NetworkError(e) => ClientError::NetworkError(e),
            OllamaError::IoError(e) => ClientError::IoError(e),
            OllamaError::Utf8Error(e) => ClientError::Utf8Error(e),
            OllamaError::DataError(e) => ClientError::DataError(e),
        }
    }
}

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: String,
    stream: bool,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    #[allow(dead_code)]
    done: bool,
}

fn default_http_client() -> Client {
    Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new())
}

fn decode_utf8_chunk(
    raw_buffer: &mut Vec<u8>,
    chunk: &[u8],
    str_buffer: &mut String,
) -> Result<(), std::str::Utf8Error> {
    raw_buffer.extend_from_slice(chunk);
    match std::str::from_utf8(raw_buffer) {
        Ok(valid_str) => {
            str_buffer.push_str(valid_str);
            raw_buffer.clear();
            Ok(())
        }
        Err(e) => {
            let valid = e.valid_up_to();
            if valid > 0 {
                let valid_str = std::str::from_utf8(&raw_buffer[..valid]).unwrap();
                str_buffer.push_str(valid_str);
                raw_buffer.drain(..valid);
            }
            if e.error_len().is_some() {
                std::str::from_utf8(raw_buffer).map(|_| ())
            } else {
                Ok(())
            }
        }
    }
}

fn think_tag_prefix_len(s: &str) -> usize {
    let tag = "<think>";
    for len in (1..tag.len()).rev() {
        if s.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
}

pub struct OllamaClient {
    host: String,
    model: String,
    client: Client,
}

impl OllamaClient {
    pub fn new(host: String, model: String) -> Self {
        Self {
            host,
            model,
            client: default_http_client(),
        }
    }

    pub async fn get_embeddings(&self, model: &str, prompt: &str) -> Result<Vec<f32>, OllamaError> {
        let request_payload = EmbeddingRequest {
            model,
            prompt: prompt.to_string(),
        };

        let url = format!("{}/api/embeddings", self.host);
        let response = self.client
            .post(&url)
            .json(&request_payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() || e.is_timeout() {
                    OllamaError::NotRunning { host: self.host.clone(), source: e }
                } else {
                    OllamaError::NetworkError(e)
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            if status == StatusCode::NOT_FOUND {
                return Err(OllamaError::ApiError(format!("Model '{}' not found. You may need to pull it first using 'ollama pull {}'", model, model)));
            }
            return Err(OllamaError::ApiError(status.to_string()));
        }

        let emb_response: EmbeddingResponse = response.json().await?;
        Ok(emb_response.embedding)
    }

    pub async fn stream_command(&self, question: &str) -> Result<String, OllamaError> {
        let (os, shell, cwd) = self.get_env_context();
        let prompt = format!(
            "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n\n\
             Task: Provide the most common command line command for this query. \
             If there are other very distinct or necessary alternative ways, you may list them as well, each on a new line starting with 'Command: '. \
             Otherwise, provide just the single raw command.\n\n\
             Ensure the command is compatible with the specified OS and Shell. \
             Do not include any markdown formatting, backticks, or explanations. \
             Query: {}",
            os, shell, cwd, question
        );

        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn refine_command(&self, last_command: &str, refinement: &str) -> Result<String, OllamaError> {
        let (os, shell, cwd) = self.get_env_context();
        let prompt = format!(
            "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n- Previous Command: {}\n\n\
             Task: Based on the previous command and the following refinement instruction, provide the most direct updated command line command. \
             If there are other very distinct or necessary alternative refinements, you may list them as well, each on a new line starting with 'Command: '. \
             Otherwise, provide just the single raw command.\n\n\
             Instruction: {}\n\
             Ensure the command is compatible with the specified OS and Shell. \
             Do not include any markdown formatting, backticks, or explanations.",
            os, shell, cwd, last_command, refinement
        );

        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn explain_command(&self, command: &str) -> Result<(), OllamaError> {
        let os = std::env::consts::OS;
        let prompt = format!(
            "Context:\n- Operating System: {}\n\n\
             Task: Provide a concise, high-signal technical breakdown of this command. \
             Assume the user is an expert CLI user. Focus on flag functions and non-obvious behavior. \
             No introductory fluff or basic definitions. \
             Command: {}",
            os, command
        );

        self.stream_raw(&prompt, &mut io::stdout()).await?;
        Ok(())
    }

    pub async fn fix_command(&self, error_output: &str) -> Result<String, OllamaError> {
        let (os, shell, cwd) = self.get_env_context();
        let prompt = format!(
            "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n\n\
             Task: The following error occurred. \
             1. Shortly explain what went wrong (max 2 sentences). \
             2. Provide the fixed raw command line command. \
             Ensure the command is compatible with the specified OS and Shell.\n\n\
             Error Output:\n{}\n\n\
             Format your response exactly like this:\n\
             Explanation: [Your short explanation]\n\
             Command: [The raw command]",
            os, shell, cwd, error_output
        );

        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    fn get_env_context(&self) -> (String, String, String) {
        get_env_context()
    }

    async fn stream_raw<W: Write>(&self, prompt: &str, writer: &mut W) -> Result<String, OllamaError> {
        let request_payload = GenerateRequest {
            model: &self.model,
            prompt: prompt.to_string(),
            stream: true,
        };

        let url = format!("{}/api/generate", self.host);
        let response = self.client
            .post(&url)
            .json(&request_payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() || e.is_timeout() {
                    OllamaError::NotRunning { host: self.host.clone(), source: e }
                } else {
                    OllamaError::NetworkError(e)
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            if status == StatusCode::NOT_FOUND {
                return Err(OllamaError::ApiError(format!("Model '{}' not found. You may need to pull it first using 'ollama pull {}'", self.model, self.model)));
            }
            return Err(OllamaError::ApiError(status.to_string()));
        }

        let mut response_stream = response.bytes_stream();

        let mut in_thought = false;
        let mut raw_buffer = Vec::new();
        let mut response_buffer = String::new(); // Buffer for JSON lines
        let mut content_buffer = String::new();  // Buffer for raw content (thinking/command)
        let mut final_response = String::new();   // Accumulate response for return

        while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result.map_err(OllamaError::NetworkError)?;
            decode_utf8_chunk(&mut raw_buffer, &chunk, &mut response_buffer)?;
            
            while let Some(pos) = response_buffer.find('\n') {
                let line = response_buffer[..pos].to_string();
                response_buffer = response_buffer[pos + 1..].to_string();
                
                if line.is_empty() { continue; }
                let gen_response: GenerateResponse = serde_json::from_str(&line)?;
                let content = gen_response.response;
                content_buffer.push_str(&content);

                if !in_thought && content_buffer.contains("<think>") {
                    in_thought = true;
                    if let Some(pos) = content_buffer.find("<think>") {
                        let pre_thought = &content_buffer[..pos];
                        write!(writer, "{}", pre_thought)?;
                        final_response.push_str(pre_thought);
                        content_buffer = content_buffer[pos + 7..].to_string();
                    }
                }

                if in_thought && content_buffer.contains("</think>") {
                    if let Some(pos) = content_buffer.find("</think>") {
                        let thought = &content_buffer[..pos];
                        write!(writer, "{}", thought.dimmed().cyan())?;
                        write!(writer, "\n{}", "---".dimmed().cyan())?;
                        in_thought = false;
                        content_buffer = content_buffer[pos + 8..].to_string();
                    }
                } else if in_thought {
                    if content_buffer.len() > 8 {
                        let mut split_idx = content_buffer.len() - 8;
                        while !content_buffer.is_char_boundary(split_idx) {
                            split_idx -= 1;
                        }
                        if split_idx > 0 {
                            let to_print = &content_buffer[..split_idx];
                            write!(writer, "{}", to_print.dimmed().cyan())?;
                            content_buffer = content_buffer[split_idx..].to_string();
                        }
                    }
                } else {
                    let prefix_len = think_tag_prefix_len(&content_buffer);
                    if prefix_len > 0 {
                        let flush_len = content_buffer.len() - prefix_len;
                        if flush_len > 0 {
                            let to_print = &content_buffer[..flush_len];
                            write!(writer, "{}", to_print)?;
                            final_response.push_str(to_print);
                            content_buffer = content_buffer[flush_len..].to_string();
                        }
                    } else {
                        write!(writer, "{}", content_buffer)?;
                        final_response.push_str(&content_buffer);
                        content_buffer.clear();
                    }
                }
                writer.flush()?;
            }
        }
        
        if !content_buffer.is_empty() {
            if in_thought {
                write!(writer, "{}", content_buffer.dimmed().cyan())?;
            } else {
                write!(writer, "{}", content_buffer)?;
                final_response.push_str(&content_buffer);
            }
            writer.flush()?;
        }
        writeln!(writer)?;

        Ok(final_response.trim().to_string())
    }
}

pub fn get_env_context() -> (String, String, String) {
    let os_name = std::env::consts::OS;
    let mut distro = String::new();

    if os_name == "linux"
        && let Ok(content) = std::fs::read_to_string("/etc/os-release")
    {
        for line in content.lines() {
            if line.starts_with("PRETTY_NAME=") {
                distro = line.replace("PRETTY_NAME=", "").replace('\"', "").to_string();
                break;
            }
        }
    }

    let context_os = if distro.is_empty() {
        os_name.to_string()
    } else {
        format!("{} ({})", os_name, distro)
    };

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    (context_os, shell, cwd)
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
}

#[derive(Deserialize, Default, Debug)]
struct ChatDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    #[serde(default)]
    delta: ChatDelta,
}

#[derive(Deserialize, Debug)]
struct StreamError {
    message: String,
    #[serde(default)]
    code: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    error: Option<StreamError>,
}

pub struct OpenAiClient {
    pub provider_name: String,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub client: Client,
}

impl OpenAiClient {
    pub fn new(provider_name: String, base_url: String, model: String, api_key: Option<String>) -> Self {
        Self {
            provider_name,
            base_url,
            model,
            api_key,
            client: default_http_client(),
        }
    }

    pub async fn stream_command(&self, question: &str) -> Result<String, ClientError> {
        let (os, shell, cwd) = get_env_context();
        let prompt = format!(
            "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n\n\
             Task: Provide the most common command line command for this query. \
             If there are other very distinct or necessary alternative ways, you may list them as well, each on a new line starting with 'Command: '. \
             Otherwise, provide just the single raw command.\n\n\
             Ensure the command is compatible with the specified OS and Shell. \
             Do not include any markdown formatting, backticks, or explanations. \
             Query: {}",
            os, shell, cwd, question
        );

        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn refine_command(&self, last_command: &str, refinement: &str) -> Result<String, ClientError> {
        let (os, shell, cwd) = get_env_context();
        let prompt = format!(
            "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n- Previous Command: {}\n\n\
             Task: Based on the previous command and the following refinement instruction, provide the most direct updated command line command. \
             If there are other very distinct or necessary alternative refinements, you may list them as well, each on a new line starting with 'Command: '. \
             Otherwise, provide just the single raw command.\n\n\
             Instruction: {}\n\
             Ensure the command is compatible with the specified OS and Shell. \
             Do not include any markdown formatting, backticks, or explanations.",
            os, shell, cwd, last_command, refinement
        );

        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn explain_command(&self, command: &str) -> Result<(), ClientError> {
        let os = std::env::consts::OS;
        let prompt = format!(
            "Context:\n- Operating System: {}\n\n\
             Task: Provide a concise, high-signal technical breakdown of this command. \
             Assume the user is an expert CLI user. Focus on flag functions and non-obvious behavior. \
             No introductory fluff or basic definitions. \
             Command: {}",
            os, command
        );

        self.stream_raw(&prompt, &mut io::stdout()).await?;
        Ok(())
    }

    pub async fn fix_command(&self, error_output: &str) -> Result<String, ClientError> {
        let (os, shell, cwd) = get_env_context();
        let prompt = format!(
            "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n\n\
             Task: The following error occurred. \
             1. Shortly explain what went wrong (max 2 sentences). \
             2. Provide the fixed raw command line command. \
             Ensure the command is compatible with the specified OS and Shell.\n\n\
             Error Output:\n{}\n\n\
             Format your response exactly like this:\n\
             Explanation: [Your short explanation]\n\
             Command: [The raw command]",
            os, shell, cwd, error_output
        );

        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn stream_raw<W: Write>(&self, prompt: &str, writer: &mut W) -> Result<String, ClientError> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let request_payload = ChatCompletionRequest {
            model: &self.model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            stream: true,
        };

        let mut req = self.client.post(&url).json(&request_payload);

        if let Some(key) = self.api_key.as_deref().map(str::trim).filter(|k| !k.is_empty()) {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        if self.base_url.contains("openrouter.ai") || self.provider_name.to_lowercase().contains("openrouter") {
            req = req.header("HTTP-Referer", "https://github.com/veskuh/ask");
            req = req.header("X-Title", "ask");
        }

        let response = req.send().await.map_err(|e| {
            if e.is_connect() || e.is_timeout() {
                ClientError::Unreachable {
                    provider: self.provider_name.clone(),
                    host: self.base_url.clone(),
                    hint: format!("Could not connect to {}. Please check your network connection and URL.", self.provider_name),
                    source: Some(e),
                }
            } else {
                ClientError::NetworkError(e)
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
                let env_hint = if self.provider_name.to_lowercase().contains("openrouter") {
                    "OPENROUTER_API_KEY"
                } else {
                    "OPENAI_API_KEY"
                };
                return Err(ClientError::AuthError {
                    provider: self.provider_name.clone(),
                    message: format!(
                        "Invalid or missing API key (HTTP {}): {}. Set {} or use 'ask config --set-api-key <key>'.",
                        status, error_text, env_hint
                    ),
                });
            }
            if status == StatusCode::NOT_FOUND {
                return Err(ClientError::ApiError {
                    provider: self.provider_name.clone(),
                    status,
                    message: format!("Endpoint or model '{}' not found at {}. Response: {}", self.model, self.base_url, error_text),
                });
            }
            return Err(ClientError::ApiError {
                provider: self.provider_name.clone(),
                status,
                message: error_text,
            });
        }

        let mut response_stream = response.bytes_stream();
        let mut raw_buffer = Vec::new();
        let mut line_buffer = String::new();

        let mut in_reasoning_field = false;
        let mut in_think_tag = false;
        let mut printed_thought_separator = false;
        let mut content_buffer = String::new();
        let mut final_response = String::new();

        'stream: while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result.map_err(ClientError::NetworkError)?;
            decode_utf8_chunk(&mut raw_buffer, &chunk, &mut line_buffer)?;

            while let Some(pos) = line_buffer.find('\n') {
                let raw_line = line_buffer[..pos].trim_end_matches('\r').to_string();
                line_buffer = line_buffer[pos + 1..].to_string();

                let trimmed = raw_line.trim();
                if trimmed.is_empty() || trimmed.starts_with(':') {
                    continue;
                }

                if let Some(data) = trimmed.strip_prefix("data:") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        break 'stream;
                    }

                    let chunk: ChatChunk = match serde_json::from_str(data) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };

                    if let Some(err) = chunk.error {
                        let message = match err.code {
                            Some(code) => format!("{}: {}", code, err.message),
                            None => err.message,
                        };
                        return Err(ClientError::ApiError {
                            provider: self.provider_name.clone(),
                            status: StatusCode::BAD_REQUEST,
                            message,
                        });
                    }

                    for choice in chunk.choices {
                        let reasoning_delta = choice.delta.reasoning_content.or(choice.delta.reasoning);

                        if let Some(thought) = reasoning_delta.filter(|t| !t.is_empty()) {
                            in_reasoning_field = true;
                            write!(writer, "{}", thought.dimmed().cyan())?;
                            writer.flush()?;
                        }

                        if let Some(content) = choice.delta.content.filter(|c| !c.is_empty()) {
                            if in_reasoning_field {
                                write!(writer, "\n{}", "---".dimmed().cyan())?;
                                writeln!(writer)?;
                                writer.flush()?;
                                in_reasoning_field = false;
                                printed_thought_separator = true;
                            }

                            content_buffer.push_str(&content);

                            if !in_think_tag && content_buffer.contains("<think>") {
                                in_think_tag = true;
                                if let Some(pos) = content_buffer.find("<think>") {
                                    let pre_thought = &content_buffer[..pos];
                                    write!(writer, "{}", pre_thought)?;
                                    final_response.push_str(pre_thought);
                                    content_buffer = content_buffer[pos + 7..].to_string();
                                }
                            }

                            if in_think_tag && content_buffer.contains("</think>") {
                                if let Some(pos) = content_buffer.find("</think>") {
                                    let thought = &content_buffer[..pos];
                                    write!(writer, "{}", thought.dimmed().cyan())?;
                                    write!(writer, "\n{}", "---".dimmed().cyan())?;
                                    in_think_tag = false;
                                    printed_thought_separator = true;
                                    content_buffer = content_buffer[pos + 8..].to_string();
                                }
                            } else if in_think_tag {
                                if content_buffer.len() > 8 {
                                    let mut split_idx = content_buffer.len() - 8;
                                    while !content_buffer.is_char_boundary(split_idx) {
                                        split_idx -= 1;
                                    }
                                    if split_idx > 0 {
                                        let to_print = &content_buffer[..split_idx];
                                        write!(writer, "{}", to_print.dimmed().cyan())?;
                                        content_buffer = content_buffer[split_idx..].to_string();
                                    }
                                }
                            } else {
                                let prefix_len = think_tag_prefix_len(&content_buffer);
                                if prefix_len > 0 {
                                    let flush_len = content_buffer.len() - prefix_len;
                                    if flush_len > 0 {
                                        let to_print = &content_buffer[..flush_len];
                                        write!(writer, "{}", to_print)?;
                                        final_response.push_str(to_print);
                                        content_buffer = content_buffer[flush_len..].to_string();
                                    }
                                } else {
                                    write!(writer, "{}", content_buffer)?;
                                    final_response.push_str(&content_buffer);
                                    content_buffer.clear();
                                }
                            }
                            writer.flush()?;
                        }
                    }
                }
            }
        }

        if in_reasoning_field && !printed_thought_separator {
            write!(writer, "\n{}", "---".dimmed().cyan())?;
            writeln!(writer)?;
            writer.flush()?;
        }

        if !content_buffer.is_empty() {
            if in_think_tag {
                write!(writer, "{}", content_buffer.dimmed().cyan())?;
            } else {
                write!(writer, "{}", content_buffer)?;
                final_response.push_str(&content_buffer);
            }
            writer.flush()?;
        }
        writeln!(writer)?;

        Ok(final_response.trim().to_string())
    }
}

pub enum LlmClient {
    Ollama(OllamaClient),
    OpenAi(OpenAiClient),
}

impl LlmClient {
    pub fn provider_name(&self) -> &str {
        match self {
            LlmClient::Ollama(_) => "Ollama",
            LlmClient::OpenAi(c) => &c.provider_name,
        }
    }

    pub fn model(&self) -> &str {
        match self {
            LlmClient::Ollama(c) => &c.model,
            LlmClient::OpenAi(c) => &c.model,
        }
    }

    pub fn host(&self) -> &str {
        match self {
            LlmClient::Ollama(c) => &c.host,
            LlmClient::OpenAi(c) => &c.base_url,
        }
    }

    pub async fn stream_command(&self, question: &str) -> Result<String, ClientError> {
        match self {
            LlmClient::Ollama(c) => c.stream_command(question).await.map_err(Into::into),
            LlmClient::OpenAi(c) => c.stream_command(question).await,
        }
    }

    pub async fn refine_command(&self, last_command: &str, refinement: &str) -> Result<String, ClientError> {
        match self {
            LlmClient::Ollama(c) => c.refine_command(last_command, refinement).await.map_err(Into::into),
            LlmClient::OpenAi(c) => c.refine_command(last_command, refinement).await,
        }
    }

    pub async fn explain_command(&self, command: &str) -> Result<(), ClientError> {
        match self {
            LlmClient::Ollama(c) => c.explain_command(command).await.map_err(Into::into),
            LlmClient::OpenAi(c) => c.explain_command(command).await,
        }
    }

    pub async fn fix_command(&self, error_output: &str) -> Result<String, ClientError> {
        match self {
            LlmClient::Ollama(c) => c.fix_command(error_output).await.map_err(Into::into),
            LlmClient::OpenAi(c) => c.fix_command(error_output).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_client_creation() {
        let client = OllamaClient::new("http://localhost:11434".to_string(), "test-model".to_string());
        assert_eq!(client.host, "http://localhost:11434");
        assert_eq!(client.model, "test-model");
    }

    #[tokio::test]
    async fn test_stream_raw_success() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let mock_body = "{\"response\": \"ls\", \"done\": false}\n{\"response\": \" -la\", \"done\": true}\n";
        let _mock = server.mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_body)
            .create_async().await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "ls -la");
        assert_eq!(String::from_utf8(output).unwrap().trim(), "ls -la");
    }

    #[tokio::test]
    async fn test_stream_raw_with_think() {
        colored::control::set_override(false);
        let mut server = Server::new_async().await;
        let url = server.url();

        let mock_body = "{\"response\": \"<think>\", \"done\": false}\n\
                         {\"response\": \"I should use ls\", \"done\": false}\n\
                         {\"response\": \"</think>\", \"done\": false}\n\
                         {\"response\": \"ls -la\", \"done\": true}\n";
        let _mock = server.mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_body)
            .create_async().await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "ls -la");
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("I should use ls"));
        assert!(output_str.contains("---"));
        assert!(output_str.contains("ls -la"));
    }

    #[tokio::test]
    async fn test_stream_command_error() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server.mock("POST", "/api/generate")
            .with_status(500)
            .create_async().await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let mut output = Vec::new();
        let result = client.stream_raw("list files", &mut output).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_get_env_context() {
        let client = OllamaClient::new("host".to_string(), "model".to_string());
        let (os, shell, cwd) = client.get_env_context();
        
        assert!(!os.is_empty());
        assert!(!shell.is_empty());
        assert!(!cwd.is_empty());
    }

    #[tokio::test]
    async fn test_fix_command() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let mock_body = "{\"response\": \"Explanation: wrong flag\\nCommand: ls\", \"done\": true}\n";
        let _mock = server.mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_body)
            .create_async().await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let result = client.fix_command("error output").await.unwrap();
        assert_eq!(result, "Explanation: wrong flag\nCommand: ls");
    }

    #[tokio::test]
    async fn test_openai_stream_raw_success() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"git \"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"status\"}}]}\n\n\
                        data: [DONE]\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async().await;

        let client = OpenAiClient::new("OpenRouter".to_string(), url, "deepseek/deepseek-v4-flash".to_string(), Some("test-key".to_string()));
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "git status");
        assert_eq!(String::from_utf8(output).unwrap().trim(), "git status");
    }

    #[tokio::test]
    async fn test_openai_stream_raw_with_reasoning_content() {
        colored::control::set_override(false);
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"I should check git status\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"git status\"}}]}\n\n\
                        data: [DONE]\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async().await;

        let client = OpenAiClient::new("OpenRouter".to_string(), url, "deepseek/deepseek-v4-flash".to_string(), Some("test-key".to_string()));
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "git status");
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("I should check git status"));
        assert!(output_str.contains("---"));
        assert!(output_str.contains("git status"));
    }

    #[tokio::test]
    async fn test_openai_stream_raw_with_think_tag() {
        colored::control::set_override(false);
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"<think>thinking about docker</think>docker ps\"}}]}\n\n\
                        data: [DONE]\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async().await;

        let client = OpenAiClient::new("OpenAI".to_string(), url, "gpt-4o-mini".to_string(), None);
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "docker ps");
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("thinking about docker"));
        assert!(output_str.contains("---"));
        assert!(output_str.contains("docker ps"));
    }

    #[tokio::test]
    async fn test_openai_auth_error() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("{\"error\": \"Unauthorized\"}")
            .create_async().await;

        let client = OpenAiClient::new("OpenRouter".to_string(), url, "deepseek/deepseek-v4-flash".to_string(), None);
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await;

        assert!(matches!(result, Err(ClientError::AuthError { .. })));
    }

    #[tokio::test]
    async fn test_openai_stream_error_in_sse() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"error\":{\"message\":\"Provider returned error: Insufficient credits\",\"code\":402}}\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async().await;

        let client = OpenAiClient::new("OpenRouter".to_string(), url, "deepseek/deepseek-v4-flash".to_string(), None);
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Insufficient credits"));
    }

    #[tokio::test]
    async fn test_stream_raw_with_multibyte_utf8_think() {
        colored::control::set_override(false);
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"<think>1234567\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"🦀\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"890</think>echo hi\"}}]}\n\n\
                        data: [DONE]\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async().await;

        let client = OpenAiClient::new("OpenAI".to_string(), url, "gpt-4o-mini".to_string(), None);
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "echo hi");
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("🦀"));
        assert!(output_str.contains("echo hi"));
    }

    #[tokio::test]
    async fn test_openai_stream_raw_split_think_tag() {
        colored::control::set_override(false);
        let mut server = Server::new_async().await;
        let url = server.url();

        // <think> is split: "<th" in chunk 1, "ink>reasoning</think>ls" in chunk 2
        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"<th\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"ink>reasoning here</think>ls -la\"}}]}\n\n\
                        data: [DONE]\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async().await;

        let client = OpenAiClient::new("OpenAI".to_string(), url, "gpt-4o-mini".to_string(), None);
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "ls -la");
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("reasoning here"));
        assert!(output_str.contains("---"));
        assert!(output_str.contains("ls -la"));
    }

    #[test]
    fn test_decode_utf8_chunk_split_code_point() {
        let mut raw_buf = Vec::new();
        let mut str_buf = String::new();

        // 🦀 emoji in UTF-8: [240, 159, 166, 128]
        let part1 = &[240, 159];
        let part2 = &[166, 128];

        assert!(decode_utf8_chunk(&mut raw_buf, part1, &mut str_buf).is_ok());
        assert_eq!(str_buf, "");
        assert_eq!(raw_buf, part1);

        assert!(decode_utf8_chunk(&mut raw_buf, part2, &mut str_buf).is_ok());
        assert_eq!(str_buf, "🦀");
        assert!(raw_buf.is_empty());
    }

    #[tokio::test]
    async fn test_openai_auth_error_hints() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("{\"error\": \"Unauthorized\"}")
            .create_async().await;

        let client_openai = OpenAiClient::new("OpenAI".to_string(), url.clone(), "gpt-4o-mini".to_string(), None);
        let mut out = Vec::new();
        let err_openai = client_openai.stream_raw("test", &mut out).await.unwrap_err();
        assert!(err_openai.to_string().contains("OPENAI_API_KEY"));

        let client_or = OpenAiClient::new("OpenRouter".to_string(), url, "deepseek/deepseek-v4-flash".to_string(), None);
        let mut out2 = Vec::new();
        let err_or = client_or.stream_raw("test", &mut out2).await.unwrap_err();
        assert!(err_or.to_string().contains("OPENROUTER_API_KEY"));
    }

    #[test]
    fn test_ollama_error_conversions() {
        let err_not_running = OllamaError::NotRunning {
            host: "http://localhost:11434".to_string(),
            source: reqwest::Client::new().get("not a valid url").build().unwrap_err(),
        };
        let client_err: ClientError = err_not_running.into();
        assert!(client_err.to_string().contains("Ollama"));

        let err_api = OllamaError::ApiError("model not found".to_string());
        let client_err: ClientError = err_api.into();
        assert!(client_err.to_string().contains("model not found"));

        let err_parse = OllamaError::ParseError(serde_json::from_str::<String>("not json").unwrap_err());
        let client_err: ClientError = err_parse.into();
        assert!(matches!(client_err, ClientError::ParseError(_)));

        let err_io = OllamaError::IoError(std::io::Error::other("io error"));
        let client_err: ClientError = err_io.into();
        assert!(matches!(client_err, ClientError::IoError(_)));
    }

    #[tokio::test]
    async fn test_ollama_get_embeddings() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server.mock("POST", "/api/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"embedding\": [0.1, 0.2, 0.3]}")
            .create_async().await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let emb = client.get_embeddings("nomic-embed-text", "find files").await.unwrap();
        assert_eq!(emb, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_ollama_stream_refine_explain() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server.mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/x-ndjson")
            .with_body("{\"response\":\"echo 1\",\"done\":false}\n{\"response\":\"\",\"done\":true}\n")
            .create_async().await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let res = client.stream_command("how to echo 1").await.unwrap();
        assert_eq!(res, "echo 1");
    }

    #[tokio::test]
    async fn test_openai_methods_and_llm_client_dispatch() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"echo 42\"}}]}\n\ndata: [DONE]\n\n";

        let _mock = server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async().await;

        let openai_client = OpenAiClient::new(
            "OpenRouter".to_string(),
            url.clone(),
            "test-model".to_string(),
            Some("key".to_string()),
        );

        // Test OpenAiClient methods
        let cmd = openai_client.stream_command("test").await.unwrap();
        assert_eq!(cmd, "echo 42");

        // Test LlmClient dispatch
        let llm = LlmClient::OpenAi(openai_client);
        assert_eq!(llm.provider_name(), "OpenRouter");
        assert_eq!(llm.model(), "test-model");
        assert_eq!(llm.host(), url);

        let ollama = LlmClient::Ollama(OllamaClient::new("http://localhost:11434".to_string(), "ollama-model".to_string()));
        assert_eq!(ollama.provider_name(), "Ollama");
        assert_eq!(ollama.model(), "ollama-model");
        assert_eq!(ollama.host(), "http://localhost:11434");
    }
}
