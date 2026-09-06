//! OpenAI and OpenRouter SSE chat completions engine and streaming parser.

use super::ClientError;
use super::stream::{decode_utf8_chunk, default_http_client, think_tag_prefix_len};
use crate::prompt;
use colored::*;
use futures_util::StreamExt;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

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
    pub fn new(
        provider_name: String,
        base_url: String,
        model: String,
        api_key: Option<String>,
    ) -> Self {
        Self {
            provider_name,
            base_url,
            model,
            api_key,
            client: default_http_client(),
        }
    }

    pub async fn stream_command(&self, question: &str) -> Result<String, ClientError> {
        let prompt = prompt::build_command_prompt(question);
        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn refine_command(
        &self,
        last_command: &str,
        refinement: &str,
    ) -> Result<String, ClientError> {
        let prompt = prompt::build_refine_prompt(last_command, refinement);
        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn explain_command(&self, command: &str) -> Result<(), ClientError> {
        let prompt = prompt::build_explain_prompt(command);
        self.stream_raw(&prompt, &mut io::stdout()).await?;
        Ok(())
    }

    pub async fn fix_command(&self, error_output: &str) -> Result<String, ClientError> {
        let prompt = prompt::build_fix_prompt(error_output);
        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn stream_raw<W: Write>(
        &self,
        prompt: &str,
        writer: &mut W,
    ) -> Result<String, ClientError> {
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

        if let Some(key) = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
        {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        if self.base_url.contains("openrouter.ai")
            || self.provider_name.to_lowercase().contains("openrouter")
        {
            req = req.header("HTTP-Referer", "https://github.com/veskuh/ask");
            req = req.header("X-Title", "ask");
        }

        let response = req.send().await.map_err(|e| {
            if e.is_connect() || e.is_timeout() {
                ClientError::Unreachable {
                    provider: self.provider_name.clone(),
                    host: self.base_url.clone(),
                    hint: format!(
                        "Could not connect to {}. Please check your network connection and URL.",
                        self.provider_name
                    ),
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
                    message: format!(
                        "Endpoint or model '{}' not found at {}. Response: {}",
                        self.model, self.base_url, error_text
                    ),
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
                        let reasoning_delta =
                            choice.delta.reasoning_content.or(choice.delta.reasoning);

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

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_openai_stream_raw_success() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"git \"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"status\"}}]}\n\n\
                        data: [DONE]\n\n";

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

        let client = OpenAiClient::new(
            "OpenRouter".to_string(),
            url,
            "deepseek/deepseek-v4-flash".to_string(),
            Some("test-key".to_string()),
        );
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

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

        let client = OpenAiClient::new(
            "OpenRouter".to_string(),
            url,
            "deepseek/deepseek-v4-flash".to_string(),
            Some("test-key".to_string()),
        );
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

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

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

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("{\"error\": \"Unauthorized\"}")
            .create_async()
            .await;

        let client = OpenAiClient::new(
            "OpenRouter".to_string(),
            url,
            "deepseek/deepseek-v4-flash".to_string(),
            None,
        );
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await;

        assert!(matches!(result, Err(ClientError::AuthError { .. })));
    }

    #[tokio::test]
    async fn test_openai_stream_error_in_sse() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body = "data: {\"error\":{\"message\":\"Provider returned error: Insufficient credits\",\"code\":402}}\n\n";

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

        let client = OpenAiClient::new(
            "OpenRouter".to_string(),
            url,
            "deepseek/deepseek-v4-flash".to_string(),
            None,
        );
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

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

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

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

        let client = OpenAiClient::new("OpenAI".to_string(), url, "gpt-4o-mini".to_string(), None);
        let mut output = Vec::new();
        let result = client.stream_raw("test prompt", &mut output).await.unwrap();

        assert_eq!(result, "ls -la");
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("reasoning here"));
        assert!(output_str.contains("---"));
        assert!(output_str.contains("ls -la"));
    }

    #[tokio::test]
    async fn test_openai_auth_error_hints() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("{\"error\": \"Unauthorized\"}")
            .create_async()
            .await;

        let client_openai = OpenAiClient::new(
            "OpenAI".to_string(),
            url.clone(),
            "gpt-4o-mini".to_string(),
            None,
        );
        let mut out = Vec::new();
        let err_openai = client_openai
            .stream_raw("test", &mut out)
            .await
            .unwrap_err();
        assert!(err_openai.to_string().contains("OPENAI_API_KEY"));

        let client_or = OpenAiClient::new(
            "OpenRouter".to_string(),
            url,
            "deepseek/deepseek-v4-flash".to_string(),
            None,
        );
        let mut out2 = Vec::new();
        let err_or = client_or.stream_raw("test", &mut out2).await.unwrap_err();
        assert!(err_or.to_string().contains("OPENROUTER_API_KEY"));
    }
}
