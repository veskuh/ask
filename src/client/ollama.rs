//! Ollama local LLM provider client, serialization types, and streaming engine.

use super::stream::{decode_utf8_chunk, default_http_client, think_tag_prefix_len};
use crate::prompt;
use colored::*;
use futures_util::StreamExt;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OllamaError {
    #[error(
        "Ollama is not running or unreachable at {host}. Please make sure Ollama is installed and running (https://ollama.com)."
    )]
    NotRunning {
        host: String,
        source: reqwest::Error,
    },

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

pub struct OllamaClient {
    pub host: String,
    pub model: String,
    pub client: Client,
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
        let response = self
            .client
            .post(&url)
            .json(&request_payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() || e.is_timeout() {
                    OllamaError::NotRunning {
                        host: self.host.clone(),
                        source: e,
                    }
                } else {
                    OllamaError::NetworkError(e)
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            if status == StatusCode::NOT_FOUND {
                return Err(OllamaError::ApiError(format!(
                    "Model '{}' not found. You may need to pull it first using 'ollama pull {}'",
                    model, model
                )));
            }
            return Err(OllamaError::ApiError(status.to_string()));
        }

        let emb_response: EmbeddingResponse = response.json().await?;
        Ok(emb_response.embedding)
    }

    pub async fn stream_command(&self, question: &str) -> Result<String, OllamaError> {
        let prompt = prompt::build_command_prompt(question);
        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn refine_command(
        &self,
        last_command: &str,
        refinement: &str,
    ) -> Result<String, OllamaError> {
        let prompt = prompt::build_refine_prompt(last_command, refinement);
        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub async fn explain_command(&self, command: &str) -> Result<(), OllamaError> {
        let prompt = prompt::build_explain_prompt(command);
        self.stream_raw(&prompt, &mut io::stdout()).await?;
        Ok(())
    }

    pub async fn fix_command(&self, error_output: &str) -> Result<String, OllamaError> {
        let prompt = prompt::build_fix_prompt(error_output);
        self.stream_raw(&prompt, &mut io::stdout()).await
    }

    pub fn get_env_context(&self) -> (String, String, String) {
        prompt::get_env_context()
    }

    pub async fn stream_raw<W: Write>(
        &self,
        prompt: &str,
        writer: &mut W,
    ) -> Result<String, OllamaError> {
        let request_payload = GenerateRequest {
            model: &self.model,
            prompt: prompt.to_string(),
            stream: true,
        };

        let url = format!("{}/api/generate", self.host);
        let response = self
            .client
            .post(&url)
            .json(&request_payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() || e.is_timeout() {
                    OllamaError::NotRunning {
                        host: self.host.clone(),
                        source: e,
                    }
                } else {
                    OllamaError::NetworkError(e)
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            if status == StatusCode::NOT_FOUND {
                return Err(OllamaError::ApiError(format!(
                    "Model '{}' not found. You may need to pull it first using 'ollama pull {}'",
                    self.model, self.model
                )));
            }
            return Err(OllamaError::ApiError(status.to_string()));
        }

        let mut response_stream = response.bytes_stream();

        let mut in_thought = false;
        let mut raw_buffer = Vec::new();
        let mut response_buffer = String::new(); // Buffer for JSON lines
        let mut content_buffer = String::new(); // Buffer for raw content (thinking/command)
        let mut final_response = String::new(); // Accumulate response for return

        while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result.map_err(OllamaError::NetworkError)?;
            decode_utf8_chunk(&mut raw_buffer, &chunk, &mut response_buffer)?;

            while let Some(pos) = response_buffer.find('\n') {
                let line = response_buffer[..pos].to_string();
                response_buffer = response_buffer[pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[tokio::test]
    async fn test_client_creation() {
        let client = OllamaClient::new(
            "http://localhost:11434".to_string(),
            "test-model".to_string(),
        );
        assert_eq!(client.host, "http://localhost:11434");
        assert_eq!(client.model, "test-model");
    }

    #[tokio::test]
    async fn test_stream_raw_success() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let mock_body =
            "{\"response\": \"ls\", \"done\": false}\n{\"response\": \" -la\", \"done\": true}\n";
        let _mock = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_body)
            .create_async()
            .await;

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
        let _mock = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_body)
            .create_async()
            .await;

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

        let _mock = server
            .mock("POST", "/api/generate")
            .with_status(500)
            .create_async()
            .await;

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

        let mock_body =
            "{\"response\": \"Explanation: wrong flag\\nCommand: ls\", \"done\": true}\n";
        let _mock = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_body)
            .create_async()
            .await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let result = client.fix_command("error output").await.unwrap();
        assert_eq!(result, "Explanation: wrong flag\nCommand: ls");
    }

    #[tokio::test]
    async fn test_ollama_get_embeddings() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server
            .mock("POST", "/api/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"embedding\": [0.1, 0.2, 0.3]}")
            .create_async()
            .await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let emb = client
            .get_embeddings("nomic-embed-text", "find files")
            .await
            .unwrap();
        assert_eq!(emb, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_ollama_stream_refine_explain() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_header("content-type", "application/x-ndjson")
            .with_body(
                "{\"response\":\"echo 1\",\"done\":false}\n{\"response\":\"\",\"done\":true}\n",
            )
            .create_async()
            .await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let res = client.stream_command("how to echo 1").await.unwrap();
        assert_eq!(res, "echo 1");
    }
}
