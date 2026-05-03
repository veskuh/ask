use serde::{Deserialize, Serialize};
use reqwest::Client;
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
    host: String,
    model: String,
    client: Client,
}

impl OllamaClient {
    pub fn new(host: String, model: String) -> Self {
        Self {
            host,
            model,
            client: Client::new(),
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
            if status == reqwest::StatusCode::NOT_FOUND {
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
        let os_name = std::env::consts::OS;
        let mut distro = String::new();

        if os_name == "linux" {
            if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
                for line in content.lines() {
                    if line.starts_with("PRETTY_NAME=") {
                        distro = line.replace("PRETTY_NAME=", "").replace("\"", "").to_string();
                        break;
                    }
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
            if status == reqwest::StatusCode::NOT_FOUND {
                return Err(OllamaError::ApiError(format!("Model '{}' not found. You may need to pull it first using 'ollama pull {}'", self.model, self.model)));
            }
            return Err(OllamaError::ApiError(status.to_string()));
        }

        let mut response_stream = response.bytes_stream();

        let mut in_thought = false;
        let mut response_buffer = String::new(); // Buffer for JSON lines
        let mut content_buffer = String::new();  // Buffer for raw content (thinking/command)
        let mut final_response = String::new();   // Accumulate response for return

        while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result.map_err(OllamaError::NetworkError)?;
            response_buffer.push_str(std::str::from_utf8(&chunk)?);
            
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
                        let to_print = &content_buffer[..content_buffer.len() - 8];
                        write!(writer, "{}", to_print.dimmed().cyan())?;
                        content_buffer = content_buffer[content_buffer.len() - 8..].to_string();
                    }
                } else {
                    write!(writer, "{}", content_buffer)?;
                    final_response.push_str(&content_buffer);
                    content_buffer.clear();
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

pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() { return 0.0; }
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude1 == 0.0 || magnitude2 == 0.0 { return 0.0; }
    dot_product / (magnitude1 * magnitude2)
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
        control::set_override(false);
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
        control::set_override(false);
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

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-6);

        let v3 = vec![0.0, 1.0];
        assert!(cosine_similarity(&v1, &v3).abs() < 1e-6);
    }
}
