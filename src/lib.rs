use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::error::Error;
use futures_util::StreamExt;
use std::io::{self, Write};
use colored::*;

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: String,
    stream: bool,
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

    pub async fn stream_command(&self, question: &str) -> Result<String, Box<dyn Error>> {
        let os = std::env::consts::OS;
        let mut distro = String::new();

        if os == "linux" {
            if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
                for line in content.lines() {
                    if line.starts_with("PRETTY_NAME=") {
                        distro = line.replace("PRETTY_NAME=", "").replace("\"", "").to_string();
                        break;
                    }
                }
            }
        }

        let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let context_os = if distro.is_empty() {
            os.to_string()
        } else {
            format!("{} ({})", os, distro)
        };

        let prompt = format!(
            "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n\n\
             Task: Provide only the command line command for this query. \
             Ensure the command is compatible with the specified OS and Shell. \
             Do not include any markdown formatting, backticks, or explanations. \
             Just the raw command itself. Query: {}",
            context_os, shell, cwd, question
        );

        let request_payload = GenerateRequest {
            model: &self.model,
            prompt,
            stream: true,
        };

        let url = format!("{}/api/generate", self.host);
        let response = self.client
            .post(&url)
            .json(&request_payload)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Ollama API returned error: {}", response.status()).into());
        }

        let mut response_stream = response.bytes_stream();

        let mut in_thought = false;
        let mut response_buffer = String::new(); // Buffer for JSON lines
        let mut content_buffer = String::new();  // Buffer for raw content (thinking/command)
        let mut final_command = String::new();   // Accumulate command for clipboard
        let mut stdout = io::stdout();

        while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result?;
            response_buffer.push_str(std::str::from_utf8(&chunk)?);
            
            while let Some(pos) = response_buffer.find('\n') {
                let line = response_buffer[..pos].to_string();
                response_buffer = response_buffer[pos + 1..].to_string();
                
                if line.is_empty() { continue; }
                if let Ok(gen_response) = serde_json::from_str::<GenerateResponse>(&line) {
                    let content = gen_response.response;
                    content_buffer.push_str(&content);

                    if !in_thought && content_buffer.contains("<think>") {
                        in_thought = true;
                        if let Some(pos) = content_buffer.find("<think>") {
                            let pre_thought = &content_buffer[..pos];
                            print!("{}", pre_thought);
                            final_command.push_str(pre_thought);
                            content_buffer = content_buffer[pos + 7..].to_string();
                        }
                    }

                    if in_thought && content_buffer.contains("</think>") {
                        if let Some(pos) = content_buffer.find("</think>") {
                            let thought = &content_buffer[..pos];
                            print!("{}", thought.dimmed().cyan());
                            println!("\n{}", "---".dimmed().cyan());
                            in_thought = false;
                            content_buffer = content_buffer[pos + 8..].to_string();
                        }
                    } else if in_thought {
                        if content_buffer.len() > 8 {
                            let to_print = &content_buffer[..content_buffer.len() - 8];
                            print!("{}", to_print.dimmed().cyan());
                            content_buffer = content_buffer[content_buffer.len() - 8..].to_string();
                        }
                    } else {
                        print!("{}", content_buffer);
                        final_command.push_str(&content_buffer);
                        content_buffer.clear();
                    }
                    stdout.flush()?;
                }
            }
        }
        
        if !content_buffer.is_empty() {
            if in_thought {
                print!("{}", content_buffer.dimmed().cyan());
            } else {
                print!("{}", content_buffer);
                final_command.push_str(&content_buffer);
            }
            stdout.flush()?;
        }
        println!();

        Ok(final_command.trim().to_string())
    }
}

// Tests updated to be minimal for now as we transition to streaming
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
    async fn test_stream_command_error() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let _mock = server.mock("POST", "/api/generate")
            .with_status(500)
            .create_async().await;

        let client = OllamaClient::new(url, "test-model".to_string());
        let result = client.stream_command("list files").await;

        assert!(result.is_err());
    }
}
