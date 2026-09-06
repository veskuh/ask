//! Client module coordinating LLM providers and streaming interfaces.

pub mod ollama;
pub mod openai;
pub mod stream;

pub use crate::prompt::get_env_context;
pub use ollama::{OllamaClient, OllamaError};
pub use openai::OpenAiClient;
pub use stream::{decode_utf8_chunk, default_http_client, think_tag_prefix_len};

use reqwest::StatusCode;
use std::io;
use thiserror::Error;

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
    AuthError { provider: String, message: String },

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
                hint: "Please make sure Ollama is installed and running (https://ollama.com)."
                    .to_string(),
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

    pub async fn refine_command(
        &self,
        last_command: &str,
        refinement: &str,
    ) -> Result<String, ClientError> {
        match self {
            LlmClient::Ollama(c) => c
                .refine_command(last_command, refinement)
                .await
                .map_err(Into::into),
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

    #[test]
    fn test_ollama_error_conversions() {
        let err_not_running = OllamaError::NotRunning {
            host: "http://localhost:11434".to_string(),
            source: reqwest::Client::new()
                .get("not a valid url")
                .build()
                .unwrap_err(),
        };
        let client_err: ClientError = err_not_running.into();
        assert!(client_err.to_string().contains("Ollama"));

        let err_api = OllamaError::ApiError("model not found".to_string());
        let client_err: ClientError = err_api.into();
        assert!(client_err.to_string().contains("model not found"));

        let err_parse =
            OllamaError::ParseError(serde_json::from_str::<String>("not json").unwrap_err());
        let client_err: ClientError = err_parse.into();
        assert!(matches!(client_err, ClientError::ParseError(_)));

        let err_io = OllamaError::IoError(std::io::Error::other("io error"));
        let client_err: ClientError = err_io.into();
        assert!(matches!(client_err, ClientError::IoError(_)));
    }

    #[tokio::test]
    async fn test_openai_methods_and_llm_client_dispatch() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"echo 42\"}}]}\n\ndata: [DONE]\n\n";

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

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

        let ollama = LlmClient::Ollama(OllamaClient::new(
            "http://localhost:11434".to_string(),
            "ollama-model".to_string(),
        ));
        assert_eq!(ollama.provider_name(), "Ollama");
        assert_eq!(ollama.model(), "ollama-model");
        assert_eq!(ollama.host(), "http://localhost:11434");
    }

    #[tokio::test]
    async fn test_llm_client_dispatch_methods() {
        let mut server = Server::new_async().await;
        let url = server.url();

        let sse_body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"git status\"}}]}\n\ndata: [DONE]\n\n";

        let _mock = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse_body)
            .create_async()
            .await;

        let openai_client = OpenAiClient::new(
            "OpenAI".to_string(),
            url.clone(),
            "gpt-4o-mini".to_string(),
            Some("sk-test".to_string()),
        );
        let llm = LlmClient::OpenAi(openai_client);

        let cmd = llm.stream_command("how to check git status").await.unwrap();
        assert_eq!(cmd, "git status");
    }
}
