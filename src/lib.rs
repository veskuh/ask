pub mod cache;
pub mod client;
pub mod config;
pub mod prompt;
pub mod ui;

pub use cache::cosine_similarity;
pub use client::{ClientError, LlmClient, OllamaClient, OllamaError, OpenAiClient};
pub use config::Config;
pub use prompt::get_env_context;
