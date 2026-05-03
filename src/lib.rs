pub mod client;
pub mod cache;
pub mod config;

pub use client::{OllamaClient, OllamaError};
pub use cache::cosine_similarity;
