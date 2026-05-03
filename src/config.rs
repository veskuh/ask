use serde::{Deserialize, Serialize};
use anyhow::{Context, Result, bail};
use url::Url;

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub model: String,
    pub host: String,
    pub auto_copy: bool,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_cache_threshold")]
    pub cache_threshold: f32,
}

fn default_embedding_model() -> String {
    "nomic-embed-text".to_string()
}

fn default_cache_threshold() -> f32 {
    0.92
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        Url::parse(&self.host).context("Invalid host URL in configuration")?;
        
        if self.cache_threshold < 0.0 || self.cache_threshold > 1.0 {
            bail!("Cache threshold must be between 0.0 and 1.0 (currently: {})", self.cache_threshold);
        }
        
        if self.model.is_empty() {
            bail!("Model name cannot be empty in configuration");
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "gemma4:e4b".to_string(),
            host: "http://localhost:11434".to_string(),
            auto_copy: true,
            embedding_model: "nomic-embed-text".to_string(),
            cache_threshold: 0.92,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct State {
    pub last_command: String,
}
