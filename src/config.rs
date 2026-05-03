use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub model: String,
    pub host: String,
    pub auto_copy: bool,
    pub embedding_model: String,
    pub cache_threshold: f32,
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
