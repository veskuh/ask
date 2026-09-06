use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(default = "default_provider")]
    pub provider: String,

    // Ollama / legacy fields
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_auto_copy")]
    pub auto_copy: bool,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_cache_threshold")]
    pub cache_threshold: f32,

    #[serde(default)]
    pub openrouter: OpenRouterConfig,
    #[serde(default)]
    pub openai: OpenAiConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterConfig {
    #[serde(default = "default_openrouter_base_url")]
    pub base_url: String,
    #[serde(default = "default_openrouter_model")]
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            base_url: default_openrouter_base_url(),
            model: default_openrouter_model(),
            api_key: String::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAiConfig {
    #[serde(default = "default_openai_base_url")]
    pub base_url: String,
    #[serde(default = "default_openai_model")]
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: default_openai_base_url(),
            model: default_openai_model(),
            api_key: String::new(),
        }
    }
}

fn default_provider() -> String {
    "ollama".to_string()
}

fn default_model() -> String {
    "gemma4:e4b".to_string()
}

fn default_host() -> String {
    "http://localhost:11434".to_string()
}

fn default_auto_copy() -> bool {
    true
}

fn default_embedding_model() -> String {
    "nomic-embed-text".to_string()
}

fn default_cache_threshold() -> f32 {
    0.92
}

fn default_openrouter_base_url() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

fn default_openrouter_model() -> String {
    "deepseek/deepseek-v4-flash".to_string()
}

fn default_openai_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

fn default_openai_model() -> String {
    "gpt-4o-mini".to_string()
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        match self.provider.to_lowercase().as_str() {
            "ollama" => {
                Url::parse(&self.host).context("Invalid Ollama host URL in configuration")?;
                if self.model.trim().is_empty() {
                    bail!("Ollama model name cannot be empty in configuration");
                }
            }
            "openrouter" => {
                Url::parse(&self.openrouter.base_url)
                    .context("Invalid OpenRouter base URL in configuration")?;
                if self.openrouter.model.trim().is_empty() {
                    bail!("OpenRouter model name cannot be empty in configuration");
                }
            }
            "openai" => {
                Url::parse(&self.openai.base_url)
                    .context("Invalid OpenAI base URL in configuration")?;
                if self.openai.model.trim().is_empty() {
                    bail!("OpenAI model name cannot be empty in configuration");
                }
            }
            other => {
                bail!(
                    "Unknown provider '{}'. Supported providers: ollama, openrouter, openai",
                    other
                );
            }
        }

        if self.cache_threshold < 0.0 || self.cache_threshold > 1.0 {
            bail!(
                "Cache threshold must be between 0.0 and 1.0 (currently: {})",
                self.cache_threshold
            );
        }

        Ok(())
    }

    /// Resolve the active API key for a provider, checking CLI override -> env var -> config.
    pub fn resolve_api_key(&self, provider: &str, cli_api_key: Option<&str>) -> Option<String> {
        if let Some(key) = cli_api_key.map(str::trim).filter(|k| !k.is_empty()) {
            return Some(key.to_string());
        }

        match provider.to_lowercase().as_str() {
            "openrouter" => {
                if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
                    let trimmed = key.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
                if let Ok(key) = std::env::var("ASK_API_KEY") {
                    let trimmed = key.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
                let configured = self.openrouter.api_key.trim();
                if !configured.is_empty() {
                    return Some(configured.to_string());
                }
            }
            "openai" => {
                if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                    let trimmed = key.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
                if let Ok(key) = std::env::var("ASK_API_KEY") {
                    let trimmed = key.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
                let configured = self.openai.api_key.trim();
                if !configured.is_empty() {
                    return Some(configured.to_string());
                }
            }
            _ => {}
        }
        None
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            host: default_host(),
            auto_copy: default_auto_copy(),
            embedding_model: default_embedding_model(),
            cache_threshold: default_cache_threshold(),
            openrouter: OpenRouterConfig::default(),
            openai: OpenAiConfig::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct State {
    pub last_command: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legacy_config_deserialization() {
        let legacy_json = r#"{
            "model": "gemma4:e4b",
            "host": "http://localhost:11434",
            "auto_copy": false,
            "embedding_model": "nomic-embed-text",
            "cache_threshold": 0.85
        }"#;
        let cfg: Config = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(cfg.provider, "ollama");
        assert_eq!(cfg.model, "gemma4:e4b");
        assert_eq!(cfg.host, "http://localhost:11434");
        assert!(!cfg.auto_copy);
        assert_eq!(cfg.cache_threshold, 0.85);
        assert_eq!(cfg.openrouter.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(cfg.openrouter.model, "deepseek/deepseek-v4-flash");
        assert!(cfg.openrouter.api_key.is_empty());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_openrouter_config() {
        let json_str = r#"{
            "provider": "openrouter",
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "deepseek/deepseek-v4-flash",
                "api_key": "test-key"
            }
        }"#;
        let cfg: Config = serde_json::from_str(json_str).unwrap();
        assert_eq!(cfg.provider, "openrouter");
        assert_eq!(cfg.openrouter.api_key, "test-key");
        assert!(cfg.validate().is_ok());

        // CLI override always takes highest precedence
        let cli_override = cfg.resolve_api_key("openrouter", Some("cli-key"));
        assert_eq!(cli_override, Some("cli-key".to_string()));

        // Resolve without CLI override returns Some key (from env if set, else config)
        let resolved = cfg.resolve_api_key("openrouter", None);
        assert!(resolved.is_some());
    }

    #[test]
    fn test_invalid_provider() {
        let cfg = Config {
            provider: "unsupported".to_string(),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_branches() {
        // Invalid Ollama host URL
        let cfg = Config {
            host: "invalid host: //".to_string(),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        // Empty Ollama model
        let cfg = Config {
            model: "   ".to_string(),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        // Invalid OpenRouter URL
        let mut cfg = Config {
            provider: "openrouter".to_string(),
            ..Default::default()
        };
        cfg.openrouter.base_url = "not a valid url".to_string();
        assert!(cfg.validate().is_err());

        // Empty OpenRouter model
        let mut cfg = Config {
            provider: "openrouter".to_string(),
            ..Default::default()
        };
        cfg.openrouter.model = "".to_string();
        assert!(cfg.validate().is_err());

        // Valid OpenAI provider
        let cfg = Config {
            provider: "openai".to_string(),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());

        // Invalid OpenAI URL
        let mut cfg = Config {
            provider: "openai".to_string(),
            ..Default::default()
        };
        cfg.openai.base_url = "bad-url".to_string();
        assert!(cfg.validate().is_err());

        // Empty OpenAI model
        let mut cfg = Config {
            provider: "openai".to_string(),
            ..Default::default()
        };
        cfg.openai.model = "".to_string();
        assert!(cfg.validate().is_err());

        // Bad cache threshold (< 0 or > 1)
        let cfg = Config {
            cache_threshold: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg = Config {
            cache_threshold: 1.5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_resolve_api_key_fallbacks() {
        let mut cfg = Config::default();
        cfg.openai.api_key = "openai-stored-key".to_string();
        cfg.openrouter.api_key = "openrouter-stored-key".to_string();

        // Empty CLI key is ignored and falls back to config or env
        let resolved = cfg.resolve_api_key("openai", Some("   "));
        assert!(resolved.is_some());

        // Unknown provider returns None
        assert!(cfg.resolve_api_key("unknown-provider", None).is_none());
    }

    #[test]
    fn test_state_struct() {
        let default_state = State::default();
        assert_eq!(default_state.last_command, "");

        let state = State {
            last_command: "git status".to_string(),
        };
        let serialized = serde_json::to_string(&state).unwrap();
        let deserialized: State = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.last_command, "git status");
    }
}
