use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Cache {
    pub entries: Vec<CacheEntry>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CacheEntry {
    pub question: String,
    pub embedding: Vec<f32>,
    pub command: String,
    pub os: String,
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

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-6);

        let v3 = vec![0.0, 1.0];
        assert!(cosine_similarity(&v1, &v3).abs() < 1e-6);
    }
}
