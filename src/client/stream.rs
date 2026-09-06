//! Low-level network client configuration, UTF-8 chunk buffering, and tag parsing utilities.

use reqwest::Client;

/// Returns a reqwest HTTP client configured with a 10s connection timeout and 30s TCP keep-alive.
pub fn default_http_client() -> Client {
    Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// Decodes incoming streaming bytes into a UTF-8 string buffer, safely retaining
/// incomplete multi-byte code points in `raw_buffer` across chunk boundaries.
pub fn decode_utf8_chunk(
    raw_buffer: &mut Vec<u8>,
    chunk: &[u8],
    str_buffer: &mut String,
) -> Result<(), std::str::Utf8Error> {
    raw_buffer.extend_from_slice(chunk);
    match std::str::from_utf8(raw_buffer) {
        Ok(valid_str) => {
            str_buffer.push_str(valid_str);
            raw_buffer.clear();
            Ok(())
        }
        Err(e) => {
            let valid = e.valid_up_to();
            if valid > 0 {
                let valid_str = std::str::from_utf8(&raw_buffer[..valid]).unwrap();
                str_buffer.push_str(valid_str);
                raw_buffer.drain(..valid);
            }
            if e.error_len().is_some() {
                std::str::from_utf8(raw_buffer).map(|_| ())
            } else {
                Ok(())
            }
        }
    }
}

/// Detects if the trailing end of a string matches a prefix of `<think>`.
/// Returns the length of the matched prefix, or 0 if no prefix matches.
pub fn think_tag_prefix_len(s: &str) -> usize {
    let tag = "<think>";
    for len in (1..tag.len()).rev() {
        if s.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_utf8_chunk_split_code_point() {
        let mut raw_buf = Vec::new();
        let mut str_buf = String::new();

        // 🦀 emoji in UTF-8: [240, 159, 166, 128]
        let part1 = &[240, 159];
        let part2 = &[166, 128];

        assert!(decode_utf8_chunk(&mut raw_buf, part1, &mut str_buf).is_ok());
        assert_eq!(str_buf, "");
        assert_eq!(raw_buf, part1);

        assert!(decode_utf8_chunk(&mut raw_buf, part2, &mut str_buf).is_ok());
        assert_eq!(str_buf, "🦀");
        assert!(raw_buf.is_empty());
    }

    #[test]
    fn test_think_tag_prefix_len() {
        assert_eq!(think_tag_prefix_len("hello <"), 1);
        assert_eq!(think_tag_prefix_len("hello <th"), 3);
        assert_eq!(think_tag_prefix_len("hello <think"), 6);
        assert_eq!(think_tag_prefix_len("hello <think>"), 0); // Full tag handled separately
        assert_eq!(think_tag_prefix_len("hello world"), 0);
        assert_eq!(think_tag_prefix_len(""), 0);
    }

    #[test]
    fn test_default_http_client_creation() {
        let client = default_http_client();
        // Client builds without panic
        let _ = client;
    }
}
