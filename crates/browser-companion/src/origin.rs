//! Validation for the dynamic Firefox extension origin.

use thiserror::Error;

const PREFIX: &str = "moz-extension://";
const MAX_HOST_LEN: usize = 128;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum OriginError {
    #[error("origin must be a canonical moz-extension origin without a path")]
    Invalid,
}

/// Accept only a canonical origin (`moz-extension://<host>`) and no URL path,
/// credentials, query, fragment, or port.
pub fn validate_extension_origin(value: &str) -> Result<(), OriginError> {
    let Some(host) = value.strip_prefix(PREFIX) else {
        return Err(OriginError::Invalid);
    };
    if host.is_empty()
        || host.len() > MAX_HOST_LEN
        || host.starts_with('.')
        || host.ends_with('.')
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_'))
    {
        return Err(OriginError::Invalid);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_firefox_profile_origins_only() {
        assert!(
            validate_extension_origin("moz-extension://00000000-0000-4000-8000-000000000001")
                .is_ok()
        );
        for invalid in [
            "https://example.test",
            "moz-extension://",
            "moz-extension://uuid/",
            "moz-extension://uuid/path",
            "moz-extension://uuid:1234",
            "moz-extension://uuid?query",
            "moz-extension://user@uuid",
        ] {
            assert_eq!(
                validate_extension_origin(invalid),
                Err(OriginError::Invalid),
                "{invalid}"
            );
        }
    }
}
