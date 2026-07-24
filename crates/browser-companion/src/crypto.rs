//! Small cryptographic helpers used at the browser trust boundary.

use std::fmt::Write as _;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;

pub const SECRET_BYTES: usize = 32;
pub const ENCODED_SECRET_LEN: usize = 43;

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("the operating system random generator failed: {0}")]
    Random(#[from] getrandom::Error),
    #[error("secret must be an unpadded base64url-encoded 256-bit value")]
    InvalidSecret,
}

/// Generate an unpredictable 256-bit secret and return both raw and wire forms.
pub fn generate_secret() -> Result<([u8; SECRET_BYTES], String), CryptoError> {
    let mut raw = [0_u8; SECRET_BYTES];
    getrandom::fill(&mut raw)?;
    let encoded = URL_SAFE_NO_PAD.encode(raw);
    debug_assert_eq!(encoded.len(), ENCODED_SECRET_LEN);
    Ok((raw, encoded))
}

/// Decode the exact token representation accepted by protocol v1.
pub fn decode_secret(value: &str) -> Result<[u8; SECRET_BYTES], CryptoError> {
    if value.len() != ENCODED_SECRET_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(CryptoError::InvalidSecret);
    }
    let decoded = URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| CryptoError::InvalidSecret)?;
    decoded.try_into().map_err(|_| CryptoError::InvalidSecret)
}

/// Compare two fixed-size secrets without data-dependent early exit.
pub fn secrets_equal(left: &[u8; SECRET_BYTES], right: &[u8; SECRET_BYTES]) -> bool {
    bool::from(left.ct_eq(right))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_secrets_are_fresh_exact_256_bit_values() {
        let (first_raw, first) = generate_secret().unwrap();
        let (second_raw, second) = generate_secret().unwrap();
        assert_ne!(first, second);
        assert_ne!(first_raw, second_raw);
        assert_eq!(decode_secret(&first).unwrap(), first_raw);
        assert_eq!(first.len(), ENCODED_SECRET_LEN);
        assert!(!first.contains('='));
    }

    #[test]
    fn rejects_noncanonical_secret_encodings() {
        assert!(decode_secret("short").is_err());
        assert!(decode_secret(&format!("{}=", "A".repeat(42))).is_err());
        assert!(decode_secret(&format!("{}+", "A".repeat(42))).is_err());
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
