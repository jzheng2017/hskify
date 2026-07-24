//! Firefox native-messaging framing.
//!
//! Firefox prefixes each JSON message with an unsigned 32-bit little-endian
//! byte length. The launcher deliberately accepts exactly one bounded request.

use std::io::{self, Read, Write};

use serde::Serialize;
use serde::de::DeserializeOwned;
use thiserror::Error;

pub const MAX_NATIVE_REQUEST_BYTES: usize = 64 * 1024;
pub const MAX_NATIVE_RESPONSE_BYTES: usize = 64 * 1024;

#[derive(Debug, Error)]
pub enum NativeFrameError {
    #[error("native message I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("native message length {actual} exceeds limit {limit}")]
    TooLarge { actual: usize, limit: usize },
    #[error("native message must not be empty")]
    Empty,
    #[error("native message contains invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn read_frame<R: Read, T: DeserializeOwned>(reader: &mut R) -> Result<T, NativeFrameError> {
    let mut prefix = [0_u8; 4];
    reader.read_exact(&mut prefix)?;
    let length = u32::from_le_bytes(prefix) as usize;
    if length == 0 {
        return Err(NativeFrameError::Empty);
    }
    if length > MAX_NATIVE_REQUEST_BYTES {
        return Err(NativeFrameError::TooLarge {
            actual: length,
            limit: MAX_NATIVE_REQUEST_BYTES,
        });
    }
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload)?;
    Ok(serde_json::from_slice(&payload)?)
}

pub fn write_frame<W: Write, T: Serialize>(
    writer: &mut W,
    value: &T,
) -> Result<(), NativeFrameError> {
    let payload = serde_json::to_vec(value)?;
    if payload.is_empty() {
        return Err(NativeFrameError::Empty);
    }
    if payload.len() > MAX_NATIVE_RESPONSE_BYTES {
        return Err(NativeFrameError::TooLarge {
            actual: payload.len(),
            limit: MAX_NATIVE_RESPONSE_BYTES,
        });
    }
    let length = u32::try_from(payload.len()).expect("native response limit fits u32");
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct Message {
        value: String,
    }

    #[test]
    fn framing_is_little_endian_and_round_trips_partial_reads() {
        let value = Message {
            value: "hello".into(),
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &value).unwrap();
        let payload_len = serde_json::to_vec(&value).unwrap().len() as u32;
        assert_eq!(&bytes[..4], &payload_len.to_le_bytes());

        struct OneByteAtATime(Cursor<Vec<u8>>);
        impl Read for OneByteAtATime {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                let length = output.len().min(1);
                self.0.read(&mut output[..length])
            }
        }
        let decoded: Message = read_frame(&mut OneByteAtATime(Cursor::new(bytes))).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn rejects_oversized_and_truncated_frames_before_json_parse() {
        let oversized = ((MAX_NATIVE_REQUEST_BYTES + 1) as u32)
            .to_le_bytes()
            .to_vec();
        assert!(matches!(
            read_frame::<_, Message>(&mut Cursor::new(oversized)),
            Err(NativeFrameError::TooLarge { .. })
        ));

        let mut truncated = 10_u32.to_le_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        assert!(matches!(
            read_frame::<_, Message>(&mut Cursor::new(truncated)),
            Err(NativeFrameError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }

    #[test]
    fn rejects_zero_length_frame() {
        assert!(matches!(
            read_frame::<_, Message>(&mut Cursor::new(0_u32.to_le_bytes())),
            Err(NativeFrameError::Empty)
        ));
    }
}
