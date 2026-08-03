//! Cross-language fixtures for the unversioned terminal browser contract.

use std::sync::OnceLock;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

use crate::contracts::{
    BrowserSetupStatus, HealthResponse, JobUpdatesResponse, LookupResult, Validate,
};

const HEALTH: &str = include_str!("../../../fixtures/contracts/health.ready.json");
const SETUP: &str = include_str!("../../../fixtures/contracts/setup.ready.json");
const UPDATES: &str = include_str!("../../../fixtures/contracts/job-updates.success.json");
const LOOKUP: &str = include_str!("../../../fixtures/contracts/lookup.valid.json");

fn parse_valid<T>(json: &str, name: &str) -> T
where
    T: serde::de::DeserializeOwned + Validate,
{
    let value: T =
        serde_json::from_str(json).unwrap_or_else(|error| panic!("parse {name}: {error}"));
    value
        .validate()
        .unwrap_or_else(|error| panic!("validate {name}: {error}"));
    value
}

pub fn health() -> HealthResponse {
    static VALUE: OnceLock<HealthResponse> = OnceLock::new();
    VALUE
        .get_or_init(|| parse_valid(HEALTH, "health.ready.json"))
        .clone()
}

pub fn setup() -> BrowserSetupStatus {
    static VALUE: OnceLock<BrowserSetupStatus> = OnceLock::new();
    VALUE
        .get_or_init(|| parse_valid(SETUP, "setup.ready.json"))
        .clone()
}

pub fn updates(job_id: &str) -> JobUpdatesResponse {
    static VALUE: OnceLock<JobUpdatesResponse> = OnceLock::new();
    let mut value = VALUE
        .get_or_init(|| parse_valid(UPDATES, "job-updates.success.json"))
        .clone();
    value.job_id = job_id.to_owned();
    value
}

pub fn lookup(selected_text: &str) -> LookupResult {
    static VALUE: OnceLock<LookupResult> = OnceLock::new();
    let mut value = VALUE
        .get_or_init(|| parse_valid(LOOKUP, "lookup.valid.json"))
        .clone();
    value.selected_text = selected_text.to_owned();
    value
}

// Generated specifically for this project with fontTools. It contains only a
// `.notdef` glyph and space, so browsers load a valid fixture font and then use
// normal CJK fallback for Chinese. A licensed CJK bank replaces it at Gate 6.
const FIXTURE_FONT_TTF: &str = "AAEAAAAKAIAAAwAgT1MvMkUAQ34AAAEoAAAAYGNtYXAADABzAAABkAAAADRnbHlmAAAAAAAAAcwAAAABaGVhZCwoxFYAAACsAAAANmhoZWEDIgGTAAAA5AAAACRobXR4A4QAAAAAAYgAAAAIbG9jYQAAAAAAAAHEAAAABm1heHAAAwACAAABCAAAACBuYW1lICk/qwAAAdAAAAHIcG9zdAAHAAAAAAOYAAAAJgABAAAAAQAAdXwHQl8PPPUAAwPoAAAAAOaJQbsAAAAA5olBuwAAAAAAAAAAAAAAAwACAAAAAAAAAAEAAAMg/zgAAAJYAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAACAAEAAAACAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAwHCAZAABQAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAPz8/PwAAACAAIAMg/zgAAAMgAMgAAAAAAAAAAAAAAAAAAAAgAAACWAAAASwAAAAAAAIAAAADAAAAFAADAAEAAAAUAAQAIAAAAAQABAABAAAAIP//AAAAIP///+EAAQAAAAAAAAAAAAAAAAAAAAAAAAAMAJYAAQAAAAAAAQAYAAAAAQAAAAAAAgAHABgAAQAAAAAAAwAcAB8AAQAAAAAABAAgADsAAQAAAAAABQALAFsAAQAAAAAABgAcAB8AAwABBAkAAQAwAGYAAwABBAkAAgAOAJYAAwABBAkAAwA4AKQAAwABBAkABABAANwAAwABBAkABQAWARwAAwABBAkABgA4AKRIU0sgTWFuZ2EgR2F0ZSAyIEZpeHR1cmVSZWd1bGFySFNLTWFuZ2FHYXRlMkZpeHR1cmUtUmVndWxhckhTSyBNYW5nYSBHYXRlIDIgRml4dHVyZSBSZWd1bGFyVmVyc2lvbiAxLjAASABTAEsAIABNAGEAbgBnAGEAIABHAGEAdABlACAAMgAgAEYAaQB4AHQAdQByAGUAUgBlAGcAdQBsAGEAcgBIAFMASwBNAGEAbgBnAGEARwBhAHQAZQAyAEYAaQB4AHQAdQByAGUALQBSAGUAZwB1AGwAYQByAEgAUwBLACAATQBhAG4AZwBhACAARwBhAHQAZQAgADIAIABGAGkAeAB0AHUAcgBlACAAUgBlAGcAdQBsAGEAcgBWAGUAcgBzAGkAbwBuACAAMQAuADAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAwAA";

/// Gate-2-only valid TrueType payload. Keeping it explicit avoids accidentally
/// shipping a proprietary system font.
pub fn font_bytes(font_id: &str) -> Option<&'static [u8]> {
    match font_id {
        "fixture-sans" | "fixture-display" => {
            static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
            Some(
                BYTES
                    .get_or_init(|| {
                        STANDARD
                            .decode(FIXTURE_FONT_TTF)
                            .expect("decode generated Gate 2 fixture font")
                    })
                    .as_slice(),
            )
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_embedded_fixtures_still_validate() {
        health().validate().unwrap();
        setup().validate().unwrap();
        updates("job").validate().unwrap();
        lookup("selected").validate().unwrap();
        assert_eq!(&font_bytes("fixture-sans").unwrap()[..4], &[0, 1, 0, 0]);
    }
}
