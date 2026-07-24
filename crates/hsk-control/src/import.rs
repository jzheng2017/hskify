use std::collections::BTreeMap;

use csv::{ReaderBuilder, StringRecord, Trim};
use sha2::{Digest, Sha256};

use crate::{
    DATA_SCHEMA_VERSION, DatasetCompleteness, DatasetKind, DictionaryArtifact, DictionaryEntry,
    HSK_STANDARD, HskArtifact, HskControlError, HskEntry, HskLevel, ImportMetadata, Result,
    TextNormalizer,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    Tab,
    Comma,
}

impl Delimiter {
    const fn byte(self) -> u8 {
        match self {
            Self::Tab => b'\t',
            Self::Comma => b',',
        }
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn parse_import_metadata(json: &[u8]) -> Result<ImportMetadata> {
    Ok(serde_json::from_slice(json)?)
}

pub fn generate_hsk_artifact(
    source: &[u8],
    metadata: &ImportMetadata,
    delimiter: Delimiter,
) -> Result<Vec<u8>> {
    metadata.validate_common(DatasetKind::Hsk20)?;
    verify_source_hash(source, metadata)?;

    let normalizer = TextNormalizer::new();
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter.byte())
        .trim(Trim::All)
        .comment(Some(b'#'))
        .from_reader(source);
    let headers = reader.headers()?.clone();
    let fields = HskFields::from_headers(
        &headers,
        metadata.completeness == DatasetCompleteness::Complete,
    )?;
    let mut entries = BTreeMap::<String, HskEntry>::new();

    for row in reader.records() {
        let row = row?;
        let level = field(&row, fields.level, "level")?.parse::<HskLevel>()?;
        let simplified = normalizer.normalize(field(&row, fields.simplified, "simplified")?);
        let pinyin = field(&row, fields.pinyin, "pinyin")?.trim().to_owned();
        let glosses = split_list(field(&row, fields.glosses, "glosses")?);
        let simpler_words = fields
            .simpler_words
            .map(|index| split_list(row.get(index).unwrap_or_default()))
            .unwrap_or_default()
            .into_iter()
            .map(|word| normalizer.normalize(&word))
            .collect::<Vec<_>>();
        let independently_usable_value = fields
            .independently_usable
            .and_then(|index| row.get(index))
            .map(str::trim);
        let independently_usable = match independently_usable_value {
            None | Some("") if metadata.completeness == DatasetCompleteness::Complete => {
                return Err(HskControlError::InvalidData(format!(
                    "complete HSK entry {simplified:?} must explicitly audit independentlyUsable"
                )));
            }
            None | Some("") => false,
            Some("true") | Some("1") | Some("yes") => true,
            Some("false") | Some("0") | Some("no") => false,
            Some(value) => {
                return Err(HskControlError::InvalidData(format!(
                    "invalid independentlyUsable value {value:?}"
                )));
            }
        };
        let frequency_rank = fields
            .frequency_rank
            .and_then(|index| row.get(index))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                value.parse::<u32>().map_err(|_| {
                    HskControlError::InvalidData(format!("invalid frequency rank {value:?}"))
                })
            })
            .transpose()?;

        if simplified.is_empty() || pinyin.is_empty() || glosses.is_empty() {
            return Err(HskControlError::InvalidData(
                "HSK source contains an empty required field".into(),
            ));
        }
        let entry = HskEntry {
            simplified: simplified.clone(),
            pinyin,
            glosses,
            level,
            simpler_words,
            independently_usable,
            frequency_rank,
        };
        if let Some(existing) = entries.get_mut(&simplified) {
            if entry.level < existing.level {
                *existing = entry;
            }
        } else {
            entries.insert(simplified, entry);
        }
    }

    let mut entries = entries.into_values().collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        left.level
            .cmp(&right.level)
            .then_with(|| left.simplified.cmp(&right.simplified))
    });
    verify_expected_counts(metadata, &entries)?;

    let artifact = HskArtifact {
        schema_version: DATA_SCHEMA_VERSION,
        standard: HSK_STANDARD.into(),
        dataset_revision: metadata.dataset_revision.clone(),
        completeness: metadata.completeness,
        source: metadata.source.clone(),
        licence: metadata.licence.clone(),
        audited_entry_count: entries.len(),
        audited_level_counts: level_counts(&entries),
        entries,
    };
    canonical_json(&artifact)
}

pub fn generate_dictionary_artifact(source: &[u8], metadata: &ImportMetadata) -> Result<Vec<u8>> {
    metadata.validate_common(DatasetKind::CcCedict)?;
    verify_source_hash(source, metadata)?;
    let text = std::str::from_utf8(source).map_err(|error| {
        HskControlError::InvalidData(format!("dictionary is not UTF-8: {error}"))
    })?;
    let normalizer = TextNormalizer::new();
    let mut entries = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        entries.push(parse_cedict_line(line, line_number, &normalizer)?);
    }

    entries.sort_by(|left, right| {
        left.simplified
            .cmp(&right.simplified)
            .then_with(|| left.pinyin.cmp(&right.pinyin))
            .then_with(|| left.definitions.cmp(&right.definitions))
    });
    entries.dedup();

    if let Some(expected) = metadata.expected_entry_count
        && entries.len() != expected
    {
        return Err(HskControlError::InvalidData(format!(
            "dictionary entry count {} does not match audited expected count {expected}",
            entries.len()
        )));
    }
    if entries.is_empty() {
        return Err(HskControlError::InvalidData(
            "dictionary source contains no entries".into(),
        ));
    }

    let artifact = DictionaryArtifact {
        schema_version: DATA_SCHEMA_VERSION,
        format: "CC-CEDICT".into(),
        dataset_revision: metadata.dataset_revision.clone(),
        completeness: metadata.completeness,
        source: metadata.source.clone(),
        licence: metadata.licence.clone(),
        audited_entry_count: entries.len(),
        entries,
    };
    canonical_json(&artifact)
}

fn verify_source_hash(source: &[u8], metadata: &ImportMetadata) -> Result<()> {
    let actual = sha256_hex(source);
    if actual != metadata.source.sha256 {
        return Err(HskControlError::SourceHashMismatch {
            expected: metadata.source.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

fn verify_expected_counts(metadata: &ImportMetadata, entries: &[HskEntry]) -> Result<()> {
    if let Some(expected) = metadata.expected_entry_count
        && entries.len() != expected
    {
        return Err(HskControlError::InvalidData(format!(
            "HSK entry count {} does not match audited expected count {expected}",
            entries.len()
        )));
    }
    if let Some(expected) = metadata.expected_level_counts {
        let actual = level_counts(entries);
        if actual != expected {
            return Err(HskControlError::InvalidData(format!(
                "HSK level counts {actual:?} do not match audited expected counts {expected:?}"
            )));
        }
    }
    Ok(())
}

fn level_counts(entries: &[HskEntry]) -> [usize; 6] {
    let mut counts = [0usize; 6];
    for entry in entries {
        counts[entry.level.index()] += 1;
    }
    counts
}

fn canonical_json<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

struct HskFields {
    level: usize,
    simplified: usize,
    pinyin: usize,
    glosses: usize,
    simpler_words: Option<usize>,
    independently_usable: Option<usize>,
    frequency_rank: Option<usize>,
}

impl HskFields {
    fn from_headers(headers: &StringRecord, require_independently_usable: bool) -> Result<Self> {
        let normalized = headers
            .iter()
            .map(|header| {
                header
                    .trim_start_matches('\u{feff}')
                    .trim()
                    .to_ascii_lowercase()
                    .replace(['-', ' '], "_")
            })
            .collect::<Vec<_>>();
        let find = |aliases: &[&str]| {
            normalized
                .iter()
                .position(|header| aliases.contains(&header.as_str()))
        };
        let required = |aliases: &[&str], name: &str| {
            find(aliases).ok_or_else(|| {
                HskControlError::InvalidData(format!(
                    "HSK source is missing required {name:?} header"
                ))
            })
        };
        let independently_usable = find(&["independently_usable", "standalone"]);
        if require_independently_usable && independently_usable.is_none() {
            return Err(HskControlError::InvalidData(
                "complete HSK sources require an independently_usable header with an explicit audited value for every entry"
                    .into(),
            ));
        }
        Ok(Self {
            level: required(&["level", "hsk_level"], "level")?,
            simplified: required(&["simplified", "word", "hanzi"], "simplified")?,
            pinyin: required(&["pinyin"], "pinyin")?,
            glosses: required(&["gloss", "glosses", "definition", "definitions"], "gloss")?,
            simpler_words: find(&["simpler_words", "suggestions"]),
            independently_usable,
            frequency_rank: find(&["frequency_rank", "rank"]),
        })
    }
}

fn field<'a>(row: &'a StringRecord, index: usize, name: &str) -> Result<&'a str> {
    row.get(index)
        .ok_or_else(|| HskControlError::InvalidData(format!("HSK row is missing field {name:?}")))
}

fn split_list(value: &str) -> Vec<String> {
    let mut values = value
        .split('|')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn parse_cedict_line(
    line: &str,
    line_number: usize,
    normalizer: &TextNormalizer,
) -> Result<DictionaryEntry> {
    let Some(first_space) = line.find(' ') else {
        return cedict_error(line_number, "missing traditional/simplified separator");
    };
    let traditional = line[..first_space].trim();
    let remainder = line[first_space..].trim_start();
    let Some(second_space) = remainder.find(' ') else {
        return cedict_error(line_number, "missing simplified/pinyin separator");
    };
    let simplified_source = remainder[..second_space].trim();
    let remainder = remainder[second_space..].trim_start();
    let Some(pinyin_end) = remainder.find(']') else {
        return cedict_error(line_number, "missing closing pinyin bracket");
    };
    if !remainder.starts_with('[') {
        return cedict_error(line_number, "pinyin must begin with '['");
    }
    let numbered_pinyin = &remainder[1..pinyin_end];
    let definitions_source = remainder[pinyin_end + 1..].trim();
    if !definitions_source.starts_with('/') || !definitions_source.ends_with('/') {
        return cedict_error(line_number, "definitions must be slash-delimited");
    }
    let mut definitions = definitions_source[1..definitions_source.len() - 1]
        .split('/')
        .map(str::trim)
        .filter(|definition| !definition.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    definitions.sort();
    definitions.dedup();
    if traditional.is_empty()
        || simplified_source.is_empty()
        || numbered_pinyin.trim().is_empty()
        || definitions.is_empty()
    {
        return cedict_error(line_number, "one or more required fields are empty");
    }

    Ok(DictionaryEntry {
        traditional: traditional.to_owned(),
        simplified: normalizer.normalize(simplified_source),
        pinyin: numbered_pinyin_to_tone_marks(numbered_pinyin),
        definitions,
        frequency_rank: None,
    })
}

fn cedict_error<T>(line: usize, message: &str) -> Result<T> {
    Err(HskControlError::CedictParse {
        line,
        message: message.into(),
    })
}

fn numbered_pinyin_to_tone_marks(input: &str) -> String {
    input
        .split_whitespace()
        .map(mark_syllable)
        .collect::<Vec<_>>()
        .join(" ")
}

fn mark_syllable(input: &str) -> String {
    let replaced = input
        .replace("u:", "ü")
        .replace("U:", "Ü")
        .replace('v', "ü");
    let Some(last) = replaced.chars().next_back() else {
        return replaced;
    };
    let Some(tone) = last.to_digit(10).filter(|tone| (1..=5).contains(tone)) else {
        return replaced;
    };
    let mut syllable = replaced
        .char_indices()
        .next_back()
        .map(|(index, _)| replaced[..index].to_owned())
        .unwrap_or_default();
    if tone == 5 {
        return syllable;
    }

    let characters = syllable.chars().collect::<Vec<_>>();
    let vowel_index = characters
        .iter()
        .position(|character| matches!(character, 'a' | 'A'))
        .or_else(|| {
            characters
                .iter()
                .position(|character| matches!(character, 'e' | 'E'))
        })
        .or_else(|| {
            characters
                .windows(2)
                .position(|pair| matches!(pair, ['o' | 'O', 'u' | 'U']))
        })
        .or_else(|| {
            characters.iter().rposition(|character| {
                matches!(
                    character,
                    'a' | 'e' | 'i' | 'o' | 'u' | 'ü' | 'A' | 'E' | 'I' | 'O' | 'U' | 'Ü'
                )
            })
        });
    let Some(vowel_index) = vowel_index else {
        return syllable;
    };
    let marked = tone_mark(characters[vowel_index], tone as usize);
    syllable = characters
        .into_iter()
        .enumerate()
        .map(|(index, character)| {
            if index == vowel_index {
                marked
            } else {
                character
            }
        })
        .collect();
    syllable
}

fn tone_mark(vowel: char, tone: usize) -> char {
    let row = match vowel {
        'a' => ['a', 'ā', 'á', 'ǎ', 'à'],
        'e' => ['e', 'ē', 'é', 'ě', 'è'],
        'i' => ['i', 'ī', 'í', 'ǐ', 'ì'],
        'o' => ['o', 'ō', 'ó', 'ǒ', 'ò'],
        'u' => ['u', 'ū', 'ú', 'ǔ', 'ù'],
        'ü' => ['ü', 'ǖ', 'ǘ', 'ǚ', 'ǜ'],
        'A' => ['A', 'Ā', 'Á', 'Ǎ', 'À'],
        'E' => ['E', 'Ē', 'É', 'Ě', 'È'],
        'I' => ['I', 'Ī', 'Í', 'Ǐ', 'Ì'],
        'O' => ['O', 'Ō', 'Ó', 'Ǒ', 'Ò'],
        'U' => ['U', 'Ū', 'Ú', 'Ǔ', 'Ù'],
        'Ü' => ['Ü', 'Ǖ', 'Ǘ', 'Ǚ', 'Ǜ'],
        _ => return vowel,
    };
    row[tone]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbered_pinyin_conversion_handles_tone_placement_and_umlaut() {
        assert_eq!(
            numbered_pinyin_to_tone_marks("li2 kai1 nu:3 peng2 you5"),
            "lí kāi nǚ péng you"
        );
        assert_eq!(numbered_pinyin_to_tone_marks("Liu2 Gui4"), "Liú Guì");
    }
}
