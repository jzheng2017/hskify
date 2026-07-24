use std::{
    array,
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

use jieba_rs::{Jieba, TokenizeMode};

use crate::{
    DictionaryArtifact, EMBEDDED_DICTIONARY_TEST_SEED, EMBEDDED_HSK_TEST_SEED, HSK_STANDARD,
    HskArtifact, HskDataset, HskException, HskLevel, HskViolation, LOOKUP_REVISION, LoadPolicy,
    LocalDictionary, LookupResult, LookupToken, NORMALIZATION_REVISION, ProperName,
    ProperNameReason, Result, SEGMENTATION_REVISION, TextNormalizer, ValidationReport,
    ViolationReason,
    normalization::{is_all_han, is_ignorable_token},
    sha256_hex,
    trie::AllowedWordTrie,
};

const OPENCC_DEPENDENCY_REVISION: &str = "opencc-fmmseg-0.8.0-tw2sp-hk2s";
const JIEBA_DEPENDENCY_REVISION: &str = "jieba-rs-0.10-default-dict";

/// Pure Rust HSK control and dictionary engine.
pub struct HskControl {
    hsk: HskDataset,
    dictionary: LocalDictionary,
    normalizer: TextNormalizer,
    jieba: Jieba,
    full_lexicon: AllowedWordTrie,
    allowed_tries: [AllowedWordTrie; 6],
    cache_revision: String,
}

impl HskControl {
    /// Constructs from generated artifacts and rejects incomplete resources.
    pub fn from_json(hsk_json: &str, dictionary_json: &str) -> Result<Self> {
        Self::from_json_with_policy(hsk_json, dictionary_json, LoadPolicy::RequireComplete)
    }

    /// Constructs with an explicit policy. `AllowIncompleteTestSeed` is intended
    /// only for tests and local fixture development.
    pub fn from_json_with_policy(
        hsk_json: &str,
        dictionary_json: &str,
        policy: LoadPolicy,
    ) -> Result<Self> {
        let hsk_artifact = serde_json::from_str::<HskArtifact>(hsk_json)?;
        let dictionary_artifact = serde_json::from_str::<DictionaryArtifact>(dictionary_json)?;
        let normalizer = TextNormalizer::new();
        let hsk = HskDataset::from_artifact(hsk_artifact, policy, &normalizer)?;
        let dictionary = LocalDictionary::from_artifact(dictionary_artifact, policy, &normalizer)?;
        Self::from_datasets(hsk, dictionary, normalizer)
    }

    /// Loads the repository's deliberately incomplete seed with an explicit
    /// test-only policy.
    pub fn from_embedded_test_seed() -> Result<Self> {
        Self::from_json_with_policy(
            EMBEDDED_HSK_TEST_SEED,
            EMBEDDED_DICTIONARY_TEST_SEED,
            LoadPolicy::AllowIncompleteTestSeed,
        )
    }

    fn from_datasets(
        hsk: HskDataset,
        dictionary: LocalDictionary,
        normalizer: TextNormalizer,
    ) -> Result<Self> {
        let mut word_frequencies = BTreeMap::<String, usize>::new();
        for word in dictionary.words() {
            word_frequencies.insert(word.to_owned(), 10);
        }
        for entry in hsk.entries() {
            let default_frequency = 1_000_000usize / usize::from(entry.level.get());
            let frequency = entry
                .frequency_rank
                .map(|rank| 2_000_000usize.saturating_sub(rank as usize).max(100))
                .unwrap_or(default_frequency);
            word_frequencies
                .entry(entry.simplified.clone())
                .and_modify(|existing| *existing = (*existing).max(frequency))
                .or_insert(frequency);
        }

        let mut jieba = Jieba::new();
        let mut full_lexicon = AllowedWordTrie::new();
        for (word, frequency) in &word_frequencies {
            jieba.add_word(word, Some(*frequency), Some("hsk-control"));
            full_lexicon.insert(word);
        }

        let allowed_tries = array::from_fn(|level_index| {
            let selected = HskLevel::ALL[level_index];
            let mut trie = AllowedWordTrie::new();
            for entry in hsk.entries() {
                if entry.level <= selected && entry.independently_usable {
                    trie.insert(&entry.simplified);
                }
            }
            trie
        });

        let cache_bytes = [
            HSK_STANDARD,
            hsk.cache_revision(),
            dictionary.cache_revision(),
            NORMALIZATION_REVISION,
            SEGMENTATION_REVISION,
            LOOKUP_REVISION,
            OPENCC_DEPENDENCY_REVISION,
            JIEBA_DEPENDENCY_REVISION,
        ]
        .join("\n");
        let cache_revision = format!("hsk-control-sha256:{}", sha256_hex(cache_bytes.as_bytes()));

        Ok(Self {
            hsk,
            dictionary,
            normalizer,
            jieba,
            full_lexicon,
            allowed_tries,
            cache_revision,
        })
    }

    pub fn hsk_dataset(&self) -> &HskDataset {
        &self.hsk
    }

    pub fn dictionary(&self) -> &LocalDictionary {
        &self.dictionary
    }

    pub fn cache_revision(&self) -> &str {
        &self.cache_revision
    }

    pub fn normalize_text(&self, text: &str) -> String {
        self.normalizer.normalize(text)
    }

    pub fn allowed_words(&self, level: HskLevel) -> &BTreeSet<String> {
        self.hsk.allowed_words(level)
    }

    /// Validates against character offsets in the returned normalized text.
    pub fn validate(
        &self,
        text: &str,
        selected_level: HskLevel,
        proper_names: &[ProperName],
    ) -> ValidationReport {
        let normalized_text = self.normalizer.normalize(text);
        let characters = normalized_text.chars().collect::<Vec<_>>();
        let normalized_names = self.normalize_names(proper_names);
        let name_matches = find_name_matches(&characters, &normalized_names);
        let exceptions = name_matches
            .iter()
            .map(|name| HskException {
                text: name.text.clone(),
                start_char: name.start,
                end_char: name.end,
                reason: name.reason,
            })
            .collect::<Vec<_>>();
        let boundaries = char_boundaries(&normalized_text);
        let mut violations = Vec::new();
        let mut cursor = 0;

        for name in &name_matches {
            self.validate_gap(
                &normalized_text,
                &characters,
                &boundaries,
                cursor,
                name.start,
                selected_level,
                &mut violations,
            );
            cursor = name.end;
        }
        self.validate_gap(
            &normalized_text,
            &characters,
            &boundaries,
            cursor,
            characters.len(),
            selected_level,
            &mut violations,
        );

        violations.sort_by(|left, right| {
            left.start_char
                .cmp(&right.start_char)
                .then_with(|| right.end_char.cmp(&left.end_char))
                .then_with(|| left.text.cmp(&right.text))
        });
        violations.dedup_by(|left, right| {
            left.start_char == right.start_char
                && left.end_char == right.end_char
                && left.reason == right.reason
        });

        ValidationReport {
            normalized_text,
            requested_level: selected_level,
            strictly_valid: violations.is_empty(),
            violations,
            exceptions,
            cache_revision: self.cache_revision.clone(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_gap(
        &self,
        text: &str,
        characters: &[char],
        boundaries: &[usize],
        start: usize,
        end: usize,
        selected_level: HskLevel,
        violations: &mut Vec<HskViolation>,
    ) {
        if start >= end {
            return;
        }
        let gap = &text[boundaries[start]..boundaries[end]];
        let tokens = self
            .jieba
            .tokenize(gap, TokenizeMode::Default, true)
            .into_iter()
            .map(|token| PrimaryToken {
                text: token.word.to_owned(),
                start: start + token.start,
                end: start + token.end,
            })
            .collect::<Vec<_>>();
        if tokens.is_empty() {
            return;
        }

        let guards = self.compound_guards(characters, &tokens, end, selected_level);
        for guard in &guards {
            violations.push(self.make_violation(
                characters,
                guard.start,
                guard.end,
                guard.reason.clone(),
                selected_level,
            ));
        }

        for token in tokens {
            if guards
                .iter()
                .any(|guard| token.start >= guard.start && token.end <= guard.end)
            {
                continue;
            }
            if is_ignorable_token(&token.text)
                || crate::is_numeric_token(&token.text)
                || self.numeric_allowed_decomposition(&token.text, selected_level)
            {
                continue;
            }
            if self.hsk.is_allowed(&token.text, selected_level) {
                continue;
            }

            let reason = if let Some(required_level) = self.hsk.level_of(&token.text)
                && required_level > selected_level
            {
                Some(ViolationReason::AboveSelectedHskLevel { required_level })
            } else if self.dictionary.contains_word(&token.text) {
                Some(ViolationReason::KnownDictionaryWord)
            } else if is_all_han(&token.text) {
                let decomposition =
                    self.allowed_tries[selected_level.index()].best_decomposition(&token.text);
                if decomposition.as_ref().is_some_and(|parts| parts.len() >= 2)
                    && !self.contains_known_disallowed_subspan(&token.text, selected_level)
                {
                    None
                } else {
                    Some(ViolationReason::UnknownChineseWord)
                }
            } else {
                Some(ViolationReason::NonChineseLexicalToken)
            };

            if let Some(reason) = reason {
                violations.push(self.make_violation(
                    characters,
                    token.start,
                    token.end,
                    reason,
                    selected_level,
                ));
            }
        }
    }

    fn numeric_allowed_decomposition(&self, token: &str, selected_level: HskLevel) -> bool {
        let characters = token.chars().collect::<Vec<_>>();
        if characters.len() < 2 || self.contains_known_disallowed_subspan(token, selected_level) {
            return false;
        }
        let trie = &self.allowed_tries[selected_level.index()];
        let mut reachable = vec![false; characters.len() + 1];
        let mut used_numeric = vec![false; characters.len() + 1];
        reachable[0] = true;

        for start in 0..characters.len() {
            if !reachable[start] {
                continue;
            }
            for end in trie.matches_from(&characters, start) {
                reachable[end] = true;
                used_numeric[end] |= used_numeric[start];
            }
            for end in start + 1..=characters.len() {
                let candidate = characters[start..end].iter().collect::<String>();
                if crate::is_numeric_token(&candidate) {
                    reachable[end] = true;
                    used_numeric[end] = true;
                }
            }
        }

        reachable[characters.len()] && used_numeric[characters.len()]
    }

    fn contains_known_disallowed_subspan(&self, token: &str, selected_level: HskLevel) -> bool {
        let characters = token.chars().collect::<Vec<_>>();
        (0..characters.len()).any(|start| {
            self.full_lexicon
                .matches_from(&characters, start)
                .into_iter()
                .filter(|end| *end > start + 1)
                .any(|end| {
                    let word = characters[start..end].iter().collect::<String>();
                    self.known_disallowed_reason(&word, selected_level)
                        .is_some()
                })
        })
    }

    fn compound_guards(
        &self,
        characters: &[char],
        tokens: &[PrimaryToken],
        gap_end: usize,
        selected_level: HskLevel,
    ) -> Vec<CompoundGuard> {
        let token_boundaries = tokens
            .iter()
            .flat_map(|token| [token.start, token.end])
            .collect::<BTreeSet<_>>();
        let mut result = Vec::new();
        let mut token_index = 0;

        while token_index < tokens.len() {
            let start = tokens[token_index].start;
            let candidate = self
                .full_lexicon
                .matches_from(characters, start)
                .into_iter()
                .rev()
                .filter(|end| {
                    *end <= gap_end
                        && *end > start + 1
                        && token_boundaries.contains(end)
                        && is_all_han(&characters[start..*end].iter().collect::<String>())
                })
                .find_map(|end| {
                    let word = characters[start..end].iter().collect::<String>();
                    self.known_disallowed_reason(&word, selected_level)
                        .map(|reason| (end, reason))
                });

            if let Some((end, reason)) = candidate {
                result.push(CompoundGuard { start, end, reason });
                while token_index < tokens.len() && tokens[token_index].end <= end {
                    token_index += 1;
                }
            } else {
                token_index += 1;
            }
        }
        result
    }

    fn known_disallowed_reason(
        &self,
        word: &str,
        selected_level: HskLevel,
    ) -> Option<ViolationReason> {
        if self.hsk.is_allowed(word, selected_level) {
            return None;
        }
        if let Some(required_level) = self.hsk.level_of(word)
            && required_level > selected_level
        {
            return Some(ViolationReason::AboveSelectedHskLevel { required_level });
        }
        self.dictionary
            .contains_word(word)
            .then_some(ViolationReason::KnownDictionaryWord)
    }

    fn make_violation(
        &self,
        characters: &[char],
        start: usize,
        end: usize,
        reason: ViolationReason,
        selected_level: HskLevel,
    ) -> HskViolation {
        let text = characters[start..end].iter().collect::<String>();
        HskViolation {
            suggested_words: self.suggested_words(&text, selected_level),
            text,
            start_char: start,
            end_char: end,
            reason,
        }
    }

    fn suggested_words(&self, word: &str, selected_level: HskLevel) -> Vec<String> {
        let mut suggestions = Vec::new();
        if let Some(entry) = self.hsk.entry(word) {
            for candidate in &entry.simpler_words {
                if self.hsk.is_allowed(candidate, selected_level) {
                    push_unique(&mut suggestions, candidate);
                }
            }
        }
        if let Some(parts) = self.allowed_tries[selected_level.index()].best_decomposition(word)
            && parts.len() >= 2
        {
            for part in parts {
                push_unique(&mut suggestions, &part);
            }
        }

        let source_glosses = self.glosses_for(word);
        let source_terms = gloss_terms(&source_glosses);
        let mut scored = self
            .hsk
            .entries()
            .iter()
            .filter(|entry| entry.level <= selected_level && entry.simplified != word)
            .filter_map(|entry| {
                let candidate_terms = gloss_terms(&entry.glosses);
                let overlap = source_terms.intersection(&candidate_terms).count();
                let exact = entry
                    .glosses
                    .iter()
                    .any(|candidate| source_glosses.iter().any(|source| source == candidate));
                (overlap > 0 || exact).then_some(ScoredSuggestion {
                    word: entry.simplified.clone(),
                    exact,
                    overlap,
                    level: entry.level,
                    frequency_rank: entry.frequency_rank.unwrap_or(u32::MAX),
                })
            })
            .collect::<Vec<_>>();
        scored.sort_by(ScoredSuggestion::compare);

        for candidate in scored {
            push_unique(&mut suggestions, &candidate.word);
            if suggestions.len() == 3 {
                break;
            }
        }
        suggestions.truncate(3);
        suggestions
    }

    fn glosses_for(&self, word: &str) -> Vec<String> {
        let mut result = BTreeSet::new();
        if let Some(entry) = self.hsk.entry(word) {
            result.extend(entry.glosses.iter().map(|gloss| gloss.to_ascii_lowercase()));
        }
        for entry in self.dictionary.entries_for(word) {
            result.extend(
                entry
                    .definitions
                    .iter()
                    .map(|definition| definition.to_ascii_lowercase()),
            );
        }
        result.into_iter().collect()
    }

    /// Longest-match local lookup with HSK metadata and explicit proper-name
    /// labels. Punctuation/whitespace are omitted from token results.
    pub fn lookup(&self, selected_text: &str, proper_names: &[ProperName]) -> LookupResult {
        let selected_text = self.normalizer.normalize(selected_text);
        let characters = selected_text.chars().collect::<Vec<_>>();
        let names = self.normalize_names(proper_names);
        let name_matches = find_name_matches(&characters, &names);
        let names_by_start = name_matches
            .into_iter()
            .map(|name| (name.start, name))
            .collect::<BTreeMap<_, _>>();
        let mut tokens = Vec::new();
        let mut start = 0;

        while start < characters.len() {
            if let Some(name) = names_by_start.get(&start) {
                tokens.push(self.lookup_token(&name.text, true));
                start = name.end;
                continue;
            }
            let one = characters[start].to_string();
            if is_ignorable_token(&one) {
                start += 1;
                continue;
            }
            let end = self
                .full_lexicon
                .longest_match(&characters, start)
                .or_else(|| self.dictionary.longest_match(&characters, start))
                .unwrap_or(start + 1);
            let word = characters[start..end].iter().collect::<String>();
            tokens.push(self.lookup_token(&word, false));
            start = end;
        }

        LookupResult {
            selected_text,
            tokens,
        }
    }

    fn lookup_token(&self, word: &str, proper_name: bool) -> LookupToken {
        let (mut pinyin, mut definitions) = self.dictionary.merged_fields(word);
        if let Some(entry) = self.hsk.entry(word) {
            if pinyin.is_empty() {
                pinyin.clone_from(&entry.pinyin);
            }
            let mut merged = definitions.into_iter().collect::<BTreeSet<_>>();
            merged.extend(entry.glosses.iter().cloned());
            definitions = merged.into_iter().collect();
        }
        if proper_name && definitions.is_empty() {
            definitions.push("Proper name · outside HSK list".into());
        }
        LookupToken {
            simplified: word.into(),
            pinyin,
            definitions,
            hsk_level: self.hsk.level_of(word),
            proper_name,
        }
    }

    fn normalize_names(&self, names: &[ProperName]) -> Vec<NormalizedName> {
        let mut result = names
            .iter()
            .filter_map(|name| {
                let text = self.normalizer.normalize(&name.text);
                (!text.is_empty()).then_some(NormalizedName {
                    characters: text.chars().collect(),
                    text,
                    reason: name.reason,
                })
            })
            .collect::<Vec<_>>();
        result.sort_by(|left, right| {
            right
                .characters
                .len()
                .cmp(&left.characters.len())
                .then_with(|| left.text.cmp(&right.text))
                .then_with(|| reason_rank(left.reason).cmp(&reason_rank(right.reason)))
        });
        result.dedup_by(|left, right| left.text == right.text);
        result
    }
}

#[derive(Debug)]
struct PrimaryToken {
    text: String,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct CompoundGuard {
    start: usize,
    end: usize,
    reason: ViolationReason,
}

#[derive(Debug)]
struct NormalizedName {
    text: String,
    characters: Vec<char>,
    reason: ProperNameReason,
}

#[derive(Debug)]
struct NameMatch {
    text: String,
    start: usize,
    end: usize,
    reason: ProperNameReason,
}

fn find_name_matches(characters: &[char], names: &[NormalizedName]) -> Vec<NameMatch> {
    let mut result = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let matched = names.iter().find(|name| {
            start + name.characters.len() <= characters.len()
                && characters[start..start + name.characters.len()] == name.characters
        });
        if let Some(name) = matched {
            let end = start + name.characters.len();
            result.push(NameMatch {
                text: name.text.clone(),
                start,
                end,
                reason: name.reason,
            });
            start = end;
        } else {
            start += 1;
        }
    }
    result
}

fn char_boundaries(text: &str) -> Vec<usize> {
    let mut result = text
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    result.push(text.len());
    result
}

fn reason_rank(reason: ProperNameReason) -> u8 {
    match reason {
        ProperNameReason::PersonName => 0,
        ProperNameReason::PlaceName => 1,
        ProperNameReason::Title => 2,
        ProperNameReason::UnavoidableProperNoun => 3,
    }
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_owned());
    }
}

fn gloss_terms(glosses: &[String]) -> BTreeSet<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "and", "be", "for", "in", "of", "on", "or", "the", "to",
    ];
    glosses
        .iter()
        .flat_map(|gloss| {
            gloss
                .split(|character: char| !character.is_alphanumeric())
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
        })
        .filter(|term| term.len() > 1 && !STOP_WORDS.contains(&term.as_str()))
        .collect()
}

struct ScoredSuggestion {
    word: String,
    exact: bool,
    overlap: usize,
    level: HskLevel,
    frequency_rank: u32,
}

impl ScoredSuggestion {
    fn compare(left: &Self, right: &Self) -> Ordering {
        right
            .exact
            .cmp(&left.exact)
            .then_with(|| right.overlap.cmp(&left.overlap))
            .then_with(|| left.level.cmp(&right.level))
            .then_with(|| left.frequency_rank.cmp(&right.frequency_rank))
            .then_with(|| left.word.cmp(&right.word))
    }
}
