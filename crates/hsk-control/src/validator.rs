use std::{
    array,
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

use jieba_rs::{Jieba, TokenizeMode};

use crate::{
    DictionaryArtifact, HSK_STANDARD, HskArtifact, HskDataset, HskException, HskLevel,
    HskViolation, JIEBA_CRATE_VERSION, JIEBA_EMBEDDED_DICTIONARY_SHA256, LOOKUP_REVISION,
    LoadPolicy, LocalDictionary, LookupRegionContext, LookupResult, LookupToken,
    NORMALIZATION_REVISION, ProperName, ProperNameReason, Result, SEGMENTATION_REVISION,
    TextNormalizer, UNICODE_NORMALIZATION_CRATE_VERSION, UNICODE_NORMALIZATION_TABLES_SHA256,
    ValidationReport, ViolationReason,
    normalization::{is_all_han, is_ignorable_token},
    sha256_hex,
    trie::AllowedWordTrie,
};
#[cfg(feature = "test-seeds")]
use crate::{EMBEDDED_DICTIONARY_TEST_SEED, EMBEDDED_HSK_TEST_SEED};

const OPENCC_DEPENDENCY_REVISION: &str = "opencc-fmmseg-0.8.0-tw2sp-hk2s";

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
    /// test-only policy. This API and its embedded resources exist only when
    /// the non-default `test-seeds` feature is enabled.
    #[cfg(feature = "test-seeds")]
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
            JIEBA_CRATE_VERSION,
            JIEBA_EMBEDDED_DICTIONARY_SHA256,
            UNICODE_NORMALIZATION_CRATE_VERSION,
            UNICODE_NORMALIZATION_TABLES_SHA256,
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

        let guards = self.disallowed_spans(characters, start, end, selected_level);
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
            if span_intersects_guard(token.start, token.end, &guards) {
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
                if decomposition.as_ref().is_some_and(|parts| parts.len() >= 2) {
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
        if characters.len() < 2 {
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

    fn disallowed_spans(
        &self,
        characters: &[char],
        gap_start: usize,
        gap_end: usize,
        selected_level: HskLevel,
    ) -> Vec<CompoundGuard> {
        let mut allowed_spans = Vec::new();
        let mut candidates = Vec::new();

        // Scan every Unicode character position in the gap. Jieba remains the
        // primary tokenizer, but it is not an authority on where a known
        // HSK/dictionary span is allowed to begin or end.
        for start in gap_start..gap_end {
            for end in self.full_lexicon.matches_from(characters, start) {
                if end > gap_end {
                    break;
                }
                let word = characters[start..end].iter().collect::<String>();
                if !is_all_han(&word) {
                    continue;
                }
                if self.hsk.is_allowed(&word, selected_level) {
                    allowed_spans.push((start, end));
                } else if let Some(reason) = self.known_disallowed_reason(&word, selected_level) {
                    candidates.push(CompoundGuard { start, end, reason });
                }
            }
        }

        // A selected-level HSK headword is valid as a whole even if a shorter
        // dictionary spelling happens to occur inside it. Cross-boundary spans
        // (the failure mode this guard addresses) are not contained and remain
        // candidates.
        candidates.retain(|candidate| {
            !allowed_spans.iter().any(|(start, end)| {
                *start <= candidate.start
                    && candidate.end <= *end
                    && (*start < candidate.start || candidate.end < *end)
            })
        });
        candidates.sort_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then_with(|| right.end.cmp(&left.end))
        });
        candidates.dedup_by(|left, right| {
            left.start == right.start && left.end == right.end && left.reason == right.reason
        });
        best_nonoverlapping_guard_path(candidates, gap_start, gap_end)
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
        self.lookup_with_region_context(selected_text, proper_names, None)
    }

    /// Adds optional immutable region context to the pure lookup result. The
    /// Chinese/English context is carried verbatim; only `selected_text` is
    /// normalized for dictionary lookup.
    pub fn lookup_with_region_context(
        &self,
        selected_text: &str,
        proper_names: &[ProperName],
        region: Option<LookupRegionContext>,
    ) -> LookupResult {
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
            region,
        }
    }

    /// Resolves exactly one dictionary expression beginning at a hovered
    /// Unicode-scalar offset. Longest-match is intentionally anchored at the
    /// hovered character: hovering the first character of a compound returns
    /// the compound, while hovering a later component starts a fresh lookup
    /// from that component.
    pub fn lookup_at_with_region_context(
        &self,
        displayed_text: &str,
        character_offset: usize,
        proper_names: &[ProperName],
        region: Option<LookupRegionContext>,
    ) -> Option<LookupResult> {
        let characters = displayed_text.chars().collect::<Vec<_>>();
        let first = *characters.get(character_offset)?;
        let first_text = first.to_string();
        if is_ignorable_token(&first_text) {
            return None;
        }

        let suffix = characters[character_offset..].iter().collect::<String>();
        let mut result = self.lookup_with_region_context(&suffix, proper_names, region);
        let token = result.tokens.into_iter().next()?;
        result.selected_text.clone_from(&token.simplified);
        result.tokens = vec![token];
        Some(result)
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
        if proper_name && pinyin.is_empty() {
            pinyin = self.composed_pinyin(word);
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

    fn composed_pinyin(&self, word: &str) -> String {
        let characters = word.chars().collect::<Vec<_>>();
        let mut parts = Vec::new();
        let mut start = 0;
        while start < characters.len() {
            let end = self
                .full_lexicon
                .longest_match(&characters, start)
                .unwrap_or(start + 1);
            let component = characters[start..end].iter().collect::<String>();
            let (mut pinyin, _) = self.dictionary.merged_fields(&component);
            if pinyin.is_empty()
                && let Some(entry) = self.hsk.entry(&component)
            {
                pinyin.clone_from(&entry.pinyin);
            }
            if pinyin.trim().is_empty() {
                return String::new();
            }
            parts.push(pinyin);
            start = end;
        }
        parts.join(" ")
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

#[derive(Clone, Debug)]
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

fn span_intersects_guard(start: usize, end: usize, guards: &[CompoundGuard]) -> bool {
    guards
        .iter()
        .any(|guard| guard.start < end && start < guard.end)
}

#[derive(Clone, Debug, Default)]
struct GuardPath {
    lexical_weight: usize,
    covered_characters: usize,
    guards: Vec<CompoundGuard>,
}

fn best_nonoverlapping_guard_path(
    candidates: Vec<CompoundGuard>,
    gap_start: usize,
    gap_end: usize,
) -> Vec<CompoundGuard> {
    let mut starts = vec![Vec::<CompoundGuard>::new(); gap_end.saturating_add(1)];
    for candidate in candidates {
        starts[candidate.start].push(candidate);
    }
    for candidates in &mut starts {
        candidates.sort_by(|left, right| {
            right
                .end
                .cmp(&left.end)
                .then_with(|| left.start.cmp(&right.start))
        });
    }

    let mut best = vec![GuardPath::default(); gap_end.saturating_add(1)];
    for position in (gap_start..gap_end).rev() {
        let mut selected = best[position + 1].clone();
        for candidate in &starts[position] {
            let length = candidate.end - candidate.start;
            let mut path = best[candidate.end].clone();
            path.lexical_weight = path
                .lexical_weight
                .saturating_add(length.saturating_mul(length));
            path.covered_characters = path.covered_characters.saturating_add(length);
            path.guards.insert(0, candidate.clone());
            if guard_path_is_better(&path, &selected) {
                selected = path;
            }
        }
        best[position] = selected;
    }
    best[gap_start].guards.clone()
}

fn guard_path_is_better(candidate: &GuardPath, current: &GuardPath) -> bool {
    candidate
        .lexical_weight
        .cmp(&current.lexical_weight)
        .then_with(|| {
            candidate
                .covered_characters
                .cmp(&current.covered_characters)
        })
        .then_with(|| current.guards.len().cmp(&candidate.guards.len()))
        == Ordering::Greater
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
