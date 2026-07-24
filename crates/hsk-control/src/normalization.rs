use opencc_fmmseg::OpenCC;
use unicode_normalization::UnicodeNormalization;

/// Bump whenever normalization order or mappings change.
pub const NORMALIZATION_REVISION: &str = "nfkc-zero-width-opencc-tw2sp-hk2s-surface-v2";

/// Unicode/OpenCC-compatible normalizer used by import, validation, and lookup.
pub struct TextNormalizer {
    opencc: OpenCC,
}

impl TextNormalizer {
    pub fn new() -> Self {
        Self {
            opencc: OpenCC::new(),
        }
    }

    /// Produces NFKC, removes zero-width controls, converts Traditional
    /// variants with OpenCC-compatible Taiwan/Hong Kong-to-mainland Simplified
    /// conversion, and canonicalizes punctuation and whitespace.
    pub fn normalize(&self, input: &str) -> String {
        let unicode = input
            .nfkc()
            .filter(|character| !is_zero_width(*character))
            .collect::<String>();
        // `tw2sp` includes general Traditional-to-Simplified conversion plus
        // mainland phrase mappings (for example 軟體 -> 软件). A following
        // `hk2s` covers Hong Kong variants that are not in the Taiwan config.
        let taiwan_simplified = self.opencc.tw2sp(&unicode, false);
        let simplified = self.opencc.hk2s(&taiwan_simplified, false);
        normalize_surface(&simplified)
    }
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

fn is_zero_width(character: char) -> bool {
    matches!(
        character,
        '\u{00ad}'
            | '\u{034f}'
            | '\u{061c}'
            | '\u{180e}'
            | '\u{200b}'
            | '\u{200c}'
            | '\u{200d}'
            | '\u{2060}'
            | '\u{feff}'
    ) || ('\u{2061}'..='\u{2064}').contains(&character)
}

fn normalize_surface(input: &str) -> String {
    let characters = input.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(input.len());
    let mut pending_space = false;

    for (index, character) in characters.iter().copied().enumerate() {
        if character.is_whitespace() {
            pending_space = !output.is_empty();
            continue;
        }
        if pending_space {
            output.push(' ');
            pending_space = false;
        }

        let previous = index
            .checked_sub(1)
            .and_then(|i| characters.get(i))
            .copied();
        let next = characters.get(index + 1).copied();
        output.push(canonical_punctuation(character, previous, next));
    }

    output
}

fn canonical_punctuation(character: char, previous: Option<char>, next: Option<char>) -> char {
    match character {
        ',' | '﹐' | '､' => '，',
        '!' | '﹗' => '！',
        '?' | '﹖' => '？',
        ':' | '﹕' => '：',
        ';' | '﹔' => '；',
        '(' | '﹙' => '（',
        ')' | '﹚' => '）',
        '[' | '﹝' => '【',
        ']' | '﹞' => '】',
        '.' | '﹒' | '｡'
            if !(previous.is_some_and(is_numeric_core) && next.is_some_and(is_numeric_core)) =>
        {
            '。'
        }
        other => other,
    }
}

fn is_numeric_core(character: char) -> bool {
    character.is_numeric() || is_chinese_numeric_core(character)
}

/// Returns whether a scalar belongs to a Han ideograph block.
pub fn is_han(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xf900..=0xfaff
            | 0x20000..=0x2fa1f
            | 0x30000..=0x323af
    )
}

fn is_chinese_numeric_core(character: char) -> bool {
    matches!(
        character,
        '零' | '〇'
            | '一'
            | '二'
            | '两'
            | '三'
            | '四'
            | '五'
            | '六'
            | '七'
            | '八'
            | '九'
            | '十'
            | '百'
            | '千'
            | '万'
            | '亿'
            | '兆'
    )
}

fn is_numeric_separator(character: char) -> bool {
    matches!(
        character,
        '.' | ',' | '，' | '+' | '-' | '−' | '/' | '%' | '％' | '点' | '分' | '之'
    )
}

/// Chinese and Arabic numeric forms are lexical exceptions, but the token must
/// contain a real digit/numeral and may contain only recognized separators.
pub fn is_numeric_token(token: &str) -> bool {
    let mut has_numeric_core = false;
    let mut has_character = false;

    for character in token.chars() {
        has_character = true;
        if character.is_numeric() || is_chinese_numeric_core(character) {
            has_numeric_core = true;
        } else if !is_numeric_separator(character) {
            return false;
        }
    }

    has_character && has_numeric_core
}

pub(crate) fn is_ignorable_token(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|character| character.is_whitespace() || !character.is_alphanumeric())
}

pub(crate) fn is_all_han(token: &str) -> bool {
    !token.is_empty() && token.chars().all(is_han)
}
