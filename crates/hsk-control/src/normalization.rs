use opencc_fmmseg::OpenCC;
use unicode_normalization::UnicodeNormalization;

/// Exact normalization crate release used by the NFKC stage.
pub const UNICODE_NORMALIZATION_CRATE_VERSION: &str = "0.1.25";

/// Unicode Character Database version compiled into `unicode-normalization`.
pub const UNICODE_NORMALIZATION_UNICODE_VERSION: (u8, u8, u8) = (17, 0, 0);

/// SHA-256 of `unicode-normalization` 0.1.25's generated `src/tables.rs`.
pub const UNICODE_NORMALIZATION_TABLES_SHA256: &str =
    "177d5f08019cc8e335444fcab61aabb7f6309f158f6ebbd7525c73c0e532ec44";

/// Bump whenever normalization order or mappings change.
pub const NORMALIZATION_REVISION: &str = "nfkc17-zero-width-opencc-script-aware-surface-v6";

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
    /// variants with OpenCC-compatible Taiwan-to-mainland Simplified
    /// conversion, and canonicalizes punctuation and whitespace.
    pub fn normalize(&self, input: &str) -> String {
        let unicode = input
            .nfkc()
            .filter(|character| !is_zero_width(*character))
            .collect::<String>();
        // Locale reverse dictionaries are only valid for Traditional input.
        // Applying `tw2sp` to already-Simplified text rewrites correct 什么 as
        // 什幺. Detect explicit Simplified evidence with the inverse mapping;
        // mixed/Simplified text takes the idempotent general T2S path, while
        // genuinely Traditional text receives Taiwan phrase normalization.
        let has_simplified_evidence = self.opencc.s2t(&unicode, false) != unicode;
        let simplified = if has_simplified_evidence {
            self.opencc.t2s(&unicode, false)
        } else {
            self.opencc.tw2sp(&unicode, false)
        };
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
    let mut index = 0;

    while let Some(&character) = characters.get(index) {
        if character.is_whitespace() {
            pending_space = !output.is_empty();
            index += 1;
            continue;
        }
        if pending_space {
            output.push(' ');
            pending_space = false;
        }

        // Generative models commonly spell an ellipsis as three or six ASCII
        // periods. Converting each period independently creates six full-width
        // stops, which changes the punctuation and can make short balloon text
        // impossible to fit. Treat a contiguous run as the single Chinese
        // ellipsis it represents.
        if is_ellipsis_dot(character) {
            let end = characters[index..]
                .iter()
                .take_while(|candidate| is_ellipsis_dot(**candidate))
                .count()
                + index;
            if end - index >= 3 {
                output.push_str("……");
                index = end;
                continue;
            }
        }

        let previous = index
            .checked_sub(1)
            .and_then(|i| characters.get(i))
            .copied();
        let next = characters.get(index + 1).copied();
        output.push(canonical_punctuation(character, previous, next));
        index += 1;
    }

    output
}

fn is_ellipsis_dot(character: char) -> bool {
    matches!(character, '.' | '\u{3002}' | '\u{fe52}' | '\u{ff61}')
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
