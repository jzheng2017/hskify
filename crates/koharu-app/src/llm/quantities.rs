//! Conservative checks of explicit numeric evidence. These do not certify meaning.
fn digit(character: char) -> Option<u64> {
    match character {
        '0' | '零' | '〇' => Some(0),
        '1' | '一' => Some(1),
        '2' | '二' | '两' => Some(2),
        '3' | '三' => Some(3),
        '4' | '四' => Some(4),
        '5' | '五' => Some(5),
        '6' | '六' => Some(6),
        '7' | '七' => Some(7),
        '8' | '八' => Some(8),
        '9' | '九' => Some(9),
        _ => None,
    }
}
fn numeral(c: char) -> bool {
    digit(c).is_some() || "十百千万亿点负正".contains(c)
}
fn canonical(value: &str) -> Option<String> {
    let negative = value.starts_with(['-', '−', '－', '负']);
    let value = value.trim_start_matches(['-', '−', '－', '+', '＋', '负', '正']);
    let mut parts = value.split(['.', '点']);
    let integer = parts.next()?;
    let integer = if integer.chars().all(|c| digit(c).is_some()) {
        integer.chars().try_fold(0u64, |value, c| {
            value.checked_mul(10)?.checked_add(digit(c)?)
        })?
    } else {
        let mut total = 0u64;
        let mut section = 0u64;
        let mut number = None;
        let mut previous_small = 10_000;
        let mut previous_large = u64::MAX;
        for c in integer.chars() {
            if let Some(value) = digit(c) {
                number = Some(number.unwrap_or(0u64).checked_mul(10)?.checked_add(value)?);
            } else {
                let unit = match c {
                    '十' => 10,
                    '百' => 100,
                    '千' => 1000,
                    '万' => 10_000,
                    '亿' => 100_000_000,
                    _ => return None,
                };
                if unit < 10_000 {
                    if unit >= previous_small {
                        return None;
                    }
                    section = section.checked_add(number.take().unwrap_or(1).checked_mul(unit)?)?;
                    previous_small = unit;
                } else {
                    if unit >= previous_large {
                        return None;
                    }
                    section = section.checked_add(number.take().unwrap_or(0))?;
                    total = total.checked_add(section.checked_mul(unit)?)?;
                    section = 0;
                    previous_small = 10_000;
                    previous_large = unit;
                }
            }
        }
        total
            .checked_add(section)?
            .checked_add(number.unwrap_or(0))?
    };
    let decimal = parts
        .next()
        .map(|part| {
            part.chars()
                .map(|c| digit(c).map(|d| char::from(b'0' + d as u8)))
                .collect::<Option<String>>()
        })
        .transpose_option()?;
    if parts.next().is_some() {
        return None;
    }
    let decimal = decimal.unwrap_or_default();
    let decimal = decimal.trim_end_matches('0');
    let sign = if negative && (integer != 0 || !decimal.is_empty()) {
        "-"
    } else {
        ""
    };
    Some(if decimal.is_empty() {
        format!("{sign}{integer}")
    } else {
        format!("{sign}{integer}.{decimal}")
    })
}
trait TransposeOption<T> {
    fn transpose_option(self) -> Option<Option<T>>;
}
impl<T> TransposeOption<T> for Option<Option<T>> {
    fn transpose_option(self) -> Option<Option<T>> {
        match self {
            Some(Some(v)) => Some(Some(v)),
            Some(None) => None,
            None => Some(None),
        }
    }
}

fn scale_zeros(word: &str) -> Option<usize> {
    match word
        .trim_matches(|c: char| !c.is_ascii_alphabetic())
        .to_ascii_lowercase()
        .as_str()
    {
        "hundred" => Some(2),
        "thousand" => Some(3),
        "million" => Some(6),
        "billion" => Some(9),
        _ => None,
    }
}

fn decimal_scale(value: &str, zeros: usize) -> Option<String> {
    let negative = value.starts_with('-');
    let unsigned = value.trim_start_matches('-');
    let (integer, decimal) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let position = integer.len() + zeros;
    let mut digits = format!("{integer}{decimal}");
    if digits.len() < position {
        digits.extend(std::iter::repeat_n('0', position - digits.len()));
    }
    if digits.len() > position {
        digits.insert(position, '.');
    }
    if negative {
        digits.insert(0, '-');
    }
    canonical(&digits)
}

pub(super) fn explicit_numbers(text: &str, chinese: bool) -> Vec<String> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut numbers = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let (start, c) = chars[index];
        let is_start = c.is_ascii_digit()
            || chinese
                && (digit(c).is_some()
                    || "十百千万亿".contains(c)
                    || "负正点".contains(c)
                        && chars.get(index + 1).is_some_and(|(_, next)| {
                            digit(*next).is_some() || "十百千万亿".contains(*next)
                        }))
            || c == '.'
                && chars
                    .get(index + 1)
                    .is_some_and(|(_, next)| next.is_ascii_digit())
            || matches!(c, '-' | '−' | '－' | '+' | '＋')
                && chars
                    .get(index + 1)
                    .is_some_and(|(_, next)| next.is_ascii_digit());
        if !is_start {
            index += 1;
            continue;
        }
        let first = index;
        index += 1;
        while index < chars.len()
            && (chars[index].1.is_ascii_digit()
                || chinese && numeral(chars[index].1)
                || chars[index].1 == '.'
                    && chars
                        .get(index + 1)
                        .is_some_and(|(_, next)| next.is_ascii_digit())
                || chars[index].1 == ','
                    && (index - first <= 3 || chars[first..index].iter().any(|(_, c)| *c == ','))
                    && (1..=3).all(|offset| {
                        chars
                            .get(index + offset)
                            .is_some_and(|(_, next)| next.is_ascii_digit())
                    })
                    && !chars
                        .get(index + 4)
                        .is_some_and(|(_, next)| next.is_ascii_digit()))
        {
            index += 1;
        }
        let end = chars.get(index).map_or(text.len(), |(offset, _)| *offset);
        let left = first.checked_sub(1).map(|i| chars[i].1);
        let right = chars.get(index).map(|(_, c)| *c);
        let multiplier = left.is_some_and(|c| c == 'x' || c == 'X')
            || right.is_some_and(|c| c == 'x' || c == 'X');
        if !chinese
            && !multiplier
            && (left.is_some_and(|c| c.is_ascii_alphabetic())
                || right.is_some_and(|c| c.is_ascii_alphabetic()))
        {
            continue;
        }
        let literal = &text[start..end];
        if chinese && !literal.starts_with(['负', '正']) && !literal.chars().any(|c| c.is_ascii_digit()) && right.is_some_and(|c| !c.is_whitespace() && !"，。！？、：；,.!?;:%％个名位岁年月天次本只条张部辆瓶杯件米元分秒斤吨倍层章页号组队箱种人生".contains(c)) { continue; }
        if literal.contains(',') {
            let groups = literal
                .trim_start_matches(['-', '−', '－', '+', '＋'])
                .split('.')
                .next()
                .unwrap_or_default()
                .split(',')
                .collect::<Vec<_>>();
            if groups.len() < 2
                || groups[0].len() > 3
                || groups
                    .iter()
                    .skip(1)
                    .any(|group| group.len() != 3 || !group.bytes().all(|b| b.is_ascii_digit()))
            {
                continue;
            }
        }
        if let Some(mut value) = canonical(&literal.replace(',', "")) {
            let preceding_sign = text[..start].split_whitespace().next_back();
            if !chinese
                && !value.starts_with('-')
                && value != "0"
                && preceding_sign.is_some_and(|word| {
                    word.eq_ignore_ascii_case("minus") || word.eq_ignore_ascii_case("negative")
                })
            {
                value.insert(0, '-');
            }
            if !chinese
                && let Some(scale) = text[end..].split_whitespace().next().and_then(scale_zeros)
            {
                let Some(scaled) = decimal_scale(&value, scale) else {
                    continue;
                };
                value = scaled;
            }
            numbers.push(value);
        }
    }
    if !chinese {
        numbers.extend(english_numbers(text));
    }

    numbers
}

fn english_number(word: &str) -> Option<u64> {
    match word.to_ascii_lowercase().as_str() {
        "zero" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        _ => None,
    }
}

fn english_numbers(text: &str) -> Vec<String> {
    let mut numbers = Vec::new();
    let (mut total, mut section, mut previous) = (0u64, None::<u64>, None::<u64>);
    let mut negative = false;
    let flush = |numbers: &mut Vec<String>,
                 total: &mut u64,
                 section: &mut Option<u64>,
                 previous: &mut Option<u64>,
                 negative: &mut bool| {
        if section.is_some() || *total != 0 {
            let value = total.saturating_add(section.take().unwrap_or(0));
            numbers.push(if *negative && value != 0 {
                format!("-{value}")
            } else {
                value.to_string()
            });
        }
        *total = 0;
        *previous = None;
        *negative = false;
    };
    let mut tokens = text
        .split_inclusive(|c: char| !c.is_ascii_alphabetic())
        .chain(std::iter::once(""))
        .peekable();
    while let Some(token) = tokens.next() {
        let word = token.trim_matches(|c: char| !c.is_ascii_alphabetic());
        if let Some(value) = english_number(word) {
            // Standalone "one" may be a pronoun ("no one", "the one"). Only a numeric group or explicit magnitude makes it unambiguous.
            let next_scale = tokens.peek().and_then(|next| {
                scale_zeros(next.trim_matches(|c: char| !c.is_ascii_alphabetic()))
            });
            if value == 1
                && !negative
                && total == 0
                && section.is_none_or(|number| number < 20)
                && next_scale.is_none()
            {
                continue;
            }
            if previous.is_some_and(|last| !(last >= 20 && last % 10 == 0 && value < 10)) {
                flush(
                    &mut numbers,
                    &mut total,
                    &mut section,
                    &mut previous,
                    &mut negative,
                );
            }
            section = Some(section.unwrap_or(0).saturating_add(value));
            previous = Some(value);
        } else if word.eq_ignore_ascii_case("hundred")
            && section.is_some_and(|value| (1..=9).contains(&value))
        {
            section = section.map(|value| value * 100);
            previous = None;
        } else if let Some(scale) =
            scale_zeros(word).filter(|scale| *scale >= 3 && section.is_some())
        {
            total = total.saturating_add(
                section
                    .take()
                    .unwrap_or(0)
                    .saturating_mul(10u64.pow(scale as u32)),
            );
            previous = None;
        } else if ["minus", "negative", "plus", "positive"]
            .iter()
            .any(|sign| word.eq_ignore_ascii_case(sign))
        {
            flush(
                &mut numbers,
                &mut total,
                &mut section,
                &mut previous,
                &mut negative,
            );
            negative = word.eq_ignore_ascii_case("minus") || word.eq_ignore_ascii_case("negative");
        } else if !(word.eq_ignore_ascii_case("and")
            && (total > 0 || section.is_some_and(|value| value >= 100)))
        {
            flush(
                &mut numbers,
                &mut total,
                &mut section,
                &mut previous,
                &mut negative,
            );
        }
        if token.ends_with([',', ';', '.', '!', '?']) {
            flush(
                &mut numbers,
                &mut total,
                &mut section,
                &mut previous,
                &mut negative,
            );
        }
    }
    numbers
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ambiguous_pronouns_do_not_invent_quantities_and_explicit_magnitudes_are_complete_values() {
        assert!(explicit_numbers("No one knew who was the one.", false).is_empty());
        assert_eq!(
            explicit_numbers("negative 3, minus one; 3.5 million.", false),
            ["-3", "3500000", "-1"]
        );
        assert_eq!(
            explicit_numbers(
                "3.5 million years and one million people, minus three; positive five.",
                false
            ),
            ["3500000", "1000000", "-3", "5"]
        );
    }
    #[test]
    fn complete_values_signs_and_multiplicity() {
        assert_eq!(
            explicit_numbers("三十，三，三，负三点五，二千零六。", true),
            ["30", "3", "3", "-3.5", "2006"]
        );
        assert_eq!(
            explicit_numbers("3, 3, -3.50 and 1,200", false),
            ["3", "3", "-3.5", "1200"]
        );
        assert_eq!(
            explicit_numbers(
                "3,3 and three three; one hundred and one; twenty-three",
                false
            ),
            ["3", "3", "3", "3", "101", "23"]
        );
        assert_eq!(explicit_numbers("1,200,000.50", false), ["1200000.5"]);
        assert_eq!(explicit_numbers(".5 and -0.50", false), ["0.5", "-0.5"]);
        assert!(explicit_numbers("负责。点心。正是时候。", true).is_empty());
        assert_eq!(
            explicit_numbers("IDENTIT4 and M4; X3 and 3x", false),
            ["3", "3"]
        );
    }
}

#[cfg(test)]
mod sign_tests {
    use super::explicit_numbers;
    #[test]
    fn unicode_minus_and_fullwidth_signs_preserve_numeric_signs() {
        assert_eq!(
            explicit_numbers("−3 and －4 and ＋5", false),
            ["-3", "-4", "5"]
        );
        assert_eq!(
            explicit_numbers("负三、负四和正五。", true),
            ["-3", "-4", "5"]
        );
    }
}
