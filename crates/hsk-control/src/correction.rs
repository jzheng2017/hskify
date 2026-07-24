use crate::{HskControl, HskLevel, ProperName, ValidationReport, is_numeric_token};

/// Initial rewrite plus at most two correction attempts.
pub const MAX_CORRECTION_ATTEMPTS: u8 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreservationViolation {
    NumbersChanged {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    ProperNameOccurrencesChanged {
        text: String,
        expected: usize,
        actual: usize,
    },
    NegationMarkersChanged {
        expected: Vec<String>,
        actual: Vec<String>,
    },
}

/// Preservation requirements derived from the faithful Chinese reference.
#[derive(Debug, Clone)]
pub struct CorrectionContext {
    proper_names: Vec<ProperName>,
    required_name_forms: Vec<RequiredNameForm>,
    reference_numbers: Vec<String>,
    reference_negation_markers: Vec<String>,
}

#[derive(Debug, Clone)]
struct RequiredNameForm {
    text: String,
    occurrences: usize,
}

impl CorrectionContext {
    pub fn from_reference(
        control: &HskControl,
        faithful_chinese: &str,
        proper_names: &[ProperName],
    ) -> Self {
        let reference = control.normalize_text(faithful_chinese);
        let mut normalized_name_forms = proper_names
            .iter()
            .map(|name| control.normalize_text(&name.text))
            .filter(|name| !name.is_empty())
            .collect::<Vec<_>>();
        normalized_name_forms.sort();
        normalized_name_forms.dedup();
        let required_name_forms = normalized_name_forms
            .into_iter()
            .filter_map(|text| {
                let occurrences = count_occurrences(&reference, &text);
                (occurrences > 0).then_some(RequiredNameForm { text, occurrences })
            })
            .collect();

        Self {
            proper_names: proper_names.to_vec(),
            required_name_forms,
            reference_numbers: extract_numeric_forms(&reference),
            reference_negation_markers: control.negation_markers(&reference),
        }
    }

    pub fn proper_names(&self) -> &[ProperName] {
        &self.proper_names
    }

    pub fn reference_numbers(&self) -> &[String] {
        &self.reference_numbers
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectionOutcome {
    Accepted {
        report: ValidationReport,
    },
    Retry {
        /// One-based correction attempt that should be requested next.
        correction_attempt: u8,
        final_attempt: bool,
        report: ValidationReport,
        preservation_violations: Vec<PreservationViolation>,
    },
    Failed {
        report: ValidationReport,
        preservation_violations: Vec<PreservationViolation>,
    },
    Terminated,
}

/// Stateful bound around validator feedback. It never asks for more than two
/// correction attempts after the initial rewrite.
pub struct CorrectionLoop<'a> {
    control: &'a HskControl,
    level: HskLevel,
    context: CorrectionContext,
    corrections_issued: u8,
    terminal: bool,
}

impl<'a> CorrectionLoop<'a> {
    pub(crate) fn new(
        control: &'a HskControl,
        level: HskLevel,
        context: CorrectionContext,
    ) -> Self {
        Self {
            control,
            level,
            context,
            corrections_issued: 0,
            terminal: false,
        }
    }

    pub fn evaluate(&mut self, candidate: &str) -> CorrectionOutcome {
        if self.terminal {
            return CorrectionOutcome::Terminated;
        }

        let report = self
            .control
            .validate(candidate, self.level, self.context.proper_names());
        let preservation_violations = self.preservation_violations(&report.normalized_text);

        if report.strictly_valid && preservation_violations.is_empty() {
            self.terminal = true;
            return CorrectionOutcome::Accepted { report };
        }

        if self.corrections_issued < MAX_CORRECTION_ATTEMPTS {
            self.corrections_issued += 1;
            CorrectionOutcome::Retry {
                correction_attempt: self.corrections_issued,
                final_attempt: self.corrections_issued == MAX_CORRECTION_ATTEMPTS,
                report,
                preservation_violations,
            }
        } else {
            self.terminal = true;
            CorrectionOutcome::Failed {
                report,
                preservation_violations,
            }
        }
    }

    pub fn corrections_issued(&self) -> u8 {
        self.corrections_issued
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn preservation_violations(&self, candidate: &str) -> Vec<PreservationViolation> {
        let mut violations = Vec::new();
        let actual_numbers = extract_numeric_forms(candidate);
        if actual_numbers != self.context.reference_numbers {
            violations.push(PreservationViolation::NumbersChanged {
                expected: self.context.reference_numbers.clone(),
                actual: actual_numbers,
            });
        }
        for name in &self.context.required_name_forms {
            let actual = count_occurrences(candidate, &name.text);
            if actual != name.occurrences {
                violations.push(PreservationViolation::ProperNameOccurrencesChanged {
                    text: name.text.clone(),
                    expected: name.occurrences,
                    actual,
                });
            }
        }
        let actual_negation_markers = self.control.negation_markers(candidate);
        if actual_negation_markers != self.context.reference_negation_markers {
            violations.push(PreservationViolation::NegationMarkersChanged {
                expected: self.context.reference_negation_markers.clone(),
                actual: actual_negation_markers,
            });
        }
        violations
    }
}

impl HskControl {
    pub fn correction_loop(
        &self,
        level: HskLevel,
        faithful_chinese: &str,
        proper_names: &[ProperName],
    ) -> CorrectionLoop<'_> {
        CorrectionLoop::new(
            self,
            level,
            CorrectionContext::from_reference(self, faithful_chinese, proper_names),
        )
    }
}

fn extract_numeric_forms(text: &str) -> Vec<String> {
    let mut forms = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if is_numeric_component(character) {
            current.push(character);
        } else {
            flush_numeric(&mut current, &mut forms);
        }
    }
    flush_numeric(&mut current, &mut forms);
    forms
}

fn count_occurrences(text: &str, needle: &str) -> usize {
    text.match_indices(needle).count()
}

fn flush_numeric(current: &mut String, forms: &mut Vec<String>) {
    if is_numeric_token(current) {
        forms.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn is_numeric_component(character: char) -> bool {
    character.is_numeric()
        || matches!(
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
                | '.'
                | ','
                | '，'
                | '+'
                | '-'
                | '−'
                | '/'
                | '%'
                | '％'
                | '点'
                | '分'
                | '之'
        )
}
