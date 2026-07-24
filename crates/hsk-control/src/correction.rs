use crate::{HskControl, HskLevel, ProperName, ValidationReport, is_numeric_token};

/// Initial rewrite plus at most two correction attempts.
pub const MAX_CORRECTION_ATTEMPTS: u8 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreservationViolation {
    NumbersChanged {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    MissingProperName {
        text: String,
    },
    RemovedNegation,
    AddedNegation,
}

/// Preservation requirements derived from the faithful Chinese reference.
#[derive(Debug, Clone)]
pub struct CorrectionContext {
    proper_names: Vec<ProperName>,
    required_name_forms: Vec<String>,
    reference_numbers: Vec<String>,
    reference_has_negation: bool,
}

impl CorrectionContext {
    pub fn from_reference(
        control: &HskControl,
        faithful_chinese: &str,
        proper_names: &[ProperName],
    ) -> Self {
        let reference = control.normalize_text(faithful_chinese);
        let mut required_name_forms = proper_names
            .iter()
            .map(|name| control.normalize_text(&name.text))
            .filter(|name| !name.is_empty() && reference.contains(name))
            .collect::<Vec<_>>();
        required_name_forms.sort();
        required_name_forms.dedup();

        Self {
            proper_names: proper_names.to_vec(),
            required_name_forms,
            reference_numbers: extract_numeric_forms(&reference),
            reference_has_negation: control.has_negation(&reference),
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
            if !candidate.contains(name) {
                violations.push(PreservationViolation::MissingProperName { text: name.clone() });
            }
        }
        match (
            self.context.reference_has_negation,
            self.control.has_negation(candidate),
        ) {
            (true, false) => violations.push(PreservationViolation::RemovedNegation),
            (false, true) => violations.push(PreservationViolation::AddedNegation),
            _ => {}
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
