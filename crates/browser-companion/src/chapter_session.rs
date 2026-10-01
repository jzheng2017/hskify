//! Chapter-owned state shared by page analyses.
//!
//! A browser job is an execution unit, not a document.  This module keeps
//! the immutable page surfaces, ordered region plans, and dialogue links in
//! one chapter session so completion order cannot change the meaning of a
//! later page.

use std::collections::{BTreeMap, HashMap};

use koharu_app::llm::HskPrecedingUtterance;
use serde::Serialize;

pub const MAX_CONTEXT_UTTERANCES: usize = 6;

/// One immutable source unit plus its optional terminal translation. Every
/// modality registers units before language dispatch; execution priority may
/// change, but context is always read in `(source_index, item_order)` order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChapterContextUnit {
    pub item_id: String,
    pub source_index: u32,
    pub item_order: u32,
    pub source_text: String,
    pub displayed_text: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct ChapterContextStore {
    chapters: HashMap<String, BTreeMap<(u32, u32), ChapterContextUnit>>,
}

impl ChapterContextStore {
    pub fn register<I>(&mut self, chapter_id: &str, units: I)
    where
        I: IntoIterator<Item = ChapterContextUnit>,
    {
        let chapter = self.chapters.entry(chapter_id.to_owned()).or_default();
        for unit in units {
            let position = (unit.source_index, unit.item_order);
            match chapter.get_mut(&position) {
                Some(existing)
                    if existing.item_id == unit.item_id
                        && existing.source_text == unit.source_text =>
                {
                    // Registration is immutable apart from preserving an
                    // already-published translation during replay/refresh.
                    let displayed_text = existing.displayed_text.take();
                    *existing = unit;
                    existing.displayed_text = displayed_text;
                }
                Some(existing) => {
                    *existing = unit;
                }
                None => {
                    chapter.insert(position, unit);
                }
            }
        }
    }

    pub fn publish(
        &mut self,
        chapter_id: &str,
        position: (u32, u32),
        item_id: &str,
        displayed_text: String,
    ) {
        if let Some(unit) = self
            .chapters
            .get_mut(chapter_id)
            .and_then(|chapter| chapter.get_mut(&position))
            && unit.item_id == item_id
        {
            unit.displayed_text = Some(displayed_text);
        }
    }

    pub fn preceding(&self, chapter_id: &str, position: (u32, u32)) -> Vec<HskPrecedingUtterance> {
        let Some(chapter) = self.chapters.get(chapter_id) else {
            return Vec::new();
        };
        let mut context = chapter
            .range(..position)
            .rev()
            .filter_map(|(_, unit)| {
                let chinese = unit.displayed_text.as_deref()?.trim();
                let source = unit.source_text.trim();
                (!source.is_empty() && !chinese.is_empty()).then(|| HskPrecedingUtterance {
                    source_english: source.to_owned(),
                    chinese: chinese.to_owned(),
                })
            })
            .take(MAX_CONTEXT_UTTERANCES)
            .collect::<Vec<_>>();
        context.reverse();
        context
    }

    pub fn following_source(
        &self,
        chapter_id: &str,
        position: (u32, u32),
        limit: usize,
    ) -> Vec<String> {
        self.chapters
            .get(chapter_id)
            .map(|chapter| {
                chapter
                    .range((
                        std::ops::Bound::Excluded(position),
                        std::ops::Bound::Unbounded,
                    ))
                    .map(|(_, unit)| unit.source_text.trim())
                    .filter(|text| !text.is_empty())
                    .take(limit)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn snapshot_excluding_source(
        &self,
        chapter_id: &str,
        source_index: u32,
        source_order: &[u32],
    ) -> Vec<ChapterContextUnit> {
        let Some(chapter) = self.chapters.get(chapter_id) else {
            return Vec::new();
        };
        let Some(position) = source_order.iter().position(|index| *index == source_index) else {
            return Vec::new();
        };
        let mut preceding = source_order[..position]
            .iter()
            .rev()
            .flat_map(|index| {
                chapter
                    .range((*index, 0)..=(*index, u32::MAX))
                    .rev()
                    .map(|(_, unit)| unit.clone())
            })
            .take(MAX_CONTEXT_UTTERANCES)
            .collect::<Vec<_>>();
        preceding.reverse();
        preceding.extend(
            source_order[position + 1..]
                .iter()
                .flat_map(|index| {
                    chapter
                        .range((*index, 0)..=(*index, u32::MAX))
                        .map(|(_, unit)| unit.clone())
                })
                .take(MAX_CONTEXT_UTTERANCES),
        );
        preceding
    }

    pub fn remove_source(&mut self, chapter_id: &str, source_index: u32) {
        if let Some(chapter) = self.chapters.get_mut(chapter_id) {
            let positions = chapter
                .range((source_index, 0)..=(source_index, u32::MAX))
                .map(|(position, _)| *position)
                .collect::<Vec<_>>();
            for position in positions {
                chapter.remove(&position);
            }
        }
    }

    pub fn remove(&mut self, chapter_id: &str) {
        self.chapters.remove(chapter_id);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageSurfaceKind {
    Image,
    ContinuousStrip,
    Frame,
    Canvas,
    WebGl,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSurface {
    pub session_id: String,
    pub page_index: u32,
    pub source_sha256: String,
    pub width: u32,
    pub height: u32,
    pub kind: PageSurfaceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionRole {
    Dialogue,
    Narration,
    System,
    SoundEffect,
    TechniqueArtwork,
    Exclusion,
    Unreadable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionPlan {
    pub id: String,
    pub reading_order: u32,
    pub role: RegionRole,
    pub source_english: String,
    pub continuation_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageAnalysis {
    pub surface: PageSurface,
    pub regions: Vec<RegionPlan>,
    pub complete: bool,
}

#[derive(Debug, Default, Clone)]
pub struct ChapterSession {
    pub id: String,
    pub surfaces: BTreeMap<u32, PageSurface>,
    pub analyses: BTreeMap<u32, PageAnalysis>,
}

impl ChapterSession {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    pub fn register_surface(&mut self, surface: PageSurface) -> bool {
        let changed = self
            .surfaces
            .get(&surface.page_index)
            .is_some_and(|previous| previous.source_sha256 != surface.source_sha256);
        if changed {
            self.analyses.remove(&surface.page_index);
        }
        self.surfaces.insert(surface.page_index, surface);
        changed
    }

    pub fn record_analysis(&mut self, analysis: PageAnalysis) {
        let PageAnalysis {
            surface,
            regions: incoming_regions,
            complete,
        } = analysis;
        let page_index = surface.page_index;
        self.register_surface(surface.clone());
        let entry = self
            .analyses
            .entry(page_index)
            .or_insert_with(|| PageAnalysis {
                surface: surface.clone(),
                regions: Vec::new(),
                complete: false,
            });
        entry.surface = surface;
        let mut regions = std::mem::take(&mut entry.regions);
        for region in incoming_regions {
            if let Some(existing) = regions.iter_mut().find(|existing| existing.id == region.id) {
                *existing = region;
            } else {
                regions.push(region);
            }
        }
        regions.sort_by(|left, right| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        entry.regions = regions;
        entry.complete |= complete;
    }
}

#[derive(Debug, Default)]
pub struct ChapterSessionStore {
    sessions: HashMap<String, ChapterSession>,
}

impl ChapterSessionStore {
    pub fn session_mut(&mut self, id: &str) -> &mut ChapterSession {
        self.sessions
            .entry(id.to_owned())
            .or_insert_with(|| ChapterSession::new(id))
    }

    pub fn session(&self, id: &str) -> Option<&ChapterSession> {
        self.sessions.get(id)
    }

    pub fn remove(&mut self, id: &str) {
        self.sessions.remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(index: u32, order: u32, id: &str, source: &str) -> ChapterContextUnit {
        ChapterContextUnit {
            item_id: id.to_owned(),
            source_index: index,
            item_order: order,
            source_text: source.to_owned(),
            displayed_text: None,
        }
    }

    #[test]
    fn context_is_ordered_by_source_position_not_completion_time() {
        let mut store = ChapterContextStore::default();
        store.register(
            "chapter",
            [
                unit(2, 0, "later", "later source"),
                unit(1, 0, "earlier", "earlier source"),
            ],
        );
        store.publish("chapter", (2, 0), "later", "later Chinese".to_owned());
        store.publish("chapter", (1, 0), "earlier", "earlier Chinese".to_owned());

        let context = store.preceding("chapter", (3, 0));
        assert_eq!(
            context
                .iter()
                .map(|entry| entry.source_english.as_str())
                .collect::<Vec<_>>(),
            ["earlier source", "later source"]
        );
    }

    #[test]
    fn context_registers_all_source_before_visible_first_publication() {
        let mut store = ChapterContextStore::default();
        store.register(
            "chapter",
            [
                unit(0, 0, "first", "first source"),
                unit(0, 1, "visible", "visible source"),
                unit(0, 2, "following", "following source"),
            ],
        );
        store.publish("chapter", (0, 1), "visible", "visible Chinese".to_owned());

        assert!(store.preceding("chapter", (0, 1)).is_empty());
        assert_eq!(
            store.following_source("chapter", (0, 1), 6),
            vec!["following source".to_owned()]
        );
    }

    #[test]
    fn late_insertions_use_canonical_dom_order_instead_of_admission_ids() {
        let mut store = ChapterContextStore::default();
        store.register(
            "chapter",
            [
                unit(0, 0, "first", "first"),
                unit(1, 0, "last", "last"),
                unit(2, 0, "inserted", "inserted"),
            ],
        );
        assert_eq!(
            store
                .snapshot_excluding_source("chapter", 0, &[2, 0, 1])
                .iter()
                .map(|unit| unit.item_id.as_str())
                .collect::<Vec<_>>(),
            ["inserted", "last"]
        );
    }

    #[test]
    fn changing_a_surface_discards_old_region_tails_and_context_only_for_that_source() {
        let mut session = ChapterSession::new("chapter");
        let surface = PageSurface {
            session_id: "chapter".into(),
            page_index: 0,
            source_sha256: "old".into(),
            width: 100,
            height: 100,
            kind: PageSurfaceKind::Image,
        };
        session.record_analysis(PageAnalysis {
            surface: surface.clone(),
            regions: vec![RegionPlan {
                id: "old-tail".into(),
                reading_order: 1,
                role: RegionRole::Dialogue,
                source_english: "old source".into(),
                continuation_group: None,
            }],
            complete: true,
        });
        let mut changed = surface;
        changed.source_sha256 = "new".into();
        assert!(session.register_surface(changed.clone()));
        assert!(!session.analyses.contains_key(&0));
        assert!(!session.register_surface(changed));
        let mut context = ChapterContextStore::default();
        context.register(
            "chapter",
            [
                unit(0, 0, "old", "old"),
                unit(0, 1, "tail", "tail"),
                unit(1, 0, "other", "other"),
            ],
        );
        context.remove_source("chapter", 0);
        assert_eq!(context.following_source("chapter", (0, 0), 6), ["other"]);
    }

    #[test]
    fn page_analysis_merges_incremental_regions_without_a_chapter_barrier() {
        let mut session = ChapterSession::new("chapter");
        session.record_analysis(PageAnalysis {
            surface: PageSurface {
                session_id: "chapter".to_owned(),
                page_index: 0,
                source_sha256: "a".to_owned(),
                width: 100,
                height: 100,
                kind: PageSurfaceKind::Image,
            },
            regions: vec![RegionPlan {
                id: "region".to_owned(),
                reading_order: 0,
                role: RegionRole::Dialogue,
                source_english: "Hello".to_owned(),
                continuation_group: None,
            }],
            complete: false,
        });
        assert!(!session.analyses[&0].complete);
        session.record_analysis(PageAnalysis {
            surface: session.analyses[&0].surface.clone(),
            regions: session.analyses[&0].regions.clone(),
            complete: true,
        });
        assert!(session.analyses[&0].complete);
        assert_eq!(session.analyses[&0].regions.len(), 1);
    }
}
