//! Chapter-owned state shared by page analyses.
//!
//! A browser job is an execution unit, not a document.  This module keeps
//! the immutable page surfaces, ordered region plans, dialogue links, and
//! entity memory in one chapter session so completion order cannot change the
//! meaning of a later page.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use koharu_app::llm::HskPrecedingUtterance;

pub const MAX_CONTEXT_UTTERANCES: usize = 6;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogueNode {
    pub page_index: u32,
    pub reading_order: u32,
    pub region_id: String,
    pub source_english: String,
    pub chinese: String,
    pub continuation_group: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DialogueGraph {
    pages: BTreeMap<u32, Vec<DialogueNode>>,
}

impl DialogueGraph {
    pub fn record_page(&mut self, page_index: u32, mut nodes: Vec<DialogueNode>) {
        // A page can be translated in several bounded language batches. Merge
        // terminal nodes by region identity instead of replacing an earlier
        // batch; publication order must never erase context that was already
        // accepted for the same page.
        let mut merged = self.pages.remove(&page_index).unwrap_or_default();
        for node in nodes.drain(..) {
            if let Some(existing) = merged
                .iter_mut()
                .find(|existing| existing.region_id == node.region_id)
            {
                *existing = node;
            } else {
                merged.push(node);
            }
        }
        nodes = merged;
        nodes.sort_by(|left, right| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.region_id.cmp(&right.region_id))
        });
        self.pages.insert(page_index, nodes);
        self.normalize_continuation_groups();
    }

    pub fn before(&self, page_index: u32) -> Vec<HskPrecedingUtterance> {
        let nodes = self
            .pages
            .range(..page_index)
            .flat_map(|(_, nodes)| nodes.iter())
            .filter(|node| {
                !node.source_english.trim().is_empty() && !node.chinese.trim().is_empty()
            })
            .collect::<Vec<_>>();
        self.context_from_nodes(nodes)
    }

    /// Return accepted dialogue strictly before a page/reading position.  A
    /// page can be translated in several language batches, so same-page
    /// terminal nodes are included when they precede the next batch.  The
    /// graph is already canonicalized by page and reading order; completion
    /// timing cannot reorder this context.
    pub fn before_position(
        &self,
        page_index: u32,
        reading_order: u32,
    ) -> Vec<HskPrecedingUtterance> {
        let nodes = self
            .pages
            .iter()
            .flat_map(|(page, nodes)| {
                nodes.iter().filter(move |node| {
                    *page < page_index
                        || (*page == page_index && node.reading_order < reading_order)
                })
            })
            .filter(|node| {
                !node.source_english.trim().is_empty() && !node.chinese.trim().is_empty()
            })
            .collect::<Vec<_>>();
        self.context_from_nodes(nodes)
    }

    fn context_from_nodes(&self, nodes: Vec<&DialogueNode>) -> Vec<HskPrecedingUtterance> {
        let mut utterances: Vec<HskPrecedingUtterance> = Vec::with_capacity(nodes.len());
        let mut last_group: Option<String> = None;
        for node in nodes {
            if let Some(group) = node.continuation_group.as_deref()
                && last_group.as_deref() == Some(group)
                && let Some(previous) = utterances.last_mut()
            {
                previous.source_english.push('\n');
                previous.source_english.push_str(node.source_english.trim());
                previous.chinese.push('\n');
                previous.chinese.push_str(node.chinese.trim());
                continue;
            }
            last_group = node.continuation_group.clone();
            utterances.push(HskPrecedingUtterance {
                source_english: node.source_english.trim().to_owned(),
                chinese: node.chinese.trim().to_owned(),
            });
        }
        if utterances.len() > MAX_CONTEXT_UTTERANCES {
            utterances.drain(..utterances.len() - MAX_CONTEXT_UTTERANCES);
        }
        utterances
    }

    /// Model continuation links are parent references (`child -> parent`),
    /// while translation context needs one stable group key for every member.
    /// Resolve those links after each terminal insertion so completion order
    /// cannot leave a child detached from its earlier bubble.
    fn normalize_continuation_groups(&mut self) {
        let links = self
            .pages
            .values()
            .flat_map(|nodes| nodes.iter())
            .filter_map(|node| {
                node.continuation_group
                    .as_ref()
                    .map(|parent| (node.region_id.clone(), parent.clone()))
            })
            .collect::<HashMap<_, _>>();
        let referenced = links.values().cloned().collect::<BTreeSet<_>>();
        for nodes in self.pages.values_mut() {
            for node in nodes {
                let root = if links.contains_key(&node.region_id) {
                    Some(resolve_continuation_root(&links, &node.region_id))
                } else if referenced.contains(&node.region_id) {
                    Some(resolve_continuation_root(&links, &node.region_id))
                } else {
                    None
                };
                node.continuation_group = root;
            }
        }
    }

    pub fn pages(&self) -> &BTreeMap<u32, Vec<DialogueNode>> {
        &self.pages
    }
}

fn resolve_continuation_root(links: &HashMap<String, String>, id: &str) -> String {
    let mut current = id.to_owned();
    let mut seen = BTreeSet::new();
    while let Some(parent) = links.get(&current) {
        if !seen.insert(current.clone()) {
            break;
        }
        current = parent.clone();
    }
    current
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChapterEntityType {
    Person,
    Place,
    Organization,
    CoinedEntity,
    Relationship,
    Occupation,
    Rank,
    Title,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterEntity {
    pub source_english: String,
    pub entity_type: ChapterEntityType,
    pub chinese: Option<String>,
    pub first_page: u32,
    pub first_reading_order: u32,
    pub pages: BTreeSet<u32>,
}

#[derive(Debug, Default, Clone)]
pub struct ChapterSession {
    pub id: String,
    pub surfaces: BTreeMap<u32, PageSurface>,
    /// Canonical page indexes admitted by the browser for this chapter run.
    /// This barrier is registered with every job request, before any model
    /// work starts, so concurrent jobs cannot make context depend on which
    /// upload happened to reach the daemon first.
    pub expected_pages: BTreeSet<u32>,
    /// Pages whose detector/OCR/page-understanding frontier is available.
    /// This is intentionally separate from `analyses`: a page can expose a
    /// partial analysis while its ordered language stream is still waiting on
    /// cleanup or translation.
    pub analysis_ready_pages: BTreeSet<u32>,
    pub analyses: BTreeMap<u32, PageAnalysis>,
    /// A language stream is committed in document order. Future pages may
    /// analyze ahead, but they must not translate until every earlier page has
    /// reached a terminal language state (success or explicit failure).
    pub language_complete_pages: BTreeSet<u32>,
    pub dialogue: DialogueGraph,
    pub entities: HashMap<String, ChapterEntity>,
}

impl ChapterSession {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    pub fn register_surface(&mut self, surface: PageSurface) {
        self.surfaces.insert(surface.page_index, surface);
    }

    pub fn register_expected_pages(&mut self, page_indexes: &[u32]) {
        self.expected_pages.extend(page_indexes.iter().copied());
    }

    pub fn record_analysis(&mut self, analysis: PageAnalysis) {
        let PageAnalysis {
            surface,
            regions: incoming_regions,
            complete,
        } = analysis;
        let page_index = surface.page_index;
        self.register_surface(surface.clone());
        self.analysis_ready_pages.insert(page_index);
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

    pub fn mark_language_complete(&mut self, page_index: u32) {
        self.language_complete_pages.insert(page_index);
    }

    pub fn analysis_ready(&self, page_index: u32) -> bool {
        self.analysis_ready_pages.contains(&page_index)
    }

    /// A partial detector/OCR frontier is useful for the page that is
    /// currently visible, but it is not a safe context boundary for a later
    /// page.  Ordered translation waits for the immutable page analysis to
    /// be complete so no continuation/entity decision can be omitted merely
    /// because one concurrent job finished its first viewport batch.
    pub fn analysis_complete(&self, page_index: u32) -> bool {
        self.analyses
            .get(&page_index)
            .is_some_and(|analysis| analysis.complete)
    }

    pub fn language_complete(&self, page_index: u32) -> bool {
        self.language_complete_pages.contains(&page_index)
    }

    pub fn record_dialogue(&mut self, page_index: u32, nodes: Vec<DialogueNode>) {
        self.dialogue.record_page(page_index, nodes);
    }

    pub fn remember_entity(&mut self, mut entity: ChapterEntity) {
        let key = entity.source_english.to_ascii_lowercase();
        if let Some(existing) = self.entities.get_mut(&key) {
            existing.pages.extend(entity.pages);
            let incoming_position = (entity.first_page, entity.first_reading_order);
            let existing_position = (existing.first_page, existing.first_reading_order);
            let incoming_is_earlier = incoming_position < existing_position;
            if incoming_is_earlier {
                existing.first_page = entity.first_page;
                existing.first_reading_order = entity.first_reading_order;
            }
            if existing.chinese.is_none() || incoming_is_earlier {
                existing.chinese = entity.chinese.take();
            }
            if existing.entity_type == ChapterEntityType::Unknown {
                existing.entity_type = entity.entity_type;
            }
        } else {
            self.entities.insert(key, entity);
        }
    }

    /// Return only entity decisions that are earlier than a translation
    /// window. A page job may finish ahead of an earlier page, so exposing the
    /// whole hash map would leak future names into the current prompt.
    pub fn entities_before_position(
        &self,
        page_index: u32,
        reading_order: u32,
    ) -> impl Iterator<Item = &ChapterEntity> {
        self.entities.values().filter(move |entity| {
            (entity.first_page, entity.first_reading_order) < (page_index, reading_order)
        })
    }

    /// Return source-language regions that follow a translation window in the
    /// immutable chapter analysis. These are intentionally untranslated source
    /// lines: they give the language model a bounded look-ahead for pronouns,
    /// sentence continuations, and connected bubbles without exposing future
    /// Chinese/entity decisions.
    pub fn following_source(
        &self,
        page_index: u32,
        reading_order: u32,
        limit: usize,
    ) -> Vec<String> {
        let mut values = Vec::new();
        for (page, analysis) in &self.analyses {
            for region in &analysis.regions {
                if *page < page_index
                    || (*page == page_index && region.reading_order <= reading_order)
                {
                    continue;
                }
                if matches!(
                    region.role,
                    RegionRole::Exclusion | RegionRole::TechniqueArtwork | RegionRole::Unreadable
                ) {
                    continue;
                }
                let source = region.source_english.trim();
                if source.is_empty() || values.iter().any(|value| value == source) {
                    continue;
                }
                values.push(source.to_owned());
                if values.len() >= limit {
                    return values;
                }
            }
        }
        values
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

    pub fn before(&self, id: &str, page_index: u32) -> Vec<HskPrecedingUtterance> {
        self.session(id)
            .map(|session| session.dialogue.before(page_index))
            .unwrap_or_default()
    }

    pub fn before_position(
        &self,
        id: &str,
        page_index: u32,
        reading_order: u32,
    ) -> Vec<HskPrecedingUtterance> {
        self.session(id)
            .map(|session| session.dialogue.before_position(page_index, reading_order))
            .unwrap_or_default()
    }

    pub fn remove(&mut self, id: &str) {
        self.sessions.remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(page_index: u32, order: u32, id: &str, source: &str, chinese: &str) -> DialogueNode {
        DialogueNode {
            page_index,
            reading_order: order,
            region_id: id.to_owned(),
            source_english: source.to_owned(),
            chinese: chinese.to_owned(),
            continuation_group: None,
        }
    }

    #[test]
    fn dialogue_context_is_document_ordered_not_completion_ordered() {
        let mut graph = DialogueGraph::default();
        graph.record_page(2, vec![node(2, 0, "p2", "later", "后面")]);
        graph.record_page(1, vec![node(1, 0, "p1", "earlier", "前面")]);
        let context = graph.before(3);
        assert_eq!(
            context
                .iter()
                .map(|entry| entry.source_english.as_str())
                .collect::<Vec<_>>(),
            ["earlier", "later"]
        );
    }

    #[test]
    fn entity_memory_merges_occurrences_without_changing_type() {
        let mut session = ChapterSession::new("chapter");
        session.remember_entity(ChapterEntity {
            source_english: "Wife".to_owned(),
            entity_type: ChapterEntityType::Relationship,
            chinese: Some("妻子".to_owned()),
            first_page: 4,
            first_reading_order: 0,
            pages: [4].into_iter().collect(),
        });
        session.remember_entity(ChapterEntity {
            source_english: "Wife".to_owned(),
            entity_type: ChapterEntityType::Unknown,
            chinese: None,
            first_page: 2,
            first_reading_order: 0,
            pages: [2].into_iter().collect(),
        });
        let entity = session.entities.get("wife").unwrap();
        assert_eq!(entity.first_page, 2);
        assert_eq!(entity.pages, [2, 4].into_iter().collect());
        assert_eq!(entity.entity_type, ChapterEntityType::Relationship);
    }

    #[test]
    fn entity_context_never_exposes_a_future_document_position() {
        let mut session = ChapterSession::new("chapter");
        for (source, page, order) in [("Earlier", 1, 0), ("Later", 3, 0)] {
            session.remember_entity(ChapterEntity {
                source_english: source.to_owned(),
                entity_type: ChapterEntityType::Person,
                chinese: Some(source.to_owned()),
                first_page: page,
                first_reading_order: order,
                pages: [page].into_iter().collect(),
            });
        }
        let visible = session
            .entities_before_position(2, 0)
            .map(|entity| entity.source_english.as_str())
            .collect::<Vec<_>>();
        assert_eq!(visible, vec!["Earlier"]);
    }

    #[test]
    fn expected_page_barrier_is_registered_before_concurrent_analysis() {
        let mut session = ChapterSession::new("chapter");
        session.register_expected_pages(&[0, 1, 4]);
        session.register_expected_pages(&[2, 3]);
        assert_eq!(
            session.expected_pages,
            [0, 1, 2, 3, 4].into_iter().collect()
        );
    }

    #[test]
    fn analysis_frontier_and_language_frontier_are_independent() {
        let mut session = ChapterSession::new("chapter");
        session.register_expected_pages(&[0, 1]);
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
        assert!(session.analysis_ready(0));
        assert!(!session.analysis_complete(0));
        assert!(!session.language_complete(0));
        session.record_analysis(PageAnalysis {
            surface: session.analyses.get(&0).unwrap().surface.clone(),
            regions: session.analyses.get(&0).unwrap().regions.clone(),
            complete: true,
        });
        assert!(session.analysis_complete(0));
        session.mark_language_complete(0);
        assert!(session.language_complete(0));
    }

    #[test]
    fn continuation_links_are_canonicalized_and_context_is_joined() {
        let mut graph = DialogueGraph::default();
        graph.record_page(
            1,
            vec![
                node(1, 0, "first", "Wait", "等一下"),
                DialogueNode {
                    page_index: 1,
                    reading_order: 1,
                    region_id: "second".to_owned(),
                    source_english: "for me.".to_owned(),
                    chinese: "等我。".to_owned(),
                    continuation_group: Some("first".to_owned()),
                },
            ],
        );
        let context = graph.before(2);
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].source_english, "Wait\nfor me.");
        assert_eq!(context[0].chinese, "等一下\n等我。");
        assert_eq!(
            graph.pages()[&1]
                .iter()
                .map(|node| node.continuation_group.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("first"), Some("first")]
        );
    }

    #[test]
    fn following_source_is_canonical_and_excludes_non_language_regions() {
        let mut session = ChapterSession::new("chapter");
        session.record_analysis(PageAnalysis {
            surface: PageSurface {
                session_id: "chapter".to_owned(),
                page_index: 2,
                source_sha256: "c".to_owned(),
                width: 100,
                height: 100,
                kind: PageSurfaceKind::Image,
            },
            regions: vec![
                RegionPlan {
                    id: "later".to_owned(),
                    reading_order: 0,
                    role: RegionRole::Dialogue,
                    source_english: "Later dialogue".to_owned(),
                    continuation_group: None,
                },
                RegionPlan {
                    id: "art".to_owned(),
                    reading_order: 1,
                    role: RegionRole::TechniqueArtwork,
                    source_english: "SWORD TECHNIQUE".to_owned(),
                    continuation_group: None,
                },
            ],
            complete: true,
        });
        assert_eq!(
            session.following_source(1, 0, 4),
            vec!["Later dialogue".to_owned()]
        );
    }
}
