use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
struct TrieNode {
    children: BTreeMap<char, usize>,
    terminal: bool,
}

/// Deterministic trie used for allowed-word decomposition and longest match.
#[derive(Debug, Clone)]
pub struct AllowedWordTrie {
    nodes: Vec<TrieNode>,
}

impl AllowedWordTrie {
    pub fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    pub fn insert(&mut self, word: &str) {
        if word.is_empty() {
            return;
        }
        let mut node_index = 0;
        for character in word.chars() {
            let next = if let Some(next) = self.nodes[node_index].children.get(&character) {
                *next
            } else {
                let next = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[node_index].children.insert(character, next);
                next
            };
            node_index = next;
        }
        self.nodes[node_index].terminal = true;
    }

    pub fn contains(&self, word: &str) -> bool {
        let mut node_index = 0;
        for character in word.chars() {
            let Some(next) = self.nodes[node_index].children.get(&character) else {
                return false;
            };
            node_index = *next;
        }
        self.nodes[node_index].terminal
    }

    pub(crate) fn matches_from(&self, characters: &[char], start: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut node_index = 0;
        for (offset, character) in characters[start..].iter().enumerate() {
            let Some(next) = self.nodes[node_index].children.get(character) else {
                break;
            };
            node_index = *next;
            if self.nodes[node_index].terminal {
                result.push(start + offset + 1);
            }
        }
        result
    }

    pub(crate) fn longest_match(&self, characters: &[char], start: usize) -> Option<usize> {
        self.matches_from(characters, start).into_iter().max()
    }

    /// Returns a complete word break, preferring fewer/longer components and a
    /// lexical tie-break for stable cache behavior.
    pub fn best_decomposition(&self, text: &str) -> Option<Vec<String>> {
        let characters = text.chars().collect::<Vec<_>>();
        if characters.is_empty() {
            return None;
        }

        let mut paths: Vec<Option<Vec<String>>> = vec![None; characters.len() + 1];
        paths[0] = Some(Vec::new());

        for start in 0..characters.len() {
            let Some(prefix) = paths[start].clone() else {
                continue;
            };
            for end in self.matches_from(&characters, start) {
                let mut candidate = prefix.clone();
                candidate.push(characters[start..end].iter().collect());
                let replace = paths[end].as_ref().is_none_or(|current| {
                    candidate.len() < current.len()
                        || (candidate.len() == current.len() && candidate < *current)
                });
                if replace {
                    paths[end] = Some(candidate);
                }
            }
        }

        paths.pop().flatten()
    }

    pub fn can_decompose(&self, text: &str) -> bool {
        self.best_decomposition(text).is_some()
    }
}

impl Default for AllowedWordTrie {
    fn default() -> Self {
        Self::new()
    }
}
