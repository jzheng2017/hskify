use std::collections::HashMap;
use std::sync::Arc;

use image::{DynamicImage, GenericImageView};

pub(crate) const DECODED_IMAGE_CACHE_MAX_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug)]
struct Entry {
    image: Arc<DynamicImage>,
    bytes: u64,
    last_used: u64,
}

#[derive(Debug)]
pub(crate) struct DecodedImageCache {
    entries: HashMap<String, Entry>,
    retained_bytes: u64,
    max_bytes: u64,
    clock: u64,
}

impl Default for DecodedImageCache {
    fn default() -> Self {
        Self::new(DECODED_IMAGE_CACHE_MAX_BYTES)
    }
}

impl DecodedImageCache {
    pub(crate) fn new(max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            retained_bytes: 0,
            max_bytes,
            clock: 0,
        }
    }

    pub(crate) fn get(&mut self, source_sha256: &str) -> Option<Arc<DynamicImage>> {
        self.clock = self.clock.saturating_add(1);
        let entry = self.entries.get_mut(source_sha256)?;
        entry.last_used = self.clock;
        Some(Arc::clone(&entry.image))
    }

    pub(crate) fn insert(
        &mut self,
        source_sha256: String,
        image: Arc<DynamicImage>,
    ) -> Arc<DynamicImage> {
        let bytes = rgb_bytes(image.as_ref());
        if bytes > self.max_bytes {
            return image;
        }
        self.clock = self.clock.saturating_add(1);
        if let Some(replaced) = self.entries.remove(&source_sha256) {
            self.retained_bytes = self.retained_bytes.saturating_sub(replaced.bytes);
        }
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        self.entries.insert(
            source_sha256,
            Entry {
                image: Arc::clone(&image),
                bytes,
                last_used: self.clock,
            },
        );
        self.evict_to_limit();
        image
    }

    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    fn evict_to_limit(&mut self) {
        while self.retained_bytes > self.max_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.retained_bytes = self.retained_bytes.saturating_sub(removed.bytes);
            }
        }
    }
}

fn rgb_bytes(image: &DynamicImage) -> u64 {
    let (width, height) = image.dimensions();
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn image(width: u32, height: u32) -> Arc<DynamicImage> {
        Arc::new(DynamicImage::ImageRgb8(RgbImage::new(width, height)))
    }

    #[test]
    fn least_recently_used_images_are_evicted_by_bytes() {
        let mut cache = DecodedImageCache::new(18);
        cache.insert("a".to_owned(), image(2, 2));
        cache.insert("b".to_owned(), image(1, 2));
        assert!(cache.get("a").is_some());
        cache.insert("c".to_owned(), image(1, 2));

        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
        assert!(cache.retained_bytes() <= 18);
    }

    #[test]
    fn oversized_images_are_returned_without_retention() {
        let mut cache = DecodedImageCache::new(3);
        let original = image(2, 2);
        let returned = cache.insert("too-large".to_owned(), Arc::clone(&original));

        assert!(Arc::ptr_eq(&original, &returned));
        assert!(cache.get("too-large").is_none());
        assert_eq!(cache.retained_bytes(), 0);
    }
}
