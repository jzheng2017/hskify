#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PixelRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl PixelRect {
    pub(super) fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Self> {
        let rect = Self { x0, y0, x1, y1 };
        (rect.x0.is_finite()
            && rect.y0.is_finite()
            && rect.x1.is_finite()
            && rect.y1.is_finite()
            && rect.x1 > rect.x0
            && rect.y1 > rect.y0)
            .then_some(rect)
    }

    pub(super) fn width(self) -> f32 {
        self.x1 - self.x0
    }

    pub(super) fn height(self) -> f32 {
        self.y1 - self.y0
    }

    pub(super) fn pixel_bounds(self, image_width: u32, image_height: u32) -> PixelBounds {
        let x = self.x0.floor().max(0.0) as u32;
        let y = self.y0.floor().max(0.0) as u32;
        let x1 = self.x1.ceil().min(image_width as f32) as u32;
        let y1 = self.y1.ceil().min(image_height as f32) as u32;
        PixelBounds {
            x,
            y,
            width: x1.saturating_sub(x).max(1),
            height: y1.saturating_sub(y).max(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PixelBounds {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
