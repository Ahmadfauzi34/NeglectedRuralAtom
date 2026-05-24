use wasm_bindgen::prelude::*;

pub const DRAW_VERSION: u32 = 1;

// Tags — TS tidak hardcode makna, hanya switch pada angka
pub const TAG_CIRCLE: u32 = 0;
pub const TAG_LINE: u32 = 1;
pub const TAG_POLY: u32 = 2;
pub const TAG_TEXT: u32 = 3;
pub const TAG_RECT: u32 = 4;

#[repr(C)]
pub struct DrawHeader {
    pub version: u32,
    pub cmd_count: u32,
    pub payload_used: u32,
}

#[repr(C)]
pub struct DrawCmd {
    pub tag: u32,        // 0=circle, 1=line, dll
    pub color: u32,      // 0xRRGGBBAA
    pub x: f32,
    pub y: f32,
    pub z: f32,          // semantic tergantung tag (radius, x2, width)
    pub w: f32,          // semantic tergantung tag (height, y2)
    pub payload_idx: u32,// offset ke payload buffer (untuk variable data seperti polygon points)
    pub payload_len: u32,// jumlah item di payload
}

/// Ring buffer encoder. Tidak alloc di hot path karena pakai Vec dengan capacity.
pub struct CanvasEncoder {
    cmds: Vec<<DrawCmd>,
    payload: Vec<u8>,
    flat: Vec<u8>, // reusable output buffer
}

impl CanvasEncoder {
    pub fn new(capacity: usize) -> Self {
        Self {
            cmds: Vec::with_capacity(capacity),
            payload: Vec::with_capacity(capacity * 32),
            flat: Vec::with_capacity(capacity * 40 + 12),
        }
    }
    
    pub fn clear(&mut self) {
        self.cmds.clear();
        self.payload.clear();
    }
    
    pub fn circle(&mut self, x: f32, y: f32, r: f32, color: u32) {
        self.cmds.push(DrawCmd {
            tag: TAG_CIRCLE, color, x, y, z: r, w: 0.0,
            payload_idx: 0, payload_len: 0,
        });
    }
    
    pub fn line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, color: u32) {
        self.cmds.push(DrawCmd {
            tag: TAG_LINE, color,
            x: x1, y: y1, z: x2, w: y2,
            payload_idx: 0, payload_len: 0,
        });
    }
    
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: u32) {
        self.cmds.push(DrawCmd {
            tag: TAG_RECT, color,
            x, y, z: w, w: h,
            payload_idx: 0, payload_len: 0,
        });
    }
    
    pub fn polygon(&mut self, points: &[(f32, f32)], color: u32) {
        let start = self.payload.len();
        for &(x, y) in points {
            self.payload.extend_from_slice(&x.to_le_bytes());
            self.payload.extend_from_slice(&y.to_le_bytes());
        }
        self.cmds.push(DrawCmd {
            tag: TAG_POLY, color,
            x: 0.0, y: 0.0, z: 0.0, w: 0.0,
            payload_idx: start as u32,
            payload_len: points.len() as u32,
        });
    }
    
    /// Flatten semua command ke continuous byte buffer. Return (ptr, len).
    /// Buffer valid sampai encode() berikutnya dipanggil.
    pub fn encode(&mut self) -> (*const u8, usize) {
        self.flat.clear();
        
        let header = DrawHeader {
            version: DRAW_VERSION,
            cmd_count: self.cmds.len() as u32,
            payload_used: self.payload.len() as u32,
        };
        
        unsafe {
            let h = std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                std::mem::size_of::<DrawHeader>()
            );
            self.flat.extend_from_slice(h);
            
            if !self.cmds.is_empty() {
                let c = std::slice::from_raw_parts(
                    self.cmds.as_ptr() as *const u8,
                    self.cmds.len() * std::mem::size_of::<DrawCmd>()
                );
                self.flat.extend_from_slice(c);
            }
        }
        
        self.flat.extend_from_slice(&self.payload);
        (self.flat.as_ptr(), self.flat.len())
    }
}

