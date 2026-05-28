use crate::field::soa::AgentField;

/// Prepares an interleaved buffer of floats for zero-copy WebGL/WebGPU instanced rendering.
/// Layout per active agent (4 floats = 16 bytes):
/// [x, y, vx, vy]
///
/// This allows JS to simply create a Float32Array view over this memory
/// and bind it directly to an instanced WebGL array buffer without JS iteration.
pub struct GpuBuffer {
    data: Vec<f32>,
}

impl GpuBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity * 4),
        }
    }

    /// Updates the buffer with active agents' data.
    pub fn update(&mut self, field: &AgentField) {
        self.data.clear();
        for i in 0..field.len {
            if field.active[i] == 0 {
                continue;
            }
            self.data.push(field.pos_x[i]);
            self.data.push(field.pos_y[i]);
            self.data.push(field.vel_x[i]);
            self.data.push(field.vel_y[i]);
        }
    }

    pub fn ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[allow(dead_code)]
    pub fn instance_count(&self) -> usize {
        self.data.len() / 4
    }
}
