use wasm_bindgen::prelude::*;
use crate::field::{AgentField, KernelConfig, step_agents};

/// Bridge antara JS host dan Rust kernel
/// Tidak melakukan alloc di hot path — hanya pointer exchange
#[wasm_bindgen]
pub struct KernelBridge {
    field: AgentField,
    config: KernelConfig,
    scratch_acc_x: Vec<f32>, // pre-allocated scratch untuk kernel
    scratch_acc_y: Vec<f32>,
}

#[wasm_bindgen]
impl KernelBridge {
    #[wasm_bindgen(constructor)]
    pub fn new(max_agents: usize) -> Self {
        let mut field = AgentField::new(max_agents);
        field.reserve(max_agents);
        
        Self {
            field,
            config: KernelConfig::default(),
            scratch_acc_x: Vec::with_capacity(max_agents),
            scratch_acc_y: Vec::with_capacity(max_agents),
        }
    }
    
    /// Spawn agent — return index
    pub fn spawn(&mut self, x: f32, y: f32, health: f32) -> usize {
        self.field.spawn(x, y, health)
    }
    
    /// Kill agent via swap-drop (O(1))
    pub fn kill(&mut self, idx: usize) {
        self.field.kill_swap(idx);
    }
    
    /// Set config value
    pub fn set_config(&mut self, dt: f32, friction: f32, max_speed: f32, influence_radius: f32) {
        self.config = KernelConfig {
            dt,
            friction,
            max_speed,
            influence_radius,
            ..self.config
        };
    }
    
    /// Step simulation — satu tick
    pub fn step(&mut self) {
        step_agents(&mut self.field, &self.config);
    }
    
    // === Zero-copy memory views untuk JS ===
    
    #[wasm_bindgen(getter)]
    pub fn agent_count(&self) -> usize {
        self.field.agent_count()
    }
    
    #[wasm_bindgen(getter)]
    pub fn pos_x_ptr(&self) -> *const f32 {
        self.field.pos_x_ptr()
    }
    
    #[wasm_bindgen(getter)]
    pub fn pos_y_ptr(&self) -> *const f32 {
        self.field.pos_y_ptr()
    }
    
    #[wasm_bindgen(getter)]
    pub fn active_ptr(&self) -> *const u8 {
        self.field.active_ptr()
    }
    
    /// Serialize state untuk snapshot (jarang dipanggil, boleh alloc)
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.field.len * 4 * 4); // rough estimate
        for i in 0..self.field.len {
            buf.extend_from_slice(&self.field.pos_x[i].to_le_bytes());
            buf.extend_from_slice(&self.field.pos_y[i].to_le_bytes());
            buf.extend_from_slice(&self.field.vel_x[i].to_le_bytes());
            buf.extend_from_slice(&self.field.vel_y[i].to_le_bytes());
        }
        buf
    }
}

/// MemoryView untuk JS direct access
#[wasm_bindgen]
pub struct MemoryView;

#[wasm_bindgen]
impl MemoryView {
    /// Helper: buat Float32Array dari pointer + length
    /// Dipanggil dari JS: memoryView.float32Array(ptr, count)
    pub fn float32_array(ptr: *const f32, len: usize) -> js_sys::Float32Array {
        unsafe { js_sys::Float32Array::view(std::slice::from_raw_parts(ptr, len)) }
    }
    
    /// Helper: buat Uint8Array dari pointer + length
    pub fn uint8_array(ptr: *const u8, len: usize) -> js_sys::Uint8Array {
        unsafe { js_sys::Uint8Array::view(std::slice::from_raw_parts(ptr, len)) }
    }
}
