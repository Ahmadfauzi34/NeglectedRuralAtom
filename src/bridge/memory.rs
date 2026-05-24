use wasm_bindgen::prelude::*;
use crate::field::{AgentField, KernelConfig, step_agents};
use crate::command::{CommandBus, Command};
use crate::render::{CanvasEncoder, DrawHeader, DrawCmd};
use crate::render::agent_renderer::encode_agents;

#[wasm_bindgen]
pub struct KernelBridge {
    field: AgentField,
    config: KernelConfig,
    cmd_bus: CommandBus,
    encoder: CanvasEncoder,
    
    // Render buffer pointer cache
    render_ptr: *const u8,
    render_len: usize,
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
            cmd_bus: CommandBus::new(),
            encoder: CanvasEncoder::new(max_agents * 2 + 64),
            render_ptr: std::ptr::null(),
            render_len: 0,
        }
    }
    
    // === Command Interface (Agen Bebas Kirim Perintah) ===
    
    /// Eksekusi single command dari JSON string.
    pub fn execute_command(&mut self, json: &str) {
        if let Err(e) = self.cmd_bus.parse(json) {
            web_sys::console::error_1(&format!("Command parse error: {:?}", e).into());
        }
    }
    
    /// Eksekusi batch commands.
    pub fn execute_batch(&mut self, json: &str) {
        if let Err(e) = self.cmd_bus.parse_batch(json) {
            web_sys::console::error_1(&format!("Batch parse error: {:?}", e).into());
        }
    }
    
    // === Legacy Spawn (masih bisa dipakai untuk init cepat) ===
    
    pub fn spawn(&mut self, x: f32, y: f32, health: f32) -> usize {
        self.field.spawn(x, y, health)
    }
    
    pub fn kill(&mut self, idx: usize) {
        self.field.kill_swap(idx);
    }
    
    pub fn set_config(&mut self, dt: f32, friction: f32, max_speed: f32, influence_radius: f32) {
        self.config = KernelConfig {
            dt,
            friction,
            max_speed,
            influence_radius,
            ..self.config
        };
    }
    
    // === Simulation Step ===
    
    pub fn step(&mut self) {
        // 1. Jalankan command queue
        self.cmd_bus.execute(&mut self.field, &mut self.config);
        
        // 2. Physics / AI kernel
        step_agents(&mut self.field, &self.config);
        
        // 3. Render encoding (SOA → DrawCmd)
        encode_agents(&mut self.encoder, &self.field, 0xFF6366F1); // default indigo color
        
        // 4. Flatten ke buffer
        let (ptr, len) = self.encoder.encode();
        self.render_ptr = ptr;
        self.render_len = len;
    }
    
    // === Zero-Copy Render Buffer Access ===
    
    #[wasm_bindgen(getter)]
    pub fn render_ptr(&self) -> *const u8 {
        self.render_ptr
    }
    
    #[wasm_bindgen(getter)]
    pub fn render_len(&self) -> usize {
        self.render_len
    }
    
    // === Legacy direct buffer access (opsional, kalau TS mau baca SOA langsung) ===
    
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
}

/// MemoryView helpers untuk JS
#[wasm_bindgen]
pub struct MemoryView;

#[wasm_bindgen]
impl MemoryView {
    pub fn float32_array(ptr: *const f32, len: usize) -> js_sys::Float32Array {
        unsafe { js_sys::Float32Array::view(std::slice::from_raw_parts(ptr, len)) }
    }
    
    pub fn uint8_array(ptr: *const u8, len: usize) -> js_sys::Uint8Array {
        unsafe { js_sys::Uint8Array::view(std::slice::from_raw_parts(ptr, len)) }
    }
    
    /// Baca u32 dari pointer (little-endian)
    pub fn read_u32(ptr: *const u8) -> u32 {
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, 4);
            u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])
        }
    }
    
    /// Baca f32 dari pointer (little-endian)
    pub fn read_f32(ptr: *const u8) -> f32 {
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, 4);
            f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])
        }
    }
}
