use wasm_bindgen::prelude::*;
use std::sync::{Arc, RwLock};

use crate::field::{AgentField, KernelConfig, step_agents, SpatialGrid, DataWorkerField, MessageBus, EnvironmentGrid, vector_memory::VectorMemory};
use crate::command::CommandBus;
use crate::render::{CanvasEncoder, agent_renderer::encode_agents, GpuBuffer};
use crate::scripting::ScriptEngine;
use crate::prompt::PromptBuilder;

/// Represents the shared internal state of the simulation.
/// Wrapped in Arc<RwLock> to allow async tasks (like LLM fetch)
/// to read the state safely without blocking rendering.
pub struct SharedState {
    pub field: AgentField,
    pub workers: DataWorkerField,
    pub messages: MessageBus,
    pub env_grid: EnvironmentGrid,
    pub vector_mem: VectorMemory,
    pub config: KernelConfig,
    pub spatial_grid: SpatialGrid,
}

#[wasm_bindgen]
pub struct KernelBridge {
    state: Arc<RwLock<SharedState>>,
    cmd_bus: CommandBus,
    encoder: CanvasEncoder,
    render_ptr: *const u8,
    render_len: usize,
    gpu_buffer: GpuBuffer,
    script_engine: ScriptEngine,
    prompt_builder: PromptBuilder,
    use_webgl: bool,
}

#[wasm_bindgen]
impl KernelBridge {
    #[wasm_bindgen(constructor)]
    pub fn new(max_agents: usize) -> Self {
        let mut field = AgentField::new(max_agents);
        field.reserve(max_agents);
        
        let state = SharedState {
            field,
            workers: DataWorkerField::new(max_agents),
            messages: MessageBus::new(1024),
            env_grid: EnvironmentGrid::new(100, 100, 10.0), // 100x100 grid, 10px per cell
            vector_mem: VectorMemory::new(1024),
            config: KernelConfig::default(),
            spatial_grid: SpatialGrid::new(80.0),
        };

        Self {
            state: Arc::new(RwLock::new(state)),
            cmd_bus: CommandBus::new(),
            encoder: CanvasEncoder::new(max_agents * 2 + 64),
            render_ptr: std::ptr::null(),
            render_len: 0,
            gpu_buffer: GpuBuffer::new(max_agents),
            script_engine: ScriptEngine::new(),
            prompt_builder: PromptBuilder::new(max_agents * 128), // 128 bytes roughly covers each agent printout
            use_webgl: false,
        }
    }
    
    pub fn execute_command(&mut self, json: &str) {
        if let Err(e) = self.cmd_bus.parse(json) {
            web_sys::console::error_1(&format!("Command parse error: {:?}", e).into());
        }
    }
    
    pub fn execute_batch(&mut self, json: &str) {
        if let Err(e) = self.cmd_bus.parse_batch(json) {
            web_sys::console::error_1(&format!("Batch parse error: {:?}", e).into());
        }
    }
    
    /// Evaluates a dynamic LLM-generated script against the WASM engine.
    pub fn eval_llm_script(&mut self, script: &str) -> String {
        let mut state = self.state.write().unwrap();
        let SharedState { field, workers, messages, env_grid, vector_mem, .. } = &mut *state;
        match self.script_engine.eval(script, field, workers, messages, env_grid, vector_mem) {
            Ok(res) => res,
            Err(e) => e,
        }
    }

    /// Builds a prompt string containing the active agent states.
    /// Uses a snapshot approach to quickly release the `RwLock` and avoid blocking `step()`.
    pub fn generate_llm_prompt(&mut self) -> String {
        // Scope the lock to copy only what we need (Snapshot)
        let snapshot = {
            let state = self.state.read().unwrap();

            // Limit snapshot size to avoid cloning huge arrays if not necessary
            let limit = state.field.len.min(50);
            let mut snap = Vec::with_capacity(limit);

            let mut count = 0;
            for i in 0..state.field.len {
                if count >= limit { break; }
                if state.field.active[i] == 1 {
                    snap.push((
                        i,
                        state.field.pos_x[i], state.field.pos_y[i],
                        state.field.vel_x[i], state.field.vel_y[i],
                        state.field.health[i]
                    ));
                    count += 1;
                }
            }
            (snap, state.field.len)
        }; // Read Lock is DROPPED here immediately!

        // Now build the prompt text (which takes formatting time) without locking the main thread
        self.prompt_builder.build_from_snapshot(&snapshot.0, snapshot.1).to_string()
    }

    pub fn spawn(&mut self, x: f32, y: f32, health: f32) -> usize {
        let mut state = self.state.write().unwrap();
        state.field.spawn(x, y, health)
    }
    
    pub fn kill(&mut self, idx: usize) {
        let mut state = self.state.write().unwrap();
        state.field.kill_swap(idx);
    }
    
    pub fn set_render_mode(&mut self, use_webgl: bool) {
        self.use_webgl = use_webgl;
    }

    pub fn set_config(&mut self, dt: f32, friction: f32, max_speed: f32, influence_radius: f32) {
        let mut state = self.state.write().unwrap();
        state.config = KernelConfig {
            dt,
            friction,
            max_speed,
            influence_radius,
            ..state.config
        };
    }
    
    pub fn step(&mut self) {
        let mut state = self.state.write().unwrap();

        // Destructure state to avoid borrow checker conflicts
        let SharedState { field, workers: _, messages: _, env_grid, vector_mem: _, config, spatial_grid } = &mut *state;

        // Execute pending commands
        self.cmd_bus.execute(field, config);

        // Decay environment pheromones slightly every frame
        env_grid.decay(0.99);

        // Step physics and AI
        step_agents(field, config, spatial_grid, env_grid);

        // Render pass optimization: Branch execution based on chosen target
        if self.use_webgl {
            // Zero-copy Instanced Buffer Rendering for WebGL
            self.gpu_buffer.update(field);
        } else {
            // Classic CPU Canvas Rendering
            encode_agents(&mut self.encoder, field, 0xFF6366F1);
            let (ptr, len) = self.encoder.encode();
            self.render_ptr = ptr;
            self.render_len = len;
        }
    }

    #[wasm_bindgen(getter)]
    pub fn gpu_buffer_ptr(&self) -> *const f32 {
        self.gpu_buffer.ptr()
    }

    #[wasm_bindgen(getter)]
    pub fn gpu_buffer_len(&self) -> usize {
        self.gpu_buffer.len()
    }
    
    #[wasm_bindgen(getter)]
    pub fn render_ptr(&self) -> *const u8 {
        self.render_ptr
    }
    
    #[wasm_bindgen(getter)]
    pub fn render_len(&self) -> usize {
        self.render_len
    }
    
    #[wasm_bindgen(getter)]
    pub fn agent_count(&self) -> usize {
        let state = self.state.read().unwrap();
        state.field.agent_count()
    }
    
    #[wasm_bindgen(getter)]
    pub fn pos_x_ptr(&self) -> *const f32 {
        // Warning: Exposing raw pointers to a lock-protected struct is inherently risky
        // if JS accesses them while a Write lock is held or if a reallocation occurs.
        // As long as SOA buffer is pre-allocated and JS only reads during idle time, it's safe.
        // WebAssembly linear memory does not move unless the Vec reallocates.
        let state = self.state.read().unwrap();
        state.field.pos_x_ptr()
    }
    
    #[wasm_bindgen(getter)]
    pub fn pos_y_ptr(&self) -> *const f32 {
        let state = self.state.read().unwrap();
        state.field.pos_y_ptr()
    }
}

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
    
    pub fn read_u32(ptr: *const u8) -> u32 {
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, 4);
            u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])
        }
    }
    
    pub fn read_f32(ptr: *const u8) -> f32 {
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, 4);
            f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])
        }
    }
}
