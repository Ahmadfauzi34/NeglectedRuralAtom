use wasm_bindgen::prelude::*;
use std::sync::{Arc, RwLock};

use crate::field::{AgentField, KernelConfig, step_agents, SpatialGrid, DataWorkerField, MessageBus, EnvironmentGrid, vector_memory::VectorMemory};
use crate::command::CommandBus;
use crate::render::{CanvasEncoder, agent_renderer::encode_agents, GpuBuffer};
use crate::scripting::ScriptEngine;
use crate::prompt::PromptBuilder;
use crate::telemetry::Telemetry;
use crate::graph::{GraphExecutor, ScriptNode};

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
    // Made internal to avoid wasm_bindgen trait issues with struct exposure.
    // We will expose it via a getter that returns a JSON string instead.
    telemetry: Telemetry,
    graph_executor: GraphExecutor,
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
            telemetry: Telemetry::new(),
            graph_executor: GraphExecutor::new(),
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
    
    /// Instantly allocates and distributes tasks mapped from a JS Array of strings
    /// across idle Data Workers. Zero-copy parsing via JsValue.
    pub fn spawn_workers_batch(&mut self, base_task_id: u32, payloads_js: JsValue) -> i32 {
        if let Ok(payloads) = serde_wasm_bindgen::from_value::<Vec<String>>(payloads_js) {
            let mut state = self.state.write().unwrap();
            let mut spawned_count = 0;

            for payload in payloads {
                // If it fails to spawn (buffer full), it halts early and returns count
                if state.workers.spawn_worker(base_task_id, &payload) == -1 {
                    break;
                }
                spawned_count += 1;
            }

            spawned_count
        } else {
            -1 // Deserialization error
        }
    }

    /// Evaluates a dynamic LLM-generated script against the WASM engine.
    pub fn eval_llm_script(&mut self, script: &str) -> String {
        let start_time = Telemetry::start_timer();

        let mut state = self.state.write().unwrap();
        let SharedState { field, workers, messages, env_grid, vector_mem, .. } = &mut *state;
        let metrics_copy = self.telemetry.metrics.clone();

        let result = match self.script_engine.eval(script, field, workers, messages, env_grid, vector_mem, metrics_copy) {
            Ok(res) => res,
            Err(e) => e,
        };

        self.telemetry.record_script_eval(start_time);
        result
    }

    /// Evaluates a JS Array representing a DAG of `ScriptNode` blocks.
    /// This brings advanced Pipeline behavior matching ComfyUI or LangChain
    /// directly to the edge inside WASM without heavy JSON parsing overhead.
    pub fn eval_graph(&mut self, nodes_js: JsValue, start_node_id: &str) -> String {
        let start_time = Telemetry::start_timer();

        let result = match serde_wasm_bindgen::from_value::<Vec<ScriptNode>>(nodes_js) {
            Ok(nodes) => {
                let mut state = self.state.write().unwrap();
                let SharedState { field, workers, messages, env_grid, vector_mem, .. } = &mut *state;
                let metrics_copy = self.telemetry.metrics.clone();

                let mut scope = rhai::Scope::new();
                // We utilize the ScriptEngine's existing capability to inject field bindings into a base scope
                match self.script_engine.eval_with_injected_scope(
                    &mut scope, "", field, workers, messages, env_grid, vector_mem, metrics_copy
                ) {
                    Ok(_) => {
                        // Base scope is now primed with `field`, `workers`, etc.
                        match self.graph_executor.run_graph(&self.script_engine.engine, &mut scope, nodes, start_node_id) {
                            Ok(res) => res,
                            Err(e) => e,
                        }
                    },
                    Err(e) => e,
                }
            },
            Err(_) => "JsValue Parsing Error: Invalid Graph Format".to_string(),
        };

        self.telemetry.record_script_eval(start_time);
        result
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

    pub fn set_config(&mut self, dt: f32, friction: f32, max_speed: f32, influence_radius: f32, cursor_x: f32, cursor_y: f32, cursor_weight: f32) {
        let mut state = self.state.write().unwrap();
        state.config = KernelConfig {
            dt,
            friction,
            max_speed,
            influence_radius,
            cursor_x,
            cursor_y,
            cursor_weight,
            ..state.config
        };
    }
    
    pub fn step(&mut self) {
        let start_time = Telemetry::start_timer();
        let mut state = self.state.write().unwrap();

        // Record structural counts into telemetry
        self.telemetry.metrics.active_physics_agents = state.field.agent_count();
        self.telemetry.metrics.active_data_workers = state.workers.len - state.workers.free_slots.len();
        self.telemetry.metrics.text_arena_bytes = state.workers.text_arena.len();
        self.telemetry.metrics.total_messages = state.messages.len;
        self.telemetry.metrics.memory_vector_count = state.vector_mem.len;

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

        self.telemetry.record_physics_step(start_time);
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

    /// Exposes a serialized JSON of the engine metrics to Javascript
    pub fn get_metrics_json(&self) -> String {
        self.telemetry.get_metrics_json()
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
