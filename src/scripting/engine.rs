use crate::business;
use crate::dom::DomContext;
use crate::field::tensor_logic::{OrthogonalFusion, SpectralCore, Tensor3D, ZeroParamBridge};
use crate::field::{
    vector_memory::VectorMemory, AgentField, DataWorkerField, EnvironmentGrid, MessageBus,
    BROADCAST_ID,
};
use crate::render::CanvasEncoder;
use crate::svg_generator::SvgGenerator;
use crate::telemetry::EngineMetrics;
use rhai::{Array, CustomType, Engine, Scope};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Safe sandboxed environment to evaluate dynamic scripts (e.g. from LLM).
use super::meta_optimizer::MetaScriptEngine;

pub struct ScriptEngine {
    pub meta: MetaScriptEngine,
    pub max_regex_cache_items: Arc<AtomicUsize>,
}

/// A wrapper pointer context to allow Rhai to safely manipulate the SOA `AgentField`.
/// Using raw pointer here because Rhai requires types to be `'static + Clone`.
/// We guarantee safety because the lifetime of this context is strictly bound
/// to the `eval` function scope.
#[derive(Clone, CustomType)]
pub struct FieldContext {
    ptr: *mut AgentField,
}

impl FieldContext {
    pub fn new(field: &mut AgentField) -> Self {
        Self {
            ptr: std::ptr::from_mut::<AgentField>(field),
        }
    }

    #[inline]
    fn get_field(&mut self) -> &mut AgentField {
        unsafe { &mut *self.ptr }
    }

    // --- PRIMITIVE API FOR RHAI ---

    pub fn get_count(&mut self) -> i64 {
        self.get_field().len as i64
    }

    pub fn get_x(&mut self, idx: i64) -> f64 {
        let field = self.get_field();
        if idx >= 0 {
            f64::from(field.pos_x.get(idx as usize).copied().unwrap_or(0.0))
        } else {
            0.0
        }
    }

    pub fn get_y(&mut self, idx: i64) -> f64 {
        let field = self.get_field();
        if idx >= 0 {
            f64::from(field.pos_y.get(idx as usize).copied().unwrap_or(0.0))
        } else {
            0.0
        }
    }

    pub fn get_behavior(&mut self, idx: i64) -> i64 {
        let field = self.get_field();
        if idx >= 0 {
            i64::from(field.behavior_state.get(idx as usize).copied().unwrap_or(0))
        } else {
            0
        }
    }

    pub fn set_behavior(&mut self, idx: i64, state: i64) {
        let field = self.get_field();
        if idx >= 0 {
            if let Some(beh) = field.behavior_state.get_mut(idx as usize) {
                *beh = state as u8;
            }
        }
    }

    pub fn set_pos(&mut self, idx: i64, x: f64, y: f64) {
        let field = self.get_field();
        if idx >= 0 {
            let u_idx = idx as usize;
            if let (Some(x_ref), Some(y_ref)) =
                (field.pos_x.get_mut(u_idx), field.pos_y.get_mut(u_idx))
            {
                *x_ref = x as f32;
                *y_ref = y as f32;
            }
        }
    }

    pub fn set_health(&mut self, idx: i64, health: f64) {
        let field = self.get_field();
        if idx >= 0 {
            let u_idx = idx as usize;
            if let Some(h_ref) = field.health.get_mut(u_idx) {
                *h_ref = health as f32;
            }
        }
    }

    pub fn set_velocity(&mut self, idx: i64, vx: f64, vy: f64) {
        let field = self.get_field();
        if idx >= 0 {
            let u_idx = idx as usize;
            if let (Some(vx_ref), Some(vy_ref)) =
                (field.vel_x.get_mut(u_idx), field.vel_y.get_mut(u_idx))
            {
                *vx_ref = vx as f32;
                *vy_ref = vy as f32;
            }
        }
    }

    pub fn spawn(&mut self, x: f64, y: f64, health: f64) -> i64 {
        self.get_field().spawn(x as f32, y as f32, health as f32) as i64
    }

    pub fn kill(&mut self, idx: i64) {
        let field = self.get_field();
        if idx >= 0 && (idx as usize) < field.len {
            field.kill_swap(idx as usize);
        }
    }
}

/// A wrapper pointer context to allow Rhai to safely manipulate Data Worker Agents.
#[derive(Clone, CustomType)]
pub struct WorkerContext {
    ptr: *mut DataWorkerField,
}

impl WorkerContext {
    pub fn new(workers: &mut DataWorkerField) -> Self {
        Self {
            ptr: std::ptr::from_mut::<DataWorkerField>(workers),
        }
    }

    #[inline]
    fn get_workers(&mut self) -> &mut DataWorkerField {
        unsafe { &mut *self.ptr }
    }

    pub fn spawn_worker(&mut self, task_id: i64, payload: &str) -> i64 {
        i64::from(self.get_workers().spawn_worker(task_id as u32, payload))
    }

    /// Allows LLM scripts to recursively assign batch tasks to free workers.
    pub fn spawn_workers_batch(&mut self, task_id: i64, payloads: Array) -> i64 {
        let workers = self.get_workers();
        let mut spawned_count = 0;

        for dyn_val in payloads {
            if let Ok(payload_str) = dyn_val.into_string() {
                if workers.spawn_worker(task_id as u32, &payload_str) == -1 {
                    break; // Buffer is full
                }
                spawned_count += 1;
            }
        }

        spawned_count
    }

    pub fn get_worker_state(&mut self, idx: i64) -> i64 {
        let workers = self.get_workers();
        if idx >= 0 {
            i64::from(workers.states.get(idx as usize).copied().unwrap_or(0))
        } else {
            0
        }
    }

    pub fn get_worker_payload(&mut self, idx: i64) -> String {
        let workers = self.get_workers();
        if idx >= 0 && (idx as usize) < workers.len {
            if let Some(&(start, end)) = workers.payload_slices.get(idx as usize) {
                if let Some(text) = workers.text_arena.get(start as usize..end as usize) {
                    return text.to_string();
                }
            }
            String::new()
        } else {
            String::new()
        }
    }

    pub fn set_worker_result(&mut self, idx: i64, result: &str) {
        let workers = self.get_workers();
        if idx >= 0 && (idx as usize) < workers.len {
            let start = workers.text_arena.len() as u32;
            workers.text_arena.push_str(result);
            let end = workers.text_arena.len() as u32;

            workers.result_slices[idx as usize] = (start, end);
            workers.states[idx as usize] = 2; // Done
        }
    }

    pub fn kill_worker(&mut self, idx: i64) {
        if idx >= 0 {
            self.get_workers().kill_worker(idx as usize);
        }
    }

    pub fn get_worker_memory(&mut self, idx: i64, slot: i64) -> f64 {
        if idx >= 0 && (0..8).contains(&slot) {
            if let Some(mem) = self.get_workers().memory.get(idx as usize) {
                return f64::from(mem[slot as usize]);
            }
        }
        0.0
    }

    pub fn set_worker_memory(&mut self, idx: i64, slot: i64, value: f64) {
        if idx >= 0 && (0..8).contains(&slot) {
            if let Some(mem) = self.get_workers().memory.get_mut(idx as usize) {
                mem[slot as usize] = value as f32;
            }
        }
    }
}

/// A wrapper pointer context to allow Rhai to safely manipulate the `MessageBus`.
#[derive(Clone, CustomType)]
pub struct MessageContext {
    ptr: *mut MessageBus,
}

impl MessageContext {
    pub fn new(bus: &mut MessageBus) -> Self {
        Self {
            ptr: std::ptr::from_mut::<MessageBus>(bus),
        }
    }

    #[inline]
    fn get_bus(&mut self) -> &mut MessageBus {
        unsafe { &mut *self.ptr }
    }

    pub fn send(&mut self, sender_id: i64, receiver_id: i64, msg_type: i64, payload: &str) {
        if sender_id >= 0 && receiver_id >= -1 {
            let r_id = if receiver_id == -1 {
                BROADCAST_ID
            } else {
                receiver_id as u32
            };
            self.get_bus()
                .send_message(sender_id as u32, r_id, msg_type as u8, payload);
        }
    }

    pub fn get_message_count(&mut self) -> i64 {
        self.get_bus().len as i64
    }

    pub fn get_payload(&mut self, idx: i64) -> String {
        let bus = self.get_bus();
        if idx >= 0 && (idx as usize) < bus.len {
            if let Some(&(start, end)) = bus.payload_slices.get(idx as usize) {
                if let Some(text) = bus.text_arena.get(start as usize..end as usize) {
                    return text.to_string();
                }
            }
            String::new()
        } else {
            String::new()
        }
    }

    pub fn get_sender(&mut self, idx: i64) -> i64 {
        let bus = self.get_bus();
        if idx >= 0 && (idx as usize) < bus.len {
            i64::from(bus.sender_ids.get(idx as usize).copied().unwrap_or(0))
        } else {
            -1
        }
    }

    pub fn get_type(&mut self, idx: i64) -> i64 {
        let bus = self.get_bus();
        if idx >= 0 && (idx as usize) < bus.len {
            i64::from(bus.message_types.get(idx as usize).copied().unwrap_or(0))
        } else {
            -1
        }
    }
}

/// A wrapper pointer context to allow Rhai to safely manipulate the `EnvironmentGrid`.
#[derive(Clone, CustomType)]
pub struct EnvironmentContext {
    ptr: *mut EnvironmentGrid,
}

impl EnvironmentContext {
    pub fn new(env: &mut EnvironmentGrid) -> Self {
        Self {
            ptr: std::ptr::from_mut::<EnvironmentGrid>(env),
        }
    }

    #[inline]
    fn get_env(&mut self) -> &mut EnvironmentGrid {
        unsafe { &mut *self.ptr }
    }

    pub fn get_value(&mut self, x: f32, y: f32) -> f32 {
        self.get_env().read_value(x, y)
    }

    pub fn set_value(&mut self, x: f32, y: f32, value: f32) {
        self.get_env().set_value(x, y, value);
    }

    pub fn add_value(&mut self, x: f32, y: f32, amount: f32) {
        self.get_env().add_value(x, y, amount);
    }
}

/// A wrapper pointer context to allow Rhai to safely manipulate the `VectorMemory` (RAG).
#[derive(Clone, CustomType)]
pub struct VectorMemoryContext {
    ptr: *mut VectorMemory,
}

impl VectorMemoryContext {
    pub fn new(mem: &mut VectorMemory) -> Self {
        Self {
            ptr: std::ptr::from_mut::<VectorMemory>(mem),
        }
    }

    #[inline]
    fn get_mem(&mut self) -> &mut VectorMemory {
        unsafe { &mut *self.ptr }
    }

    pub fn store(&mut self, id: &str, vector_array: Array) {
        let mut rust_vec = Vec::with_capacity(16);
        for dyn_val in vector_array {
            if let Ok(val) = dyn_val.as_float() {
                rust_vec.push(val as f32);
            } else if let Ok(val) = dyn_val.as_int() {
                rust_vec.push(val as f32);
            }
        }
        self.get_mem().store(id, &rust_vec);
    }

    pub fn search(&mut self, query_array: Array) -> String {
        let mut rust_vec = Vec::with_capacity(16);
        for dyn_val in query_array {
            if let Ok(val) = dyn_val.as_float() {
                rust_vec.push(val as f32);
            } else if let Ok(val) = dyn_val.as_int() {
                rust_vec.push(val as f32);
            }
        }

        if let Some((id, _score)) = self.get_mem().search(&rust_vec) {
            id
        } else {
            String::new()
        }
    }
}

impl ScriptEngine {
    pub fn update_max_regex_cache_items(&mut self, new_val: usize) {
        self.max_regex_cache_items.store(new_val, Ordering::Relaxed);
    }

    pub fn new(max_regex_cache_items: usize) -> Self {
        let mut engine = Engine::new();
        let cache_limit = Arc::new(AtomicUsize::new(max_regex_cache_items));
        let cache_limit_clone1 = cache_limit.clone();
        let cache_limit_clone2 = cache_limit.clone();

        // --- HARDCODE / PRE-REGISTER APIS FOR LLM TO USE ---

        // Register the FieldContext type so Rhai scripts can use it
        engine.build_type::<FieldContext>();

        // Register the methods so LLM scripts can actually call them
        engine.register_fn("get_count", FieldContext::get_count);
        engine.register_fn("get_x", FieldContext::get_x);
        engine.register_fn("get_y", FieldContext::get_y);
        engine.register_fn("get_behavior", FieldContext::get_behavior);
        engine.register_fn("set_behavior", FieldContext::set_behavior);
        engine.register_fn("set_velocity", FieldContext::set_velocity);
        engine.register_fn("set_pos", FieldContext::set_pos);
        engine.register_fn("set_health", FieldContext::set_health);
        engine.register_fn("agent_spawn", FieldContext::spawn);
        engine.register_fn("agent_kill", FieldContext::kill);

        // Expose field object methods directly so script can call field.get_x(...)
        engine.register_fn("get_count", |ctx: &mut FieldContext| ctx.get_count());
        engine.register_fn("get_x", |ctx: &mut FieldContext, idx: i64| ctx.get_x(idx));
        engine.register_fn("get_y", |ctx: &mut FieldContext, idx: i64| ctx.get_y(idx));
        engine.register_fn("get_behavior", |ctx: &mut FieldContext, idx: i64| {
            ctx.get_behavior(idx)
        });
        engine.register_fn(
            "set_behavior",
            |ctx: &mut FieldContext, idx: i64, state: i64| ctx.set_behavior(idx, state),
        );
        engine.register_fn(
            "set_velocity",
            |ctx: &mut FieldContext, idx: i64, vx: f64, vy: f64| ctx.set_velocity(idx, vx, vy),
        );
        engine.register_fn(
            "set_pos",
            |ctx: &mut FieldContext, idx: i64, x: f64, y: f64| ctx.set_pos(idx, x, y),
        );
        engine.register_fn(
            "set_health",
            |ctx: &mut FieldContext, idx: i64, health: f64| ctx.set_health(idx, health),
        );
        engine.register_fn(
            "agent_spawn",
            |ctx: &mut FieldContext, x: f64, y: f64, health: f64| ctx.spawn(x, y, health),
        );
        engine.register_fn("agent_kill", |ctx: &mut FieldContext, idx: i64| {
            ctx.kill(idx);
        });

        // --- WORKER AGENT APIS FOR RHAI ---
        engine.build_type::<WorkerContext>();
        engine.register_fn("spawn_worker", WorkerContext::spawn_worker);
        engine.register_fn("spawn_workers_batch", WorkerContext::spawn_workers_batch);
        engine.register_fn("get_worker_state", WorkerContext::get_worker_state);
        engine.register_fn("get_worker_payload", WorkerContext::get_worker_payload);
        engine.register_fn("set_worker_result", WorkerContext::set_worker_result);
        engine.register_fn("kill_worker", WorkerContext::kill_worker);
        engine.register_fn("get_worker_memory", WorkerContext::get_worker_memory);
        engine.register_fn("set_worker_memory", WorkerContext::set_worker_memory);

        // --- MESSAGE BUS APIS FOR RHAI ---
        engine.build_type::<MessageContext>();
        engine.register_fn("msg_send", MessageContext::send);
        engine.register_fn("msg_count", MessageContext::get_message_count);
        engine.register_fn("msg_payload", MessageContext::get_payload);
        engine.register_fn("msg_sender", MessageContext::get_sender);
        engine.register_fn("msg_type", MessageContext::get_type);

        // --- ENVIRONMENT GRID APIS FOR RHAI ---
        engine.build_type::<EnvironmentContext>();
        engine.register_fn("env_get", EnvironmentContext::get_value);
        engine.register_fn("env_set", EnvironmentContext::set_value);
        engine.register_fn("env_add", EnvironmentContext::add_value);

        // --- VECTOR MEMORY APIS FOR RHAI ---
        engine.build_type::<VectorMemoryContext>();
        engine.register_fn("mem_store", VectorMemoryContext::store);
        engine.register_fn("mem_search", VectorMemoryContext::search);

        // --- DOM MANIPULATION APIS FOR RHAI ---

        engine.register_fn(
            "dom_get_html",
            |target_id: &str| -> Result<String, String> {
                if let Some(dom) = DomContext::new() {
                    dom.get_html(target_id)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_insert_html",
            |target_id: &str, position: &str, html: &str| -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.insert_html_at(target_id, position, html)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_diff_replace_html",
            |target_id: &str, old_str: &str, new_str: &str| -> Result<bool, String> {
                if let Some(dom) = DomContext::new() {
                    dom.diff_and_replace_html(target_id, old_str, new_str)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_append_html",
            |target_id: &str, html: &str| -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.append_html(target_id, html)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_set_inner_html",
            |target_id: &str, html: &str| -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.set_inner_html(target_id, html)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_set_text",
            |target_id: &str, text: &str| -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.set_text_content(target_id, text)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_set_style",
            |target_id: &str, property: &str, value: &str| -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.set_style(target_id, property, value)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_get_value",
            |target_id: &str| -> Result<String, String> {
                if let Some(dom) = DomContext::new() {
                    dom.get_value(target_id)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_canvas_fill_rect",
            |target_id: &str,
             x: f64,
             y: f64,
             w: f64,
             h: f64,
             fill_style: &str|
             -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.canvas_fill_rect(target_id, x, y, w, h, fill_style)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_canvas_clear_rect",
            |target_id: &str, x: f64, y: f64, w: f64, h: f64| -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.canvas_clear_rect(target_id, x, y, w, h)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        engine.register_fn(
            "dom_canvas_draw_text",
            |target_id: &str,
             text: &str,
             x: f64,
             y: f64,
             font: &str,
             color: &str|
             -> Result<(), String> {
                if let Some(dom) = DomContext::new() {
                    dom.canvas_draw_text(target_id, text, x, y, font, color)
                } else {
                    Err("DOM Context unavailable".to_string())
                }
            },
        );

        // --- BUSINESS ANALYTICS APIS FOR RHAI ---
        engine.register_fn("json_extract_string", business::json_extract_string);
        engine.register_fn(
            "regex_extract",
            move |pattern: rhai::ImmutableString, text: rhai::ImmutableString| -> String {
                let limit = cache_limit_clone1.load(Ordering::Relaxed);
                business::regex_extract(&pattern, &text, limit)
            },
        );
        engine.register_fn(
            "regex_extract_all",
            move |pattern: rhai::ImmutableString, text: rhai::ImmutableString| -> rhai::Array {
                let limit = cache_limit_clone2.load(Ordering::Relaxed);
                business::regex_extract_all(&pattern, &text, limit)
            },
        );
        engine.register_fn("sum_number_strings", business::sum_number_strings);
        engine.register_fn("multiply_matrix_1d", business::multiply_matrix_1d);
        engine.register_fn("dot_product", business::dot_product);
        engine.register_fn("sigmoid", business::sigmoid);
        engine.register_fn("q_learning_update", business::q_learning_update);
        engine.register_fn("context_evolution", business::context_evolution);

        // --- TELEMETRY APIS FOR RHAI ---
        engine.build_type::<EngineMetrics>();
        engine.register_fn("get_physics_ms", |m: &mut EngineMetrics| -> f64 {
            m.physics_step_ms
        });
        engine.register_fn("get_script_ms", |m: &mut EngineMetrics| -> f64 {
            m.scripting_eval_ms
        });
        engine.register_fn("get_active_workers", |m: &mut EngineMetrics| -> i64 {
            m.active_data_workers as i64
        });
        engine.register_fn("get_arena_bytes", |m: &mut EngineMetrics| -> i64 {
            m.text_arena_bytes as i64
        });

        // --- TENSOR LOGIC / ASYMMETRIC SPARSE CORE APIS FOR RHAI ---
        engine.build_type::<Tensor3D>();
        engine.register_fn("tensor_zeros", Tensor3D::zeros);

        engine.build_type::<SpectralCore>();
        engine.register_fn("spectral_core", SpectralCore::new);
        engine.register_fn("forward_sparse", SpectralCore::forward_sparse);

        engine.build_type::<ZeroParamBridge>();
        engine.register_fn("zero_param_bridge", ZeroParamBridge::new);
        engine.register_fn("forward", ZeroParamBridge::forward);

        engine.build_type::<OrthogonalFusion>();
        engine.register_fn("orthogonal_fusion", OrthogonalFusion::new);
        engine.register_fn("fuse_sparse", OrthogonalFusion::fuse_sparse);

        // --- RENDER CONTEXT APIS FOR RHAI ---
        engine.build_type::<RenderContext>();
        engine.register_fn("canvas_clear", |cx: &mut RenderContext| cx.clear());
        engine.register_fn(
            "canvas_circle",
            |cx: &mut RenderContext, x: f64, y: f64, r: f64, c: i64| cx.circle(x, y, r, c),
        );
        engine.register_fn(
            "canvas_line",
            |cx: &mut RenderContext, x1: f64, y1: f64, x2: f64, y2: f64, c: i64| {
                cx.line(x1, y1, x2, y2, c);
            },
        );
        engine.register_fn(
            "canvas_rect",
            |cx: &mut RenderContext, x: f64, y: f64, w: f64, h: f64, c: i64| cx.rect(x, y, w, h, c),
        );

        // --- KERNEL CONFIG APIS FOR RHAI ---
        engine.build_type::<ConfigContext>();
        engine.register_fn("get_cursor_x", ConfigContext::get_cursor_x);
        engine.register_fn("get_cursor_y", ConfigContext::get_cursor_y);
        engine.register_fn("get_cursor_weight", ConfigContext::get_cursor_weight);

        // Graph Context (for AST caching parallel execution)
        engine.build_type::<crate::graph::GraphContext>();
        engine.register_fn("get_var", crate::graph::GraphContext::get_var);
        engine.register_fn("set_var", crate::graph::GraphContext::set_var);

        // --- SVG AND PLOTTING APIS FOR RHAI ---
        // Allows LLM to dynamically generate generic statistical charts
        engine.register_fn(
            "svg_draw_scatter",
            |cx: f32, cy: f32, positions: Array| -> String {
                let mut rust_positions = Vec::new();
                for item in positions {
                    // Expects an array of [x, y] coordinates
                    if let Ok(arr) = item.into_array() {
                        if arr.len() >= 2 {
                            let x = arr[0].as_float().unwrap_or(0.0) as f32;
                            let y = arr[1].as_float().unwrap_or(0.0) as f32;
                            rust_positions.push((x, y));
                        }
                    }
                }
                SvgGenerator::build_scatter_svg(cx, cy, &rust_positions)
            },
        );

        engine.register_fn(
            "svg_draw_line_chart",
            |title: &str, data: Array| -> String {
                let mut rust_points = Vec::new();
                for item in data {
                    if let Ok(arr) = item.into_array() {
                        if arr.len() >= 2 {
                            let x = arr[0].as_float().unwrap_or(0.0) as f32;
                            let y = arr[1].as_float().unwrap_or(0.0) as f32;
                            rust_points.push((x, y));
                        }
                    }
                }
                SvgGenerator::build_line_chart_svg(&rust_points, title)
            },
        );

        engine.register_fn("svg_foreign_object", SvgGenerator::build_foreign_object_svg);

        // We can still keep utility functions
        engine.register_fn("render_html_card", |title: &str, content: &str| -> String {
            format!("<div class='agent-card'><h3>{title}</h3><p>{content}</p></div>")
        });

        // Security limits:
        engine.set_max_operations(1_000_000); // Expanded to allow deep learning/while loops before aborting
        engine.set_max_string_size(50_000); // Prevent memory exhaustion

        let meta = MetaScriptEngine::new(engine);
        Self {
            meta,
            max_regex_cache_items: cache_limit,
        }
    }

    /// Evaluates a Rhai script string and returns the resulting String.
    /// Helper to inject memory bindings into a borrowed scope.
    pub fn eval_with_injected_scope(
        &mut self,
        scope: &mut Scope,
        script: &str,
        field: &mut AgentField,
        workers: &mut DataWorkerField,
        messages: &mut MessageBus,
        env_grid: &mut EnvironmentGrid,
        vector_mem: &mut VectorMemory,
        encoder: &mut CanvasEncoder,
        config: &crate::field::KernelConfig,
        metrics: EngineMetrics,
    ) -> Result<String, String> {
        let f_ctx = FieldContext::new(field);
        let w_ctx = WorkerContext::new(workers);
        let m_ctx = MessageContext::new(messages);
        let e_ctx = EnvironmentContext::new(env_grid);
        let v_ctx = VectorMemoryContext::new(vector_mem);
        let r_ctx = RenderContext::new(encoder);
        let k_ctx = ConfigContext::new(config);

        scope.push("field", f_ctx);
        scope.push("workers", w_ctx);
        scope.push("messages", m_ctx);
        scope.push("env_grid", e_ctx);
        scope.push("vector_mem", v_ctx);
        scope.push("canvas", r_ctx);
        scope.push("kernel", k_ctx);
        scope.push("metrics", metrics);

        if script.is_empty() {
            return Ok(String::new());
        }

        match self.meta.eval_with_scope(script, scope) {
            Ok(result) => Ok(result.to_string()),
            Err(e) => Err(format!("LLM Script Error: {e}")),
        }
    }

    pub fn eval_agent(
        &mut self,
        script: &str,
        field: &mut AgentField,
        workers: &mut DataWorkerField,
        messages: &mut MessageBus,
        env_grid: &mut EnvironmentGrid,
        vector_mem: &mut VectorMemory,
        encoder: &mut CanvasEncoder,
        config: &crate::field::KernelConfig,
        metrics: EngineMetrics,
        agent_idx: usize,
    ) -> Result<String, String> {
        let behavior = field.behavior_state.get(agent_idx).copied().unwrap_or(0);
        let x = field.pos_x.get(agent_idx).copied().unwrap_or(0.0);
        let y = field.pos_y.get(agent_idx).copied().unwrap_or(0.0);
        let health = field.health.get(agent_idx).copied().unwrap_or(0.0);

        let mut scope = Scope::new();

        let f_ctx = FieldContext::new(field);
        let w_ctx = WorkerContext::new(workers);
        let m_ctx = MessageContext::new(messages);
        let e_ctx = EnvironmentContext::new(env_grid);
        let v_ctx = VectorMemoryContext::new(vector_mem);
        let r_ctx = RenderContext::new(encoder);
        let k_ctx = ConfigContext::new(config);

        scope.push("field", f_ctx);
        scope.push("workers", w_ctx);
        scope.push("messages", m_ctx);
        scope.push("env_grid", e_ctx);
        scope.push("vector_mem", v_ctx);
        scope.push("canvas", r_ctx);
        scope.push("kernel", k_ctx);
        scope.push("metrics", metrics);
        scope.push("agent_idx", agent_idx as i64); // Provide context of who is executing

        if script.is_empty() {
            return Ok(String::new());
        }

        match self
            .meta
            .eval_for_agent(script, &mut scope, behavior.into(), x, y, health)
        {
            Ok(result) => Ok(result.to_string()),
            Err(e) => Err(format!("LLM Agent Eval Error: {e}")),
        }
    }

    /// Evaluates a Rhai script string and returns the resulting String.
    /// Passes the contexts as dynamic variables to the script.
    pub fn eval(
        &mut self,
        script: &str,
        field: &mut AgentField,
        workers: &mut DataWorkerField,
        messages: &mut MessageBus,
        env_grid: &mut EnvironmentGrid,
        vector_mem: &mut VectorMemory,
        encoder: &mut CanvasEncoder,
        config: &crate::field::KernelConfig,
        metrics: EngineMetrics,
    ) -> Result<String, String> {
        let mut scope = Scope::new();
        self.eval_with_injected_scope(
            &mut scope, script, field, workers, messages, env_grid, vector_mem, encoder, config,
            metrics,
        )
    }
}

/// A wrapper pointer context to allow Rhai to safely command the `CanvasEncoder`.
#[derive(Clone, CustomType)]
pub struct RenderContext {
    ptr: *mut CanvasEncoder,
}

impl RenderContext {
    pub fn new(encoder: &mut CanvasEncoder) -> Self {
        Self {
            ptr: std::ptr::from_mut::<CanvasEncoder>(encoder),
        }
    }

    #[inline]
    fn get_encoder(&mut self) -> &mut CanvasEncoder {
        unsafe { &mut *self.ptr }
    }

    pub fn clear(&mut self) {
        self.get_encoder().clear();
    }

    pub fn circle(&mut self, x: f64, y: f64, r: f64, color: i64) {
        self.get_encoder()
            .circle(x as f32, y as f32, r as f32, color as u32);
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, color: i64) {
        self.get_encoder()
            .line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, color as u32);
    }

    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, color: i64) {
        self.get_encoder()
            .rect(x as f32, y as f32, w as f32, h as f32, color as u32);
    }
}

/// A wrapper pointer context to allow Rhai to safely read the Kernel Config.
#[derive(Clone, CustomType)]
pub struct ConfigContext {
    ptr: *const crate::field::KernelConfig,
}

impl ConfigContext {
    pub fn new(config: &crate::field::KernelConfig) -> Self {
        Self {
            ptr: std::ptr::from_ref(config),
        }
    }

    #[inline]
    fn get_config(&self) -> &crate::field::KernelConfig {
        unsafe { &*self.ptr }
    }

    pub fn get_cursor_x(&mut self) -> f64 {
        f64::from(self.get_config().cursor_x)
    }

    pub fn get_cursor_y(&mut self) -> f64 {
        f64::from(self.get_config().cursor_y)
    }

    pub fn get_cursor_weight(&mut self) -> f64 {
        f64::from(self.get_config().cursor_weight)
    }
}
