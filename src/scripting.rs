use rhai::{Engine, Scope, Dynamic, CustomType, Array};
use crate::field::{AgentField, DataWorkerField, MessageBus, EnvironmentGrid, vector_memory::VectorMemory, BROADCAST_ID};
use crate::dom::DomContext;
use crate::business;

/// Safe sandboxed environment to evaluate dynamic scripts (e.g. from LLM).
pub struct ScriptEngine {
    engine: Engine,
}

/// A wrapper pointer context to allow Rhai to safely manipulate the SOA AgentField.
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
            ptr: field as *mut AgentField,
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

    pub fn get_x(&mut self, idx: i64) -> f32 {
        let field = self.get_field();
        if idx >= 0 {
            field.pos_x.get(idx as usize).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    pub fn get_y(&mut self, idx: i64) -> f32 {
        let field = self.get_field();
        if idx >= 0 {
            field.pos_y.get(idx as usize).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    pub fn get_behavior(&mut self, idx: i64) -> i64 {
        let field = self.get_field();
        if idx >= 0 {
            field.behavior_state.get(idx as usize).copied().unwrap_or(0) as i64
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

    pub fn set_velocity(&mut self, idx: i64, vx: f32, vy: f32) {
        let field = self.get_field();
        if idx >= 0 {
            let u_idx = idx as usize;
            if let (Some(vx_ref), Some(vy_ref)) = (field.vel_x.get_mut(u_idx), field.vel_y.get_mut(u_idx)) {
                *vx_ref = vx;
                *vy_ref = vy;
            }
        }
    }

    pub fn spawn(&mut self, x: f32, y: f32, health: f32) -> i64 {
        self.get_field().spawn(x, y, health) as i64
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
            ptr: workers as *mut DataWorkerField,
        }
    }

    #[inline]
    fn get_workers(&mut self) -> &mut DataWorkerField {
        unsafe { &mut *self.ptr }
    }

    pub fn spawn_worker(&mut self, task_id: i64, payload: &str) -> i64 {
        self.get_workers().spawn_worker(task_id as u32, payload) as i64
    }

    pub fn get_worker_state(&mut self, idx: i64) -> i64 {
        let workers = self.get_workers();
        if idx >= 0 {
            workers.states.get(idx as usize).copied().unwrap_or(0) as i64
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
}

/// A wrapper pointer context to allow Rhai to safely manipulate the MessageBus.
#[derive(Clone, CustomType)]
pub struct MessageContext {
    ptr: *mut MessageBus,
}

impl MessageContext {
    pub fn new(bus: &mut MessageBus) -> Self {
        Self {
            ptr: bus as *mut MessageBus,
        }
    }

    #[inline]
    fn get_bus(&mut self) -> &mut MessageBus {
        unsafe { &mut *self.ptr }
    }

    pub fn send(&mut self, sender_id: i64, receiver_id: i64, msg_type: i64, payload: &str) {
        if sender_id >= 0 && receiver_id >= -1 {
            let r_id = if receiver_id == -1 { BROADCAST_ID } else { receiver_id as u32 };
            self.get_bus().send_message(sender_id as u32, r_id, msg_type as u8, payload);
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
            bus.sender_ids.get(idx as usize).copied().unwrap_or(0) as i64
        } else {
            -1
        }
    }

    pub fn get_type(&mut self, idx: i64) -> i64 {
        let bus = self.get_bus();
        if idx >= 0 && (idx as usize) < bus.len {
            bus.message_types.get(idx as usize).copied().unwrap_or(0) as i64
        } else {
            -1
        }
    }
}

/// A wrapper pointer context to allow Rhai to safely manipulate the EnvironmentGrid.
#[derive(Clone, CustomType)]
pub struct EnvironmentContext {
    ptr: *mut EnvironmentGrid,
}

impl EnvironmentContext {
    pub fn new(env: &mut EnvironmentGrid) -> Self {
        Self {
            ptr: env as *mut EnvironmentGrid,
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

/// A wrapper pointer context to allow Rhai to safely manipulate the VectorMemory (RAG).
#[derive(Clone, CustomType)]
pub struct VectorMemoryContext {
    ptr: *mut VectorMemory,
}

impl VectorMemoryContext {
    pub fn new(mem: &mut VectorMemory) -> Self {
        Self {
            ptr: mem as *mut VectorMemory,
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
    pub fn new() -> Self {
        let mut engine = Engine::new();

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
        engine.register_fn("spawn", FieldContext::spawn);
        engine.register_fn("kill", FieldContext::kill);

        // Expose field object methods directly so script can call field.get_x(...)
        engine.register_fn("get_count", |ctx: &mut FieldContext| ctx.get_count());
        engine.register_fn("get_x", |ctx: &mut FieldContext, idx: i64| ctx.get_x(idx));
        engine.register_fn("get_y", |ctx: &mut FieldContext, idx: i64| ctx.get_y(idx));
        engine.register_fn("get_behavior", |ctx: &mut FieldContext, idx: i64| ctx.get_behavior(idx));
        engine.register_fn("set_behavior", |ctx: &mut FieldContext, idx: i64, state: i64| ctx.set_behavior(idx, state));
        engine.register_fn("set_velocity", |ctx: &mut FieldContext, idx: i64, vx: f32, vy: f32| ctx.set_velocity(idx, vx, vy));
        engine.register_fn("spawn", |ctx: &mut FieldContext, x: f32, y: f32, health: f32| ctx.spawn(x, y, health));
        engine.register_fn("kill", |ctx: &mut FieldContext, idx: i64| ctx.kill(idx));

        // --- WORKER AGENT APIS FOR RHAI ---
        engine.build_type::<WorkerContext>();
        engine.register_fn("spawn_worker", WorkerContext::spawn_worker);
        engine.register_fn("get_worker_state", WorkerContext::get_worker_state);
        engine.register_fn("get_worker_payload", WorkerContext::get_worker_payload);
        engine.register_fn("set_worker_result", WorkerContext::set_worker_result);
        engine.register_fn("kill_worker", WorkerContext::kill_worker);

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

        engine.register_fn("dom_append_html", |target_id: &str, html: &str| -> Result<(), String> {
            if let Some(dom) = DomContext::new() {
                dom.append_html(target_id, html)
            } else {
                Err("DOM Context unavailable".to_string())
            }
        });

        engine.register_fn("dom_set_inner_html", |target_id: &str, html: &str| -> Result<(), String> {
            if let Some(dom) = DomContext::new() {
                dom.set_inner_html(target_id, html)
            } else {
                Err("DOM Context unavailable".to_string())
            }
        });

        engine.register_fn("dom_set_text", |target_id: &str, text: &str| -> Result<(), String> {
            if let Some(dom) = DomContext::new() {
                dom.set_text_content(target_id, text)
            } else {
                Err("DOM Context unavailable".to_string())
            }
        });

        engine.register_fn("dom_set_style", |target_id: &str, property: &str, value: &str| -> Result<(), String> {
            if let Some(dom) = DomContext::new() {
                dom.set_style(target_id, property, value)
            } else {
                Err("DOM Context unavailable".to_string())
            }
        });

        engine.register_fn("dom_get_value", |target_id: &str| -> Result<String, String> {
            if let Some(dom) = DomContext::new() {
                dom.get_value(target_id)
            } else {
                Err("DOM Context unavailable".to_string())
            }
        });

        // --- BUSINESS ANALYTICS APIS FOR RHAI ---
        engine.register_fn("json_extract_string", business::json_extract_string);
        engine.register_fn("regex_extract", business::regex_extract);
        engine.register_fn("regex_extract_all", business::regex_extract_all);
        engine.register_fn("sum_number_strings", business::sum_number_strings);

        // We can still keep utility functions
        engine.register_fn("render_html_card", |title: &str, content: &str| -> String {
            format!("<div class='agent-card'><h3>{}</h3><p>{}</p></div>", title, content)
        });

        // Security limits:
        engine.set_max_operations(10_000); // Prevent infinite loops from bad LLM scripts
        engine.set_max_string_size(50_000); // Prevent memory exhaustion

        Self { engine }
    }

    /// Evaluates a Rhai script string and returns the resulting String.
    /// Passes the contexts as dynamic variables to the script.
    pub fn eval(&mut self, script: &str, field: &mut AgentField, workers: &mut DataWorkerField, messages: &mut MessageBus, env_grid: &mut EnvironmentGrid, vector_mem: &mut VectorMemory) -> Result<String, String> {
        let mut scope = Scope::new();

        // Push the contexts into the scope so the script can access them
        let f_ctx = FieldContext::new(field);
        let w_ctx = WorkerContext::new(workers);
        let m_ctx = MessageContext::new(messages);
        let e_ctx = EnvironmentContext::new(env_grid);
        let v_ctx = VectorMemoryContext::new(vector_mem);
        scope.push("field", f_ctx);
        scope.push("workers", w_ctx);
        scope.push("messages", m_ctx);
        scope.push("env_grid", e_ctx);
        scope.push("vector_mem", v_ctx);

        // Execute the script and format the output as a string to return to JS
        match self.engine.eval_with_scope::<Dynamic>(&mut scope, script) {
            Ok(result) => Ok(result.to_string()),
            Err(e) => {
                // If it's just a variable not found (like when checking bindings in tests), return empty instead of failing
                let err_str = e.to_string();
                if err_str.contains("Function not found") || err_str.contains("Variable not found") {
                    Err(format!("LLM Script Error: {}", err_str))
                } else {
                    Err(format!("LLM Script Error: {}", err_str))
                }
            },
        }
    }
}
