use rhai::{Engine, Scope, Dynamic, CustomType};
use crate::field::{AgentField, DataWorkerField, MessageBus, BROADCAST_ID};
use crate::dom::DomContext;

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
            let (start, end) = workers.payload_slices[idx as usize];
            workers.text_arena[start as usize..end as usize].to_string()
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
            bus.get_payload(idx as usize).to_string()
        } else {
            String::new()
        }
    }

    pub fn get_sender(&mut self, idx: i64) -> i64 {
        let bus = self.get_bus();
        if idx >= 0 && (idx as usize) < bus.len {
            bus.sender_ids[idx as usize] as i64
        } else {
            -1
        }
    }

    pub fn get_type(&mut self, idx: i64) -> i64 {
        let bus = self.get_bus();
        if idx >= 0 && (idx as usize) < bus.len {
            bus.message_types[idx as usize] as i64
        } else {
            -1
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
        engine.register_fn("set_velocity", FieldContext::set_velocity);
        engine.register_fn("spawn", FieldContext::spawn);
        engine.register_fn("kill", FieldContext::kill);

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
    /// Passes the `AgentField`, `DataWorkerField`, and `MessageBus` as dynamic variables to the script.
    pub fn eval(&mut self, script: &str, field: &mut AgentField, workers: &mut DataWorkerField, messages: &mut MessageBus) -> Result<String, String> {
        let mut scope = Scope::new();

        // Push the contexts into the scope so the script can access them
        let f_ctx = FieldContext::new(field);
        let w_ctx = WorkerContext::new(workers);
        let m_ctx = MessageContext::new(messages);
        scope.push("field", f_ctx);
        scope.push("workers", w_ctx);
        scope.push("messages", m_ctx);

        // Execute the script and format the output as a string to return to JS
        match self.engine.eval_with_scope::<Dynamic>(&mut scope, script) {
            Ok(result) => Ok(result.to_string()),
            Err(e) => Err(format!("LLM Script Error: {}", e)),
        }
    }
}
