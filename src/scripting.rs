use rhai::{Engine, Scope, Dynamic, CustomType};
use crate::field::AgentField;
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
        if idx >= 0 && (idx as usize) < field.len {
            field.pos_x[idx as usize]
        } else {
            0.0
        }
    }

    pub fn get_y(&mut self, idx: i64) -> f32 {
        let field = self.get_field();
        if idx >= 0 && (idx as usize) < field.len {
            field.pos_y[idx as usize]
        } else {
            0.0
        }
    }

    pub fn set_velocity(&mut self, idx: i64, vx: f32, vy: f32) {
        let field = self.get_field();
        if idx >= 0 && (idx as usize) < field.len {
            field.vel_x[idx as usize] = vx;
            field.vel_y[idx as usize] = vy;
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
    /// Passes the `AgentField` as a dynamic variable `field` to the script.
    pub fn eval(&mut self, script: &str, field: &mut AgentField) -> Result<String, String> {
        let mut scope = Scope::new();

        // Push the field context into the scope so the script can access it
        let ctx = FieldContext::new(field);
        scope.push("field", ctx);

        // Execute the script and format the output as a string to return to JS
        match self.engine.eval_with_scope::<Dynamic>(&mut scope, script) {
            Ok(result) => Ok(result.to_string()),
            Err(e) => Err(format!("LLM Script Error: {}", e)),
        }
    }
}
