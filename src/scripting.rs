use rhai::{Engine, Scope, Dynamic};

/// Safe sandboxed environment to evaluate dynamic scripts (e.g. from LLM).
pub struct ScriptEngine {
    engine: Engine,
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();

        // --- HARDCODE / PRE-REGISTER APIS FOR LLM TO USE ---

        // Example 1: Rendering HTML string building for agents
        engine.register_fn("render_html_card", |title: &str, content: &str| -> String {
            format!("<div class='agent-card'><h3>{}</h3><p>{}</p></div>", title, content)
        });

        // Example 2: Lightweight data processing
        engine.register_fn("process_agent_data", |health: f32, energy: f32| -> f32 {
            (health * 0.7) + (energy * 0.3)
        });

        // We can add many utility functions here: regex simulation,
        // string escaping, mathematical models, state machine evaluations, etc.

        // Security limits:
        engine.set_max_operations(10_000); // Prevent infinite loops from bad LLM scripts
        engine.set_max_string_size(50_000); // Prevent memory exhaustion

        Self { engine }
    }

    /// Evaluates a Rhai script string and returns the resulting String.
    /// Used when LLM wants to execute logic or generate data on the fly.
    pub fn eval(&mut self, script: &str) -> Result<String, String> {
        let mut scope = Scope::new();

        // Execute the script and format the output as a string to return to JS
        match self.engine.eval_with_scope::<Dynamic>(&mut scope, script) {
            Ok(result) => Ok(result.to_string()),
            Err(e) => Err(format!("LLM Script Error: {}", e)),
        }
    }
}
