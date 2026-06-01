use rhai::CustomType;
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use web_sys::window;

/// A structure to hold precise performance and memory metrics
/// for the WebAssembly simulation engine.
#[derive(Default, Serialize, Clone, Copy, CustomType)]
pub struct EngineMetrics {
    pub physics_step_ms: f64,
    pub scripting_eval_ms: f64,
    pub arena_compaction_ms: f64,

    pub active_physics_agents: usize,
    pub active_data_workers: usize,
    pub total_messages: usize,

    pub text_arena_bytes: usize,
    pub memory_vector_count: usize,
}

pub struct Telemetry {
    pub metrics: EngineMetrics,

    // Internal counters
    #[allow(dead_code)]
    last_frame_time: f64,
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            metrics: EngineMetrics::default(),
            last_frame_time: Self::now(),
        }
    }

    /// Fetches the current high-resolution timestamp in milliseconds from `window.performance.now()`.
    /// Returns 0.0 if not running in a browser environment.
    pub fn now() -> f64 {
        // Under pure cargo test (native x86_64, not wasm32), `js-sys` panics when attempting to fetch JS globals.
        // We conditionally disable this native panic during cargo tests.
        #[cfg(all(target_arch = "wasm32", not(test)))]
        {
            if let Some(win) = window() {
                if let Some(perf) = win.performance() {
                    return perf.now();
                }
            }
        }
        0.0
    }

    /// Starts a timer and returns the start time.
    #[inline(always)]
    pub fn start_timer() -> f64 {
        Self::now()
    }

    /// Measures the elapsed time since `start_time` and updates the physics step metric.
    #[inline(always)]
    pub fn record_physics_step(&mut self, start_time: f64) {
        self.metrics.physics_step_ms = Self::now() - start_time;
    }

    /// Measures the elapsed time for evaluating Rhai scripts.
    #[inline(always)]
    pub fn record_script_eval(&mut self, start_time: f64) {
        self.metrics.scripting_eval_ms = Self::now() - start_time;
    }

    /// Measures the elapsed time for garbage collection routines.
    #[inline(always)]
    pub fn record_compaction(&mut self, start_time: f64) {
        self.metrics.arena_compaction_ms = Self::now() - start_time;
    }

    /// Serializes the current metrics state into a JSON string for JS consumption.
    pub fn get_metrics_json(&self) -> String {
        serde_json::to_string(&self.metrics).unwrap_or_else(|_| "{}".to_string())
    }
}
