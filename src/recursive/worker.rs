//! Worker Agent — Layer 1
//!
//! Recursive capability:
//! - Local script cache (MetaScriptEngine)
//! - Q-Learning memory (8-slot weights)
//! - Self-report status ke Orchestrator via MessageBus
//! - Request help (escalate) jika stuck

use crate::field::{
    DataWorkerField, MessageBus, BROADCAST_ID,
};
use crate::scripting::ScriptEngine;

/// Worker bisa dalam mode:
/// 0 = Idle, 1 = Working, 2 = Done, 3 = Error, 4 = Escalated (need help)
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkerMode {
    Idle = 0,
    Working = 1,
    Done = 2,
    Error = 3,
    Escalated = 4,      // Recursive: minta bantuan layer atas
    Learning = 5,       // Recursive: sedang update weights
}

pub struct WorkerAgent {
    pub idx: usize,
    pub task_id: u32,

    // Local state machine
    pub mode: WorkerMode,
    pub retry_count: u8,
    pub max_retries: u8,        // Configurable via JS/Rhai

    // Q-Learning weights (8 slots dari DataWorkerField)
    pub weights: [f32; 8],
    pub learning_rate: f32,

    // Local memory untuk RAG konteks task
    pub context_embeddings: Vec<f32>,

    // Script yang sedang dieksekusi (untuk self-modification)
    pub current_script: String,
    pub last_result: String,

    // Telemetry
    pub exec_time_ms: f64,
    pub success_rate: f32,      // Running average
}

impl WorkerAgent {
    pub fn new(idx: usize, task_id: u32) -> Self {
        Self {
            idx,
            task_id,
            mode: WorkerMode::Idle,
            retry_count: 0,
            max_retries: 3,
            weights: [0.0; 8],
            learning_rate: 0.1,
            context_embeddings: Vec::new(),
            current_script: String::new(),
            last_result: String::new(),
            exec_time_ms: 0.0,
            success_rate: 1.0,
        }
    }

    /// Recursive step: execute script dengan self-learning
    pub fn step(
        &mut self,
        script: &str,
        script_engine: &mut ScriptEngine,
        scope: &mut rhai::Scope,
        field: &mut crate::field::AgentField,
        workers: &mut DataWorkerField,
        messages: &mut MessageBus,
        env_grid: &mut crate::field::EnvironmentGrid,
        vector_mem: &mut crate::field::vector_memory::VectorMemory,
        vfs: &mut crate::vfs::VirtualFileSystem,
        spatial_grid: &mut crate::field::SpatialGrid,
        encoder: &mut crate::render::CanvasEncoder,
        config: &crate::field::KernelConfig,
        metrics: crate::telemetry::EngineMetrics,
    ) -> Result<String, String> {
        let start = crate::telemetry::Telemetry::now();

        self.mode = WorkerMode::Working;
        self.current_script = script.to_string();

        // Eval dengan meta engine (cached)
        let result = script_engine.eval_with_injected_scope(
            scope,
            script,
            field,
            workers,
            messages,
            env_grid,
            vector_mem,
            vfs,
            spatial_grid,
            encoder,
            config,
            metrics,
        );

        let elapsed = crate::telemetry::Telemetry::now() - start;
        self.exec_time_ms = elapsed;

        match result {
            Ok(output) => {
                self.last_result = output.clone();
                self.success_rate = self.success_rate * 0.9 + 1.0 * 0.1;
                self.retry_count = 0;
                self.mode = WorkerMode::Done;

                // Q-Learning: update weights jika success
                self.update_weights(1.0, elapsed as f32);

                Ok(output)
            }
            Err(e) => {
                self.success_rate = self.success_rate * 0.9 + 0.0 * 0.1;
                self.retry_count += 1;

                if self.retry_count >= self.max_retries {
                    self.mode = WorkerMode::Escalated;
                    // Kirim message ke Orchestrator minta bantuan
                    let msg = format!("Worker {} failed task {} after {} retries: {}",
                        self.idx, self.task_id, self.max_retries, e);
                    messages.send_message(
                        self.idx as u32,
                        BROADCAST_ID, // Broadcast ke orchestrator
                        1, // msg_type: ESCALATION
                        &msg,
                    );
                } else {
                    self.mode = WorkerMode::Error;
                }

                Err(e)
            }
        }
    }

    /// Q-Learning update weights (slot 4: speed, slot 5: success)
    fn update_weights(&mut self, reward: f32, exec_time: f32) {
        // Slot 4: speed weight (faster = higher)
        let speed_reward = (100.0 / (exec_time + 1.0)).min(1.0);
        self.weights[4] = crate::business::q_learning_update(
            self.weights[4] as f64,
            speed_reward as f64,
            1.0, // max_future_q
            self.learning_rate as f64,
            0.9, // discount
        ) as f32;

        // Slot 5: success weight
        self.weights[5] = crate::business::q_learning_update(
            self.weights[5] as f64,
            reward as f64,
            1.0,
            self.learning_rate as f64,
            0.9,
        ) as f32;
    }

    /// Self-modification: tweak script berdasar weights
    pub fn generate_improved_script(&self) -> String {
        // Analisis script current, inject learned parameters
        let mut improved = self.current_script.clone();

        // Contoh: replace hardcoded values dengan learned weights
        improved = improved.replace(
            "THRESHOLD_PLACEHOLDER",
            &format!("{:.2}", self.weights[5])
        );

        improved
    }
}
