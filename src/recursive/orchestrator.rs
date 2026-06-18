//! Orchestrator Agent — Layer 2
//!
//! Recursive capability:
//! - Decompose task → sub-tasks → spawn workers
//! - Monitor via MessageBus, handle ESCALATION
//! - Retry dengan modified strategy (fork)
//! - Reduce results → aggregate

use std::collections::HashMap;

use crate::field::{DataWorkerField, MessageBus, BROADCAST_ID};
use crate::recursive::worker::{WorkerAgent, WorkerMode};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TaskStatus {
    Pending,
    Assigned,
    Running,
    Done,
    Failed,
    Forked,     // Recursive: di-fork ke multiple strategies
}

/// Strategy retry berbeda untuk recursive recovery
#[derive(Clone, Debug, PartialEq)]
pub enum RetryStrategy {
    Simple,         // Retry sama
    Modified,       // Script dengan tweak
    Forked(Vec<String>), // Multiple variants paralel
    Escalated,      // Naik ke MetaAgent
}

/// Task decomposition tree node
pub struct TaskNode {
    pub id: u32,
    pub parent_id: Option<u32>,
    pub script: String,
    pub sub_tasks: Vec<u32>,
    pub assigned_worker: Option<usize>,
    pub status: TaskStatus,
    pub result: String,
    pub retry_strategy: RetryStrategy,
    pub retry_count: u8,
}

pub struct OrchestratorAgent {
    pub task_counter: u32,
    pub task_tree: HashMap<u32, TaskNode>,
    pub workers: Vec<WorkerAgent>,

    // Configurable
    pub max_parallel_workers: usize,
    pub fork_threshold: f32,       // Success rate di bawah ini → fork
    pub escalation_threshold: u8,  // Retry count → escalate
}

impl OrchestratorAgent {
    pub fn new(max_workers: usize) -> Self {
        Self {
            task_counter: 0,
            task_tree: HashMap::new(),
            workers: Vec::with_capacity(max_workers),
            max_parallel_workers: max_workers,
            fork_threshold: 0.3,
            escalation_threshold: 3,
        }
    }

    /// Recursive decompose: pecah task besar jadi tree
    pub fn decompose(&mut self, root_script: &str, depth: u8) -> u32 {
        let id = self.task_counter;
        self.task_counter += 1;

        let node = TaskNode {
            id,
            parent_id: None,
            script: root_script.to_string(),
            sub_tasks: Vec::new(),
            assigned_worker: None,
            status: TaskStatus::Pending,
            result: String::new(),
            retry_strategy: RetryStrategy::Simple,
            retry_count: 0,
        };

        self.task_tree.insert(id, node);

        // Recursive decomposition jika depth > 0
        if depth > 0 {
            let sub_scripts = self.split_task(root_script);
            for sub in sub_scripts {
                let sub_id = self.decompose(&sub, depth - 1);
                if let Some(n) = self.task_tree.get_mut(&id) {
                    n.sub_tasks.push(sub_id);
                }
                if let Some(sub_node) = self.task_tree.get_mut(&sub_id) {
                    sub_node.parent_id = Some(id);
                }
            }
        }

        id
    }

    /// Split task via heuristik
    fn split_task(&self, script: &str) -> Vec<String> {
        let mut chunks = Vec::new();

        // Contoh dekomposisi sederhana: split by agent count ranges
        if script.contains("get_count()") {
            chunks.push(script.replace("get_count()", "(get_count() / 2)"));
            chunks.push(script.replace("get_count()", "(get_count() / 2)")); // Simplified range logic
        } else {
            chunks.push(script.to_string());
        }

        chunks
    }

    /// Step: assign, monitor, handle escalation
    pub fn step(
        &mut self,
        script_engine: &mut crate::scripting::ScriptEngine,
        scope: &mut rhai::Scope,
        field: &mut crate::field::AgentField,
        workers_field: &mut DataWorkerField,
        messages: &mut MessageBus,
        env_grid: &mut crate::field::EnvironmentGrid,
        vector_mem: &mut crate::field::vector_memory::VectorMemory,
        vfs: &mut crate::vfs::VirtualFileSystem,
        spatial_grid: &mut crate::field::SpatialGrid,
        encoder: &mut crate::render::CanvasEncoder,
        config: &crate::field::KernelConfig,
        metrics: crate::telemetry::EngineMetrics,
    ) {
        // 1. Check MessageBus untuk escalation
        self.handle_escalations(messages);

        // 2. Assign pending tasks ke idle workers
        self.assign_pending_tasks(
            script_engine, scope, field, workers_field, messages,
            env_grid, vector_mem, vfs, spatial_grid, encoder, config, metrics
        );

        // 3. Monitor running tasks, handle failures
        self.monitor_running_tasks(workers_field);

        // 4. Reduce completed sub-tasks ke parent
        self.reduce_completed();
    }

    fn handle_escalations(&mut self, messages: &mut MessageBus) {
        let mut escalations = Vec::new();

        let mut indices = Vec::new();
        messages.query_messages(BROADCAST_ID, &mut indices);

        for &idx in &indices {
            if messages.message_types[idx] == 1 { // msg_type: ESCALATION
                let sender = messages.sender_ids[idx];
                escalations.push(sender);
            }
        }

        for sender in escalations {
            let mut variants = None;
            let worker_idx = sender as usize;

            if let Some(worker) = self.workers.iter_mut().find(|w| w.idx == worker_idx) {
                if worker.success_rate < self.fork_threshold {
                    let current = worker.current_script.clone();
                    let improved = worker.generate_improved_script();
                    let alternative = format!("// Alternative strategy\n{}\n// End alternative", current);

                    variants = Some(vec![current, improved, alternative]);

                    worker.mode = WorkerMode::Idle;
                    worker.retry_count = 0;
                }
            }

            if let Some(vars) = variants {
                for node in self.task_tree.values_mut() {
                    if node.assigned_worker == Some(worker_idx) {
                        node.retry_strategy = RetryStrategy::Forked(vars);
                        node.status = TaskStatus::Forked;
                        break;
                    }
                }
            }
        }

        messages.clear();
    }

    fn assign_pending_tasks(
        &mut self,
        script_engine: &mut crate::scripting::ScriptEngine,
        scope: &mut rhai::Scope,
        field: &mut crate::field::AgentField,
        workers_field: &mut DataWorkerField,
        messages: &mut MessageBus,
        env_grid: &mut crate::field::EnvironmentGrid,
        vector_mem: &mut crate::field::vector_memory::VectorMemory,
        vfs: &mut crate::vfs::VirtualFileSystem,
        spatial_grid: &mut crate::field::SpatialGrid,
        encoder: &mut crate::render::CanvasEncoder,
        config: &crate::field::KernelConfig,
        metrics: crate::telemetry::EngineMetrics,
    ) {
        let mut tasks_to_assign = Vec::new();
        for (id, node) in &self.task_tree {
            if node.status == TaskStatus::Pending {
                tasks_to_assign.push(*id);
            }
        }

        for task_id in tasks_to_assign {
            if let Some(worker) = self.workers.iter_mut().find(|w| w.mode == WorkerMode::Idle) {
                if let Some(node) = self.task_tree.get_mut(&task_id) {
                    node.status = TaskStatus::Assigned;
                    node.assigned_worker = Some(worker.idx);

                    let payload = format!("Task {}: {}", task_id, node.script);
                    workers_field.spawn_worker(task_id, &payload);

                    node.status = TaskStatus::Running;
                    let _ = worker.step(
                        &node.script, script_engine, scope, field, workers_field, messages,
                        env_grid, vector_mem, vfs, spatial_grid, encoder, config, metrics
                    );
                }
            }
        }
    }


    fn monitor_running_tasks(&mut self, workers_field: &DataWorkerField) {
        for node in self.task_tree.values_mut() {
            if node.status != TaskStatus::Running { continue; }

            if let Some(worker_idx) = node.assigned_worker {
                if worker_idx < workers_field.capacity {
                    let state = workers_field.states[worker_idx];
                    match state {
                        2 => { // Done
                            node.status = TaskStatus::Done;
                        }
                        3 => { // Error
                            match &node.retry_strategy {
                                RetryStrategy::Forked(variants) => {
                                    if let Some(next) = variants.get(node.retry_count as usize) {
                                        node.script = next.clone();
                                        node.status = TaskStatus::Pending;
                                        node.retry_count += 1;
                                    } else {
                                        node.status = TaskStatus::Failed;
                                    }
                                }
                                _ => {
                                    node.status = TaskStatus::Failed;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn reduce_completed(&mut self) {
        let completed: Vec<u32> = self.task_tree.iter()
            .filter(|(_, n)| n.status == TaskStatus::Done && n.sub_tasks.is_empty())
            .map(|(id, _)| *id)
            .collect();

        for child_id in completed {
            if let Some(parent_id) = self.task_tree.get(&child_id).and_then(|n| n.parent_id) {
                let mut result_to_add = String::new();
                if let Some(child) = self.task_tree.get(&child_id) {
                    result_to_add = child.result.clone();
                }

                if let Some(parent) = self.task_tree.get_mut(&parent_id) {
                    parent.sub_tasks.retain(|&id| id != child_id);
                    parent.result.push_str(&result_to_add);
                    parent.result.push('\n');

                    if parent.sub_tasks.is_empty() {
                        parent.status = TaskStatus::Done;
                    }
                }
            }
        }
    }

    pub fn get_result(&self, root_id: u32) -> Option<&str> {
        self.task_tree.get(&root_id).map(|n| n.result.as_str())
    }
}
