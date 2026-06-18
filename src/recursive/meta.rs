//! Meta Agent — Layer 3
//!
//! Recursive capability:
//! - Rewrite own genome (script template) berdasar fitness
//! - Spawn child agents dengan evolved parameters
//! - Evaluate population, kill underperformers
//! - Compress learned knowledge ke VFS

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::field::{AgentField, DataWorkerField, MessageBus};
use crate::recursive::orchestrator::OrchestratorAgent;
use crate::vfs::VirtualFileSystem;

/// "Genome" = template script + learned parameters
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentGenome {
    pub template: String,
    pub weights: [f32; 8],
    pub fitness: f32,
    pub generation: u32,
    pub birth_frame: u64,
}

pub struct MetaAgent {
    pub population: Vec<OrchestratorAgent>,
    pub genomes: HashMap<u32, AgentGenome>,

    // Evolution parameters
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub selection_pressure: f32,
    pub max_population: usize,

    // Current generation
    pub generation: u32,
    pub frame: u64,

    // VFS untuk persist learned genomes
    pub vfs_ptr: *mut VirtualFileSystem,
}

impl MetaAgent {
    pub fn new(max_pop: usize, vfs: &mut VirtualFileSystem) -> Self {
        Self {
            population: Vec::with_capacity(max_pop),
            genomes: HashMap::new(),
            mutation_rate: 0.1,
            crossover_rate: 0.3,
            selection_pressure: 2.0,
            max_population: max_pop,
            generation: 0,
            frame: 0,
            vfs_ptr: std::ptr::from_mut::<VirtualFileSystem>(vfs),
        }
    }

    #[inline]
    fn get_vfs(&mut self) -> &mut VirtualFileSystem {
        unsafe { &mut *self.vfs_ptr }
    }

    /// Spawn new agent dari genome
    pub fn spawn_from_genome(&mut self, genome_id: u32) {
        if let Some(genome) = self.genomes.get(&genome_id).cloned() {
            let mut orch = OrchestratorAgent::new(8); // Default 8 workers per orchestrator

            for worker in &mut orch.workers {
                worker.weights = genome.weights;
            }

            self.population.push(orch);
        }
    }

    /// Evolution step: evaluate, select, mutate, spawn
    pub fn evolve(
        &mut self,
        script_engine: &mut crate::scripting::ScriptEngine,
        scope: &mut rhai::Scope,
        field: &mut AgentField,
        workers: &mut DataWorkerField,
        messages: &mut MessageBus,
        env_grid: &mut crate::field::EnvironmentGrid,
        vector_mem: &mut crate::field::vector_memory::VectorMemory,
        spatial_grid: &mut crate::field::SpatialGrid,
        encoder: &mut crate::render::CanvasEncoder,
        config: &crate::field::KernelConfig,
        metrics: crate::telemetry::EngineMetrics,
    ) {
        self.frame += 1;

        // 1. Evaluate fitness
        self.evaluate_fitness();

        // 2. Selection: keep top performers
        self.select();

        // 3. Breed + Mutate (Simplified)
        if self.population.len() < self.max_population / 2 && self.genomes.len() >= 2 {
            self.breed_new_genomes();
        }

        // 4. Spawn to refill population
        while self.population.len() < self.max_population {
            if let Some(best_id) = self.get_best_genome_id() {
                self.spawn_from_genome(best_id);
            } else {
                break;
            }
        }

        // 5. Step semua population
        // Safety: Raw pointer access to VFS to bypass borrow checker constraints with self.population
        let vfs = unsafe { &mut *self.vfs_ptr };
        for orch in &mut self.population {
            orch.step(
                script_engine, scope, field, workers, messages,
                env_grid, vector_mem, vfs, spatial_grid, encoder, config, metrics
            );
        }
    }

    fn evaluate_fitness(&mut self) {
        for (i, orch) in self.population.iter().enumerate() {
            let mut total_success = 0.0;
            let mut total_time = 0.0;

            for worker in &orch.workers {
                total_success += worker.success_rate;
                total_time += worker.exec_time_ms;
            }

            let avg_success = total_success / orch.workers.len().max(1) as f32;
            let avg_time = total_time / orch.workers.len().max(1) as f64;

            let fitness = if avg_time > 0.0 {
                (avg_success * 100.0) / (avg_time as f32 + 1.0)
            } else {
                avg_success
            };

            if let Some(genome) = self.genomes.get_mut(&(i as u32)) {
                genome.fitness = genome.fitness * 0.7 + fitness * 0.3;
            }
        }
    }

    fn select(&mut self) {
        let mut indexed: Vec<(usize, f32)> = self.population.iter().enumerate()
            .map(|(i, _)| {
                let fitness = self.genomes.get(&(i as u32))
                    .map(|g| g.fitness)
                    .unwrap_or(0.0);
                (i, fitness)
            })
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let keep_count = ((self.population.len() as f32) / self.selection_pressure).ceil() as usize;
        let keep_ids: Vec<usize> = indexed.into_iter().take(keep_count).map(|(i, _)| i).collect();

        let mut new_genomes: HashMap<u32, AgentGenome> = HashMap::new();

        for (new_idx, &old_idx) in keep_ids.iter().enumerate() {
            if old_idx < self.population.len() {
                // Simplified move logic
                // new_pop.push(self.population[old_idx].clone());
                if let Some(genome) = self.genomes.get(&(old_idx as u32)) {
                    new_genomes.insert(new_idx as u32, genome.clone());
                }
            }
        }

        // self.population = new_pop;
        self.genomes = new_genomes;
    }

    fn breed_new_genomes(&mut self) {
        let genome_ids: Vec<u32> = self.genomes.keys().copied().collect();
        if genome_ids.len() < 2 { return; }

        let parent_a = self.genomes.get(&genome_ids[0]).unwrap();
        let parent_b = self.genomes.get(&genome_ids[1]).unwrap();

        let mut child_weights = [0.0f32; 8];
        for i in 0..8 {
            child_weights[i] = (parent_a.weights[i] + parent_b.weights[i]) / 2.0;
            // Simplified mutation without external rand crate for now
            if (self.frame + i as u64) % 10 == 0 {
                child_weights[i] += 0.1;
            }
        }

        let child = AgentGenome {
            template: parent_a.template.clone(),
            weights: child_weights,
            fitness: 0.0,
            generation: self.generation,
            birth_frame: self.frame,
        };

        let new_id = self.genomes.len() as u32;
        self.genomes.insert(new_id, child);
    }

    fn get_best_genome_id(&self) -> Option<u32> {
        self.genomes.iter()
            .max_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| *k)
    }

    pub fn save_genome(&mut self, genome_id: u32, path: &str) {
        if let Some(genome) = self.genomes.get(&genome_id) {
            if let Ok(json) = serde_json::to_string(genome) {
                self.get_vfs().write_file(path, &json);
            }
        }
    }
}
