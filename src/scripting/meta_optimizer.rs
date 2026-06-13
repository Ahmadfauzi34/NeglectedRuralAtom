use rhai::{Dynamic, Engine, Scope, AST};
use seahash::SeaHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Layer 1: Agent Exact Cache Context Fingerprint
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ContextFingerprint {
    pub behavior_state: u32,
    pub pos_x_quantized: i32,
    pub pos_y_quantized: i32,
    pub health_quantized: i32,
}

impl ContextFingerprint {
    pub fn new(behavior: u32, x: f32, y: f32, health: f32) -> Self {
        // Quantize floats to rough integers for hashing stability
        Self {
            behavior_state: behavior,
            pos_x_quantized: (x / 10.0).floor() as i32,
            pos_y_quantized: (y / 10.0).floor() as i32,
            health_quantized: (health / 10.0).floor() as i32,
        }
    }
}

/// Cache key for L1
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ExactCacheKey {
    pub script_hash: u64,
    pub context_fingerprint: ContextFingerprint,
}

#[derive(Clone)]
pub struct CachedExecution {
    pub result: Dynamic,
    pub expires_at_frame: u64,
}

/// Telemetry untuk meta-optimization
#[derive(Default, Clone, Debug)]
pub struct MetaTelemetry {
    pub l1_hits: u64,
    pub l3_hits: u64,
    pub misses: u64,
    pub avg_compile_us: u64,
    pub cache_memory_bytes: usize,
}

/// Semua field configurable - tidak hardcoded
#[derive(Clone, Debug)]
pub struct MetaEngineConfig {
    pub l1_max_agent_caches: usize,
    pub l1_ttl_frames: u64,
    pub l3_max_global_asts: usize,
    pub l3_lru_evict_batch: usize,
    pub enable_telemetry: bool,
    pub max_script_length: usize,
}

impl Default for MetaEngineConfig {
    fn default() -> Self {
        Self {
            l1_max_agent_caches: 2048,
            l1_ttl_frames: 60, // approx 1 second at 60fps
            l3_max_global_asts: 512,
            l3_lru_evict_batch: 16,
            enable_telemetry: true,
            max_script_length: 50_000,
        }
    }
}

/// Layer 1: Agent Cache (Short-lived exact match cache)
pub struct AgentCache {
    entries: HashMap<ExactCacheKey, CachedExecution>,
    pub max_items: usize,
}

impl AgentCache {
    pub fn new(max_items: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_items,
        }
    }

    pub fn get(&self, key: &ExactCacheKey, current_frame: u64) -> Option<&Dynamic> {
        if let Some(cached) = self.entries.get(key) {
            if cached.expires_at_frame > current_frame {
                return Some(&cached.result);
            }
        }
        None
    }

    pub fn store(&mut self, key: ExactCacheKey, result: Dynamic, expires_at: u64) {
        if self.entries.len() >= self.max_items {
            // Simple clear to avoid complex LRU overhead for fast-changing L1 cache
            self.entries.clear();
        }
        self.entries.insert(
            key,
            CachedExecution {
                result,
                expires_at_frame: expires_at,
            },
        );
    }

    pub fn cleanup_expired(&mut self, current_frame: u64) {
        self.entries
            .retain(|_, v| v.expires_at_frame > current_frame);
    }
}

/// Layer 3: Global AST pool (deduplicated compiled scripts)
pub struct GlobalAstPool {
    // script_hash → rhai::Shared<AST>
    asts: HashMap<u64, rhai::Shared<AST>>,
    // LRU tracking: script_hash -> frame number
    last_accessed: HashMap<u64, u64>,
    pub max_items: usize,
    pub current_frame: u64,
}

impl GlobalAstPool {
    pub fn new(max_items: usize) -> Self {
        Self {
            asts: HashMap::new(),
            last_accessed: HashMap::new(),
            max_items,
            current_frame: 0,
        }
    }

    pub fn get(&mut self, script_hash: u64) -> Option<&rhai::Shared<AST>> {
        if self.asts.contains_key(&script_hash) {
            self.last_accessed.insert(script_hash, self.current_frame);
            self.asts.get(&script_hash)
        } else {
            None
        }
    }

    pub fn store(&mut self, script_hash: u64, ast: rhai::Shared<AST>) {
        self.asts.insert(script_hash, ast);
        self.last_accessed.insert(script_hash, self.current_frame);
    }

    pub fn evict_lru(&mut self, batch_size: usize) {
        if self.asts.len() <= self.max_items {
            return;
        }

        let mut entries: Vec<_> = self.last_accessed.iter().map(|(k, v)| (*k, *v)).collect();
        entries.sort_by_key(|(_, v)| *v);

        let to_remove = batch_size.min(entries.len());
        for (hash, _) in entries.into_iter().take(to_remove) {
            self.asts.remove(&hash);
            self.last_accessed.remove(&hash);
        }
    }
}

pub struct MetaScriptEngine {
    pub engine: Engine,
    pub agent_cache: AgentCache,
    pub global_ast_pool: GlobalAstPool,
    pub telemetry: MetaTelemetry,
    pub config: MetaEngineConfig,
    pub frame_counter: u64,
}

impl MetaScriptEngine {
    pub fn new(engine: Engine) -> Self {
        let config = MetaEngineConfig::default();
        Self {
            engine,
            agent_cache: AgentCache::new(config.l1_max_agent_caches),
            global_ast_pool: GlobalAstPool::new(config.l3_max_global_asts),
            telemetry: MetaTelemetry::default(),
            config,
            frame_counter: 0,
        }
    }

    pub fn tick(&mut self) {
        self.frame_counter += 1;
        self.global_ast_pool.current_frame = self.frame_counter;

        // L1 Cleanup (every 60 frames)
        if self.frame_counter % 60 == 0 {
            self.agent_cache.cleanup_expired(self.frame_counter);
        }

        // L3 LRU Eviction
        if self.global_ast_pool.asts.len() > self.config.l3_max_global_asts {
            self.global_ast_pool
                .evict_lru(self.config.l3_lru_evict_batch);
        }
    }

    pub fn eval_for_agent(
        &mut self,
        script: &str,
        scope: &mut Scope,
        behavior: u32,
        x: f32,
        y: f32,
        health: f32,
    ) -> Result<Dynamic, String> {
        let mut safe_script = script;
        if safe_script.len() > self.config.max_script_length {
            let mut end = self.config.max_script_length;
            while end > 0 && !safe_script.is_char_boundary(end) {
                end -= 1;
            }
            safe_script = &safe_script[..end];
        }

        let script_hash = {
            let mut h = SeaHasher::new();
            safe_script.hash(&mut h);
            h.finish()
        };

        let fingerprint = ContextFingerprint::new(behavior, x, y, health);
        let cache_key = ExactCacheKey {
            script_hash,
            context_fingerprint: fingerprint,
        };

        // L1 Cache Check (Exact Match)
        if let Some(result) = self.agent_cache.get(&cache_key, self.frame_counter) {
            if self.config.enable_telemetry {
                self.telemetry.l1_hits += 1;
            }
            return Ok(result.clone());
        }

        // L1 Miss -> Fallback to L3 Evaluation
        let result = self.eval_with_scope_hashed(safe_script, scope, script_hash)?;

        // Store in L1
        let expires = self.frame_counter + self.config.l1_ttl_frames;
        self.agent_cache.store(cache_key, result.clone(), expires);

        Ok(result)
    }

    pub fn eval_with_scope(&mut self, script: &str, scope: &mut Scope) -> Result<Dynamic, String> {
        let mut safe_script = script;
        if safe_script.len() > self.config.max_script_length {
            let mut end = self.config.max_script_length;
            while end > 0 && !safe_script.is_char_boundary(end) {
                end -= 1;
            }
            safe_script = &safe_script[..end];
        }

        let script_hash = {
            let mut h = SeaHasher::new();
            safe_script.hash(&mut h);
            h.finish()
        };

        self.eval_with_scope_hashed(safe_script, scope, script_hash)
    }

    fn eval_with_scope_hashed(
        &mut self,
        safe_script: &str,
        scope: &mut Scope,
        script_hash: u64,
    ) -> Result<Dynamic, String> {
        // L3 Cache Check
        if let Some(ast) = self.global_ast_pool.get(script_hash) {
            if self.config.enable_telemetry {
                self.telemetry.l3_hits += 1;
            }
            return self
                .engine
                .eval_ast_with_scope::<Dynamic>(scope, ast)
                .map_err(|e| format!("L3 Eval Error: {e}"));
        }

        // Cache Miss -> Compile
        if self.config.enable_telemetry {
            self.telemetry.misses += 1;
        }

        let ast = rhai::Shared::new(
            self.engine
                .compile(safe_script)
                .map_err(|e| format!("Compile Error: {e}"))?,
        );

        let result = self
            .engine
            .eval_ast_with_scope::<Dynamic>(scope, &ast)
            .map_err(|e| format!("Eval Error: {e}"))?;

        self.global_ast_pool.store(script_hash, ast);

        Ok(result)
    }
}
