use rhai::{Dynamic, Engine, Scope, AST};
use seahash::SeaHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Telemetry untuk meta-optimization
#[derive(Default, Clone, Debug)]
pub struct MetaTelemetry {
    pub l3_hits: u64,
    pub misses: u64,
    pub avg_compile_us: u64,
    pub cache_memory_bytes: usize,
}

/// Semua field configurable - tidak hardcoded
#[derive(Clone, Debug)]
pub struct MetaEngineConfig {
    pub l3_max_global_asts: usize,
    pub l3_lru_evict_batch: usize,
    pub enable_telemetry: bool,
    pub max_script_length: usize,
}

impl Default for MetaEngineConfig {
    fn default() -> Self {
        Self {
            l3_max_global_asts: 512,
            l3_lru_evict_batch: 16,
            enable_telemetry: true,
            max_script_length: 50_000,
        }
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
            global_ast_pool: GlobalAstPool::new(config.l3_max_global_asts),
            telemetry: MetaTelemetry::default(),
            config,
            frame_counter: 0,
        }
    }

    pub fn tick(&mut self) {
        self.frame_counter += 1;
        self.global_ast_pool.current_frame = self.frame_counter;
        if self.global_ast_pool.asts.len() > self.config.l3_max_global_asts {
            self.global_ast_pool
                .evict_lru(self.config.l3_lru_evict_batch);
        }
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
