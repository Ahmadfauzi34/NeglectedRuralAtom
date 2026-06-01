use rhai::{CustomType, Dynamic, Engine, Scope, AST};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a single execution step (node) in the LLM's dynamic logic graph.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScriptNode {
    pub id: String,
    pub name: String,
    pub script: String,
    // The IDs of the next nodes to execute.
    // If there are multiple, they are spawned in parallel via Data Worker Swarm.
    #[serde(default)]
    pub next: Vec<String>,
}

/// A highly optimized wrapper for `ScriptNode` that caches the compiled AST.
/// This prevents Rhai from recompiling script texts every time the graph is executed.
pub struct CompiledNode {
    pub id: String,
    pub name: String,
    pub ast: AST,
    pub next: Vec<String>,
}

/// The Shared Context allowing Stateful Intelligence across graph nodes.
/// Node 1 can save data here, and Node 5 can retrieve it.
#[derive(Clone, CustomType)]
pub struct GraphContext {
    // We use Rhai's dynamic types to store arbitrary structured data between nodes.
    // Wrapped internally in an `Arc<RwLock>` if async mutation was required,
    // but synchronous linear execution is fine with standard ownership.
    memory: HashMap<String, Dynamic>,
}

impl GraphContext {
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
        }
    }

    pub fn get_var(&mut self, key: &str) -> Dynamic {
        self.memory.get(key).cloned().unwrap_or(Dynamic::UNIT)
    }

    pub fn set_var(&mut self, key: &str, val: Dynamic) {
        // Anti-memory-leak: Prevent unbound accumulation of variables in shared context
        if self.memory.len() >= 256 {
            return; // Context memory full, ignore new variables
        }

        let mut safe_key = key.to_string();
        if safe_key.len() > 128 {
            let mut end = 128;
            while end > 0 && !safe_key.is_char_boundary(end) {
                end -= 1;
            }
            safe_key.truncate(end);
        }

        self.memory.insert(safe_key, val);
    }

    pub fn clear(&mut self) {
        self.memory.clear();
    }
}

/// The Execution Engine for parsing and traversing JSON Node Graphs dynamically.
pub struct GraphExecutor {
    pub context: GraphContext,

    // Cached compiled nodes mapped by ID.
    // If the Javascript Host sends the exact same graph layout, we skip recompilation.
    #[allow(dead_code)]
    cached_graphs: HashMap<String, HashMap<String, CompiledNode>>,
}

impl GraphExecutor {
    pub fn new() -> Self {
        Self {
            context: GraphContext::new(),
            cached_graphs: HashMap::new(),
        }
    }

    /// Pre-compiles the textual scripts inside the `ScriptNode` list into a cached AST map.
    fn compile_nodes(
        engine: &Engine,
        nodes: Vec<ScriptNode>,
    ) -> Result<HashMap<String, CompiledNode>, String> {
        let mut compiled_map = HashMap::with_capacity(nodes.len());

        for node in nodes {
            let ast = engine
                .compile(&node.script)
                .map_err(|e| format!("Failed to compile node '{}': {}", node.id, e))?;
            compiled_map.insert(
                node.id.clone(),
                CompiledNode {
                    id: node.id,
                    name: node.name,
                    ast,
                    next: node.next,
                },
            );
        }

        Ok(compiled_map)
    }

    /// Evaluates a list of nodes as a Directed Graph using the provided pre-configured Rhai Engine.
    /// Automatically caches the compiled AST of the graph by hashing the graph IDs and script lengths.
    pub fn run_graph(
        &mut self,
        engine: &Engine,
        base_scope: &mut Scope,
        nodes: Vec<ScriptNode>,
        start_node_id: &str,
    ) -> Result<String, String> {
        // 1. Generate a lightweight structural hash for caching
        let mut graph_hash = String::with_capacity(nodes.len() * 16);
        for n in &nodes {
            graph_hash.push_str(&n.id);
            graph_hash.push_str(&n.script.len().to_string());
        }

        // 2. Compile if not cached
        if !self.cached_graphs.contains_key(&graph_hash) {
            let compiled = Self::compile_nodes(engine, nodes)?;
            self.cached_graphs.insert(graph_hash.clone(), compiled);
        }

        // 3. Fetch from cache
        let Some(node_map) = self.cached_graphs.get(&graph_hash) else {
            return Err("Failed to retrieve cached graph".to_string());
        };

        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start_node_id.to_string(), Dynamic::UNIT));

        let mut final_result = Dynamic::UNIT;
        let mut execution_steps = 0;
        let max_steps = 50_000; // Expanded to safely allow for deep autonomous while loops/eval cycles

        // Push our GraphContext into the scope to act as global memory
        base_scope.push("graph_ctx", self.context.clone());

        // Process queue (Breadth-First for Parallel conceptual execution)
        while let Some((current_node_id, incoming_result)) = queue.pop_front() {
            if execution_steps > max_steps {
                return Err("Graph execution exceeded max steps (Infinite Loop?)".to_string());
            }
            execution_steps += 1;

            if let Some(node) = node_map.get(&current_node_id) {
                // Inject the result from the previous node into the scope
                base_scope.set_or_push("previous_result", incoming_result);

                // Execute the highly optimized Rhai AST for the current node
                let eval_result = engine.eval_ast_with_scope::<Dynamic>(base_scope, &node.ast);

                match eval_result {
                    Ok(res) => {
                        final_result = res.clone();

                        // Branching Logic:
                        // If the script returns a string that explicitly matches another node's ID, branch to it.
                        // Otherwise, spawn ALL `next` nodes in parallel to the execution queue.
                        if let Ok(branch_id) = res.clone().into_string() {
                            if node_map.contains_key(&branch_id) {
                                queue.push_back((branch_id, res.clone()));
                                continue;
                            }
                        }

                        // Distribute parallel branches to the queue
                        for next_id in &node.next {
                            queue.push_back((next_id.clone(), res.clone()));
                        }
                    }
                    Err(e) => {
                        return Err(format!(
                            "ERROR in node '{}' ({}): {}",
                            node.id, node.name, e
                        ));
                    }
                }
            }
        }

        // Save the manipulated context back into self for future queries
        if let Some(ctx) = base_scope.get_value::<GraphContext>("graph_ctx") {
            self.context = ctx;
        }

        Ok(final_result.to_string())
    }
}
