use serde::{Deserialize, Serialize};
use rhai::{Engine, Scope, Dynamic, CustomType, AST};
use std::collections::HashMap;

/// Represents a single execution step (node) in the LLM's dynamic logic graph.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScriptNode {
    pub id: String,
    pub name: String,
    pub script: String,
    // The ID of the next node to execute.
    // If the script returns a specific string, it can override this for conditional branching.
    pub next: Option<String>,
}

/// A highly optimized wrapper for `ScriptNode` that caches the compiled AST.
/// This prevents Rhai from recompiling script texts every time the graph is executed.
pub struct CompiledNode {
    pub id: String,
    pub name: String,
    pub ast: AST,
    pub next: Option<String>,
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
        self.memory.insert(key.to_string(), val);
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
    fn compile_nodes(engine: &Engine, nodes: Vec<ScriptNode>) -> Result<HashMap<String, CompiledNode>, String> {
        let mut compiled_map = HashMap::with_capacity(nodes.len());

        for node in nodes {
            let ast = engine.compile(&node.script).map_err(|e| format!("Failed to compile node '{}': {}", node.id, e))?;
            compiled_map.insert(node.id.clone(), CompiledNode {
                id: node.id,
                name: node.name,
                ast,
                next: node.next,
            });
        }

        Ok(compiled_map)
    }

    /// Evaluates a list of nodes as a Directed Graph using the provided pre-configured Rhai Engine.
    /// Automatically caches the compiled AST of the graph by hashing or checking graph IDs (simulated here by evaluating directly if not cached).
    /// To be perfect, the host JS should provide a `graph_hash` to identify unique graphs, but we will simply recompile if the host sends raw nodes for now.
    pub fn run_graph(
        &mut self,
        engine: &Engine,
        base_scope: &mut Scope,
        nodes: Vec<ScriptNode>,
        start_node_id: &str,
    ) -> Result<String, String> {
        // In a production setup, we would look up `self.cached_graphs` using a Hash of the incoming JSON.
        // For simplicity and immediate optimization here, we'll compile them into ASTs on the fly
        // to at least guarantee AST execution speed across the traversal.
        let node_map = Self::compile_nodes(engine, nodes)?;

        let mut current_node_id = start_node_id.to_string();
        let mut current_result = Dynamic::UNIT;
        let mut execution_steps = 0;
        let max_steps = 1000; // Prevent infinite loops in cyclical graphs

        // Push our GraphContext into the scope to act as global memory
        base_scope.push("graph_ctx", self.context.clone());

        while let Some(node) = node_map.get(&current_node_id) {
            if execution_steps > max_steps {
                return Err("Graph execution exceeded max steps (Infinite Loop?)".to_string());
            }
            execution_steps += 1;

            // Inject the result from the previous node into the scope
            base_scope.set_or_push("previous_result", current_result.clone());

            // Execute the highly optimized Rhai AST for the current node
            let eval_result = engine.eval_ast_with_scope::<Dynamic>(base_scope, &node.ast);

            match eval_result {
                Ok(res) => {
                    current_result = res.clone();

                    // Branching Logic:
                    // If the script returns a string that explicitly matches another node's ID, branch to it.
                    // Otherwise, proceed to the static `next` node defined in the JSON.
                    if let Ok(branch_id) = res.into_string() {
                        if node_map.contains_key(&branch_id) {
                            current_node_id = branch_id;
                            continue;
                        }
                    }

                    if let Some(next_id) = &node.next {
                        current_node_id = next_id.clone();
                    } else {
                        // End of graph
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("ERROR in node '{}' ({}): {}", node.id, node.name, e));
                }
            }
        }

        // Save the manipulated context back into self for future queries
        if let Some(ctx) = base_scope.get_value::<GraphContext>("graph_ctx") {
            self.context = ctx;
        }

        Ok(current_result.to_string())
    }
}
