use serde::{Deserialize, Serialize};
use rhai::{Engine, Scope, Dynamic, CustomType};
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
}

impl GraphExecutor {
    pub fn new() -> Self {
        Self {
            context: GraphContext::new(),
        }
    }

    /// Evaluates a list of nodes as a Directed Graph using the provided pre-configured Rhai Engine.
    /// It shares the simulation state scopes (like AgentField) inherited from KernelBridge.
    pub fn run_graph(
        &mut self,
        engine: &Engine,
        base_scope: &mut Scope,
        nodes: Vec<ScriptNode>,
        start_node_id: &str,
    ) -> Result<String, String> {
        // Map nodes for O(1) traversal lookup
        let mut node_map: HashMap<String, ScriptNode> = HashMap::new();
        for node in nodes {
            node_map.insert(node.id.clone(), node);
        }

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

            // Execute the Rhai script for the current node
            let eval_result = engine.eval_with_scope::<Dynamic>(base_scope, &node.script);

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
