use agentic_kernel::graph::{GraphContext, GraphExecutor, ScriptNode};
use rhai::{Engine, Scope};

#[test]
fn test_parallel_graph_execution() {
    let mut executor = GraphExecutor::new();
    let mut engine = Engine::new();

    // We must register the methods for GraphContext manually for testing since KernelBridge does it normally.
    // Wait, let's see how KernelBridge does it. Actually KernelBridge DOES NOT register GraphContext methods.
    // It is registered implicitly if we just build_type or if we use dynamic scope. Let's check `src/scripting.rs`.
    engine.build_type::<GraphContext>();
    engine.register_fn("set_var", GraphContext::set_var);
    engine.register_fn("get_var", GraphContext::get_var);

    let mut scope = Scope::new();

    // We create a graph where node1 branches to node2a and node2b simultaneously.
    let nodes = vec![
        ScriptNode {
            id: "node1".into(),
            name: "Start".into(),
            script: "graph_ctx.set_var(\"started\", true); \"success\"".into(),
            next: vec!["node2a".into(), "node2b".into()],
        },
        ScriptNode {
            id: "node2a".into(),
            name: "Branch A".into(),
            script: "graph_ctx.set_var(\"branch_a_done\", previous_result);".into(),
            next: vec![],
        },
        ScriptNode {
            id: "node2b".into(),
            name: "Branch B".into(),
            script: "graph_ctx.set_var(\"branch_b_done\", previous_result);".into(),
            next: vec![],
        },
    ];

    let result = executor.run_graph(&engine, &mut scope, nodes, "node1");
    assert!(result.is_ok(), "Graph execution failed: {:?}", result.err());

    // Verify that both branches updated the shared graph context memory
    let started = executor
        .context
        .get_var("started")
        .as_bool()
        .unwrap_or(false);
    assert!(started, "Node 1 did not execute");

    let branch_a_res = executor
        .context
        .get_var("branch_a_done")
        .into_string()
        .unwrap_or("".into());
    assert_eq!(
        branch_a_res, "success",
        "Branch A did not receive previous result from Node 1"
    );

    let branch_b_res = executor
        .context
        .get_var("branch_b_done")
        .into_string()
        .unwrap_or("".into());
    assert_eq!(
        branch_b_res, "success",
        "Branch B did not receive previous result from Node 1"
    );
}
