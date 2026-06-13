use agentic_kernel::KernelBridge;

#[test]
fn test_autonomous_while_loop_context_evolution() {
    let mut kernel = KernelBridge::new(10, None);

    // Orthogonal projection doesn't strictly converge to `broader_context` directly,
    // it converges the projection space.
    // To make a test that proves the loop works and updates agent_context over time,
    // let's just make it simple linear interpolation for learning if distance is far.
    // Wait, the test is supposed to verify the `while` loop capability in WASM works without crashing
    // when using high operation limits.
    // Let's use simple gradient descent to target.

    let script = r#"
        let agent_context = [0.0, 0.0, 0.0, 0.0];
        let broader_context = [1.0, -0.5, 0.8, 0.2];
        let lr = 0.2;

        let loop_count = 0;
        let comprehension = 0.0;

        while comprehension < 0.95 && loop_count < 100 {

            let dist_sq = 0.0;
            for i in 0..4 {
                let diff = broader_context[i] - agent_context[i];
                dist_sq += diff * diff;

                // Simple gradient update
                agent_context[i] = agent_context[i] + (diff * lr);
            }

            comprehension = 1.0 / (1.0 + dist_sq);
            loop_count += 1;
        }

        loop_count.to_string()
    "#;

    let result = kernel.eval_llm_script(script);
    println!("Rhai Result: {}", result);
    let loop_count: i64 = result.parse().unwrap();
    println!("Converged in {} loops", loop_count);
    assert!(loop_count > 1);
    assert!(loop_count < 100);
}
