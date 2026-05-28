use agentic_kernel::KernelBridge;

#[test]
fn test_agent_autonomy_q_learning() {
    let mut kernel = KernelBridge::new(10);

    // Call the spawn method that works without JsValue easily
    // We can just use rhai to spawn it via KernelBridge eval
    let spawn_script = "workers.spawn_worker(1, \"Initial Payload\");";
    kernel.eval_llm_script(spawn_script);

    // Rhai script to simulate an autonomous Q-learning step
    // Slot 0 = q_current, Slot 1 = best_future_q
    let script = r#"
        let idx = 0; // Our spawned worker is at index 0

        // Let's pretend the worker observed a reward of 10.0
        let reward = 10.0;
        let lr = 0.1;
        let discount = 0.9;

        let q_current = workers.get_worker_memory(idx, 0);
        let max_future_q = workers.get_worker_memory(idx, 1);

        let new_q = q_learning_update(q_current, reward, max_future_q, lr, discount);

        // Save the learned Q-value back to the agent's brain
        workers.set_worker_memory(idx, 0, new_q);

        new_q
    "#;

    let result = kernel.eval_llm_script(script);

    // The initial q_current is 0.0, max_future_q is 0.0.
    // new_q = 0.0 + 0.1 * (10.0 + 0.9 * 0.0 - 0.0) = 1.0
    assert_eq!(result, "1.0");
}

#[test]
fn test_agent_autonomy_neural_forward_pass() {
    let mut kernel = KernelBridge::new(10);

    let script = r#"
        let inputs = [0.5, 0.8, -0.2];
        let weights = [0.1, 0.5, 0.9];

        // Dot product: (0.5 * 0.1) + (0.8 * 0.5) + (-0.2 * 0.9)
        // 0.05 + 0.4 - 0.18 = 0.27
        let dot = dot_product(inputs, weights);

        // Pass through sigmoid activation
        let prediction = sigmoid(dot);

        prediction
    "#;

    let result = kernel.eval_llm_script(script);

    // Check if result is a valid number string close to sigmoid(0.27) = ~0.567
    let pred: f64 = result.parse().unwrap();
    assert!(pred > 0.56 && pred < 0.57);
}
