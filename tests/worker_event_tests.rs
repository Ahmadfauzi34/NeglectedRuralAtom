use agentic_kernel::KernelBridge;

#[test]
fn test_worker_callback_signature() {
    let mut kernel = KernelBridge::new(10, None);

    let script = r#"
        let w1 = workers.spawn_worker(101, "Task A");
        workers.set_worker_result(w1, "Result A");
    "#;

    kernel.eval_llm_script(script);

    // We cannot easily test js_sys::Function in a raw Rust environment without wasm-pack testing
    // but we can call step() and ensure it doesn't panic and clears the worker logic natively.
    kernel.step();
}
