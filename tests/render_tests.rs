use agentic_kernel::KernelBridge;

#[test]
fn test_generalized_rendering_via_rhai() {
    let mut kernel = KernelBridge::new(10);

    // Test the logic using the object syntax for custom types.
    // In Rhai, the context was pushed as "canvas". So it's `canvas.canvas_clear()` or `canvas.clear()`
    // depending on how we registered it. We registered it as a method on RenderContext.
    // Let's use `canvas.canvas_clear()` because that's the fn name we registered.

    let script = r#"
        canvas.canvas_clear();
        canvas.canvas_rect(10.0, 10.0, 100.0, 50.0, 4278255360);
        canvas.canvas_circle(50.0, 50.0, 25.0, 4294901760);
        canvas.canvas_line(0.0, 0.0, 100.0, 100.0, 4294967295);

        "render_complete"
    "#;

    let result = kernel.eval_llm_script(script);
    println!("{}", result);
    assert_eq!(result, "render_complete");

    // Step to finalize
    kernel.step();

    // The render length shouldn't be 0 since we drew things.
    let len = kernel.render_len();
    assert!(len > 0, "Canvas encoding buffer is empty, expected drawing commands!");
}
