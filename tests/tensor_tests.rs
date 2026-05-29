use agentic_kernel::KernelBridge;

#[test]
fn test_asymmetric_trm_components() {
    let mut kernel = KernelBridge::new(10);

    // Oh, Rhai parser doesn't accept "1_i64". You just write "1".
    // The previous error "Syntax error: Expecting ; to terminate this statement" on `let mut spectral = spectral_core(16, 4)` is NOT about `16_i64`, it is about `let mut spectral`.
    // Rhai does not use `mut` in `let`. Everything is mutable by default unless declared as `const`.
    // This is why `let mut` was throwing a syntax error!

    let script = r#"
        let y_tensor = tensor_zeros(1, 8, 16);
        let z_tensor = tensor_zeros(1, 8, 16);

        let core_spec = spectral_core(16, 4);
        let spec_out = forward_sparse(core_spec, z_tensor, 1.0, 0.05);

        let bridge_comp = zero_param_bridge(16, 4);
        let translated_z = forward(bridge_comp, y_tensor, spec_out);

        let fusion_comp = orthogonal_fusion(0.02);
        let stream = tensor_zeros(1, 8, 16);
        let fused_tensor = fuse_sparse(fusion_comp, stream, translated_z);

        "success"
    "#;

    let result = kernel.eval_llm_script(script);
    println!("Rhai evaluation result: {}", result);
    assert_eq!(result, "success");
}
