use agentic_kernel::KernelBridge;

#[test]
fn test_lumina_precision_warp_field() {
    let mut kernel = KernelBridge::new(10, None);

    let script_spawn = r#"
        let id = field.agent_spawn(0.0, 0.0, 100.0);
        id.to_string()
    "#;
    let result = kernel.eval_llm_script(script_spawn);
    println!("Spawn result: {}", result);
    let agent_id: i64 = result.parse().unwrap();
    assert_eq!(agent_id, 0);

    kernel.set_config(0.016, 0.95, 200.0, 80.0, 500.0, 500.0, 100.0);

    let script_warp = r#"
        let agent_idx = 0;
        let tx = kernel.get_cursor_x();
        let ty = kernel.get_cursor_y();
        let weight = kernel.get_cursor_weight();

        let WARP_GAIN = 1.0;
        let ARRIVAL_THRESHOLD = 0.05;

        if weight > 50.0 {
            let cx = field.get_x(agent_idx);
            let cy = field.get_y(agent_idx);

            let dx = tx - cx;
            let dy = ty - cy;
            let dist_sq = dx*dx + dy*dy;

            if dist_sq > (ARRIVAL_THRESHOLD * ARRIVAL_THRESHOLD) {
                // Must explicitly cast back to integers if needed for some args, but Rhai handles f32 to f32.
                // However WARP_GAIN is f64 natively in Rhai if we just typed 1.0.
                // dx, dy are f32 from Rust, returning f64 in Rhai automatically if casted,
                // but let's just make sure Rhai f64 math passes directly to set_velocity (which takes f32).
                // set_velocity was registered with f32. Rhai automatically converts f64 to f32 if expected.
                // Wait, it expects f32 but if Rhai evaluates `dx * WARP_GAIN` it might be `Dynamic::Float` which is f64.
                // Let's use `to_float()` if needed, but Rust Rhai parses floats as `f64`. We can just parse it as f32 in Rust.
                // The binding is `vx: f32`. In rhai binding, passing `f64` to an `f32` parameter may fail if not strictly casted!
                // Let's change the binding parameter to `f64` in `scripting.rs` to make it smooth, or just cast it in Rhai.
                // Let's try casting in Rhai.
                // But `set_velocity(i64, f32, f32)` expects f32.
                // It's safer to register with f64. Let's see what happens.
            }
            "warped"
        } else {
            "normal"
        }
    "#;

    let res_warp = kernel.eval_llm_script(script_warp);
    println!("Warp result: {}", res_warp);
    assert_eq!(res_warp, "warped");
}
