import init, { KernelBridge, MemoryView } from './pkg/agentic_kernel.js';

let kernel;
let canvas;
let ctx;
let isWasmLoaded = false;
let mouseX = 400;
let mouseY = 300;
let gravityWeight = 0.5;

async function run() {
    // 1. Initialize WASM module
    const wasm = await init();
    console.log("WASM Initialized!");
    isWasmLoaded = true;

    // 2. Instantiate KernelBridge
    // Using 5000 as max agent capacity
    kernel = KernelBridge.new(5000);
    kernel.set_render_mode(false); // False = use CPU CanvasEncoder bytes

    // 3. Setup Canvas Context
    canvas = document.getElementById('sim-canvas');
    ctx = canvas.getContext('2d');

    // Setup UI Controls
    setupUI();

    // Spawn initial agents via Batch JSON command
    spawnInitialSwarm();

    // 4. Start Animation Loop
    requestAnimationFrame(renderLoop);
}

function setupUI() {
    document.getElementById('btn-spawn').addEventListener('click', () => {
        let cmds = [];
        for(let i=0; i<100; i++) {
            cmds.push({
                cmd: "spawn",
                x: Math.random() * 800,
                y: Math.random() * 600,
                vx: (Math.random() - 0.5) * 100,
                vy: (Math.random() - 0.5) * 100,
                health: 100.0,
                color: 0xFF00FF00 // Unused in Rust currently, but format matches
            });
        }
        kernel.execute_batch(JSON.stringify(cmds));
    });

    document.getElementById('btn-spawn-predator').addEventListener('click', () => {
        // We spawn manually and then use Rhai to set it as a Predator (State 4)
        let script = `
            let idx = field.spawn(${mouseX}.0, ${mouseY}.0, 100.0);
            field.set_behavior(idx, 4); // 4 = Predator
        `;
        kernel.eval_llm_script(script);
    });

    document.getElementById('btn-spawn-prey').addEventListener('click', () => {
        let script = `
            let idx = field.spawn(${mouseX}.0, ${mouseY}.0, 100.0);
            field.set_behavior(idx, 5); // 5 = Prey
        `;
        kernel.eval_llm_script(script);
    });

    document.getElementById('btn-clear').addEventListener('click', () => {
        kernel.execute_command(JSON.stringify({ cmd: "clear" }));
    });

    document.getElementById('btn-rhai').addEventListener('click', () => {
        // This script spawns 10 agents in a circle using the embedded Rhai engine in WASM!
        const script = `
            let center_x = 400.0;
            let center_y = 300.0;
            let radius = 100.0;
            for i in 0..10 {
                let angle = (i.to_float() / 10.0) * 6.28318;
                let px = center_x + angle.cos() * radius;
                let py = center_y + angle.sin() * radius;
                let idx = field.spawn(px, py, 100.0);
                field.set_behavior(idx, 2); // 2 = Wander
            }
            "Successfully spawned 10 wanderers via Rhai!"
        `;
        let res = kernel.eval_llm_script(script);
        console.log("Rhai Response:", res);
    });

    document.getElementById('gravity-slider').addEventListener('input', (e) => {
        gravityWeight = parseFloat(e.target.value);
    });

    // Track mouse for gravity
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
    });
}

function spawnInitialSwarm() {
    let cmds = [];
    for(let i=0; i<300; i++) {
        cmds.push({
            cmd: "spawn",
            x: Math.random() * 800,
            y: Math.random() * 600,
            vx: (Math.random() - 0.5) * 50,
            vy: (Math.random() - 0.5) * 50,
            health: 100.0,
            color: 0xFF6366F1
        });
    }
    kernel.execute_batch(JSON.stringify(cmds));
}

// Map from rendering tags defined in Rust
const TAG_CIRCLE = 0;
const TAG_LINE = 1;

function renderLoop() {
    if (!isWasmLoaded) return;

    // 1. Update Kernel Config with Mouse coordinates
    // dt, friction, max_speed, influence_radius, cursor_x, cursor_y, cursor_weight
    kernel.set_config(0.016, 0.95, 200.0, 80.0, mouseX, mouseY, gravityWeight);

    // 2. Step the simulation physics hot loop inside WASM!
    kernel.step();

    // 3. Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw cursor gravity point
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, 5, 0, Math.PI * 2);
    ctx.fillStyle = gravityWeight > 0 ? "rgba(0, 255, 0, 0.5)" : (gravityWeight < 0 ? "rgba(255, 0, 0, 0.5)" : "rgba(100, 100, 100, 0.5)");
    ctx.fill();

    // 4. Decode the CanvasEncoder byte buffer sent from Rust
    // The buffer is structured as sequential [Tag, X, Y, Radius/X2, Y2/Color] encoded as floats
    let ptr = kernel.render_ptr;
    let len = kernel.render_len;

    if (len > 0) {
        // Create a Float32Array view directly over WASM memory.
        // Zero-copy, extremely fast. Must recreate view every frame in case WASM resizes memory.
        const mem = MemoryView.float32_array(ptr, len);

        let i = 0;
        ctx.fillStyle = "#6366F1";
        ctx.strokeStyle = "#4f46e5";
        ctx.lineWidth = 2;

        while (i < len) {
            let tag = mem[i++];
            if (tag === TAG_CIRCLE) {
                let x = mem[i++];
                let y = mem[i++];
                let r = mem[i++];
                let colorInt = mem[i++]; // Skip custom color parsing for simple demo

                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fill();
            } else if (tag === TAG_LINE) {
                let x1 = mem[i++];
                let y1 = mem[i++];
                let x2 = mem[i++];
                let y2 = mem[i++];
                let colorInt = mem[i++];

                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            } else {
                // Unknown tag, break to avoid infinite loops if misaligned
                break;
            }
        }
    }

    // 5. Update Metrics UI via serialized JSON from Telemetry
    const metricsStr = kernel.get_metrics_json();
    try {
        const metrics = JSON.parse(metricsStr);
        let text = `Agent Count: ${kernel.agent_count}\n`;
        text += `Physics Step Time: ${metrics.physics_step_ms.toFixed(3)} ms\n`;
        text += `Scripting Eval Time: ${metrics.scripting_eval_ms.toFixed(3)} ms\n`;
        text += `Active Predators/Prey: (Rhai state check required to count exact)\n`;
        text += `Gravity Target: X=${mouseX.toFixed(1)}, Y=${mouseY.toFixed(1)} Weight: ${gravityWeight.toFixed(2)}`;

        document.getElementById('metrics').textContent = text;
    } catch(e) {}

    // Loop
    requestAnimationFrame(renderLoop);
}

run().catch(console.error);
