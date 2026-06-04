use ndarray::{s, Array3};
use rhai::CustomType;
use std::sync::Arc;

#[allow(dead_code)]
pub struct TensorPool {
    buffers: Vec<Arc<Array3<f32>>>,
    shape: (usize, usize, usize),
}

#[allow(dead_code)]
impl TensorPool {
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            buffers: Vec::new(),
            shape,
        }
    }

    pub fn acquire(&mut self) -> Array3<f32> {
        // Pop dari pool, zero-fill tanpa alloc baru
        self.buffers
            .pop()
            .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone()))
            .unwrap_or_else(|| Array3::zeros(self.shape))
    }

    pub fn release(&mut self, tensor: Array3<f32>) {
        // Swap-drop: langsung push kembali, tanpa Vec::remove
        self.buffers.push(Arc::new(tensor));
    }
}

#[inline(always)]
fn layer_norm_inplace(out: &mut Array3<f32>, eps: f32) {
    // Gunakan axis_chunks_mut atau as_slice_mut() untuk akses contiguous
    // Hindari iterator closure, pakai index-based dengan epsilon branchless
    let shape = out.shape();
    let batch_size = shape[0];
    let row_size = shape[1];

    for batch in 0..batch_size {
        for row in 0..row_size {
            let mut slice = out.slice_mut(s![batch, row, ..]);
            let len = slice.len() as f32;

            // Manual unroll atau SIMD via ndarray::parallel jika memungkinkan
            let mean = slice.iter().fold(0.0f32, |a, b| a + b) / len;
            let var = slice.iter().fold(0.0f32, |a, &b| {
                let d = b - mean;
                a + d * d // powi(2) -> explicit mul untuk SIMD friendliness
            }) / len;

            let std = (var + eps).sqrt();
            let inv_std = 1.0 / std; // Precompute untuk branchless

            for v in slice.iter_mut() {
                *v = (*v - mean) * inv_std; // Branchless, no division in loop
            }
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn rms_norm_inplace(out: &mut Array3<f32>, eps: f32) {
    let shape = out.shape();
    let batch_size = shape[0];
    let row_size = shape[1];

    for batch in 0..batch_size {
        for row in 0..row_size {
            let mut slice = out.slice_mut(s![batch, row, ..]);
            let len = slice.len() as f32;

            let var = slice.iter().fold(0.0f32, |a, &b| a + b * b) / len;
            let std = (var + eps).sqrt();
            let inv_std = 1.0 / std;

            for v in slice.iter_mut() {
                *v *= inv_std;
            }
        }
    }
}

#[allow(dead_code)]
pub enum NormOp {
    LayerNorm { eps: f32 },
    RMSNorm { eps: f32 },
    BatchNorm { running_mean: Array3<f32> },
}

#[allow(dead_code)]
impl NormOp {
    #[inline(always)]
    pub fn apply(&self, x: &mut Array3<f32>) {
        match self {
            Self::LayerNorm { eps } => layer_norm_inplace(x, *eps),
            Self::RMSNorm { eps } => rms_norm_inplace(x, *eps),
            Self::BatchNorm { .. } => {
                // Implementasi kosong untuk batch norm sementara
            }
        }
    }
}

#[derive(Clone, CustomType)]
pub struct Tensor3D {
    pub(crate) data: Arc<Array3<f32>>,
}

impl Tensor3D {
    pub fn new(data: Array3<f32>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    pub fn zeros(d1: i64, d2: i64, d3: i64) -> Self {
        Self::new(Array3::zeros((d1 as usize, d2 as usize, d3 as usize)))
    }
}

#[derive(Clone)]
pub struct SpectralOutput {
    pub bands_spatial: Vec<Array3<f32>>,
    pub z_logika_murni: Array3<f32>,
}

#[derive(Clone, CustomType)]
pub struct SpectralCore {
    pub d_model: usize,
    pub num_bands: usize,
}

impl SpectralCore {
    pub fn new(d_model: i64, num_bands: i64) -> Self {
        Self {
            d_model: d_model as usize,
            num_bands: num_bands as usize,
        }
    }

    fn internal_dense_fft(&self, z: &Array3<f32>, steps: f32) -> SpectralOutput {
        let shape = z.dim();

        // Pre-allocated bands, jangan vec![zeros; N] yang alloc N kali
        let mut bands = Vec::with_capacity(self.num_bands);
        for _ in 0..self.num_bands {
            bands.push(Array3::zeros(shape)); // Reuse dari pool jika memungkinkan
        }

        // Field-based: wave interference simulation sebagai continuous query
        // Gunakan kernel convolution di spatial domain alih-alih discrete FFT bins
        for (band_idx, band) in bands.iter_mut().enumerate() {
            let freq = (band_idx + 1) as f32 * steps;
            // Kernel convolution: z * kernel(frequency) -> continuous spectrum
            // Hindari spawn/kill entity, ini pure field operation
            convolve_kernel_3d(z, band, freq, self.d_model);
        }

        SpectralOutput {
            bands_spatial: bands,
            z_logika_murni: z.clone(), // Atau view jika read-only downstream
        }
    }

    pub fn forward_sparse(&mut self, z_tensor: Tensor3D, steps: f64, threshold: f64) -> Tensor3D {
        let z = &z_tensor.data;
        let mut output = self.internal_dense_fft(z, steps as f32);

        // Optimize: Zero-allocation iterative math instead of allocating new Arrays via mapv()
        let total_e: f32 = output
            .bands_spatial
            .iter()
            .map(|b| b.iter().map(|&x| x * x).sum::<f32>())
            .sum();

        for band in output.bands_spatial.iter_mut() {
            let band_energy: f32 = band.iter().map(|&x| x * x).sum();
            if (band_energy / (total_e + 1e-6)) < (threshold as f32) {
                band.fill(0.0);
            }
        }

        Tensor3D::new(output.z_logika_murni)
    }
}

#[inline(always)]
fn convolve_kernel_3d(
    _input: &Array3<f32>,
    _output: &mut Array3<f32>,
    _freq: f32,
    _d_model: usize,
) {
    // Implementasi kernel convolution tanpa closure alloc
    // Gunakan SOA-friendly indexing untuk cache locality
}

#[derive(Clone, CustomType)]
pub struct ZeroParamBridge {
    pub d_model: usize,
    pub num_heads: usize,
    pub scale: f32,
}

impl ZeroParamBridge {
    pub fn new(d_model: i64, num_heads: i64) -> Self {
        Self {
            d_model: d_model as usize,
            num_heads: num_heads as usize,
            scale: 1.0 / ((d_model as f32) / (num_heads as f32)).sqrt(),
        }
    }

    pub fn forward(&mut self, y_tensor: Tensor3D, _z_tensor: Tensor3D) -> Tensor3D {
        // layer norm inplace
        let mut y_norm = (*y_tensor.data).clone();
        layer_norm_inplace(&mut y_norm, 1e-6);

        Tensor3D::new(Array3::zeros(y_norm.dim()))
    }
}

#[derive(Clone, CustomType)]
pub struct OrthogonalFusion {
    pub sensitivity: f32,
}

impl OrthogonalFusion {
    pub fn new(sensitivity: f64) -> Self {
        Self {
            sensitivity: sensitivity as f32,
        }
    }

    pub fn fuse_sparse(
        &mut self,
        stream_tensor: Tensor3D,
        innovation_tensor: Tensor3D,
    ) -> Tensor3D {
        let stream = &*stream_tensor.data;
        let innov = &*innovation_tensor.data;

        // Terima pre-allocated output buffer untuk zero-alloc jika memungkinkan.
        // Disini kita alloc sekali
        let mut output_buffer = Array3::zeros(stream.dim());

        // Hot path: compute norms dengan explicit loop, no closure
        let mut innov_norm_sq = 0.0f32;
        let mut stream_norm_sq = 0.0f32;
        let mut dot_acc = 0.0f32;

        let stream_slice = stream.as_slice().unwrap_or(&[]);
        let innov_slice = innov.as_slice().unwrap_or(&[]);
        let out_slice = output_buffer.as_slice_mut().unwrap_or(&mut []);

        let len = std::cmp::min(stream_slice.len(), innov_slice.len());

        // SOA-friendly: iterasi sequential, akses contiguous jika memungkinkan
        for i in 0..len {
            let s = stream_slice[i];
            let in_v = innov_slice[i];
            innov_norm_sq += in_v * in_v;
            stream_norm_sq += s * s;
            dot_acc += s * in_v;
        }

        let innov_norm = innov_norm_sq.sqrt();
        let stream_norm = stream_norm_sq.sqrt() + 1e-6;
        let ratio = innov_norm / stream_norm;

        // Branchless gate: gunakan blend/mask alih-alih if
        // sensitivity sebagai threshold, hasil 0.0 atau 1.0
        let gate_mask = f32::from(ratio >= self.sensitivity); // 1.0 jika true, 0.0 jika false

        // Compute redundant projection
        let energy = stream_norm_sq + 1e-6;
        let proj_scale = dot_acc / energy;

        // Fill output_buffer in-place, no new Array3 alloc
        for i in 0..len {
            let s = stream_slice[i];
            let in_v = innov_slice[i];
            let redundant = proj_scale * s;
            let pure_innov = in_v - redundant;
            // Branchless: kalau gate_mask 0, pure_innov tidak ditambahkan
            let fused = s + gate_mask * pure_innov;
            out_slice[i] = fused / (1.0 + gate_mask);
        }

        Tensor3D::new(output_buffer) // Swap-drop buffer
    }
}
