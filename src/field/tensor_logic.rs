use ndarray::{Array3, Axis};
use std::sync::Arc;
use rhai::CustomType;

fn layer_norm(x: &Array3<f32>, eps: f32) -> Array3<f32> {
    let mut out = x.clone();
    for mut batch in out.axis_iter_mut(Axis(0)) {
        for mut row in batch.axis_iter_mut(Axis(0)) {
            let len = row.len() as f32;
            let mean = row.iter().sum::<f32>() / len;
            let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / len;
            let std = (var + eps).sqrt();
            for v in row.iter_mut() { *v = (*v - mean) / std; }
        }
    }
    out
}

#[derive(Clone, CustomType)]
pub struct Tensor3D {
    pub(crate) data: Arc<Array3<f32>>,
}

impl Tensor3D {
    pub fn new(data: Array3<f32>) -> Self {
        Self { data: Arc::new(data) }
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

    fn internal_dense_fft(&self, z: &Array3<f32>, _steps: f32) -> SpectralOutput {
        let shape = z.dim();
        SpectralOutput {
            bands_spatial: vec![Array3::zeros(shape.clone()); self.num_bands],
            z_logika_murni: z.clone(),
        }
    }

    pub fn forward_sparse(&mut self, z_tensor: Tensor3D, steps: f64, threshold: f64) -> Tensor3D {
        let z = &z_tensor.data;
        let mut output = self.internal_dense_fft(z, steps as f32);

        let total_e: f32 = output.bands_spatial.iter().map(|b| b.mapv(|x| x*x).sum()).sum();
        for band in output.bands_spatial.iter_mut() {
            if (band.mapv(|x| x*x).sum() / (total_e + 1e-6)) < (threshold as f32) {
                band.fill(0.0);
            }
        }

        Tensor3D::new(output.z_logika_murni)
    }
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
        let y_norm = layer_norm(&y_tensor.data, 1e-6);
        Tensor3D::new(Array3::zeros(y_norm.dim()))
    }
}

#[derive(Clone, CustomType)]
pub struct OrthogonalFusion {
    pub sensitivity: f32,
}

impl OrthogonalFusion {
    pub fn new(sensitivity: f64) -> Self {
        Self { sensitivity: sensitivity as f32 }
    }

    pub fn fuse_sparse(
        &mut self,
        stream_tensor: Tensor3D,
        innovation_tensor: Tensor3D,
    ) -> Tensor3D {
        let stream = &stream_tensor.data;
        let gate = Array3::<f32>::ones(stream.dim());

        let stream_ref = &*stream_tensor.data;
        let innovation_ref = &*innovation_tensor.data;

        let innov_norm = innovation_ref.mapv(|x| x*x).sum().sqrt();
        let stream_norm = stream_ref.mapv(|x| x*x).sum().sqrt() + 1e-6;

        if (innov_norm / stream_norm) < self.sensitivity {
            return stream_tensor;
        }

        let dot = (stream_ref * innovation_ref).sum_axis(Axis(2)).insert_axis(Axis(2));
        let energy = (stream_ref * stream_ref).sum_axis(Axis(2)).insert_axis(Axis(2)) + 1e-6;
        let redundant = (&dot / &energy) * stream_ref;
        let pure_innovation = innovation_ref - &redundant;

        let fused = (stream_ref + &(&gate * &pure_innovation)) / (1.0 + gate);
        Tensor3D::new(fused)
    }
}
