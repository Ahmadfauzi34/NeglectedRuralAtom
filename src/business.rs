use ndarray::Array1;
use regex::Regex;
use rhai::Array;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static REGEX_CACHE: RefCell<HashMap<String, Regex>> = RefCell::new(HashMap::new());
}

fn get_cached_regex(pattern: &str, max_cache_items: usize) -> Option<Regex> {
    let mut safe_pattern = pattern;
    if safe_pattern.len() > 256 {
        let mut end = 256;
        while end > 0 && !safe_pattern.is_char_boundary(end) {
            end -= 1;
        }
        safe_pattern = &safe_pattern[..end];
    }

    REGEX_CACHE.with(|cache_ref| {
        let mut cache = cache_ref.borrow_mut();
        if !cache.contains_key(safe_pattern) {
            if cache.len() >= max_cache_items {
                cache.clear();
            }
            if let Ok(re) = Regex::new(safe_pattern) {
                cache.insert(safe_pattern.to_string(), re);
            }
        }
        cache.get(safe_pattern).cloned()
    })
}

/// Extracts a specific string value from a raw JSON payload using a top-level key.
/// Returns an empty string if the key doesn't exist or parsing fails.
/// This relies on a fast string search instead of parsing the entire DOM to avoid heavy allocations.
pub fn json_extract_string(json_str: &str, key: &str) -> String {
    let search_key = format!("\"{key}\":");
    if let Some(start_idx) = json_str.find(&search_key) {
        let value_start = start_idx + search_key.len();
        let slice = &json_str[value_start..];

        // Very basic string extraction for flat JSON fields (ignores nested objects)
        let trimmed = slice.trim_start();
        if trimmed.starts_with('"') {
            if let Some(end_idx) = trimmed[1..].find('"') {
                return trimmed[1..=end_idx].to_string();
            }
        } else if let Some(end_idx) = trimmed.find([',', '}']) {
            return trimmed[..end_idx].trim().to_string();
        }
    }
    String::new()
}

/// Matches a regex pattern against a payload and returns the first match found.
pub fn regex_extract(pattern: &str, text: &str, max_cache_items: usize) -> String {
    if let Some(re) = get_cached_regex(pattern, max_cache_items) {
        if let Some(mat) = re.find(text) {
            return mat.as_str().to_string();
        }
    }
    String::new()
}

/// Matches a regex pattern against a payload and returns all matches as a Rhai Array.
pub fn regex_extract_all(pattern: &str, text: &str, max_cache_items: usize) -> Array {
    let mut results = Array::new();
    if let Some(re) = get_cached_regex(pattern, max_cache_items) {
        for mat in re.find_iter(text) {
            results.push(mat.as_str().to_string().into());
        }
    }
    results
}

/// Aggregates multiple number strings into a sum.
/// Useful for quick `MapReduce` math operations from a payload array.
pub fn sum_number_strings(arr: Array) -> f64 {
    let mut sum = 0.0;
    for item in arr {
        if let Ok(f) = item.as_float() {
            sum += f;
        } else if let Ok(i) = item.as_int() {
            sum += i as f64;
        } else if let Ok(s) = item.into_string() {
            if let Ok(num) = s.parse::<f64>() {
                sum += num;
            }
        }
    }
    sum
}

/// Demonstrates using `ndarray` to quickly compute matrix multiplication.
/// This would be used internally by the Data Workers to calculate
/// weights or feature embeddings independently from the JS Thread.
pub fn multiply_matrix_1d(arr_a: Array, arr_b: Array) -> Array {
    let mut vec_a = Vec::with_capacity(arr_a.len());
    let mut vec_b = Vec::with_capacity(arr_b.len());

    for item in arr_a {
        if let Ok(f) = item.as_float() {
            vec_a.push(f as f32);
        } else if let Ok(i) = item.as_int() {
            vec_a.push(i as f32);
        } else {
            vec_a.push(0.0);
        }
    }

    for item in arr_b {
        if let Ok(f) = item.as_float() {
            vec_b.push(f as f32);
        } else if let Ok(i) = item.as_int() {
            vec_b.push(i as f32);
        } else {
            vec_b.push(0.0);
        }
    }

    if vec_a.len() != vec_b.len() || vec_a.is_empty() {
        return rhai::Array::new();
    }

    let a1 = Array1::from_vec(vec_a);
    let a2 = Array1::from_vec(vec_b);
    let result = &a1 * &a2; // Element-wise multiplication

    let mut out = rhai::Array::new();
    for val in result {
        out.push(f64::from(val).into());
    }

    out
}

/// Computes the Dot Product of two 1D Arrays.
/// This acts as a fast linear Forward Pass for a single perceptron node.
pub fn dot_product(arr_a: Array, arr_b: Array) -> f64 {
    let mut dot = 0.0;
    let len = arr_a.len().min(arr_b.len());
    for i in 0..len {
        let mut a = 0.0;
        let mut b = 0.0;

        if let Ok(f) = arr_a[i].as_float() {
            a = f;
        } else if let Ok(v) = arr_a[i].as_int() {
            a = v as f64;
        }

        if let Ok(f) = arr_b[i].as_float() {
            b = f;
        } else if let Ok(v) = arr_b[i].as_int() {
            b = v as f64;
        }

        dot += a * b;
    }
    dot
}

/// Applies the Sigmoid activation function commonly used in neural nets.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Updates a Q-Learning Table value using the Bellman Equation directly in Rust WASM.
/// `q_current` = `q_current` + `learning_rate` * (reward + `discount_factor` * `max_future_q` - `q_current`)
pub fn q_learning_update(
    q_current: f64,
    reward: f64,
    max_future_q: f64,
    learning_rate: f64,
    discount_factor: f64,
) -> f64 {
    q_current + learning_rate * (reward + discount_factor * max_future_q - q_current)
}

/// Contextual Meta-Learning (Context Evolution) via Orthogonal Projection.
/// Allows an agent to evolve its understanding by finding the "novelty" in the broader context
/// (the orthogonal rejection) and shifting its internal vector towards it.
pub fn context_evolution(
    agent_context: Array,
    broader_context: Array,
    learning_rate: f64,
) -> Array {
    let mut vec_a = Vec::with_capacity(agent_context.len());
    let mut vec_b = Vec::with_capacity(broader_context.len());

    for item in agent_context {
        if let Ok(f) = item.as_float() {
            vec_a.push(f);
        } else if let Ok(i) = item.as_int() {
            vec_a.push(i as f64);
        } else {
            vec_a.push(0.0);
        }
    }

    for item in broader_context {
        if let Ok(f) = item.as_float() {
            vec_b.push(f);
        } else if let Ok(i) = item.as_int() {
            vec_b.push(i as f64);
        } else {
            vec_b.push(0.0);
        }
    }

    let len = vec_a.len().min(vec_b.len());
    if len == 0 {
        return rhai::Array::new();
    }

    // 1. Calculate dot product (Overlap / Current Understanding)
    let mut dot_a_b = 0.0;
    let mut dot_a_a = 0.0;
    for i in 0..len {
        dot_a_b += vec_a[i] * vec_b[i];
        dot_a_a += vec_a[i] * vec_a[i];
    }

    // Prevent division by zero
    if dot_a_a < 1e-8 {
        dot_a_a = 1.0;
    }

    // 2. Projection of Broader Context onto Agent Context (What is already known)
    let scalar = dot_a_b / dot_a_a;
    let mut projection = vec![0.0; len];
    for i in 0..len {
        projection[i] = vec_a[i] * scalar;
    }

    // 3. Orthogonal Rejection (The "Novelty" or "Broader Framework" missing from the agent)
    let mut novelty = vec![0.0; len];
    for i in 0..len {
        novelty[i] = vec_b[i] - projection[i];
    }

    // 4. Evolve the agent's context towards the novelty
    let mut out = rhai::Array::new();
    for i in 0..len {
        let evolved_value = vec_a[i] + (novelty[i] * learning_rate);
        out.push(evolved_value.into());
    }

    out
}
