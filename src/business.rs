use rhai::Array;
use serde_json::Value;
use regex::Regex;
use ndarray::{Array1, Array2};

/// Extracts a specific string value from a raw JSON payload using a top-level key.
/// Returns an empty string if the key doesn't exist or parsing fails.
/// This prevents heavy allocations of parsing the entire JSON object back into Rhai types.
pub fn json_extract_string(json_str: &str, key: &str) -> String {
    if let Ok(val) = serde_json::from_str::<Value>(json_str) {
        if let Some(field) = val.get(key) {
            if let Some(s) = field.as_str() {
                return s.to_string();
            } else {
                return field.to_string(); // Fallback to raw representation
            }
        }
    }
    String::new()
}

/// Matches a regex pattern against a payload and returns the first match found.
pub fn regex_extract(pattern: &str, text: &str) -> String {
    if let Ok(re) = Regex::new(pattern) {
        if let Some(mat) = re.find(text) {
            return mat.as_str().to_string();
        }
    }
    String::new()
}

/// Matches a regex pattern against a payload and returns all matches as a Rhai Array.
pub fn regex_extract_all(pattern: &str, text: &str) -> Array {
    let mut results = Array::new();
    if let Ok(re) = Regex::new(pattern) {
        for mat in re.find_iter(text) {
            results.push(mat.as_str().to_string().into());
        }
    }
    results
}

/// Aggregates multiple number strings into a sum.
/// Useful for quick MapReduce math operations from a payload array.
pub fn sum_number_strings(arr: Array) -> f64 {
    let mut sum = 0.0;
    for item in arr {
        if let Ok(f) = item.as_float() {
            sum += f as f64;
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
        out.push((val as f64).into());
    }

    out
}
