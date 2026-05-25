use rhai::Array;
use serde_json::Value;
use regex::Regex;

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
