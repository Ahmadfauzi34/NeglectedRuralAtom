# "Buku Hitam" - Hybrid Tensor-Logic Agentic Architecture

Dokumen ini adalah pedoman absolut (The Absolute Guidelines) untuk pengembangan sistem `agentic-kernel` ini. Sistem ini merupakan perpaduan antara **Tensor-Driven Physics Engine** dan **LLM-Driven Dynamic Swarm**.

Setiap modifikasi, fitur baru, atau refaktor yang dilakukan pada *codebase* ini di masa depan **wajib mematuhi pilar-pilar di bawah ini** agar arsitektur tidak keluar jalur (degrade).

---

## 1. Structure of Arrays (SOA) is the Law
Segala bentuk state dari agen (baik visual maupun data worker) **wajib** dipisahkan menjadi struktur array paralel (`Vec<T>`).
* **Mengapa:** Memberikan *cache-locality* yang ekstrem di CPU (WASM) dan memungkinkan *zero-copy pass-through* (via pointer linear memori) secara langsung ke Javascript WebGL/WebGPU (Instanced Rendering).
* **Pantangan:** Dilarang menggunakan pola `Array of Structs (AOS)` seperti `Vec<Agent>` di mana `Agent { x, y, hp }` berada dalam satu blok.

## 2. Zero-Allocation di Hot Loop (Pre-allocated Buffers)
*Hot loop* (seperti `step_agents` atau perenderan frame demi frame) harus berjalan tanpa memicu `malloc` atau *garbage collection*.
* **Mengapa:** Alokasi memori berulang di dalam frame loop WASM akan menghancurkan 60FPS target dan menyebabkan patah-patah (stuttering).
* **Implementasi:** Gunakan kapasitas pradefinisi (`Vec::with_capacity`), `String::with_capacity` (pada Prompt Builder), serta penggunaan fungsi-fungsi in-place (`clear()`, iterasi menimpa indeks).

## 3. Pseudo-Discrete Lifecycles (Ghost Tracking via `kill_swap`)
Meskipun LLM akan sering melakukan *spawn* dan *kill* untuk agen pekerja secara logikal (discrete), secara internal di memori, elemen array tidak boleh di-drop atau digeser secara berantai.
* **Mengapa:** Menghindari realokasi (*shifting overhead*) O(N) dan menjaga agar linear memory allocator tidak "lelah".
* **Implementasi:** Selalu gunakan pola `kill_swap` (menukar elemen mati dengan elemen paling belakang di array aktif, lalu memotong `len` logika). Kapasitas *buffer* tidak boleh menyusut, hanya nilai *logical length* yang berubah.

## 4. Defensive Bounds Checking untuk Sandbox Skrip
Skrip logika dinamis yang digenerate oleh eksternal (LLM) lalu disuntikkan ke dalam Runtime (Rhai) **tidak dapat dipercaya 100%**.
* **Mengapa:** WASM akan memicu `panic` dan memblokir browser secara fatal jika sebuah indeks mencoba mengakses memori *out-of-bounds*.
* **Implementasi:** Semua *setter/getter* yang di-expose ke LLM (misalnya di dalam `FieldContext` atau `WorkerContext`) **wajib** menggunakan `.get().copied().unwrap_or(...)` atau `.get_mut()` untuk memuluskan penanganan kegagalan. Dilarang keras menggunakan operator indeks native `[]` di area *script interface*.

## 5. Parallel Pipeline Processing & O(N) Querying
Hindari sebisa mungkin perulangan berkalang (`nested loops`) seperti $O(N^2)$ pada logika *swarm/flocking*.
* **Mengapa:** Membuat kerumitan eksponensial yang akan menghambat skalabilitas jumlah agen.
* **Implementasi:** Gunakan partisi spasial seperti `Spatial Hash Grid` untuk komputasi fisik, atau pisahkan pekerja abstrak (data crawling) ke pipeline spesifik `DataWorkerField`. Komputasi fisik dan komputasi kognitif (LLM) tidak boleh saling memblokir (gunakan `Arc<RwLock>` secara asinkron saat dibutuhkan).

## 6. Shared State Asynchronous Read (JS Fragment Gotchas)
Sistem dapat memproses Prompt LLM secara asinkronus agar UI Javascript tidak *freeze*.
* **Mengapa:** Membuat deskripsi raksasa berukuran 10MB dari 10.000 agen butuh waktu.
* **Implementasi:**
  1. Bungkus state dalam `Arc<RwLock<SharedState>>`.
  2. Peringatan Host JS: Jika `reserve()` di Rust memicu perluasan memori (memory grow), pointer `Float32Array` Javascript akan *detached*. Eksekusi pointer-getter WASM wajib dilakukan secara *late binding* (tepat di dalam animation frame) untuk menghindari pointer yang kedaluwarsa.