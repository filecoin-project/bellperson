#![cfg(any(feature = "cuda", feature = "opencl"))]
// This test is in its own file as the `RUST_GPU_TOOLS_CUSTOM_GPU` variable is checked only when
// `get_core_count` is accessed for the first time. Subsequent calls will return the values it was
// set to on the first call.
use std::env;

use bellperson::gpu::get_core_count;

/// Make sure that `BELLMAN_CUSTOM_GPU` env var is ignored if `RUST_GPU_TOOLS_CUSTOM_GPU` is
/// set.
#[test]
fn belllman_custom_gpu_env_var_ignored() {
    temp_env::with_vars(
        vec![
            ("RUST_GPU_TOOLS_CUSTOM_GPU", Some("My custom GPU:444")),
            ("BELLMAN_CUSTOM_GPU", Some("My custom GPU:3242")),
        ],
        || {
            let cores = get_core_count("My custom GPU");
            env::var("RUST_GPU_TOOLS_CUSTOM_GPU")
                .expect("RUST_GPU_TOOLS_CUSTOM_GPU is set after `get_core_count` was called.");
            assert_eq!(cores, 444, "Cores of custom GPU were set correctly to the `RUST_GPU_TOOLS_CUSTOM_GPU` env var.");
        },
    );
}
