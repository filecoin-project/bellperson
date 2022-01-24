#![cfg(any(feature = "cuda", feature = "opencl"))]
// This test is in its own file as the `RUST_GPU_TOOLS_CUSTOM_GPU` variable is checked only when
// `get_core_count` is accessed for the first time. Subsequent calls will return the values it was
// set to on the first call.
use std::env;

use bellperson::gpu::get_core_count;

/// Make sure that setting the `BELLMAN_CUSTOM_GPU` env var still works.
#[test]
fn belllman_custom_gpu_env_var() {
    temp_env::with_vars(
        vec![
            ("BELLMAN_CUSTOM_GPU", Some("My custom GPU:3241")),
            ("RUST_GPU_TOOLS_CUSTOM_GPU", None),
        ],
        || {
            let cores = get_core_count("My custom GPU");
            let rust_gpu_tools_custom_gpu = env::var("RUST_GPU_TOOLS_CUSTOM_GPU")
                .expect("RUST_GPU_TOOLS_CUSTOM_GPU is set after `get_core_count` was called.");
            assert_eq!(rust_gpu_tools_custom_gpu, "My custom GPU:3241");
            assert_eq!(cores, 3241, "Cores of custom GPU were set correctly");
        },
    );
}
