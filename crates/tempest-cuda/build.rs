use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> String {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let candidate = Path::new(&cuda_home).join("bin/nvcc");
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    let default_nvcc = Path::new("/usr/local/cuda/bin/nvcc");
    if default_nvcc.exists() {
        return default_nvcc.to_string_lossy().into_owned();
    }
    "nvcc".to_string()
}

fn find_cuda_lib_dir() -> String {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let candidate = Path::new(&cuda_home).join("lib64");
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "/usr/local/cuda/lib64".to_string()
}

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/nh_core.cu");
    println!("cargo:rerun-if-changed=src/cuda/nh_core.h");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=TEMPEST_CUDA_ARCH");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cu_path = manifest_dir.join("src/cuda/nh_core.cu");
    let obj_path = out_dir.join("nh_core.o");
    let static_lib_path = out_dir.join("libtempest_nh_cuda.a");
    let shared_lib_path = manifest_dir.join("libtempest_cuda.so");

    let nvcc = find_nvcc();
    let cuda_lib_dir = find_cuda_lib_dir();
    let arch = env::var("TEMPEST_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    let arch_flag = format!("-arch={arch}");

    // 1. Compile object file for static linking into Rust crate & cargo test.
    let status = Command::new(&nvcc)
        .args([
            "-O3",
            &arch_flag,
            "-std=c++17",
            "--compiler-options",
            "-fPIC",
            "-c",
            cu_path.to_str().unwrap(),
            "-o",
            obj_path.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute nvcc for static object build");
    assert!(
        status.success(),
        "nvcc failed compiling nh_core.cu to object file (status={status})"
    );

    // 2. Archive into libtempest_nh_cuda.a.
    let ar_status = Command::new("ar")
        .args([
            "rcs",
            static_lib_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute ar for libtempest_nh_cuda.a");
    assert!(
        ar_status.success(),
        "ar failed creating libtempest_nh_cuda.a (status={ar_status})"
    );

    // 3. Build standalone shared library for direct Python ctypes bridge.
    let so_status = Command::new(&nvcc)
        .args([
            "-O3",
            &arch_flag,
            "-std=c++17",
            "--shared",
            "--compiler-options",
            "-fPIC",
            cu_path.to_str().unwrap(),
            "-o",
            shared_lib_path.to_str().unwrap(),
            &format!("-L{cuda_lib_dir}"),
            "-lcudart",
        ])
        .status()
        .expect("Failed to execute nvcc for shared library libtempest_cuda.so");
    assert!(
        so_status.success(),
        "nvcc failed building shared library libtempest_cuda.so (status={so_status})"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=tempest_nh_cuda");
    println!("cargo:rustc-link-search=native={cuda_lib_dir}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
