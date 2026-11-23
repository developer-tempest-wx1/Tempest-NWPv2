use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn cuda_home_from_env() -> Option<String> {
    env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn find_nvcc() -> String {
    if let Some(cuda_home) = cuda_home_from_env() {
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
    if let Some(cuda_home) = cuda_home_from_env() {
        let candidate = Path::new(&cuda_home).join("lib64");
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "/usr/local/cuda/lib64".to_string()
}

fn query_nvcc_supported_arches(nvcc: &str) -> Vec<u32> {
    let Ok(out) = Command::new(nvcc).arg("--list-gpu-arch").output() else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut arches = BTreeSet::new();
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("compute_") {
            if let Ok(cap) = rest.parse::<u32>() {
                arches.insert(cap);
            }
        }
    }
    arches.into_iter().collect()
}

fn detect_gpu_arches() -> Vec<u32> {
    let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut arches = BTreeSet::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(cap) = parse_arch_token(trimmed) {
            arches.insert(cap);
        }
    }
    arches.into_iter().collect()
}

fn parse_arch_token(token: &str) -> Result<u32, String> {
    let s = token.trim().to_ascii_lowercase();
    if s.is_empty() {
        return Err("Empty CUDA architecture token".to_string());
    }
    if let Some(rest) = s.strip_prefix("sm_").or_else(|| s.strip_prefix("compute_")) {
        return rest
            .parse::<u32>()
            .map_err(|_| format!("Invalid CUDA architecture token: {token:?}"));
    }
    if let Some((maj_s, min_s)) = s.split_once('.') {
        let maj = maj_s
            .parse::<u32>()
            .map_err(|_| format!("Invalid CUDA compute capability: {token:?}"))?;
        let min = min_s
            .parse::<u32>()
            .map_err(|_| format!("Invalid CUDA compute capability: {token:?}"))?;
        if min >= 10 {
            return Err(format!("Invalid CUDA compute capability minor version: {token:?}"));
        }
        return Ok(maj * 10 + min);
    }
    s.parse::<u32>()
        .map_err(|_| format!("Invalid CUDA architecture token: {token:?}"))
}

fn resolve_target_arches(nvcc: &str, requested: &str, supported: &[u32]) -> Vec<u32> {
    const PORTABLE_CANONICAL: &[u32] = &[75, 80, 86, 89, 90, 100, 120];
    const HEADLESS_CANONICAL: &[u32] = &[80, 86, 89, 90, 120];

    let spec = requested.trim().to_ascii_lowercase();
    if spec.is_empty() || spec == "auto" {
        let detected = detect_gpu_arches();
        if detected.is_empty() {
            if supported.is_empty() {
                return vec![86];
            }
            let headless: Vec<u32> = HEADLESS_CANONICAL
                .iter()
                .copied()
                .filter(|c| supported.contains(c))
                .collect();
            return if headless.is_empty() {
                vec![*supported.last().unwrap()]
            } else {
                headless
            };
        }

        if supported.is_empty() {
            return detected;
        }

        let max_sup = *supported.last().unwrap();
        let min_sup = *supported.first().unwrap();
        let mut chosen = BTreeSet::new();
        for cap in detected {
            if supported.contains(&cap) {
                chosen.insert(cap);
            } else if cap > max_sup {
                println!(
                    "cargo:warning=Detected GPU sm_{cap} (RTX 50xx/newer) requires CUDA Toolkit >= 12.8 for native code; falling back to forward-compatible PTX compute_{max_sup}"
                );
                chosen.insert(max_sup);
            } else if cap < min_sup {
                println!(
                    "cargo:warning=Detected GPU sm_{cap} is older than {nvcc} minimum sm_{min_sup}; targeting sm_{min_sup}"
                );
                chosen.insert(min_sup);
            } else {
                let nearest = *supported
                    .iter()
                    .rev()
                    .find(|&&s| s <= cap)
                    .unwrap_or(&min_sup);
                chosen.insert(nearest);
            }
        }
        return chosen.into_iter().collect();
    }

    if spec == "all" || spec == "portable" {
        if supported.is_empty() {
            return PORTABLE_CANONICAL.to_vec();
        }
        let multi: Vec<u32> = PORTABLE_CANONICAL
            .iter()
            .copied()
            .filter(|c| supported.contains(c))
            .collect();
        return if multi.is_empty() {
            vec![*supported.last().unwrap()]
        } else {
            multi
        };
    }

    let mut explicit = BTreeSet::new();
    for raw in spec.split([',', ';']) {
        let tok = raw.trim();
        if tok.is_empty() {
            continue;
        }
        let cap = parse_arch_token(tok).unwrap_or_else(|e| panic!("{e}"));
        if !supported.is_empty() && !supported.contains(&cap) {
            if cap == 120 {
                panic!(
                    "Requested sm_120 (RTX 50xx Blackwell) is not supported by {nvcc}; \
                     install CUDA Toolkit >= 12.8 or build with TEMPEST_CUDA_ARCH=auto"
                );
            }
            panic!(
                "Requested sm_{cap} is not supported by {nvcc} (supported: {supported:?})"
            );
        }
        explicit.insert(cap);
    }
    assert!(
        !explicit.is_empty(),
        "TEMPEST_CUDA_ARCH={requested:?} yielded no CUDA architectures"
    );
    explicit.into_iter().collect()
}

fn build_gencode_args(arches: &[u32]) -> Vec<String> {
    let mut args = Vec::with_capacity(arches.len() * 2 + 2);
    for &cap in arches {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{cap},code=sm_{cap}"));
    }
    if let Some(&max_cap) = arches.iter().max() {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{max_cap},code=compute_{max_cap}"));
    }
    args
}

fn compile_cu_obj(nvcc: &str, gencode_args: &[String], cu_path: &Path, obj_path: &Path) {
    let mut cmd = Command::new(nvcc);
    cmd.arg("-O3");
    cmd.args(gencode_args);
    cmd.args([
        "-std=c++17",
        "--compiler-options",
        "-fPIC",
        "-c",
        cu_path.to_str().unwrap(),
        "-o",
        obj_path.to_str().unwrap(),
    ]);
    let status = cmd
        .status()
        .expect("Failed to execute nvcc for CUDA object build");
    assert!(
        status.success(),
        "nvcc failed compiling {} (status={status})",
        cu_path.display()
    );
}

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/nh_core.cu");
    println!("cargo:rerun-if-changed=src/cuda/nh_physics.cu");
    println!("cargo:rerun-if-changed=src/cuda/nh_ocean_si3.cu");
    println!("cargo:rerun-if-changed=src/cuda/nh_ocean_si3.h");
    println!("cargo:rerun-if-changed=src/cuda/nh_internal.cuh");
    println!("cargo:rerun-if-changed=src/cuda/nh_core.h");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=TEMPEST_CUDA_ARCH");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let core_cu = manifest_dir.join("src/cuda/nh_core.cu");
    let phys_cu = manifest_dir.join("src/cuda/nh_physics.cu");
    let oce_cu = manifest_dir.join("src/cuda/nh_ocean_si3.cu");
    let core_obj = out_dir.join("nh_core.o");
    let phys_obj = out_dir.join("nh_physics.o");
    let oce_obj = out_dir.join("nh_ocean_si3.o");
    let static_lib_path = out_dir.join("libtempest_nh_cuda.a");
    let shared_lib_path = manifest_dir.join("libtempest_cuda.so");

    let nvcc = find_nvcc();
    let cuda_lib_dir = find_cuda_lib_dir();
    let supported = query_nvcc_supported_arches(&nvcc);
    let requested = env::var("TEMPEST_CUDA_ARCH").unwrap_or_else(|_| "auto".to_string());
    let target_arches = resolve_target_arches(&nvcc, &requested, &supported);
    let gencode_args = build_gencode_args(&target_arches);

    let arch_summary: Vec<String> = target_arches.iter().map(|c| format!("sm_{c}")).collect();
    println!(
        "cargo:warning=tempest-cuda targeting [{}] (+PTX) via {}",
        arch_summary.join(", "),
        nvcc
    );

    // 1. Compile all CUDA translation units cleanly.
    compile_cu_obj(&nvcc, &gencode_args, &core_cu, &core_obj);
    compile_cu_obj(&nvcc, &gencode_args, &phys_cu, &phys_obj);
    compile_cu_obj(&nvcc, &gencode_args, &oce_cu, &oce_obj);

    // 2. Archive objects into libtempest_nh_cuda.a.
    let ar_status = Command::new("ar")
        .args([
            "rcs",
            static_lib_path.to_str().unwrap(),
            core_obj.to_str().unwrap(),
            phys_obj.to_str().unwrap(),
            oce_obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute ar for libtempest_nh_cuda.a");
    assert!(
        ar_status.success(),
        "ar failed creating libtempest_nh_cuda.a (status={ar_status})"
    );

    // 3. Link compiled objects directly into standalone shared library for Python ctypes bridge.
    let so_status = Command::new(&nvcc)
        .args([
            "--shared",
            core_obj.to_str().unwrap(),
            phys_obj.to_str().unwrap(),
            oce_obj.to_str().unwrap(),
            "-o",
            shared_lib_path.to_str().unwrap(),
            &format!("-L{cuda_lib_dir}"),
            "-lcudart",
            "-lcufft",
        ])
        .status()
        .expect("Failed to execute nvcc for shared library libtempest_cuda.so");
    assert!(
        so_status.success(),
        "nvcc failed linking shared library libtempest_cuda.so (status={so_status})"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=tempest_nh_cuda");
    println!("cargo:rustc-link-search=native={cuda_lib_dir}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
