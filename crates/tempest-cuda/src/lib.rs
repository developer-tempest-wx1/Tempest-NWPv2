#![deny(unsafe_op_in_unsafe_fn)]

//! CUDA split-explicit RK3 non-hydrostatic atmospheric core bindings for Tempest-NWP v2.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;
use thiserror::Error;

pub use tempest_core::constants::{CP_D, GRAVITY, P0, R_D};

#[derive(Debug, Error)]
pub enum CudaCoreError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("CUDA backend error: {0}")]
    BackendError(String),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NhGridConfig {
    pub nz: i32,
    pub ny: i32,
    pub nx: i32,
    pub has_moist: i32,
    pub bc_x: i32,
    pub bc_y: i32,
    pub lbc_edges: i32,
    pub use_coriolis: i32,
    pub adv_order: i32,
    pub lbc_rim_width: i32,
    pub has_iau: i32,
    pub dx_m: f64,
    pub dy_m: f64,
    pub z_top_m: f64,
    pub theta_sfc_k: f64,
    pub p_sfc_pa: f64,
    pub n_ref_s: f64,
    pub beta_offcenter: f64,
    pub div_damp_coeff: f64,
    pub km_const: f64,
    pub kh_const: f64,
    pub hyperdiff_coeff: f64,
    pub sponge_depth_m: f64,
    pub sponge_tau_max_s: f64,
    pub lbc_tau_max_s: f64,
    pub is_spherical: i32,
    pub polar_filter_lat_deg: f64,
}

#[repr(C)]
pub struct NhStatePtrs {
    pub u: *mut f32,
    pub v: *mut f32,
    pub w: *mut f32,
    pub theta_p: *mut f32,
    pub pi_p: *mut f32,
    pub qv: *mut f32,
    pub qc: *mut f32,
    pub qr: *mut f32,
    pub qi: *mut f32,
    pub qs: *mut f32,
    pub qg: *mut f32,
}

#[repr(C)]
pub struct NhConstStatePtrs {
    pub u: *const f32,
    pub v: *const f32,
    pub w: *const f32,
    pub theta_p: *const f32,
    pub pi_p: *const f32,
    pub qv: *const f32,
    pub qc: *const f32,
    pub qr: *const f32,
    pub qi: *const f32,
    pub qs: *const f32,
    pub qg: *const f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NhPhysicsConfig {
    pub enable_microphysics: i32,
    pub enable_pbl: i32,
    pub enable_radiation: i32,
    pub enable_surface: i32,
    pub rad_interval_s: f64,
    pub cloud_num_land: f64,
    pub cloud_num_sea: f64,
    pub julian_day: f64,
    pub utc_seconds: f64,
    pub co2_ppm: f64,
}

impl Default for NhPhysicsConfig {
    fn default() -> Self {
        Self {
            enable_microphysics: 1,
            enable_pbl: 1,
            enable_radiation: 1,
            enable_surface: 1,
            rad_interval_s: 900.0,
            cloud_num_land: 200.0e6,
            cloud_num_sea: 80.0e6,
            julian_day: 172.0,
            utc_seconds: 43_200.0,
            co2_ppm: 425.0,
        }
    }
}

#[repr(C)]
pub struct NhSurfaceStaticPtrs {
    pub lat_deg: *const f32,
    pub lon_deg: *const f32,
    pub lsm: *const f32,
    pub lake_mask: *const f32,
    pub soil_type: *const f32,
    pub bathy_depth_m: *const f32,
    pub z0m_base: *const f32,
    pub albedo_base: *const f32,
    pub veg_frac: *const f32,
}

#[repr(C)]
pub struct NhSurfaceStatePtrs {
    pub t_skin: *mut f32,
    pub soil_t: *mut f32,
    pub soil_m: *mut f32,
    pub snow_swe: *mut f32,
    pub sst_bulk: *mut f32,
    pub dt_cool: *mut f32,
    pub dt_warm: *mut f32,
    pub lake_tbot: *mut f32,
    pub lake_hmld: *mut f32,
    pub lake_ctheta: *mut f32,
    pub lake_hice: *mut f32,
    pub rain_acc: *mut f32,
    pub snow_acc: *mut f32,
    pub graupel_acc: *mut f32,
    pub sw_down: *mut f32,
    pub lw_down: *mut f32,
    pub shf: *mut f32,
    pub lhf: *mut f32,
    pub ustar: *mut f32,
    pub pblh: *mut f32,
    pub t2m: *mut f32,
    pub q2m: *mut f32,
    pub u10m: *mut f32,
    pub v10m: *mut f32,
}

#[repr(C)]
pub struct NhConstSurfaceStatePtrs {
    pub t_skin: *const f32,
    pub soil_t: *const f32,
    pub soil_m: *const f32,
    pub snow_swe: *const f32,
    pub sst_bulk: *const f32,
    pub dt_cool: *const f32,
    pub dt_warm: *const f32,
    pub lake_tbot: *const f32,
    pub lake_hmld: *const f32,
    pub lake_ctheta: *const f32,
    pub lake_hice: *const f32,
    pub rain_acc: *const f32,
    pub snow_acc: *const f32,
    pub graupel_acc: *const f32,
    pub sw_down: *const f32,
    pub lw_down: *const f32,
    pub shf: *const f32,
    pub lhf: *const f32,
    pub ustar: *const f32,
    pub pblh: *const f32,
    pub t2m: *const f32,
    pub q2m: *const f32,
    pub u10m: *const f32,
    pub v10m: *const f32,
}

#[repr(C)]
pub struct NhCudaHandle {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn nh_cuda_vram_info(free_bytes: *mut u64, total_bytes: *mut u64) -> c_int;
    pub fn nh_cuda_estimate_vram_bytes(
        nz: i32,
        ny: i32,
        nx: i32,
        has_moist: i32,
        has_lbc: i32,
        has_iau: i32,
    ) -> u64;
    pub fn nh_cuda_estimate_vram_bytes_ex(
        nz: i32,
        ny: i32,
        nx: i32,
        has_moist: i32,
        has_lbc: i32,
        has_iau: i32,
        has_physics: i32,
    ) -> u64;
    pub fn nh_cuda_create(
        cfg: *const NhGridConfig,
        zeta_w: *const f64,
        zeta_s: *const f64,
        dzeta: *const f64,
        b_w: *const f64,
        b_s: *const f64,
        db_dzeta_w: *const f64,
        db_dzeta_s: *const f64,
        hs_m: *const f64,
        msf: *const f64,
        f_cor: *const f64,
    ) -> *mut NhCudaHandle;
    pub fn nh_cuda_destroy(handle: *mut NhCudaHandle);
    pub fn nh_cuda_upload_state(handle: *mut NhCudaHandle, src: *const NhConstStatePtrs) -> c_int;
    pub fn nh_cuda_download_state(handle: *mut NhCudaHandle, dst: *const NhStatePtrs) -> c_int;
    pub fn nh_cuda_set_lbc_target(handle: *mut NhCudaHandle, target: *const NhConstStatePtrs) -> c_int;
    pub fn nh_cuda_set_iau_forcing(
        handle: *mut NhCudaHandle,
        ddt_state: *const NhConstStatePtrs,
    ) -> c_int;
    pub fn nh_cuda_set_physics_config(handle: *mut NhCudaHandle, pcfg: *const NhPhysicsConfig) -> c_int;
    pub fn nh_cuda_set_surface_static(
        handle: *mut NhCudaHandle,
        sstatic: *const NhSurfaceStaticPtrs,
    ) -> c_int;
    pub fn nh_cuda_upload_surface(
        handle: *mut NhCudaHandle,
        src: *const NhConstSurfaceStatePtrs,
    ) -> c_int;
    pub fn nh_cuda_download_surface(
        handle: *mut NhCudaHandle,
        dst: *const NhSurfaceStatePtrs,
    ) -> c_int;
    pub fn nh_cuda_step_physics(handle: *mut NhCudaHandle, dt_s: f64) -> c_int;
    pub fn nh_cuda_step_rk3(handle: *mut NhCudaHandle, dt_s: f64, n_substeps: i32) -> c_int;
    pub fn nh_cuda_last_error() -> *const c_char;
}

pub fn estimate_vram_bytes_ex(
    nz: i32,
    ny: i32,
    nx: i32,
    has_moist: bool,
    has_lbc: bool,
    has_iau: bool,
    has_physics: bool,
) -> u64 {
    unsafe {
        nh_cuda_estimate_vram_bytes_ex(
            nz,
            ny,
            nx,
            i32::from(has_moist),
            i32::from(has_lbc),
            i32::from(has_iau),
            i32::from(has_physics),
        )
    }
}

fn last_cuda_error() -> String {
    unsafe {
        let ptr = nh_cuda_last_error();
        if ptr.is_null() {
            return "unknown CUDA error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Query active CUDA device free and total VRAM in bytes.
pub fn vram_info() -> Result<(u64, u64), CudaCoreError> {
    let mut free = 0u64;
    let mut total = 0u64;
    let rc = unsafe { nh_cuda_vram_info(&mut free, &mut total) };
    if rc != 0 {
        return Err(CudaCoreError::BackendError(last_cuda_error()));
    }
    Ok((free, total))
}

/// Safe RAII wrapper around `NhCudaHandle`.
pub struct NhSolver {
    handle: *mut NhCudaHandle,
    cfg: NhGridConfig,
}

impl NhSolver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: NhGridConfig,
        zeta_w: &[f64],
        zeta_s: &[f64],
        dzeta: &[f64],
        b_w: &[f64],
        b_s: &[f64],
        db_dzeta_w: &[f64],
        db_dzeta_s: &[f64],
        hs_m: &[f64],
        msf: &[f64],
        f_cor: &[f64],
    ) -> Result<Self, CudaCoreError> {
        let nz = cfg.nz as usize;
        let n2 = (cfg.ny as usize) * (cfg.nx as usize);
        if zeta_w.len() != nz + 1
            || b_w.len() != nz + 1
            || db_dzeta_w.len() != nz + 1
            || zeta_s.len() != nz
            || dzeta.len() != nz
            || b_s.len() != nz
            || db_dzeta_s.len() != nz
            || hs_m.len() != n2
            || msf.len() != n2
            || f_cor.len() != n2
        {
            return Err(CudaCoreError::DimensionMismatch(
                "Static array lengths do not match NhGridConfig".to_string(),
            ));
        }

        let handle = unsafe {
            nh_cuda_create(
                &cfg,
                zeta_w.as_ptr(),
                zeta_s.as_ptr(),
                dzeta.as_ptr(),
                b_w.as_ptr(),
                b_s.as_ptr(),
                db_dzeta_w.as_ptr(),
                db_dzeta_s.as_ptr(),
                hs_m.as_ptr(),
                msf.as_ptr(),
                f_cor.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(Self { handle, cfg })
    }

    pub fn upload_dry(
        &mut self,
        u: &[f32],
        v: &[f32],
        w: &[f32],
        theta_p: &[f32],
        pi_p: &[f32],
    ) -> Result<(), CudaCoreError> {
        let ptrs = NhConstStatePtrs {
            u: u.as_ptr(),
            v: v.as_ptr(),
            w: w.as_ptr(),
            theta_p: theta_p.as_ptr(),
            pi_p: pi_p.as_ptr(),
            qv: ptr::null(),
            qc: ptr::null(),
            qr: ptr::null(),
            qi: ptr::null(),
            qs: ptr::null(),
            qg: ptr::null(),
        };
        let rc = unsafe { nh_cuda_upload_state(self.handle, &ptrs) };
        if rc != 0 {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(())
    }

    pub fn download_dry(
        &self,
        u: &mut [f32],
        v: &mut [f32],
        w: &mut [f32],
        theta_p: &mut [f32],
        pi_p: &mut [f32],
    ) -> Result<(), CudaCoreError> {
        let ptrs = NhStatePtrs {
            u: u.as_mut_ptr(),
            v: v.as_mut_ptr(),
            w: w.as_mut_ptr(),
            theta_p: theta_p.as_mut_ptr(),
            pi_p: pi_p.as_mut_ptr(),
            qv: ptr::null_mut(),
            qc: ptr::null_mut(),
            qr: ptr::null_mut(),
            qi: ptr::null_mut(),
            qs: ptr::null_mut(),
            qg: ptr::null_mut(),
        };
        let rc = unsafe { nh_cuda_download_state(self.handle, &ptrs) };
        if rc != 0 {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_moist(
        &mut self,
        u: &[f32],
        v: &[f32],
        w: &[f32],
        theta_p: &[f32],
        pi_p: &[f32],
        qv: &[f32],
        qc: &[f32],
        qr: &[f32],
        qi: &[f32],
        qs: &[f32],
        qg: &[f32],
    ) -> Result<(), CudaCoreError> {
        let ptrs = NhConstStatePtrs {
            u: u.as_ptr(),
            v: v.as_ptr(),
            w: w.as_ptr(),
            theta_p: theta_p.as_ptr(),
            pi_p: pi_p.as_ptr(),
            qv: qv.as_ptr(),
            qc: qc.as_ptr(),
            qr: qr.as_ptr(),
            qi: qi.as_ptr(),
            qs: qs.as_ptr(),
            qg: qg.as_ptr(),
        };
        let rc = unsafe { nh_cuda_upload_state(self.handle, &ptrs) };
        if rc != 0 {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn download_moist(
        &self,
        u: &mut [f32],
        v: &mut [f32],
        w: &mut [f32],
        theta_p: &mut [f32],
        pi_p: &mut [f32],
        qv: &mut [f32],
        qc: &mut [f32],
        qr: &mut [f32],
        qi: &mut [f32],
        qs: &mut [f32],
        qg: &mut [f32],
    ) -> Result<(), CudaCoreError> {
        let ptrs = NhStatePtrs {
            u: u.as_mut_ptr(),
            v: v.as_mut_ptr(),
            w: w.as_mut_ptr(),
            theta_p: theta_p.as_mut_ptr(),
            pi_p: pi_p.as_mut_ptr(),
            qv: qv.as_mut_ptr(),
            qc: qc.as_mut_ptr(),
            qr: qr.as_mut_ptr(),
            qi: qi.as_mut_ptr(),
            qs: qs.as_mut_ptr(),
            qg: qg.as_mut_ptr(),
        };
        let rc = unsafe { nh_cuda_download_state(self.handle, &ptrs) };
        if rc != 0 {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(())
    }

    pub fn set_physics_config(&mut self, cfg: &NhPhysicsConfig) -> Result<(), CudaCoreError> {
        let rc = unsafe { nh_cuda_set_physics_config(self.handle, cfg) };
        if rc != 0 {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(())
    }

    pub fn step_physics(&mut self, dt_s: f64) -> Result<(), CudaCoreError> {
        let rc = unsafe { nh_cuda_step_physics(self.handle, dt_s) };
        if rc != 0 {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(())
    }

    pub fn step_rk3(&mut self, dt_s: f64, n_substeps: i32) -> Result<(), CudaCoreError> {
        let rc = unsafe { nh_cuda_step_rk3(self.handle, dt_s, n_substeps) };
        if rc != 0 {
            return Err(CudaCoreError::BackendError(last_cuda_error()));
        }
        Ok(())
    }

    pub fn config(&self) -> NhGridConfig {
        self.cfg
    }
}

impl Drop for NhSolver {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nh_cuda_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_resting_atmosphere_zero_drift() {
        let (free_b, total_b) = vram_info().expect("CUDA GPU required for tempest-cuda tests");
        assert!(total_b > 0 && free_b > 0);

        let nz = 16usize;
        let ny = 12usize;
        let nx = 12usize;
        let z_top = 12_000.0;
        let dz = z_top / nz as f64;
        let zeta_w: Vec<f64> = (0..=nz).map(|k| k as f64 * dz).collect();
        let zeta_s: Vec<f64> = (0..nz).map(|k| (k as f64 + 0.5) * dz).collect();
        let dzeta = vec![dz; nz];
        let b_w: Vec<f64> = zeta_w.iter().map(|z| 1.0 - z / z_top).collect();
        let b_s: Vec<f64> = zeta_s.iter().map(|z| 1.0 - z / z_top).collect();
        let db_w = vec![-1.0 / z_top; nz + 1];
        let db_s = vec![-1.0 / z_top; nz];

        let mut hs = vec![0.0; ny * nx];
        for j in 0..ny {
            for i in 0..nx {
                let dx = i as f64 - 5.5;
                let dy = j as f64 - 5.5;
                hs[j * nx + i] = 800.0 * (-0.15 * (dx * dx + dy * dy)).exp();
            }
        }
        let msf = vec![1.0; ny * nx];
        let fcor = vec![0.0; ny * nx];

        let cfg = NhGridConfig {
            nz: nz as i32,
            ny: ny as i32,
            nx: nx as i32,
            has_moist: 0,
            bc_x: 0,
            bc_y: 0,
            lbc_edges: 0,
            use_coriolis: 0,
            adv_order: 5,
            lbc_rim_width: 0,
            has_iau: 0,
            dx_m: 1500.0,
            dy_m: 1500.0,
            z_top_m: z_top,
            theta_sfc_k: 288.15,
            p_sfc_pa: 101_325.0,
            n_ref_s: 0.01,
            beta_offcenter: 0.55,
            div_damp_coeff: 0.03,
            km_const: 0.0,
            kh_const: 0.0,
            hyperdiff_coeff: 0.0,
            sponge_depth_m: 0.0,
            sponge_tau_max_s: 0.0,
            lbc_tau_max_s: 0.0,
            is_spherical: 0,
            polar_filter_lat_deg: 0.0,
        };

        let mut solver = NhSolver::new(
            cfg, &zeta_w, &zeta_s, &dzeta, &b_w, &b_s, &db_w, &db_s, &hs, &msf, &fcor,
        )
        .expect("Failed creating NhSolver");

        let mut u = vec![0.0f32; nz * ny * (nx + 1)];
        let mut v = vec![0.0f32; nz * (ny + 1) * nx];
        let mut w = vec![0.0f32; (nz + 1) * ny * nx];
        let mut thp = vec![0.0f32; nz * ny * nx];
        let mut pip = vec![0.0f32; nz * ny * nx];

        solver.upload_dry(&u, &v, &w, &thp, &pip).unwrap();
        for _ in 0..5 {
            solver.step_rk3(4.0, 0).unwrap();
        }
        solver
            .download_dry(&mut u, &mut v, &mut w, &mut thp, &mut pip)
            .unwrap();

        let max_u = u.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_w = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_u < 1e-6 && max_w < 1e-6, "max_u={max_u}, max_w={max_w}");
    }

    #[test]
    fn cuda_physics_smoke_no_nan() {
        let nz = 16usize;
        let ny = 8usize;
        let nx = 8usize;
        let z_top = 12_000.0;
        let dz = z_top / nz as f64;
        let zeta_w: Vec<f64> = (0..=nz).map(|k| k as f64 * dz).collect();
        let zeta_s: Vec<f64> = (0..nz).map(|k| (k as f64 + 0.5) * dz).collect();
        let dzeta = vec![dz; nz];
        let b_w: Vec<f64> = zeta_w.iter().map(|z| 1.0 - z / z_top).collect();
        let b_s: Vec<f64> = zeta_s.iter().map(|z| 1.0 - z / z_top).collect();
        let db_w = vec![-1.0 / z_top; nz + 1];
        let db_s = vec![-1.0 / z_top; nz];

        let hs = vec![0.0; ny * nx];
        let msf = vec![1.0; ny * nx];
        let fcor = vec![1.1e-4; ny * nx];
        let cfg = NhGridConfig {
            nz: nz as i32,
            ny: ny as i32,
            nx: nx as i32,
            has_moist: 1,
            bc_x: 0,
            bc_y: 0,
            lbc_edges: 0,
            use_coriolis: 1,
            adv_order: 5,
            lbc_rim_width: 0,
            has_iau: 0,
            dx_m: 1500.0,
            dy_m: 1500.0,
            z_top_m: z_top,
            theta_sfc_k: 288.15,
            p_sfc_pa: 101_325.0,
            n_ref_s: 0.01,
            beta_offcenter: 0.55,
            div_damp_coeff: 0.03,
            km_const: 0.0,
            kh_const: 0.0,
            hyperdiff_coeff: 0.0,
            sponge_depth_m: 0.0,
            sponge_tau_max_s: 0.0,
            lbc_tau_max_s: 0.0,
            is_spherical: 0,
            polar_filter_lat_deg: 0.0,
        };
        let mut solver = NhSolver::new(
            cfg, &zeta_w, &zeta_s, &dzeta, &b_w, &b_s, &db_w, &db_s, &hs, &msf, &fcor,
        )
        .expect("Failed creating moist NhSolver");

        let mut u = vec![5.0f32; nz * ny * (nx + 1)];
        let mut v = vec![2.0f32; nz * (ny + 1) * nx];
        let mut w = vec![0.0f32; (nz + 1) * ny * nx];
        let mut thp = vec![0.0f32; nz * ny * nx];
        let mut pip = vec![0.0f32; nz * ny * nx];
        let mut qv = vec![0.008f32; nz * ny * nx];
        let mut qc = vec![0.0f32; nz * ny * nx];
        let mut qr = vec![0.0f32; nz * ny * nx];
        let mut qi = vec![0.0f32; nz * ny * nx];
        let mut qs = vec![0.0f32; nz * ny * nx];
        let mut qg = vec![0.0f32; nz * ny * nx];

        solver
            .upload_moist(&u, &v, &w, &thp, &pip, &qv, &qc, &qr, &qi, &qs, &qg)
            .unwrap();
        solver
            .set_physics_config(&NhPhysicsConfig::default())
            .unwrap();
        for _ in 0..3 {
            solver.step_physics(10.0).unwrap();
        }
        solver
            .download_moist(
                &mut u, &mut v, &mut w, &mut thp, &mut pip, &mut qv, &mut qc, &mut qr, &mut qi,
                &mut qs, &mut qg,
            )
            .unwrap();

        assert!(thp.iter().all(|x| x.is_finite()));
        assert!(qv.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(u.iter().all(|x| x.is_finite()));
    }
}
