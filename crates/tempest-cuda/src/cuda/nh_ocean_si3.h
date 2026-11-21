#ifndef TEMPEST_NWP_V2_NH_OCEAN_SI3_H
#define TEMPEST_NWP_V2_NH_OCEAN_SI3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TEMPEST_SI3_NCAT 5
#define TEMPEST_SI3_NLAY 4
#define TEMPEST_OCEAN_MAX_NZ 64

typedef struct NhOceanConfig {
    int32_t nz;              /* Ocean z-levels (surface k=0 .. bottom k=nz-1), <= 64 */
    int32_t ny;
    int32_t nx;
    int32_t bc_x;            /* 1 = periodic in x (global D1), 0 = regional closed/clamped */
    int32_t bc_y;            /* 0 = wall / pole */
    int32_t is_spherical;    /* 1 = spherical lat-lon metric (dx_eff = dx_m / msf, dy_eff = dy_m) */
    int32_t enable_dynamics; /* 1 = 3D baroclinic + split-explicit barotropic + TKE; 0 = 1D column + ice */
    int32_t enable_si3;      /* 1 = SI3 5-category sea-ice rheology + BL99 + ITD */
    int32_t n_bt_substeps;   /* Barotropic split-explicit substeps per ocean step (e.g. 12) */
    int32_t n_evp_substeps;  /* SI3 Bouillon mEVP rheology substeps per step (e.g. 60) */
    double dx_m;             /* Equatorial / nominal grid spacing in x [m] */
    double dy_m;             /* Grid spacing in y [m] */
    double rho0;             /* Reference seawater density [kg m^-3], NEMO 1026.0 */
    double cp_o;             /* Seawater specific heat [J kg^-1 K^-1], NEMO 3991.867957 */
    double ah_lap;           /* Horizontal Laplacian viscosity/diffusivity [m^2 s^-1] */
    double avm_bg;           /* Background vertical eddy viscosity [m^2 s^-1] */
    double avt_bg;           /* Background vertical eddy diffusivity [m^2 s^-1] */
    double cd_bot;           /* Quadratic ocean bottom drag coefficient [-] */
    double p_ref_pa;         /* Inverse-barometer reference SLP [Pa], 101325.0 */
    double rn_pstar;         /* Hibler/SI3 ice strength scale [N m^-2], 2.0e4 */
    double rn_crhg;          /* Ice strength concentration decay [-], 20.0 */
    double rn_ecc;           /* Yield ellipse eccentricity [-], 2.0 */
    double rn_cd_io;         /* Ice-ocean quadratic drag coefficient [-], 5.0e-3 */
    double rn_hinew;         /* Frazil new-ice formation thickness [m], 0.10 */
    double rn_amax;          /* Maximum sea-ice concentration cap [-], 0.997 */
} NhOceanConfig;

/* Static grid & bathymetry arrays (all [ny, nx] float32, dz_m is [nz] float64). */
typedef struct NhOceanStaticPtrs {
    const double* dz_m;       /* Level thicknesses from surface k=0 downward [m], len nz */
    const float* msf;         /* Map scale factor (1/cos(lat) for D1, LCC k for D2/D3) */
    const float* f_cor;       /* Coriolis parameter 2*Omega*sin(lat) [s^-1] */
    const float* bathy_m;     /* Positive ocean depth [m]; 0 on land/inland lakes */
    const float* ocean_mask;  /* 1.0 = wet ocean cell, 0.0 = land or inland lake */
} NhOceanStaticPtrs;

/* Mutable / constant host pointers for coupled NEMO 5.0.2 + SI3 5-category state.
 * Strict SI units throughout:
 *   temp_k     [K]      ([nz, ny, nx])
 *   sal_psu    [g kg-1] ([nz, ny, nx])
 *   u_oce,v_oce[m s-1]  ([nz, ny, nx])
 *   tke        [m2 s-2] ([nz, ny, nx])
 *   ssh        [m]      ([ny, nx])
 *   ub, vb     [m s-1]  ([ny, nx])
 *   mld        [m]      ([ny, nx])
 *   a_i..v_ip  5-cat    ([5, ny, nx]), e_i ([20, ny, nx] J m^-2, >0 to melt)
 *   t_su, t_s  [K]      ([5, ny, nx])
 *   u_ice,v_ice[m s-1]  ([ny, nx])
 *   stress*    [N m^-1] ([ny, nx])
 *   ice_frac, ice_thick, ice_tsfc, ice_albedo: aggregate surface state ([ny, nx])
 */
typedef struct NhOceanStatePtrs {
    float* temp_k;
    float* sal_psu;
    float* u_oce;
    float* v_oce;
    float* tke;
    float* avm;
    float* avt;
    float* ssh;
    float* ub;
    float* vb;
    float* mld;
    float* a_i;
    float* v_i;
    float* v_s;
    float* sv_i;
    float* e_i;
    float* t_su;
    float* t_s;
    float* a_ip;
    float* v_ip;
    float* u_ice;
    float* v_ice;
    float* stress1;
    float* stress2;
    float* stress12;
    float* ice_frac;
    float* ice_thick;
    float* ice_tsfc;
    float* ice_albedo;
} NhOceanStatePtrs;

typedef struct NhOceanConstStatePtrs {
    const float* temp_k;
    const float* sal_psu;
    const float* u_oce;
    const float* v_oce;
    const float* tke;
    const float* avm;
    const float* avt;
    const float* ssh;
    const float* ub;
    const float* vb;
    const float* mld;
    const float* a_i;
    const float* v_i;
    const float* v_s;
    const float* sv_i;
    const float* e_i;
    const float* t_su;
    const float* t_s;
    const float* a_ip;
    const float* v_ip;
    const float* u_ice;
    const float* v_ice;
    const float* stress1;
    const float* stress2;
    const float* stress12;
    const float* ice_frac;
    const float* ice_thick;
    const float* ice_tsfc;
    const float* ice_albedo;
} NhOceanConstStatePtrs;

/* Atmospheric surface forcing arrays [ny, nx] float32 in strict SI units. */
typedef struct NhOceanForcingPtrs {
    const float* u10m;       /* 10 m grid-x wind [m s^-1] */
    const float* v10m;       /* 10 m grid-y wind [m s^-1] */
    const float* t2m;        /* 2 m air temperature [K] */
    const float* q2m;        /* 2 m specific humidity [kg kg^-1] */
    const float* mslp;       /* Mean sea-level pressure [Pa] */
    const float* sw_down;    /* Downward surface shortwave radiation [W m^-2] */
    const float* lw_down;    /* Downward surface longwave radiation [W m^-2] */
    const float* precip_liq; /* Liquid precipitation rate [kg m^-2 s^-1] */
    const float* precip_ice; /* Solid (snow/graupel) precipitation rate [kg m^-2 s^-1] */
    const float* cld_fra;    /* Total cloud fraction [0..1] */
} NhOceanForcingPtrs;

typedef struct NhOceanSi3Handle NhOceanSi3Handle;

uint64_t nh_ocean_si3_estimate_vram_bytes(int32_t nz, int32_t ny, int32_t nx, int32_t enable_si3);

NhOceanSi3Handle* nh_ocean_si3_create(const NhOceanConfig* cfg, const NhOceanStaticPtrs* statics);
void nh_ocean_si3_destroy(NhOceanSi3Handle* h);

int32_t nh_ocean_si3_upload_state(NhOceanSi3Handle* h, const NhOceanConstStatePtrs* src);
int32_t nh_ocean_si3_download_state(NhOceanSi3Handle* h, const NhOceanStatePtrs* dst);
int32_t nh_ocean_si3_step(NhOceanSi3Handle* h, const NhOceanForcingPtrs* forcing, double dt_s);

/* Exact Roquet et al. (2015) TEOS-10 in-situ density [kg m^-3] & NEMO TEOS-10 freezing point [C]. */
double nh_ocean_si3_teos10_rho(double ct_c, double sa_psu, double depth_m);
double nh_ocean_si3_freezing_temp_c(double sa_psu, double depth_m);
const char* nh_ocean_si3_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* TEMPEST_NWP_V2_NH_OCEAN_SI3_H */
