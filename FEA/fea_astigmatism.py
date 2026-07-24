import numpy as np
from scipy.optimize import minimize_scalar
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
E_NOM = 2275000000.0
NU = 0.4
YIELD = 65000000.0
LENS_R = 0.025
TC = 0.0025
N_IDX = 1.55
S_BASE = 0.002

def D_plate(E=E_NOM):
    return E * TC ** 3 / (12 * (1 - NU ** 2))

def smp_modulus(cycle, E0=E_NOM, E_inf=2160000000.0, k=0.002):
    return float(E_inf + (E0 - E_inf) * np.exp(-k * cycle))

def sag_to_P(s):
    if abs(s) < 1e-07:
        return 0.0
    R = (LENS_R ** 2 + s ** 2) / (2 * abs(s))
    return float((N_IDX - 1) * np.sign(s) / R)

def P_to_sag(P):
    if abs(P) < 0.01:
        return S_BASE
    R = (N_IDX - 1) / abs(P)
    d = R ** 2 - LENS_R ** 2
    return float(np.sign(P) * (R - np.sqrt(d))) if d > 0 else None

def toroidal_fea(P_sphere, P_cylinder, axis_deg, E=E_NOM, n_phi=360, nr=200):
    D = D_plate(E)
    a = LENS_R
    phi_axis = np.radians(axis_deg)
    P_flat = P_sphere
    P_steep = P_sphere + abs(P_cylinder)
    s_flat = P_to_sag(P_flat)
    s_steep = P_to_sag(P_steep)
    if s_flat is None or s_steep is None:
        return {'error': f'Prescription outside physical range', 'achievable': False}
    w0_flat = s_flat - S_BASE
    w0_steep = s_steep - S_BASE
    q_flat = 64 * D * w0_flat / a ** 4
    q_steep = 64 * D * w0_steep / a ** 4
    q_mean = (q_flat + q_steep) / 2
    q_amp = (q_steep - q_flat) / 2
    phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    q_phi = q_mean + q_amp * np.cos(2 * (phi_arr - phi_axis))
    r_arr = np.linspace(0, a, nr)
    rho = r_arr / a
    W_field = np.outer(q_phi * a ** 4 / (64 * D), (1 - rho ** 2) ** 2)
    R0 = (LENS_R ** 2 + S_BASE ** 2) / (2 * S_BASE)
    z_base = R0 - np.sqrt(np.maximum(R0 ** 2 - r_arr ** 2, 0))
    Z_total = z_base[np.newaxis, :] + W_field
    s_phi = S_BASE + W_field[:, 0]
    P_phi = np.array([sag_to_P(s) for s in s_phi])
    j_flat = np.argmin(np.abs(phi_arr - phi_axis % (2 * np.pi)))
    j_steep = np.argmin(np.abs(phi_arr - (phi_axis + np.pi / 2) % (2 * np.pi)))
    P_flat_achieved = float(P_phi[j_flat])
    P_steep_achieved = float(P_phi[j_steep])
    P_cyl_achieved = abs(P_steep_achieved - P_flat_achieved)
    q_worst = float(np.max(np.abs(q_phi)))
    vm_max = 3 * q_worst * a ** 2 / (4 * TC ** 2)
    sf = YIELD / vm_max if vm_max > 1 else 99.0
    V0 = np.pi * S_BASE / 6 * (3 * a ** 2 + S_BASE ** 2) + np.pi * a ** 2 * (TC - S_BASE)

    def tc_conserved(s):
        cap = np.pi * s / 6 * (3 * a ** 2 + s ** 2)
        return (V0 - cap) / (np.pi * a ** 2) + s
    tc_flat = tc_conserved(s_flat)
    tc_steep = tc_conserved(s_steep)
    tc_min = min(tc_flat, tc_steep)
    R2, PHI2 = np.meshgrid(r_arr, phi_arr)
    X3 = R2 * np.cos(PHI2) * 1000.0
    Y3 = R2 * np.sin(PHI2) * 1000.0
    Z3 = Z_total * 1000.0
    return {'input': {'P_sphere': P_sphere, 'P_cylinder': P_cylinder, 'axis_deg': axis_deg}, 'E_GPa': round(E / 1000000000.0, 5), 'q_flat_Pa': round(float(q_flat), 4), 'q_steep_Pa': round(float(q_steep), 4), 'q_mean_Pa': round(float(q_mean), 4), 'q_amp_Pa': round(float(q_amp), 4), 'w0_flat_um': round(w0_flat * 1000000.0, 4), 'w0_steep_um': round(w0_steep * 1000000.0, 4), 's_flat_mm': round(float(s_flat) * 1000.0, 5), 's_steep_mm': round(float(s_steep) * 1000.0, 5), 'P_flat_achieved': round(P_flat_achieved, 5), 'P_steep_achieved': round(P_steep_achieved, 5), 'P_sphere_achieved': round(P_flat_achieved, 5), 'P_cylinder_achieved': round(P_cyl_achieved, 5), 'P_sphere_error': round(abs(P_flat_achieved - P_sphere), 5), 'P_cylinder_error': round(abs(P_cyl_achieved - abs(P_cylinder)), 5), 'tc_flat_mm': round(tc_flat * 1000.0, 4), 'tc_steep_mm': round(tc_steep * 1000.0, 4), 'tc_min_mm': round(tc_min * 1000.0, 4), 'tc_ok': tc_min > 0.0005, 'max_vm_MPa': round(vm_max / 1000000.0, 5), 'safety_factor': round(min(sf, 99), 3), 'achievable': sf > 1.0 and tc_min > 0.0005, 'X3_mm': X3.tolist(), 'Y3_mm': Y3.tolist(), 'Z3_mm': Z3.tolist(), 'P_phi': P_phi.tolist(), 'phi_deg': np.degrees(phi_arr).tolist(), 'r_mm': (r_arr * 1000.0).tolist(), 'q_phi': q_phi.tolist()}

def astig_optimizer(P_sphere, P_cylinder, axis_deg, cycle=0, n_screws=5):
    E = smp_modulus(cycle)
    D = D_plate(E)
    a = LENS_R
    phi_axis = np.radians(axis_deg)
    s_flat = P_to_sag(P_sphere)
    s_steep = P_to_sag(P_sphere + abs(P_cylinder))
    if s_flat is None or s_steep is None:
        return {'error': 'Prescription outside range'}
    w0_flat = s_flat - S_BASE
    w0_steep = s_steep - S_BASE
    q_flat = 64 * D * w0_flat / a ** 4
    q_steep = 64 * D * w0_steep / a ** 4
    q_mean = (q_flat + q_steep) / 2
    q_amp = (q_steep - q_flat) / 2
    screw_angles = np.arange(n_screws) * (360 / n_screws)
    E_nom = E_NOM
    fatigue_cf = E / E_nom
    screws = []
    for ang in screw_angles:
        theta = np.radians(ang)
        q_screw = q_mean + q_amp * np.cos(2 * (theta - phi_axis))
        q_screw_corr = q_screw
        contact_area = np.pi * 0.0005 ** 2
        F_N = abs(q_screw_corr) * contact_area
        pitch = 0.0004
        eff = 0.35
        T_max = 0.05
        turns = F_N * pitch / (2 * np.pi * eff * T_max)
        turns = max(0.05, round(turns / 0.25) * 0.25)
        direction = 'CW' if q_screw_corr >= 0 else 'CCW'
        screws.append({'id': int(np.where(screw_angles == ang)[0][0] + 1), 'angle_deg': float(ang), 'q_Pa': round(float(q_screw_corr), 4), 'F_N': round(float(F_N), 6), 'turns': float(turns), 'direction': direction})
    vm_max = 3 * max(abs(q_flat), abs(q_steep)) * a ** 2 / (4 * TC ** 2)
    sf = YIELD / vm_max if vm_max > 1 else 99.0
    return {'prescription': {'sphere': P_sphere, 'cylinder': P_cylinder, 'axis': axis_deg}, 'cycle': cycle, 'E_GPa': round(E / 1000000000.0, 5), 'q_mean_Pa': round(float(q_mean), 4), 'q_amp_Pa': round(float(q_amp), 4), 'w0_flat_um': round(w0_flat * 1000000.0, 4), 'w0_steep_um': round(w0_steep * 1000000.0, 4), 'screws': screws, 'max_vm_MPa': round(vm_max / 1000000.0, 4), 'safety_factor': round(min(sf, 99), 3), 'achievable': sf > 1.0}

def gen_training_data(n=800):
    print(f'  Generating {n} samples (sphere + astigmatism)...')
    spheres = np.random.uniform(1.5, 4.5, n)
    cylinders = np.random.uniform(-2.5, 0, n)
    axes = np.random.uniform(0, 180, n)
    cycles = np.random.randint(0, 501, n)
    X, y = ([], [])
    for sph, cyl, ax, cyc in zip(spheres, cylinders, axes, cycles):
        E_c = smp_modulus(int(cyc))
        try:
            r = toroidal_fea(sph, float(cyl), float(ax), E_c)
            if not r.get('achievable', False):
                continue
            X.append([r['q_mean_Pa'] / 1000.0, r['q_amp_Pa'] / 1000.0, float(ax) / 180.0, E_c / 1000000000.0, float(cyc)])
            y.append([r['P_sphere_achieved'], r['P_cylinder_achieved'], r['max_vm_MPa'], r['w0_flat_um'], r['w0_steep_um']])
        except:
            continue
    X, y = (np.array(X), np.array(y))
    print(f'    Dataset: {len(X)} samples')
    print(f'    P_sphere:   {y[:, 0].min():.3f} – {y[:, 0].max():.3f} D')
    print(f'    P_cylinder: {y[:, 1].min():.3f} – {y[:, 1].max():.3f} D')
    return (X, y)
if __name__ == '__main__':
    np.random.seed(42)
    print('=' * 65)
    print('AdaptivEyes — Astigmatic FEA v2')
    print('=' * 65)
    test_cases = [(3.5, 0.0, 0), (2.0, -0.75, 90), (1.5, -1.25, 45), (2.5, -1.5, 30), (3.0, -0.5, 180), (2.0, -2.0, 0)]
    print('\n[1] Toroidal FEA sweep...')
    fea_out = []
    for sph, cyl, ax in test_cases:
        r = toroidal_fea(sph, cyl, ax)
        compact = {k: v for k, v in r.items() if k not in ('X3_mm', 'Y3_mm', 'Z3_mm', 'P_phi', 'phi_deg', 'r_mm', 'q_phi')}
        fea_out.append(compact)
        status = 'OK' if r.get('achievable') else 'YIELD'
        print(f"  S{sph:+.2f}C{cyl:+.2f}x{ax:03d}: P_sph={r['P_sphere_achieved']:.4f}D P_cyl={r['P_cylinder_achieved']:.4f}D err_sph={r['P_sphere_error']:.5f}D err_cyl={r['P_cylinder_error']:.5f}D SF={r['safety_factor']:.2f} [{status}]")
    print('\n[2] Astigmatic optimizer (per-screw turns)...')
    opt_out = []
    for sph, cyl, ax in test_cases[:4]:
        for cyc in [0, 250, 500]:
            o = astig_optimizer(sph, cyl, ax, cyc)
            if 'error' in o:
                continue
            opt_out.append(o)
            if cyc == 0:
                print(f'\n  S{sph:+.2f}C{cyl:+.2f}x{ax:03d}:')
                print(f"    q_mean={o['q_mean_Pa']:.1f}Pa  q_amp={o['q_amp_Pa']:.1f}Pa  SF={o['safety_factor']}")
                for s in o['screws']:
                    print(f"    Screw{s['id']}@{s['angle_deg']:3.0f}°: {s['turns']:.2f}t {s['direction']}")
    print('\n[3] Training data generation...')
    X, y = gen_training_data(800)

    def jfix(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path = os.path.join(OUT, 'astig_fea_results.json')
    with open(path, 'w') as f:
        json.dump({'fea': fea_out, 'optimizer': opt_out, 'training_X': X.tolist(), 'training_y': y.tolist()}, f, indent=2, default=jfix)
    print(f'\n[Done] → {path}')
