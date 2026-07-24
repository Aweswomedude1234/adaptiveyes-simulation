import numpy as np
import json, os
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
E_NOM = 2275000000.0
NU = 0.4
YIELD = 65000000.0
LENS_R = 0.025
LENS_TC = 0.003
N_IDX = 1.55
S_BASE = 0.002

def flexural_rigidity(E=E_NOM):
    return E * LENS_TC ** 3 / (12 * (1 - NU ** 2))

def smp_modulus(cycle, E0=E_NOM, E_inf=2160000000.0, k=0.002):
    return float(E_inf + (E0 - E_inf) * np.exp(-k * cycle))

def sag_to_power(s):
    if abs(s) < 1e-07:
        return 0.0
    R = (LENS_R ** 2 + s ** 2) / (2 * abs(s))
    return float((N_IDX - 1) * np.sign(s) / R)

def power_to_sag(P):
    if abs(P) < 0.01:
        return S_BASE
    R = (N_IDX - 1) / abs(P)
    disc = R ** 2 - LENS_R ** 2
    if disc < 0:
        return None
    return float(np.sign(P) * (R - np.sqrt(disc)))

def plate_fea(q_Pa, E=E_NOM, nr=50):
    D = flexural_rigidity(E)
    a = LENS_R
    r = np.linspace(0, a, nr)
    w = q_Pa * a ** 4 / (64 * D) * (1 - (r / a) ** 2) ** 2
    Mr = q_Pa / 16 * ((1 + NU) * a ** 2 - (3 + NU) * r ** 2)
    Mt = q_Pa / 16 * ((1 + NU) * a ** 2 - (1 + 3 * NU) * r ** 2)
    sigma_r = 6 * Mr / LENS_TC ** 2
    sigma_t = 6 * Mt / LENS_TC ** 2
    vm = np.sqrt(sigma_r ** 2 - sigma_r * sigma_t + sigma_t ** 2)
    w0 = w[0]
    s_new = S_BASE + w0
    P_new = sag_to_power(s_new)
    max_vm = float(np.max(np.abs(vm)))
    sf = float(YIELD / max_vm) if max_vm > 1000.0 else 999.0
    return {'q_Pa': q_Pa, 'q_MPa': round(q_Pa / 1000000.0, 6), 'E_GPa': round(E / 1000000000.0, 5), 'D_Nm': round(D, 5), 'w0_mm': round(w0 * 1000.0, 6), 'delta_sag_mm': round(w0 * 1000.0, 6), 'sagitta_mm': round(s_new * 1000.0, 5), 'optical_power_D': round(P_new, 5), 'max_vm_MPa': round(max_vm / 1000000.0, 5), 'safety_factor': round(min(sf, 999.0), 3), 'r_mm': (r * 1000.0).tolist(), 'w_mm': (w * 1000.0).tolist(), 'vm_MPa': (vm / 1000000.0).tolist(), 'sigma_r_MPa': (sigma_r / 1000000.0).tolist()}

def pressure_for_prescription(target_P_D, E=E_NOM):
    D = flexural_rigidity(E)
    s_t = power_to_sag(target_P_D)
    if s_t is None:
        return None
    delta_w = s_t - S_BASE
    q = 64 * D * delta_w / LENS_R ** 4
    sigma_max = 3 * abs(q) * LENS_R ** 2 / (4 * LENS_TC ** 2)
    sf = YIELD / sigma_max if sigma_max > 0 else 999.0
    return {'target_P_D': target_P_D, 's_target_mm': round(s_t * 1000.0, 4), 'delta_w_mm': round(delta_w * 1000.0, 5), 'q_Pa': round(q, 2), 'q_MPa': round(q / 1000000.0, 6), 'sigma_max_MPa': round(sigma_max / 1000000.0, 4), 'safety_factor': round(min(sf, 999.0), 3), 'achievable': sf > 1.0 and s_t is not None}

def run_cycle_simulation(target_P_D=3.5, n_cycles=500, step=10):
    records = []
    ref = pressure_for_prescription(target_P_D, E_NOM)
    q_nominal = ref['q_Pa']
    for cyc in range(0, n_cycles + 1, step):
        E_c = smp_modulus(cyc)
        r = plate_fea(q_nominal, E_c)
        drift = round(r['optical_power_D'] - target_P_D, 6)
        q_corrected = q_nominal * (E_NOM / E_c)
        records.append({'cycle': cyc, 'E_GPa': r['E_GPa'], 'q_nominal_Pa': round(q_nominal, 2), 'q_corrected_Pa': round(q_corrected, 2), 'w0_mm': r['w0_mm'], 'sagitta_mm': r['sagitta_mm'], 'optical_power_D': r['optical_power_D'], 'focal_drift_D': drift, 'max_vm_MPa': r['max_vm_MPa'], 'safety_factor': r['safety_factor'], 'within_tol': abs(drift) < 0.12})
    return records

def generate_training_data(n=800):
    q_limit = YIELD * 4 * LENS_TC ** 2 / (3 * LENS_R ** 2)
    qs = np.random.uniform(-q_limit * 0.9, q_limit * 0.9, n)
    cycs = np.random.randint(0, 501, n)
    X, y = ([], [])
    for q, cyc in zip(qs, cycs):
        E_c = smp_modulus(int(cyc))
        r = plate_fea(q, E_c)
        if r['max_vm_MPa'] > 60.0:
            continue
        X.append([q / 1000.0, E_c / 1000000000.0, float(cyc)])
        y.append([r['delta_sag_mm'], r['optical_power_D'], r['max_vm_MPa'], r['w0_mm']])
    return (np.array(X), np.array(y))

def train_ml_model(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sx = StandardScaler()
    sy = StandardScaler()
    Xtr_s = sx.fit_transform(Xtr)
    Xte_s = sx.transform(Xte)
    ytr_s = sy.fit_transform(ytr)
    mdl = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=3000, random_state=42, early_stopping=True, validation_fraction=0.15, n_iter_no_change=40, verbose=False)
    mdl.fit(Xtr_s, ytr_s)
    ypr = sy.inverse_transform(mdl.predict(Xte_s))
    r2 = r2_score(yte, ypr)
    mae = mean_absolute_error(yte, ypr)
    return (mdl, sx, sy, {'r2': round(float(r2), 4), 'mae': round(float(mae), 6), 'n_train': len(Xtr), 'n_test': len(Xte), 'n_iter': mdl.n_iter_})

def ml_predict_pressure(target_D, cycle, mdl, sx, sy, n_pts=400):
    q_lim = YIELD * 4 * LENS_TC ** 2 / (3 * LENS_R ** 2)
    qs = np.linspace(-q_lim * 0.9, q_lim * 0.9, n_pts)
    E_c = smp_modulus(cycle) / 1000000000.0
    Xq = np.column_stack([qs / 1000.0, np.full(n_pts, E_c), np.full(n_pts, float(cycle))])
    ypr = sy.inverse_transform(mdl.predict(sx.transform(Xq)))
    powers = ypr[:, 1]
    idx = int(np.argmin(np.abs(powers - target_D)))
    return {'target_D': target_D, 'q_Pa': round(float(qs[idx]), 2), 'q_MPa': round(float(qs[idx]) / 1000000.0, 6), 'pred_D': round(float(powers[idx]), 5), 'error_D': round(float(abs(powers[idx] - target_D)), 5), 'cycle': cycle, 'E_GPa': round(E_c, 5)}

def screw_rotation_map(sphere_D, cyl_D, axis_deg, screw_angles_deg, cycle, mdl, sx, sy):
    res = ml_predict_pressure(sphere_D, cycle, mdl, sx, sy)
    q_total = res['q_Pa']
    n = len(screw_angles_deg)
    screws = []
    for i, ang in enumerate(screw_angles_deg):
        theta = np.radians(ang)
        phi = np.radians(axis_deg)
        w_sph = 1 / n
        w_cyl = np.cos(2 * (theta - phi))
        q_screw = q_total * w_sph + cyl_D / 2 * abs(w_cyl) * np.sign(q_total)
        contact_r = 0.0005
        contact_area = np.pi * contact_r ** 2
        F_N = abs(q_screw) * contact_area
        pitch = 0.0004
        eff = 0.35
        T_max = 0.05
        turns = F_N * pitch / (2 * np.pi * eff * T_max)
        turns = max(0.05, round(turns, 2))
        direction = 'CW' if q_screw > 0 else 'CCW'
        screws.append({'id': i + 1, 'angle_deg': ang, 'q_Pa': round(q_screw, 2), 'turns': turns, 'direction': direction})
    return screws

def build_3d_lens_surface(s_mm, nr=60, nphi=80):
    s = s_mm * 0.001
    R = (LENS_R ** 2 + s ** 2) / (2 * abs(s)) if abs(s) > 1e-06 else 1000000.0
    r_vals = np.linspace(0, LENS_R, nr)
    phi_vals = np.linspace(0, 2 * np.pi, nphi)
    R_grid, Phi_grid = np.meshgrid(r_vals, phi_vals)
    X = R_grid * np.cos(Phi_grid) * 1000.0
    Y = R_grid * np.sin(Phi_grid) * 1000.0
    Z = np.where(R > abs(r_vals[np.newaxis, :]), (R - np.sqrt(np.maximum(R ** 2 - R_grid ** 2, 0))) * np.sign(s), 0.0) * 1000.0
    return (X, Y, Z)

def build_vm_field(q_Pa, E=E_NOM, nr=60, nphi=80):
    r_1d = np.linspace(0, LENS_R, nr)
    r_data = plate_fea(q_Pa, E, nr=nr)
    vm_1d = np.array(r_data['vm_MPa'])
    r_vals = np.linspace(0, LENS_R, nr)
    phi_vals = np.linspace(0, 2 * np.pi, nphi)
    R_grid, _ = np.meshgrid(r_vals, phi_vals)
    VM_grid = np.interp(R_grid.ravel(), r_vals, vm_1d).reshape(nphi, nr)
    return VM_grid

def create_visualization(fea_results, cycle_data, screw_results, target_Ds):
    figs = []
    baseline_sag = S_BASE * 1000.0
    target_sag = (power_to_sag(3.5) or S_BASE) * 1000.0
    Xb, Yb, Zb = build_3d_lens_surface(baseline_sag)
    Xt, Yt, Zt = build_3d_lens_surface(target_sag)
    q_used = pressure_for_prescription(3.5)['q_Pa']
    VM = build_vm_field(q_used, E_NOM)
    fig1 = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]], subplot_titles=('Baseline lens (S=2.00mm, +3.50D)', 'Deformed lens (+3.50D target) with Von Mises stress'))
    fig1.add_trace(go.Surface(x=Xb, y=Yb, z=Zb, colorscale='Blues', showscale=False, opacity=0.85, name='Baseline'), row=1, col=1)
    fig1.add_trace(go.Surface(x=Xt, y=Yt, z=Zt, surfacecolor=VM, colorscale='RdYlGn_r', colorbar=dict(title='Von Mises (MPa)', x=1.02, len=0.8), name='Deformed + VM stress'), row=1, col=2)
    fig1.update_layout(title='AdaptivEyes — 3D Lens Deformation & Stress Field', height=550, scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'), scene2=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'), paper_bgcolor='white', plot_bgcolor='white', font=dict(family='Arial', size=12))
    P_vals = np.arange(-2.0, 4.25, 0.25)
    achievable, q_vals, sf_vals, vm_vals = ([], [], [], [])
    for P in P_vals:
        res = pressure_for_prescription(P)
        achievable.append(res['achievable'])
        q_vals.append(res['q_MPa'])
        sf_vals.append(min(res['safety_factor'], 20))
        vm_vals.append(res['sigma_max_MPa'])
    fig2 = make_subplots(rows=2, cols=1, subplot_titles=('Required pressure per prescription', 'Safety factor across range'), vertical_spacing=0.15)
    colors = ['#1D9E75' if a else '#D85A30' for a in achievable]
    fig2.add_trace(go.Bar(x=[f'{p:+.2f}' for p in P_vals], y=q_vals, marker_color=colors, name='Pressure (MPa)', hovertemplate='%{x} D<br>q=%{y:.4f} MPa<extra></extra>'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=[f'{p:+.2f}' for p in P_vals], y=sf_vals, mode='lines+markers', line=dict(color='#3B8BD4', width=2), marker=dict(size=6), name='Safety factor'), row=2, col=1)
    fig2.add_hline(y=1.5, line_dash='dash', line_color='red', annotation_text='Min SF=1.5', row=2, col=1)
    fig2.update_layout(height=550, title='Prescription Range Analysis', paper_bgcolor='white', font=dict(family='Arial', size=12))
    fig2.update_yaxes(title_text='q (MPa)', row=1, col=1)
    fig2.update_yaxes(title_text='Safety factor', row=2, col=1)
    cyc_x = [r['cycle'] for r in cycle_data]
    drift_y = [r['focal_drift_D'] for r in cycle_data]
    E_y = [r['E_GPa'] for r in cycle_data]
    vm_y = [r['max_vm_MPa'] for r in cycle_data]
    qcorr_y = [r['q_corrected_Pa'] / 1000.0 for r in cycle_data]
    fig3 = make_subplots(rows=2, cols=2, subplot_titles=('Focal drift over 500 cycles', "Young's modulus degradation", 'Von Mises stress per cycle', 'Corrected pressure per cycle'), vertical_spacing=0.18, horizontal_spacing=0.12)
    fig3.add_trace(go.Scatter(x=cyc_x, y=drift_y, mode='lines', line=dict(color='#D85A30', width=2), name='Focal drift (D)'), row=1, col=1)
    fig3.add_hline(y=0.12, line_dash='dash', line_color='gray', annotation_text='+0.12D tol', row=1, col=1)
    fig3.add_hline(y=-0.12, line_dash='dash', line_color='gray', annotation_text='-0.12D tol', row=1, col=1)
    fig3.add_trace(go.Scatter(x=cyc_x, y=E_y, mode='lines', line=dict(color='#3B8BD4', width=2), name='E (GPa)'), row=1, col=2)
    fig3.add_trace(go.Scatter(x=cyc_x, y=vm_y, mode='lines', line=dict(color='#9F4FDD', width=2), name='VM (MPa)'), row=2, col=1)
    fig3.add_hline(y=65, line_dash='dash', line_color='red', annotation_text='Yield', row=2, col=1)
    fig3.add_trace(go.Scatter(x=cyc_x, y=qcorr_y, mode='lines', line=dict(color='#1D9E75', width=2), name='q corrected (kPa)'), row=2, col=2)
    fig3.update_layout(height=600, title='500-Cycle Fatigue — Cycle-by-Cycle FEA', showlegend=False, paper_bgcolor='white', font=dict(family='Arial', size=12))
    for r, c, lbl in [(1, 1, 'Drift (D)'), (1, 2, 'E (GPa)'), (2, 1, 'VM (MPa)'), (2, 2, 'q corr (kPa)')]:
        fig3.update_yaxes(title_text=lbl, row=r, col=c)
        fig3.update_xaxes(title_text='Cycle', row=r, col=c)
    fig4 = go.Figure()
    for sr in screw_results:
        ang = np.radians(sr['angle_deg'])
        r_turns = sr['turns']
        color = '#1D9E75' if sr['direction'] == 'CW' else '#D85A30'
        fig4.add_trace(go.Scatterpolar(r=[0, r_turns], theta=[sr['angle_deg'], sr['angle_deg']], mode='lines+markers+text', line=dict(color=color, width=4), marker=dict(size=[6, 14], color=color), text=['', f"S{sr['id']}<br>{r_turns:.2f}t<br>{sr['direction']}"], textposition='top center', name=f"Screw {sr['id']}", showlegend=True))
    fig4.update_layout(polar=dict(radialaxis=dict(visible=True, title='Turns')), title='Screw Rotation Map — Per-Screw Actuation', height=500, paper_bgcolor='white', font=dict(family='Arial', size=12), annotations=[dict(text='Green=CW (tighten)   Red=CCW (loosen)', xref='paper', yref='paper', x=0.5, y=-0.08, showarrow=False, font=dict(size=11))])
    frames = []
    sag_vals = np.linspace(S_BASE * 1000.0 * 0.7, S_BASE * 1000.0 * 1.5, 20)
    for sv in sag_vals:
        X3, Y3, Z3 = build_3d_lens_surface(sv)
        frames.append(go.Frame(data=[go.Surface(x=X3, y=Y3, z=Z3, colorscale='Blues', showscale=False, opacity=0.9)], name=f's={sv:.3f}'))
    Xi, Yi, Zi = build_3d_lens_surface(S_BASE * 1000.0 * 0.7)
    fig5 = go.Figure(data=[go.Surface(x=Xi, y=Yi, z=Zi, colorscale='Blues', showscale=False, opacity=0.9)], frames=frames, layout=go.Layout(title='Lens Shape Transformation (Animated)', height=500, updatemenus=[dict(type='buttons', showactive=False, y=1.1, x=0.5, xanchor='center', buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, mode='immediate')]), dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])])], sliders=[dict(steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')], label=f"{float(f.name.split('=')[1]):.2f}mm", method='animate') for f in frames], x=0.1, len=0.8, y=0, currentvalue=dict(prefix='Sagitta: ', suffix=' mm'))], scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'), paper_bgcolor='white', font=dict(family='Arial', size=12)))
    html_parts = []
    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5]):
        html_parts.append(fig.to_html(full_html=i == 0, include_plotlyjs=i == 0, div_id=f'fig{i + 1}'))
    combined = html_parts[0].replace('</body></html>', '')
    for part in html_parts[1:]:
        start = part.find('<div id=')
        end = part.rfind('</script>') + len('</script>')
        combined += '\n' + part[start:end]
    combined += '\n</body></html>'
    return combined
if __name__ == '__main__':
    np.random.seed(42)

    def jfix(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    print('=' * 60)
    print('AdaptivEyes Complete Simulation')
    print('=' * 60)
    print('\n[1] FEA baseline + prescription range sweep...')
    base = plate_fea(0, E_NOM)
    print(f"    Baseline power:  {base['optical_power_D']:.5f} D")
    print(f"    Baseline sagitta:{base['sagitta_mm']:.5f} mm")
    range_results = {}
    for P in np.arange(-2.0, 4.25, 0.25):
        res = pressure_for_prescription(round(P, 2))
        fea = plate_fea(res['q_Pa'], E_NOM)
        range_results[f'{P:+.2f}'] = {**res, **{k: v for k, v in fea.items() if k not in ('r_mm', 'w_mm', 'vm_MPa', 'sigma_r_MPa')}}
        status = 'OK' if res['achievable'] else 'YIELD'
        print(f"    {P:+.2f}D: q={res['q_MPa']:.4f}MPa  SF={res['safety_factor']:.2f}  [{status}]")
    print('\n[2] 500-cycle fatigue simulation (target=+3.50D)...')
    cycle_data = run_cycle_simulation(target_P_D=3.5, n_cycles=500, step=10)
    max_drift = max((abs(r['focal_drift_D']) for r in cycle_data))
    print(f'    Max focal drift over 500 cycles: {max_drift:.6f} D')
    print(f"    All within ±0.12D tolerance: {('YES' if all((r['within_tol'] for r in cycle_data)) else 'NO')}")
    print('\n[3] Generating FEA training data (800 samples)...')
    X, y = generate_training_data(800)
    print(f'    Dataset: {X.shape[0]} samples')
    print(f'    Power range in training: {y[:, 1].min():.4f} – {y[:, 1].max():.4f} D')
    print('\n[4] Training ML model on FEA data...')
    mdl, sx, sy, metrics = train_ml_model(X, y)
    print(f"    R² = {metrics['r2']}   MAE = {metrics['mae']:.6f}   Epochs = {metrics['n_iter']}")
    print('\n[5] Screw rotation maps for test prescriptions...')
    screw_angles = [0, 72, 144, 216, 288]
    test_rxs = [(3.5, 0.0, 0, 0), (2.0, -0.75, 90, 0), (1.5, -1.25, 45, 0), (3.5, 0.0, 0, 250), (3.5, 0.0, 0, 500)]
    ml_results = []
    for sph, cyl, ax, cyc in test_rxs:
        pred = ml_predict_pressure(sph, cyc, mdl, sx, sy)
        screws = screw_rotation_map(sph, cyl, ax, screw_angles, cyc, mdl, sx, sy)
        pred['cylinder'] = cyl
        pred['axis'] = ax
        pred['screw_map'] = screws
        ml_results.append(pred)
        print(f'\n  Rx S{sph:+.2f} C{cyl:+.2f} x{ax:03d} cyc={cyc}:')
        print(f"    q={pred['q_MPa']:.5f}MPa  pred={pred['pred_D']:.5f}D  err={pred['error_D']:.5f}D")
        for s in screws:
            print(f"    Screw {s['id']} @{s['angle_deg']:3d}°: {s['turns']:.2f} turns {s['direction']}")
    print('\n[6] Building 3D visualization...')
    demo_screws = ml_results[0]['screw_map']
    html = create_visualization(range_results, cycle_data, demo_screws, [r[0] for r in test_rxs])
    viz_path = os.path.join(OUT_DIR, 'lens_3d.html')
    with open(viz_path, 'w') as f:
        f.write(html)
    print(f'    Saved → {viz_path}')
    save = {'fea_baseline': {k: v for k, v in base.items() if not isinstance(v, list)}, 'fea_range': range_results, 'cycle_data': cycle_data, 'ml_metrics': metrics, 'ml_results': ml_results, 'training_X': X.tolist(), 'training_y': y.tolist()}
    jpath = os.path.join(OUT_DIR, 'fea_results.json')
    with open(jpath, 'w') as f:
        json.dump(save, f, indent=2, default=jfix)
    print(f'    JSON → {jpath}')
    print('\n[Complete]')
