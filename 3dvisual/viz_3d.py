import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.cm as cm
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fea_astigmatism import toroidal_fea, P_to_sag, sag_to_P
OUT = os.path.dirname(os.path.abspath(__file__))
LENS_R = 25.0
N_IDX = 1.55
S_BASE = 2.0
TC = 2.5

def make_lens_surface(s_mm, nr=80, nphi=120):
    r = np.linspace(0, LENS_R, nr)
    phi = np.linspace(0, 2 * np.pi, nphi)
    R2, P2 = np.meshgrid(r, phi)
    X = R2 * np.cos(P2)
    Y = R2 * np.sin(P2)
    R_sph = (LENS_R ** 2 + s_mm ** 2) / (2 * abs(s_mm)) if abs(s_mm) > 0.0001 else 1000000.0
    Z = R_sph - np.sqrt(np.maximum(R_sph ** 2 - R2 ** 2, 0))
    return (X, Y, Z)

def make_toroidal_surface(P_sphere, P_cyl, axis_deg, nr=80, nphi=120):
    r_arr = np.linspace(0, LENS_R * 0.001, nr)
    phi_arr = np.linspace(0, 2 * np.pi, nphi)
    R2, P2 = np.meshgrid(r_arr, phi_arr)
    phi_axis = np.radians(axis_deg)
    P_flat = P_sphere
    P_steep = P_sphere + abs(P_cyl)
    s_flat = P_to_sag(P_flat) or 0.002
    s_steep = P_to_sag(P_steep) or 0.002
    w0_flat = s_flat - 0.002
    w0_steep = s_steep - 0.002
    from fea_astigmatism import D_plate
    a = LENS_R * 0.001
    D = D_plate()
    q_flat = 64 * D * w0_flat / a ** 4
    q_steep = 64 * D * w0_steep / a ** 4
    q_mean = (q_flat + q_steep) / 2
    q_amp = (q_steep - q_flat) / 2
    q_phi = q_mean + q_amp * np.cos(2 * (P2 - phi_axis))
    rho2 = R2 / a
    W = q_phi * a ** 4 / (64 * D) * (1 - rho2 ** 2) ** 2
    R0_base = (a ** 2 + 0.002 ** 2) / (2 * 0.002)
    Z_base = R0_base - np.sqrt(np.maximum(R0_base ** 2 - R2 ** 2, 0))
    Z_total = (Z_base + W) * 1000.0
    X = R2 * np.cos(P2) * 1000.0
    Y = R2 * np.sin(P2) * 1000.0
    return (X, Y, Z_total, W * 1000000.0)

def set_equal_3d(ax, X, Y, Z):
    max_range = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())
    mid_x = (X.max() + X.min()) / 2
    mid_y = (Y.max() + Y.min()) / 2
    mid_z = (Z.max() + Z.min()) / 2
    z_range = 8.0
    ax.set_xlim(mid_x - LENS_R * 1.1, mid_x + LENS_R * 1.1)
    ax.set_ylim(mid_y - LENS_R * 1.1, mid_y + LENS_R * 1.1)
    ax.set_zlim(mid_z - z_range / 2, mid_z + z_range / 2)
fig1 = plt.figure(figsize=(20, 10), facecolor='white')
gs1 = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.35, wspace=0.3)
cases = [('Baseline\n+3.50D sph', P_to_sag(3.5), '#4A90D9', None, None, None), ('+2.00D sph', P_to_sag(2.0), '#E74C3C', None, None, None), ('+4.00D sph', P_to_sag(4.0), '#27AE60', None, None, None)]
for col, (label, s_mm, color, *_) in enumerate(cases[:3]):
    ax3d = fig1.add_subplot(gs1[0, col], projection='3d')
    if s_mm is None:
        continue
    X, Y, Z = make_lens_surface(s_mm * 1000.0)
    R2_vis = np.sqrt(X ** 2 + Y ** 2) * 0.001
    D = 2275000000.0 * (TC * 0.001) ** 3 / (12 * (1 - 0.4 ** 2))
    q = 64 * D * (s_mm - 0.002) / LENS_R ** 2 * 0.001 / (LENS_R * 0.001) ** 2
    q_Pa = 64 * D * (s_mm - 0.002) / (LENS_R * 0.001) ** 4
    Mr = q_Pa / 16 * ((1 + 0.4) * (LENS_R * 0.001) ** 2 - (3 + 0.4) * R2_vis ** 2)
    Mt = q_Pa / 16 * ((1 + 0.4) * (LENS_R * 0.001) ** 2 - (1 + 3 * 0.4) * R2_vis ** 2)
    sr = 6 * Mr / (TC * 0.001) ** 2
    st = 6 * Mt / (TC * 0.001) ** 2
    VM = np.sqrt(sr ** 2 - sr * st + st ** 2) / 1000000.0
    surf = ax3d.plot_surface(X, Y, Z, facecolors=cm.RdYlGn_r((VM - VM.min()) / (VM.max() - VM.min() + 1e-10)), alpha=0.88, linewidth=0, antialiased=True, shade=True)
    ax3d.set_xlim(-LENS_R, LENS_R)
    ax3d.set_ylim(-LENS_R, LENS_R)
    z_min = Z.min() - 0.5
    z_max = Z.max() + TC + 0.5
    ax3d.set_zlim(z_min - 1, z_min + 8)
    ax3d.set_xlabel('X (mm)', fontsize=8, labelpad=1)
    ax3d.set_ylabel('Y (mm)', fontsize=8, labelpad=1)
    ax3d.set_zlabel('Z (mm)', fontsize=8, labelpad=1)
    ax3d.set_title(f'{label}\ns={s_mm * 1000.0:.3f}mm', fontsize=9, fontweight='bold')
    ax3d.view_init(elev=22, azim=225)
    ax3d.tick_params(labelsize=7)
ax_cs = fig1.add_subplot(gs1[1, :3])
r_cs = np.linspace(-LENS_R, LENS_R, 600)
r_pos = np.abs(r_cs) * 0.001
sag_cases = [(P_to_sag(3.5), '#4A90D9', '+3.50D baseline'), (P_to_sag(2.0), '#E74C3C', '+2.00D'), (P_to_sag(3.0), '#E67E22', '+3.00D'), (P_to_sag(4.0), '#27AE60', '+4.00D')]
V0 = np.pi * 0.002 / 6 * (3 * (LENS_R * 0.001) ** 2 + 0.002 ** 2) + np.pi * (LENS_R * 0.001) ** 2 * (TC * 0.001 - 0.002)
for s_m, col, lbl in sag_cases:
    if s_m is None:
        continue
    s = s_m
    R = (LENS_R * 0.001) ** 2 + s ** 2
    R /= 2 * abs(s)
    z_top = (R - np.sqrt(np.maximum(R ** 2 - r_pos ** 2, 0))) * 1000.0
    cap = np.pi * s / 6 * (3 * (LENS_R * 0.001) ** 2 + s ** 2)
    tc_new = (V0 - cap) / (np.pi * (LENS_R * 0.001) ** 2) + s
    z_bot = z_top - tc_new * 1000.0
    ax_cs.plot(r_cs, z_top, color=col, lw=2.0, label=lbl)
    ax_cs.plot(r_cs, z_bot, color=col, lw=1.0, alpha=0.4)
    ax_cs.fill_between(r_cs, z_bot, z_top, alpha=0.06, color=col)
ax_cs.set_aspect('equal')
ax_cs.set_xlabel('r (mm)', fontsize=11)
ax_cs.set_ylabel('z (mm)', fontsize=11)
ax_cs.set_title('Cross-Section — True Proportions (equal X:Z scale)', fontsize=11, fontweight='bold')
ax_cs.legend(fontsize=9, loc='lower center', ncol=4)
ax_cs.grid(True, alpha=0.2)
ax_cs.set_xlim(-LENS_R - 2, LENS_R + 2)
s0 = P_to_sag(3.5)
R0 = (LENS_R * 0.001) ** 2 + s0 ** 2
R0 /= 2 * abs(s0)
z0top = (R0 - np.sqrt(R0 ** 2)) * 1000.0
ax_cs.annotate('', xy=(27.5, S_BASE), xytext=(27.5, 0), arrowprops=dict(arrowstyle='<->', color='navy', lw=1.5))
ax_cs.text(28.2, S_BASE / 2, f's₀={S_BASE:.1f}mm', color='navy', fontsize=9, va='center')
fig1.suptitle('AdaptivEyes — 3D Lens Geometry (Physically Accurate Scale)', fontsize=13, fontweight='bold')
p1 = os.path.join(OUT, 'viz_3d_spherical.png')
fig1.savefig(p1, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {p1}')
fig2 = plt.figure(figsize=(20, 9), facecolor='white')
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.38, wspace=0.3)
astig_cases = [('Spherical\nS+3.50 C0.00', 3.5, 0.0, 0), ('Astigmatic\nS+2.00 C-0.75 x90°', 2.0, -0.75, 90), ('Astigmatic\nS+1.50 C-1.25 x45°', 1.5, -1.25, 45)]
cmaps = [cm.Blues, cm.RdBu, cm.PiYG]
for col, (label, sph, cyl, ax_d) in enumerate(astig_cases):
    ax3 = fig2.add_subplot(gs2[0, col], projection='3d')
    X, Y, Z, W = make_toroidal_surface(sph, cyl, ax_d)
    w_norm = (W - W.min()) / (W.max() - W.min() + 1e-10)
    surf = ax3.plot_surface(X, Y, Z, facecolors=cm.RdYlBu_r(w_norm), alpha=0.88, linewidth=0, antialiased=True, shade=True)
    ax3.set_xlim(-LENS_R, LENS_R)
    ax3.set_ylim(-LENS_R, LENS_R)
    ax3.set_zlim(Z.min() - 0.5, Z.min() + 8)
    ax3.set_xlabel('X (mm)', fontsize=8, labelpad=1)
    ax3.set_ylabel('Y (mm)', fontsize=8, labelpad=1)
    ax3.set_zlabel('Z (mm)', fontsize=8, labelpad=1)
    ax3.set_title(label, fontsize=9, fontweight='bold')
    ax3.view_init(elev=22, azim=225)
    ax3.tick_params(labelsize=7)
    ax_m = fig2.add_subplot(gs2[1, col], projection='polar')
    phi_arr = np.linspace(0, 2 * np.pi, 360)
    phi_axis_r = np.radians(ax_d)
    P_flat = sph
    P_steep = sph + abs(cyl)
    s_f = P_to_sag(P_flat) or 0.002
    s_s = P_to_sag(P_steep) or 0.002
    w0_f = s_f - 0.002
    w0_s = s_s - 0.002
    from fea_astigmatism import D_plate
    a = LENS_R * 0.001
    D = D_plate()
    q_f = 64 * D * w0_f / a ** 4
    q_s = 64 * D * w0_s / a ** 4
    q_m = (q_f + q_s) / 2
    q_a = (q_s - q_f) / 2
    q_phi = q_m + q_a * np.cos(2 * (phi_arr - phi_axis_r))
    w0_phi = q_phi * a ** 4 / (64 * D)
    s_phi = 0.002 + w0_phi
    P_phi = np.array([(N_IDX - 1) * np.sign(s) / ((a ** 2 + s ** 2) / (2 * abs(s))) if abs(s) > 1e-07 else 0 for s in s_phi])
    colors_p = cm.RdYlGn((P_phi - P_phi.min()) / (P_phi.max() - P_phi.min() + 1e-10))
    for i in range(len(phi_arr) - 1):
        ax_m.plot([phi_arr[i], phi_arr[i + 1]], [P_phi[i], P_phi[i + 1]], color=colors_p[i], lw=2.5)
    ax_m.set_title(f'Power map (D)\nFlat={P_flat:.2f}D Steep={P_steep:.2f}D', fontsize=8, fontweight='bold', pad=8)
    ax_m.set_ylim(0, max(P_phi.max() * 1.1, 0.1))
    ax_m.tick_params(labelsize=7)
fig2.suptitle('AdaptivEyes — Toroidal Lens Geometry (Spherical + Astigmatic)', fontsize=13, fontweight='bold')
p2 = os.path.join(OUT, 'viz_3d_toroidal.png')
fig2.savefig(p2, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {p2}')
plt.close('all')
print('Done')
