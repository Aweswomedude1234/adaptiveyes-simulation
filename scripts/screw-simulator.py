"""
AdaptivEyes — Screw Revolution Calculator (Standalone)
=======================================================
INPUT:  Current prescription (sphere, cylinder, axis) for each eye
        Target prescription (sphere, cylinder, axis) for each eye
OUTPUT: Per-screw turns and direction diagram for each eye

Usage:
    Edit the PRESCRIPTIONS dict at the bottom, then run:
    python screw_revolution_calculator.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.dirname(os.path.abspath(__file__))

LENS_R  = 0.025
N_IDX   = 1.55
S_BASE  = 0.002
E_NOM   = 2.275e9
NU      = 0.40
TC      = 0.0025
YIELD   = 65e6

N_SCREWS     = 5
SCREW_ANGLES = np.arange(N_SCREWS) * (360/N_SCREWS)
PITCH_M      = 0.4e-3
EFF          = 0.35
T_MAX_NM     = 0.05

def D_plate(E=E_NOM):
    return E * TC**3 / (12*(1-NU**2))

def smp_modulus(cycle, E0=E_NOM, E_inf=2.16e9, k=0.002):
    return float(E_inf + (E0-E_inf)*np.exp(-k*cycle))

def P_to_sag(P):
    if abs(P) < 0.01: return S_BASE
    R = (N_IDX-1)/abs(P)
    d = R**2 - LENS_R**2
    return float(np.sign(P)*(R-np.sqrt(d))) if d > 0 else None

def sag_to_P(s):
    if abs(s) < 1e-7: return 0.0
    R = (LENS_R**2 + s**2)/(2*abs(s))
    return float((N_IDX-1)*np.sign(s)/R)

def compute_screw_plan(sph_old, cyl_old, ax_old,
                        sph_new, cyl_new, ax_new,
                        cycle=0):
    """
    Given current and new prescription, compute per-screw turns.

    Physics:
    - Flat meridian (at cylinder axis): q_flat drives P_flat = sph_new
    - Steep meridian (+90°):            q_steep drives P_steep = sph_new + |cyl_new|
    - Each screw contributes: q(θ) = q_mean + q_amp·cos(2·(θ − φ_axis))
    - Turns from force via M2 lead-screw mechanics
    - Delta from current prescription: only the CHANGE is applied
    """
    E   = smp_modulus(cycle)
    D   = D_plate(E)
    a   = LENS_R
    phi = np.radians(ax_new)

    P_flat_new  = sph_new
    P_steep_new = sph_new + abs(cyl_new)
    s_flat_new  = P_to_sag(P_flat_new)  or S_BASE
    s_steep_new = P_to_sag(P_steep_new) or S_BASE

    P_flat_old  = sph_old
    P_steep_old = sph_old + abs(cyl_old)
    s_flat_old  = P_to_sag(P_flat_old)  or S_BASE
    s_steep_old = P_to_sag(P_steep_old) or S_BASE

    delta_flat  = s_flat_new  - s_flat_old
    delta_steep = s_steep_new - s_steep_old

    dq_flat  = 64*D*delta_flat  / a**4
    dq_steep = 64*D*delta_steep / a**4
    dq_mean  = (dq_flat  + dq_steep) / 2
    dq_amp   = (dq_steep - dq_flat)  / 2

    contact_area = np.pi*(0.5e-3)**2
    screws = []

    for ang in SCREW_ANGLES:
        theta = np.radians(ang)
        dq_screw = dq_mean + dq_amp * np.cos(2*(theta - phi))

        F_N   = abs(dq_screw) * contact_area
        turns = F_N * PITCH_M / (2*np.pi*EFF*T_MAX_NM)
        turns_snapped = round(turns/0.25)*0.25
        turns_snapped = max(0.25, turns_snapped) if abs(dq_screw) > 100 else 0.0

        direction = 'CW'  if dq_screw >= 0 else 'CCW'
        no_change = abs(dq_screw) <= 100

        screws.append({
            'id':        int(np.where(SCREW_ANGLES==ang)[0][0]+1),
            'angle_deg': float(ang),
            'dq_Pa':     round(float(dq_screw), 2),
            'F_N':       round(float(F_N), 6),
            'turns':     float(turns_snapped),
            'direction': direction,
            'no_change': no_change,
        })

    dq_worst = max(abs(dq_flat), abs(dq_steep))
    vm_max   = 3*dq_worst*a**2/(4*TC**2) if dq_worst > 0 else 0
    sf       = YIELD/vm_max if vm_max > 1 else 99.0

    return {
        'from_rx': {'sphere':sph_old,'cylinder':cyl_old,'axis':ax_old},
        'to_rx':   {'sphere':sph_new,'cylinder':cyl_new,'axis':ax_new},
        'cycle':   cycle,
        'E_GPa':   round(E/1e9, 5),
        'dq_mean_Pa':  round(float(dq_mean),2),
        'dq_amp_Pa':   round(float(dq_amp),2),
        'safety_factor': round(min(sf,99),2),
        'achievable':    sf > 1.0,
        'screws':  screws,
    }

def draw_screw_diagram(ax, plan, title):
    """
    Draw polar screw diagram showing turns, direction, and lens power map.
    Green = CW (tighten = push lens out = more power)
    Red   = CCW (loosen = pull lens flat = less power)
    """
    ax.set_aspect('equal')
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.95, 1.75)
    ax.axis('off')

    ax.text(0, 1.72, title, ha='center', va='top', fontsize=9,
            fontweight='bold', color='#1A1A2E',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4FF',
                      edgecolor='#B0C0E0', linewidth=0.8))

    theta_c = np.linspace(0, 2*np.pi, 200)
    ax.fill(np.cos(theta_c), np.sin(theta_c), color='#EAF4FB', alpha=0.6, zorder=1)
    ax.plot(np.cos(theta_c), np.sin(theta_c), color='#7EB9D4', lw=1.5, zorder=2)

    ax_new = plan['to_rx']
    if abs(ax_new['cylinder']) > 0.01:
        phi_ax = np.radians(ax_new['axis'])
        ax.plot([-0.7*np.cos(phi_ax), 0.7*np.cos(phi_ax)],
                [-0.7*np.sin(phi_ax), 0.7*np.sin(phi_ax)],
                color='#9B59B6', lw=1.2, ls='--', alpha=0.6, zorder=3)
        ax.text(0.75*np.cos(phi_ax), 0.75*np.sin(phi_ax),
                f"axis\n{ax_new['axis']}°",
                ha='center', va='center', fontsize=6.5, color='#9B59B6')

    ax.plot(1.18*np.cos(theta_c), 1.18*np.sin(theta_c),
            color='#5D6D7E', lw=3.5, zorder=4, solid_capstyle='round')

    screws = plan['screws']
    max_turns = max((s['turns'] for s in screws), default=1.0) or 1.0

    for s in screws:
        ang   = np.radians(s['angle_deg'])
        px    = 1.18 * np.cos(ang)
        py    = 1.18 * np.sin(ang)
        color = '#27AE60' if s['direction']=='CW' else '#E74C3C'
        if s['no_change']:
            color = '#95A5A6'

        ax.add_patch(Circle((px,py), 0.115, color=color, zorder=6,
                             alpha=0.9, ec='white', linewidth=1.2))
        ax.text(px, py, str(s['id']), ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', zorder=7)

        if not s['no_change'] and s['turns'] > 0:

            arc_r   = 0.135 + 0.05*(s['turns']/max_turns)
            n_arc   = min(s['turns'], 2.5)
            arc_ang = n_arc * 2*np.pi
            arc_t   = np.linspace(0, arc_ang, 100)
            start_offset = np.pi/2
            arc_x = px + arc_r*np.cos(arc_t + ang + start_offset)
            arc_y = py + arc_r*np.sin(arc_t + ang + start_offset)
            ax.plot(arc_x, arc_y, color=color, lw=2.8, zorder=8,
                    solid_capstyle='round')

            ea = arc_ang + ang + start_offset
            sign_rot = 1 if s['direction']=='CW' else -1
            dx = -np.sin(ea)*0.015*sign_rot
            dy =  np.cos(ea)*0.015*sign_rot
            ax.annotate('', xy=(arc_x[-1]+dx, arc_y[-1]+dy),
                       xytext=(arc_x[-1], arc_y[-1]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
                       zorder=9)

        lx = 1.52 * np.cos(ang)
        ly = 1.52 * np.sin(ang)
        ha = 'left' if np.cos(ang) > 0.15 else ('right' if np.cos(ang) < -0.15 else 'center')
        va = 'bottom' if np.sin(ang) > 0.15 else ('top' if np.sin(ang) < -0.15 else 'center')
        if s['no_change']:
            label = '— (no change)'
            fc = '#ECF0F1'
        else:
            label = f"{s['turns']:.2f} turns\n{s['direction']}"
            fc = '#EAFAF1' if s['direction']=='CW' else '#FDEDEC'
        ax.text(lx, ly, label, ha=ha, va=va, fontsize=7.5, color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=fc,
                          edgecolor=color, alpha=0.9, linewidth=0.7))

    fr = plan['from_rx']; to = plan['to_rx']
    sf_color = '#27AE60' if plan['safety_factor'] > 2 else (
               '#E67E22' if plan['safety_factor'] > 1 else '#E74C3C')
    summary = (f"FROM: S{fr['sphere']:+.2f} C{fr['cylinder']:+.2f} x{fr['axis']:03d}\n"
               f"  TO: S{to['sphere']:+.2f} C{to['cylinder']:+.2f} x{to['axis']:03d}\n"
               f"SF: {plan['safety_factor']:.1f}  "
               f"{'✓ OK' if plan['achievable'] else '✗ check yield'}")
    ax.text(0, -1.32, summary, ha='center', va='top', fontsize=7.5,
            family='monospace', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE',
                      edgecolor='#BDC3C7', linewidth=0.8))

    for i,(col,txt) in enumerate([('#27AE60','CW = tighten (↑ power)'),
                                   ('#E74C3C','CCW = loosen (↓ power)'),
                                   ('#95A5A6','— = no change')]):
        ax.add_patch(Circle((-1.62, -1.58+i*0.14), 0.07,
                            color=col, zorder=5))
        ax.text(-1.50, -1.58+i*0.14, txt, fontsize=7,
                va='center', color=col)

def generate_diagram(prescriptions, output_path=None, cycle=0):
    """
    Main entry point.
    prescriptions: list of dicts with keys:
        'label', 'right_old', 'right_new', 'left_old', 'left_new'
        each prescription is (sphere, cylinder, axis)
    """
    n_cases = len(prescriptions)
    fig = plt.figure(figsize=(14, 6*n_cases + 1), facecolor='white')
    gs  = gridspec.GridSpec(n_cases, 2, figure=fig,
                            hspace=0.15, wspace=0.08)

    for row, case in enumerate(prescriptions):
        for col, (eye_key, eye_label) in enumerate([('right','Right Eye (OD)'),
                                                     ('left', 'Left Eye (OS)')]):
            ax = fig.add_subplot(gs[row, col])
            old = case[f'{eye_key}_old']
            new = case[f'{eye_key}_new']
            plan = compute_screw_plan(*old, *new, cycle=cycle)

            title = f"{case['label']} — {eye_label}"
            draw_screw_diagram(ax, plan, title)

    fig.suptitle('AdaptivEyes — Screw Revolution Calculator\n'
                 'Input: current & target Rx per eye  →  Output: turns per screw',
                 fontsize=13, fontweight='bold', y=1.01)

    if output_path is None:
        output_path = os.path.join(OUT, 'screw_revolutions.png')
    fig.savefig(output_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved → {output_path}")
    return output_path

if __name__ == '__main__':

    prescriptions = [
        {
            'label':     'Case 1: Pure myopia correction',
            'right_old': (3.5,  0.00,   0),
            'right_new': (2.0,  0.00,   0),
            'left_old':  (3.5,  0.00,   0),
            'left_new':  (2.5,  0.00,   0),
        },
        {
            'label':     'Case 2: Adding cylinder (new astigmatism)',
            'right_old': (3.5,  0.00,   0),
            'right_new': (2.0, -0.75,  90),
            'left_old':  (3.5,  0.00,   0),
            'left_new':  (1.5, -1.25,  45),
        },
        {
            'label':     'Case 3: Hyperopia increase with existing astigmatism',
            'right_old': (2.0, -0.50,  90),
            'right_new': (3.0, -0.75,  90),
            'left_old':  (2.5, -0.75,  45),
            'left_new':  (3.5, -1.00,  45),
        },
    ]

    print("="*60)
    print("AdaptivEyes — Screw Revolution Calculator")
    print("="*60)

    for case in prescriptions:
        print(f"\n{case['label']}:")
        for eye, eye_key in [('Right (OD)','right'), ('Left (OS)','left')]:
            old = case[f'{eye_key}_old']
            new = case[f'{eye_key}_new']
            plan = compute_screw_plan(*old, *new)
            print(f"  {eye}: S{old[0]:+.2f}→S{new[0]:+.2f}  "
                  f"C{old[1]:+.2f}→C{new[1]:+.2f}  SF={plan['safety_factor']:.1f}")
            for s in plan['screws']:
                marker = '→' if not s['no_change'] else '—'
                print(f"    Screw {s['id']} @{s['angle_deg']:3.0f}°: "
                      f"{marker} {s['turns']:.2f}t {s['direction'] if not s['no_change'] else '(no change)'}")

    path = generate_diagram(prescriptions)
    print(f"\nDiagram saved to: {path}")
