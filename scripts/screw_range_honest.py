import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT     = os.path.dirname(os.path.abspath(__file__))
E       = 2.275e9
NU      = 0.40
N_IDX   = 1.55
S_BASE  = 0.002
YIELD   = 65e6
A       = 0.025
TC_vals = [0.0015, 0.002, 0.0025, 0.003]

def D(tc): return E * tc**3 / (12*(1 - NU**2))

def sag_from_P(P):
    if abs(P) < 0.05:
        return float(S_BASE)
    R = (N_IDX - 1) / abs(P)
    disc = R**2 - A**2
    if disc <= 0:
        return None
    s = np.sign(P) * (R - np.sqrt(disc))
    return float(s)

def required_pressure(P, tc):
    s = sag_from_P(P)
    if s is None:
        return None
    delta_s = s - S_BASE
    q = 64 * D(tc) * delta_s / A**4
    return float(q)

def von_mises(q, tc):
    return 3 * abs(q) * A**2 / (4 * tc**2)

def safety_factor(q, tc):
    vm = von_mises(q, tc)
    if vm <= 0:
        return 999.0
    return float(YIELD / vm)

def achievable(P, tc):
    s = sag_from_P(P)
    if s is None:
        return False, "Lens curvature exceeds aperture (unphysical geometry)", None, None
    q = required_pressure(P, tc)
    if q is None:
        return False, "Cannot compute pressure", None, None
    SF = safety_factor(q, tc)
    if SF < 1.0:
        return False, f"YIELD EXCEEDED (SF={SF:.2f} < 1.0)", q, SF
    if SF < 1.5:
        return False, f"Unsafe margin (SF={SF:.2f} < 1.5 minimum)", q, SF
    return True, f"OK (SF={SF:.2f})", q, SF

print("="*70)
print("AdaptivEyes — Honest Prescription Range Analysis")
print("="*70)

print("\n[1] SPHERICAL RANGE vs LENS THICKNESS")
print("-"*70)

for tc in TC_vals:
    print(f"\n  Thickness = {tc*1000:.1f}mm:")
    P_range_pos = []; P_range_neg = []
    for P in np.arange(0.25, 10.0, 0.25):
        ok,_,_,_ = achievable(P, tc)
        if ok: P_range_pos.append(P)
    for P in np.arange(-0.25, -10.0, -0.25):
        ok,_,_,_ = achievable(P, tc)
        if ok: P_range_neg.append(P)
    pos_max = max(P_range_pos) if P_range_pos else 0
    neg_max = min(P_range_neg) if P_range_neg else 0
    total   = pos_max - neg_max
    print(f"    Hyperopia:  +0.25D to +{pos_max:.2f}D")
    print(f"    Myopia:     {neg_max:.2f}D to -0.25D")
    print(f"    Total range: {total:.2f}D")
    if total >= 9.0:
        print(f"    ✓ MEETS 9D target")
    else:
        print(f"    ✗ Does NOT meet 9D target (need {9.0-total:.2f}D more)")

print("\n\n[2] DETAILED ANALYSIS — each prescription, tc=2.0mm")
print("-"*70)
tc_opt = 0.002
print(f"{'P (D)':>8} {'Sag (mm)':>10} {'q (Pa)':>12} {'σ_VM (MPa)':>12} {'SF':>6} {'Status':>20}")
print("-"*70)
for P in list(np.arange(-5.0, 5.25, 0.5)):
    s = sag_from_P(P)
    if s is None:
        print(f"{P:>8.2f} {'UNPHYSICAL':>10}")
        continue
    q = required_pressure(P, tc_opt)
    vm = von_mises(q, tc_opt)
    sf = safety_factor(q, tc_opt)
    ok, reason, _, _ = achievable(P, tc_opt)
    status = "✓ OK" if ok else f"✗ {reason[:18]}"
    print(f"{P:>8.2f} {s*1000:>10.4f} {q:>12.1f} {vm/1e6:>12.3f} {sf:>6.2f} {status:>20}")

print("\n\n[3] ASTIGMATISM RANGE (cylinder correction, tc=2.0mm)")
print("-"*70)
print("Astigmatism works by applying DIFFERENT pressures to flat and steep meridians.")
print("Max cylinder = max achievable (P_steep - P_flat) where both are safe.\n")

print(f"{'Sphere':>8} {'Cylinder':>10} {'SF_flat':>8} {'SF_steep':>10} {'Achievable':>12}")
print("-"*70)
max_cyl_overall = 0
for sph in [-3.0, -1.5, 0.0, 1.5, 3.0, 4.5]:
    for cyl in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        P_flat  = sph
        P_steep = sph + cyl
        s_f = sag_from_P(P_flat);  s_s = sag_from_P(P_steep)
        if s_f is None or s_s is None:
            print(f"{sph:>8.1f} {cyl:>10.2f} {'unphysical':>8}")
            continue
        q_f = required_pressure(P_flat, tc_opt)
        q_s = required_pressure(P_steep, tc_opt)
        sf_f = safety_factor(q_f, tc_opt)
        sf_s = safety_factor(q_s, tc_opt)
        ok_f = sf_f >= 1.5
        ok_s = sf_s >= 1.5
        ok = ok_f and ok_s
        flag = "✓" if ok else "✗"
        if ok: max_cyl_overall = max(max_cyl_overall, cyl)
        print(f"{sph:>8.1f} {cyl:>10.2f} {sf_f:>8.2f} {sf_s:>10.2f} {flag:>12}")

print(f"\n  Maximum achievable cylinder (worst case): {max_cyl_overall:.2f}D")
if max_cyl_overall >= 2.5:
    print("  ✓ MEETS 2.5D cylinder target")
else:
    print(f"  ✗ Does NOT meet 2.5D target at all sphere combinations")

print("\n\n[4] HONEST SUMMARY & LIMITATIONS")
print("="*70)

results = {}
for tc in TC_vals:
    pos_ok = [P for P in np.arange(0.25,10,0.25) if achievable(P,tc)[0]]
    neg_ok = [P for P in np.arange(-0.25,-10,-0.25) if achievable(P,tc)[0]]
    results[tc] = {
        'pos_max': max(pos_ok) if pos_ok else 0,
        'neg_max': min(neg_ok) if neg_ok else 0,
        'range':   (max(pos_ok) if pos_ok else 0) - (min(neg_ok) if neg_ok else 0)
    }
    print(f"\n  tc={tc*1000:.1f}mm: range = {results[tc]['range']:.2f}D  "
          f"({results[tc]['neg_max']:.2f}D to +{results[tc]['pos_max']:.2f}D)")

best_tc = max(results, key=lambda t: results[t]['range'])
best = results[best_tc]
print(f"\n  Best thickness: {best_tc*1000:.1f}mm")
print(f"  Maximum honest range: {best['range']:.2f}D")
print(f"    ({best['neg_max']:.2f}D to +{best['pos_max']:.2f}D)")
print()
if best['range'] >= 9.0:
    print("  ✓ 9D range IS achievable with optimal thickness")
    print(f"  ✓ Covers both myopia (to {best['neg_max']:.2f}D) and hyperopia (to +{best['pos_max']:.2f}D)")
else:
    print(f"  ✗ 9D range is NOT achievable with this lens geometry.")
    print(f"    Maximum achievable range: {best['range']:.2f}D")
    print(f"    To reach 9D you would need to either:")
    print(f"      (a) increase lens thickness beyond {best_tc*1000:.1f}mm (reduces optical quality)")
    print(f"      (b) reduce lens aperture below 50mm (smaller optical zone)")
    print(f"      (c) use a higher-yield SMP material (σ_yield > {YIELD/1e6:.0f} MPa)")
    print(f"      (d) use two separate lens designs (positive + negative)")

fig = plt.figure(figsize=(18,12), facecolor='white')
gs  = gridspec.GridSpec(2,3, figure=fig, hspace=0.40, wspace=0.32)

ax1 = fig.add_subplot(gs[0,0])
colors = ['#2196F3','#4CAF50','#FF9800','#E91E63']
P_sweep = np.arange(-6, 6.1, 0.1)
for tc, col in zip(TC_vals, colors):
    SFs = []
    for P in P_sweep:
        s = sag_from_P(P)
        if s is None: SFs.append(0); continue
        q = required_pressure(P, tc)
        SFs.append(safety_factor(q, tc))
    ax1.plot(P_sweep, SFs, color=col, lw=2, label=f't={tc*1000:.1f}mm')
ax1.axhline(1.5, color='red', ls='--', lw=1.5, label='SF=1.5 (minimum safe)')
ax1.axhline(1.0, color='darkred', ls=':', lw=1, label='SF=1.0 (yield)')
ax1.fill_between(P_sweep, 0, 1.5, alpha=0.07, color='red')
ax1.set_xlabel('Prescription (D)'); ax1.set_ylabel('Safety Factor')
ax1.set_title('Safety Factor vs Prescription\n(all thickness options)', fontweight='bold')
ax1.legend(fontsize=8.5); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 15)

ax2 = fig.add_subplot(gs[0,1])
tcs  = [t*1000 for t in TC_vals]
rngs = [results[t]['range'] for t in TC_vals]
pmax = [results[t]['pos_max'] for t in TC_vals]
nmax = [-results[t]['neg_max'] for t in TC_vals]
x = np.arange(len(TC_vals))
bars = ax2.bar(x, rngs, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
ax2.axhline(9.0, color='red', ls='--', lw=2, label='9D target')
ax2.set_xticks(x); ax2.set_xticklabels([f'{t:.1f}mm' for t in tcs])
ax2.set_xlabel('Lens thickness'); ax2.set_ylabel('Total diopter range (D)')
ax2.set_title('Achievable Prescription Range\nvs Lens Thickness', fontweight='bold')
for bar, r in zip(bars, rngs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'{r:.1f}D', ha='center', fontweight='bold', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

ax3 = fig.add_subplot(gs[0,2])
tc_plot = 0.002
P_pos = np.arange(0.25, 6, 0.25); P_neg = np.arange(-0.25, -6, -0.25)
q_pos = [required_pressure(P, tc_plot) or 0 for P in P_pos]
q_neg = [required_pressure(P, tc_plot) or 0 for P in P_neg]
ax3.plot(P_pos, np.array(q_pos)/1e3,  'b-', lw=2, label='Hyperopia (+)')
ax3.plot(P_neg, np.abs(np.array(q_neg))/1e3, 'r-', lw=2, label='Myopia (-) |q|')
ax3.set_xlabel('|Prescription| (D)'); ax3.set_ylabel('Required pressure q (kPa)')
ax3.set_title('Actuation Pressure Required\n(tc=2.0mm)', fontweight='bold')
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1,0])
sph_range = np.arange(-4.5, 5.0, 0.5)
cyl_range = np.arange(0, 3.25, 0.25)
cov = np.zeros((len(sph_range), len(cyl_range)))
for i,sph in enumerate(sph_range):
    for j,cyl in enumerate(cyl_range):
        ok_f,_,_,_ = achievable(sph, tc_opt)
        ok_s,_,_,_ = achievable(sph+cyl, tc_opt)
        cov[i,j] = 1 if (ok_f and ok_s) else 0
im = ax4.pcolormesh(cyl_range, sph_range, cov, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax4, label='Achievable (1=yes)', shrink=0.85)
ax4.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
ax4.axvline(2.5, color='cyan', ls='--', lw=1.5, label='2.5D cyl target')
ax4.set_xlabel('Cylinder (D)'); ax4.set_ylabel('Sphere (D)')
ax4.set_title('Achievable Prescription Space\n(green = achievable, tc=2.0mm)', fontweight='bold')
ax4.legend(fontsize=9)

ax5 = fig.add_subplot(gs[1,1])
tc_b = best_tc
P_all = np.arange(-6, 6.1, 0.25)
SFs_b = []
colors_b = []
for P in P_all:
    s = sag_from_P(P)
    if s is None: SFs_b.append(0); colors_b.append('gray'); continue
    q = required_pressure(P, tc_b)
    sf = safety_factor(q, tc_b)
    SFs_b.append(sf)
    colors_b.append('#27AE60' if sf >= 1.5 else ('#FF9800' if sf >= 1.0 else '#E74C3C'))
ax5.bar(P_all, SFs_b, width=0.22, color=colors_b, alpha=0.85)
ax5.axhline(1.5, color='red', ls='--', lw=2, label='SF=1.5 minimum')
ax5.axhline(1.0, color='darkred', ls=':', lw=1.5, label='SF=1.0 yield')
ax5.set_xlabel('Prescription (D)'); ax5.set_ylabel('Safety Factor')
ax5.set_title(f'Safety Factor per Prescription\n(tc={tc_b*1000:.1f}mm — optimal)',fontweight='bold')
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3, axis='y')
green_p = mpatches.Patch(color='#27AE60', label='SF≥1.5 (safe)')
orange_p= mpatches.Patch(color='#FF9800', label='1.0≤SF<1.5 (marginal)')
red_p   = mpatches.Patch(color='#E74C3C', label='SF<1.0 (fails)')
ax5.legend(handles=[green_p,orange_p,red_p], fontsize=8.5)

ax6 = fig.add_subplot(gs[1,2]); ax6.axis('off')
rect = plt.Rectangle((0,0),1,1,facecolor='#FFF9E6',edgecolor='#E67E22',linewidth=2,transform=ax6.transAxes)
ax6.add_patch(rect)
best_range = results[best_tc]['range']
meets_9D   = best_range >= 9.0
summary = [
    ("HONEST RESULTS (tc=2.0mm)", "", True),
    ("", "", False),
    ("Hyperopia range:", f"+0.25 to +{results[tc_opt]['pos_max']:.2f}D", False),
    ("Myopia range:", f"{results[tc_opt]['neg_max']:.2f} to -0.25D", False),
    ("Total range:", f"{results[tc_opt]['range']:.2f}D", False),
    ("Cylinder (all Rx):", f"up to 2.5D ✓" if max_cyl_overall>=2.5 else f"up to {max_cyl_overall:.1f}D only", False),
    ("", "", False),
    ("9D target met?", "✓ YES" if meets_9D else "✗ NO", False),
    ("", "", False),
    ("LIMITING FACTOR:", "", True),
]
if not meets_9D:
    summary += [
        (f"Yield stress ({YIELD/1e6:.0f}MPa) limits", "", False),
        ("high prescriptions.", "", False),
        ("Need higher-yield SMP", "", False),
        ("or smaller aperture.", "", False),
    ]
else:
    summary += [
        ("None — design is feasible", "", False),
        ("within material limits.", "", False),
    ]

y = 0.94
for label, val, bold in summary:
    fw = 'bold' if bold else 'normal'
    ax6.text(0.06, y, label, fontsize=8.5, fontweight=fw, transform=ax6.transAxes, color='#2C3E50')
    if val:
        ax6.text(0.62, y, val, fontsize=8.5, transform=ax6.transAxes,
                 color='#27AE60' if '✓' in val else '#E74C3C' if '✗' in val else '#1A5276',
                 fontweight='bold')
    y -= 0.075

fig.suptitle('AdaptivEyes — Honest Prescription Range & Stress Analysis', fontsize=14, fontweight='bold')
p = os.path.join(OUT, 'screw_range_honest.png')
fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved → {p}")
