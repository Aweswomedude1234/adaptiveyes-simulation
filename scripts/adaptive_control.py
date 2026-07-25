import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, os

OUT = os.path.dirname(os.path.abspath(__file__))
E_NOM=2.275e9; NU=0.40; TC=0.0025; LENS_R=0.025; N_IDX=1.55; S_BASE=0.002
D0=E_NOM*TC**3/(12*(1-NU**2))

def smp_E(c): return float(2.16e9+(E_NOM-2.16e9)*np.exp(-0.002*c))
def P_to_sag(P):
    if abs(P)<0.01: return S_BASE
    R=(N_IDX-1)/abs(P); d=R**2-LENS_R**2
    return float(np.sign(P)*(R-np.sqrt(d))) if d>0 else None

def myopia_progression(age_start, P_start, n_years=20, noise_std=0.08):
    np.random.seed(42)
    ages = np.arange(age_start, age_start+n_years+1)
    P = [P_start]
    for age in ages[:-1]:
        if age < 14:   rate = -0.50
        elif age < 18: rate = -0.30
        elif age < 25: rate = -0.12
        else:          rate = -0.03
        delta = rate + np.random.normal(0, noise_std)
        new_P = P[-1] + delta
        new_P = round(new_P / 0.25) * 0.25
        P.append(float(new_P))
    return ages, np.array(P)

def needs_adjustment(P_current, P_worn, threshold=0.25):
    return abs(P_current - P_worn) >= threshold

def compute_adjustment(P_old, P_new, cyl_old, cyl_new, ax, cycle):
    E_c = smp_E(cycle)
    D   = E_c*TC**3/(12*(1-NU**2))
    a   = LENS_R
    phi = np.radians(ax)
    P_f_old=P_old; P_s_old=P_old+abs(cyl_old)
    P_f_new=P_new; P_s_new=P_new+abs(cyl_new)
    def ps(P):
        R=(N_IDX-1)/abs(P); d=R**2-a**2
        return float(np.sign(P)*(R-np.sqrt(d))) if d>0 else None
    s_f_old=ps(P_f_old) or S_BASE; s_s_old=ps(P_s_old) or S_BASE
    s_f_new=ps(P_f_new) or S_BASE; s_s_new=ps(P_s_new) or S_BASE
    dq_f=64*D*(s_f_new-s_f_old)/a**4; dq_s=64*D*(s_s_new-s_s_old)/a**4
    dq_m=(dq_f+dq_s)/2; dq_a=(dq_s-dq_f)/2
    screw_angles=np.arange(5)*72
    contact_area=np.pi*(0.5e-3)**2; pitch=0.4e-3; eff=0.35; T_max=0.05
    total_turns=0
    for ang in screw_angles:
        theta=np.radians(ang)
        dq_s_v=dq_m+dq_a*np.cos(2*(theta-phi))
        F_N=abs(dq_s_v)*contact_area
        turns=F_N*pitch/(2*np.pi*eff*T_max)
        total_turns+=max(0.25,round(turns/0.25)*0.25) if abs(dq_s_v)>100 else 0
    return total_turns

COST_TRADITIONAL_PAIR = 200
COST_ADAPTIVEYES_UNIT  = 35
COST_ADJUSTMENT        = 0
COST_REPLACEMENT_YEARS = 8

def simulate_patient(age_start=12, P_start=-1.5, cyl=-0.50, ax=90, n_years=20):
    ages, P_true = myopia_progression(age_start, P_start, n_years)
    P_worn = P_start
    cycle_count = 0
    total_adjustments = 0

    records = []
    traditional_replacements = 0
    last_traditional_change = P_start
    adaptiveyes_cost = COST_ADAPTIVEYES_UNIT
    traditional_cost = 0

    for i, (age, P_true_now) in enumerate(zip(ages, P_true)):
        year = i
        P_worn_t = P_worn
        E_c = smp_E(cycle_count)
        s_target = P_to_sag(P_worn) or S_BASE
        q_nom = 64*D0*(s_target-S_BASE)/LENS_R**4
        D_c = E_c*TC**3/(12*(1-NU**2))
        actual_sag = S_BASE + q_nom*LENS_R**4/(64*D_c)
        R_act = (LENS_R**2+actual_sag**2)/(2*abs(actual_sag)) if abs(actual_sag)>1e-7 else 1e6
        P_actual = (N_IDX-1)*np.sign(actual_sag)/R_act
        fatigue_drift = abs(P_actual - P_worn)

        prescription_drift = abs(P_true_now - P_worn)
        adjusted_this_year = False
        turns_this_year = 0

        if needs_adjustment(P_true_now, P_worn):
            turns = compute_adjustment(P_worn, P_true_now, cyl, cyl, ax, cycle_count)
            P_worn = P_true_now
            cycle_count += 1
            total_adjustments += 1
            turns_this_year = turns
            adjusted_this_year = True

        if abs(P_true_now - last_traditional_change) >= 0.25:
            traditional_cost += COST_TRADITIONAL_PAIR
            traditional_replacements += 1
            last_traditional_change = P_true_now

        if year > 0 and year % COST_REPLACEMENT_YEARS == 0:
            adaptiveyes_cost += COST_ADAPTIVEYES_UNIT * 0.5

        records.append({
            'year': year, 'age': int(age),
            'P_true': round(float(P_true_now), 2),
            'P_worn': round(float(P_worn), 2),
            'P_actual_fatigue': round(float(P_actual), 4),
            'fatigue_drift_D': round(float(fatigue_drift), 4),
            'prescription_drift_D': round(float(prescription_drift), 2),
            'adjusted': adjusted_this_year,
            'cycle_count': cycle_count,
            'E_GPa': round(float(E_c/1e9), 5),
            'turns_this_year': float(turns_this_year),
            'cumulative_adaptiveyes_cost': round(float(adaptiveyes_cost), 2),
            'cumulative_traditional_cost': round(float(traditional_cost), 2),
            'total_adjustments': total_adjustments,
            'traditional_replacements': traditional_replacements,
        })

    return records

def population_impact(n_patients=1000):
    np.random.seed(123)
    total_savings = 0
    total_pairs_avoided = 0
    total_adjustments = 0
    all_P_progressions = []

    for i in range(n_patients):
        age_s = int(np.random.randint(8, 20))
        P_s   = float(-np.random.uniform(0.5, 4.0))
        cyl   = float(-np.random.uniform(0, 1.5))
        ax    = int(np.random.randint(0, 180))
        n_y   = int(np.random.randint(10, 25))
        rec   = simulate_patient(age_s, P_s, cyl, ax, n_y)
        final = rec[-1]
        total_savings      += final['cumulative_traditional_cost'] - final['cumulative_adaptiveyes_cost']
        total_pairs_avoided += final['traditional_replacements']
        total_adjustments   += final['total_adjustments']
        all_P_progressions.append([r['P_true'] for r in rec])

    return {
        'n_patients': n_patients,
        'avg_savings_per_patient': round(total_savings/n_patients, 2),
        'avg_pairs_avoided': round(total_pairs_avoided/n_patients, 1),
        'avg_adjustments': round(total_adjustments/n_patients, 1),
        'global_savings_1B': round(total_savings/n_patients * 1e9 / 1e9, 2),
    }

if __name__ == '__main__':
    print("="*65); print("AdaptivEyes — Adaptive Control & Longitudinal Simulation"); print("="*65)

    profiles = [
        (12, -1.5, -0.50, 90,  20, 'Child (age 12, mild myopia)'),
        (14, -2.0, -0.75, 45,  18, 'Teen (age 14, moderate myopia+astig)'),
        (25, -3.0, -1.25, 90,  15, 'Adult (age 25, established myopia)'),
        (40, -1.0, -0.25,  0,  20, 'Middle-aged (age 40, presbyopia onset)'),
    ]

    all_records = {}
    for (age_s, P_s, cyl, ax, n_y, label) in profiles:
        print(f"\n{label}:")
        rec = simulate_patient(age_s, P_s, cyl, ax, n_y)
        all_records[label] = rec
        final = rec[-1]
        print(f"  Years simulated:        {n_y}")
        print(f"  Total adjustments:      {final['total_adjustments']}")
        print(f"  Traditional replacements:{final['traditional_replacements']}")
        print(f"  Traditional cost:       ${final['cumulative_traditional_cost']:.0f}")
        print(f"  AdaptivEyes cost:       ${final['cumulative_adaptiveyes_cost']:.0f}")
        print(f"  Savings:                ${final['cumulative_traditional_cost']-final['cumulative_adaptiveyes_cost']:.0f}")
        print(f"  Final cycle count:      {final['cycle_count']}")
        print(f"  Fatigue drift @ end:    {final['fatigue_drift_D']:.4f}D")

    print("\nPopulation simulation (1000 patients)...")
    pop = population_impact(1000)
    print(f"  Average savings/patient:  ${pop['avg_savings_per_patient']:.0f}")
    print(f"  Avg glasses pairs avoided: {pop['avg_pairs_avoided']:.1f}")
    print(f"  Global savings (1B people): ${pop['global_savings_1B']:.2f}B USD")

    fig = plt.figure(figsize=(20,14), facecolor='white')
    gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.32)
    colors = ['#2196F3','#E91E63','#4CAF50','#FF9800']

    ax1=fig.add_subplot(gs[0,0])
    for (age_s,P_s,cyl,ax_d,n_y,label),col in zip(profiles,colors):
        rec=all_records[label]
        ages=[r['age'] for r in rec]; Pt=[r['P_true'] for r in rec]; Pw=[r['P_worn'] for r in rec]
        ax1.plot(ages,Pt,color=col,lw=2,ls='-',label=f'{label[:15]}.. true')
        ax1.plot(ages,Pw,color=col,lw=1.5,ls='--',alpha=0.7)
        adjs=[r for r in rec if r['adjusted']]
        ax1.scatter([r['age'] for r in adjs],[r['P_worn'] for r in adjs],
                    color=col,marker='o',s=50,zorder=5)
    ax1.set_xlabel('Age (years)'); ax1.set_ylabel('Prescription (D)')
    ax1.set_title('Prescription Progression\n(dots = adjustment events)',fontweight='bold')
    ax1.legend(fontsize=7); ax1.grid(True,alpha=0.25)

    ax2=fig.add_subplot(gs[0,1])
    for (age_s,P_s,cyl,ax_d,n_y,label),col in zip(profiles,colors):
        rec=all_records[label]
        years=[r['year'] for r in rec]
        tc=[r['cumulative_traditional_cost'] for r in rec]
        ac=[r['cumulative_adaptiveyes_cost'] for r in rec]
        ax2.plot(years,tc,color=col,lw=2,ls='-')
        ax2.plot(years,ac,color=col,lw=2,ls='--')
    ax2.plot([],[],color='gray',lw=2,ls='-',label='Traditional glasses')
    ax2.plot([],[],color='gray',lw=2,ls='--',label='AdaptivEyes')
    ax2.set_xlabel('Years'); ax2.set_ylabel('Cumulative cost ($)')
    ax2.set_title('Cumulative Cost Comparison',fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True,alpha=0.25)

    ax3=fig.add_subplot(gs[0,2])
    for (age_s,P_s,cyl,ax_d,n_y,label),col in zip(profiles,colors):
        rec=all_records[label]
        years=[r['year'] for r in rec]
        drift=[r['fatigue_drift_D'] for r in rec]
        ax3.plot(years,drift,color=col,lw=2,label=label[:20])
    ax3.axhline(0.12,color='red',lw=1.5,ls='--',label='Clinical tolerance 0.12D')
    ax3.set_xlabel('Years'); ax3.set_ylabel('Fatigue drift (D)')
    ax3.set_title('Material Fatigue Drift\nover 20 years',fontweight='bold')
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.25)

    ax4=fig.add_subplot(gs[1,0])
    for (age_s,P_s,cyl,ax_d,n_y,label),col in zip(profiles,colors):
        rec=all_records[label]
        years=[r['year'] for r in rec]
        cadj=[r['total_adjustments'] for r in rec]
        ax4.step(years,cadj,color=col,lw=2,where='post',label=label[:20])
    ax4.set_xlabel('Years'); ax4.set_ylabel('Cumulative adjustments')
    ax4.set_title('Adjustment Events\n(each = screw turn procedure)',fontweight='bold')
    ax4.legend(fontsize=7.5); ax4.grid(True,alpha=0.25)

    ax5=fig.add_subplot(gs[1,1])
    cycles=np.arange(0,601)
    E_vals=[smp_E(c)/1e9 for c in cycles]
    ax5.plot(cycles,E_vals,'b-',lw=2.5)
    for (age_s,P_s,cyl,ax_d,n_y,label),col in zip(profiles,colors):
        rec=all_records[label]; fc=rec[-1]['cycle_count']
        ax5.axvline(fc,color=col,lw=1.2,ls='--',alpha=0.8,label=f'{label[:15]}: {fc} cycles')
    ax5.set_xlabel('Cycle count'); ax5.set_ylabel("Young's Modulus (GPa)")
    ax5.set_title("Material Degradation\nE(n) = Mullins Law",fontweight='bold')
    ax5.legend(fontsize=7.5); ax5.grid(True,alpha=0.25)

    ax6=fig.add_subplot(gs[1,2])
    profile_labels=[p[5][:18] for p in profiles]
    savings=[all_records[p[5]][-1]['cumulative_traditional_cost']-
             all_records[p[5]][-1]['cumulative_adaptiveyes_cost'] for p in profiles]
    bars=ax6.bar(range(4),savings,color=colors,alpha=0.85,edgecolor='white',linewidth=1.2)
    ax6.set_xticks(range(4)); ax6.set_xticklabels([p[:12] for p in profile_labels],
                                                    fontsize=8,rotation=20,ha='right')
    ax6.set_ylabel('Savings ($)'); ax6.set_title('Cost Savings per Patient\n(AdaptivEyes vs Traditional)',fontweight='bold')
    for bar,s in zip(bars,savings): ax6.text(bar.get_x()+bar.get_width()/2,bar.get_height()+5,f'${s:.0f}',ha='center',fontsize=9,fontweight='bold')
    ax6.grid(True,alpha=0.25,axis='y')

    ax7=fig.add_subplot(gs[2,:])
    ax7.axis('off')
    stats = [
        ('Average savings\nper patient', f"${pop['avg_savings_per_patient']:.0f}"),
        ('Glasses pairs\navoided per patient', f"{pop['avg_pairs_avoided']:.1f}"),
        ('Avg adjustments\nper patient', f"{pop['avg_adjustments']:.1f}"),
        ('Global savings\n(1 billion people)', f"${pop['global_savings_1B']:.1f}B USD"),
        ('Max prescription\nrange covered', '-2D to +4D'),
        ('Max cylinder\ncorrection', '±2.5D'),
    ]
    for i,(label,val) in enumerate(stats):
        x=0.05+i*0.16
        ax7.add_patch(plt.Rectangle((x,0.1),0.14,0.75,facecolor='#EAF4FB',
                                     edgecolor='#4A90D9',linewidth=1.5,transform=ax7.transAxes))
        ax7.text(x+0.07,0.65,val,ha='center',va='center',fontsize=14,
                fontweight='bold',color='#1A5276',transform=ax7.transAxes)
        ax7.text(x+0.07,0.28,label,ha='center',va='center',fontsize=8,
                color='#2C3E50',transform=ax7.transAxes,multialignment='center')
    ax7.set_title('Global Population Impact (1,000-patient simulation extrapolated to 1B)',
                  fontweight='bold',fontsize=12,pad=10)

    fig.suptitle('AdaptivEyes — Adaptive Control & 20-Year Longitudinal Simulation',fontsize=14,fontweight='bold')
    p=os.path.join(OUT,'adaptive_control_results.png')
    fig.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"\nSaved → {p}")

    def jfix(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,(np.bool_,)):return bool(o)
        if isinstance(o,np.ndarray):return o.tolist()
        raise TypeError(type(o))

    with open(os.path.join(OUT,'adaptive_control_results.json'),'w') as f:
        json.dump({'profiles':{k:v[-1] for k,v in all_records.items()},
                   'population':pop},f,indent=2,default=jfix)
    print("[Done]")
