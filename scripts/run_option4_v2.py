import sys; sys.path.insert(0,'.')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import json, os
from genetic import (sample_global_prescription_distribution, P_to_sag,
                              D0, LENS_R, TC, YIELD, S_BASE)

OUT='/home/claude'
print("="*60); print("AdaptivEyes — Genetic Algorithm Screw Optimizer"); print("="*60)

def evaluate_batch(pop_angles_rad, sph_arr, cyl_arr, ax_arr):
    """
    Vectorized: evaluate all N individuals against all M prescriptions at once.
    pop_angles_rad: (N, n_screws) in radians
    sph_arr, cyl_arr, ax_arr: (M,) prescription arrays
    Returns: (N,) fitness scores
    """
    N=pop_angles_rad.shape[0]; M=len(sph_arr)
    scores=np.zeros(N)
    for m in range(M):
        sph=sph_arr[m]; cyl=cyl_arr[m]; phi=np.radians(ax_arr[m])
        sf=P_to_sag(float(sph)); ss=P_to_sag(float(sph+abs(cyl)))
        if not sf or not ss:
            scores+=100/M; continue
        w0_f=sf-S_BASE; w0_s=ss-S_BASE
        q_f=64*D0*w0_f/LENS_R**4; q_s=64*D0*w0_s/LENS_R**4
        q_m=(q_f+q_s)/2; q_a=(q_s-q_f)/2
        vm_max=3*max(abs(q_f),abs(q_s))*LENS_R**2/(4*TC**2)
        if vm_max<=1:
            continue
        if vm_max>=YIELD:
            scores+=50/M; continue
        sf_val=YIELD/vm_max

        q_sc=q_m+q_a*np.cos(2*(pop_angles_rad-phi))

        phi_t=np.linspace(0,2*np.pi,36,endpoint=False)

        diffs=pop_angles_rad[:,:,None]-phi_t[None,None,:]
        w=np.cos(diffs)**2
        w_sum=w.sum(axis=1,keepdims=True); w_sum=np.where(w_sum<1e-10,1,w_sum)
        w=w/w_sum
        q_ach=np.sum(w*q_sc[:,:,None],axis=1)
        q_tgt=q_m+q_a*np.cos(2*(phi_t[None,:]-phi))
        rms_err=np.sqrt(np.mean((q_ach-q_tgt)**2,axis=1))/max(abs(q_m)+abs(q_a),1)
        score=rms_err*10+max(0,1.5-sf_val)*20
        scores+=score/M
    return scores

def run_ga(n_screws, sph_a, cyl_a, ax_a, pop_size=40, n_gen=50, mut=0.18, elite_frac=0.15):
    np.random.seed(42)
    n_elite=max(2,int(elite_frac*pop_size))

    base=np.linspace(0,2*np.pi*(1-1/n_screws),n_screws)
    pop=[base]+[np.sort((base+np.random.normal(0,0.3,n_screws))%(2*np.pi)) for _ in range(pop_size//3)]
    while len(pop)<pop_size:
        angles=np.sort(np.random.uniform(0,2*np.pi,n_screws))
        diffs=np.diff(np.append(angles,angles[0]+2*np.pi))
        if diffs.min()>0.2: pop.append(angles)
    pop=np.array(pop[:pop_size])
    best_score=np.inf; best_cfg=pop[0].copy(); history=[]
    for gen in range(n_gen):
        scores=evaluate_batch(pop,sph_a,cyl_a,ax_a)
        idx=np.argsort(scores); pop=pop[idx]; scores=scores[idx]
        if scores[0]<best_score: best_score=scores[0]; best_cfg=pop[0].copy()
        history.append(float(scores[0]))

        new=[pop[i].copy() for i in range(n_elite)]
        while len(new)<pop_size:
            i1,i2=np.random.choice(min(20,pop_size),2,replace=False)
            p1=pop[min(i1,i2)]; i3,i4=np.random.choice(min(20,pop_size),2,replace=False)
            p2=pop[min(i3,i4)]
            mask=np.random.rand(n_screws)>0.5
            child=np.sort(np.where(mask,p1,p2)%(2*np.pi))
            if np.random.rand()<mut:
                child=np.sort((child+np.random.normal(0,0.25,n_screws))%(2*np.pi))
            new.append(child)
        pop=np.array(new)
    return np.degrees(best_cfg), best_score, history

print("\n[1] Sampling distribution...")
spheres,cyls,axes=sample_global_prescription_distribution(800)
np.random.seed(99); idx=np.random.choice(800,30,replace=False)
sph_t=spheres[idx]; cyl_t=cyls[idx]; ax_t=axes[idx]
print(f"    Sphere: mean={np.mean(spheres):.2f}±{np.std(spheres):.2f}D | Cyl: mean={np.mean(cyls):.2f}D")

print("\n[2] Running GA (3-8 screws, 40 pop, 50 gen)...")
ga_results={}
for n_sc in range(3,9):
    best,score,hist=run_ga(n_sc,sph_t,cyl_t,ax_t,pop_size=40,n_gen=50)
    n_cov=sum(1 for (s,c) in zip(sph_t,cyl_t)
              if P_to_sag(float(s)) and P_to_sag(float(s+abs(c))) and
              3*max(abs(64*D0*(P_to_sag(float(s))-S_BASE)/LENS_R**4),
                    abs(64*D0*(P_to_sag(float(s+abs(c)))-S_BASE)/LENS_R**4))*LENS_R**2/(4*TC**2)<YIELD)
    cov=round(n_cov/len(sph_t)*100,1)
    ga_results[n_sc]={'angles_deg':[round(float(a),1) for a in best],'score':round(float(score),4),
                      'coverage_pct':cov,'history':hist}
    print(f"  {n_sc}sc: score={score:.4f}  cov={cov}%  {[round(a,1) for a in best]}")

opt_n=min(ga_results,key=lambda n:ga_results[n]['score'])
print(f"\n[3] Optimal: {opt_n} screws | score={ga_results[opt_n]['score']:.4f} | cov={ga_results[opt_n]['coverage_pct']}%")
print(f"    Angles: {ga_results[opt_n]['angles_deg']}")

print("\n[4] Generating visualization...")
fig=plt.figure(figsize=(20,14),facecolor='white')
gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.32)

ax1=fig.add_subplot(gs[0,0])
h=ax1.hist2d(spheres,cyls,bins=25,cmap='YlOrRd',density=True)
plt.colorbar(h[3],ax=ax1,label='Density',shrink=0.9)
ax1.set_xlabel('Sphere (D)'); ax1.set_ylabel('Cylinder (D)')
ax1.set_title('Global Prescription Distribution\n(Brien Holden Institute data, 800 pts)',fontweight='bold')

ns=list(ga_results.keys()); sc=[ga_results[n]['score'] for n in ns]
ax2=fig.add_subplot(gs[0,1])
cols=['#27AE60' if n==opt_n else '#E74C3C' for n in ns]
bars=ax2.bar([str(n) for n in ns],sc,color=cols,alpha=0.88,edgecolor='white')
for bar,sv in zip(bars,sc):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,f'{sv:.3f}',ha='center',fontsize=9)
ax2.set_xlabel('Number of screws'); ax2.set_ylabel('Fitness score')
ax2.set_title('GA Score vs Screw Count\n(green = optimal)',fontweight='bold'); ax2.grid(True,alpha=0.25,axis='y')

ax3=fig.add_subplot(gs[0,2])
covs=[ga_results[n]['coverage_pct'] for n in ns]
ax3.plot(ns,covs,'o-',lw=2.5,markersize=9,color='#2196F3')
ax3.axhline(95,color='green',ls='--',lw=2,label='95% target')
ax3.set_xlabel('Screws'); ax3.set_ylabel('Coverage (%)')
ax3.set_title('Population Coverage vs Screw Count',fontweight='bold')
ax3.legend(); ax3.grid(True,alpha=0.25); ax3.set_ylim(60,105)

ax4=fig.add_subplot(gs[1,0])
cc=['#E91E63','#FF9800','#2196F3','#4CAF50','#9C27B0','#00BCD4']
for n,col in zip(ns,cc):
    ax4.plot(ga_results[n]['history'],color=col,lw=2,label=f'{n} screws',alpha=0.85)
ax4.set_xlabel('Generation'); ax4.set_ylabel('Best fitness')
ax4.set_title('GA Convergence Curves',fontweight='bold'); ax4.legend(fontsize=9); ax4.grid(True,alpha=0.25)

ax5=fig.add_subplot(gs[1,1],projection='polar')
opt_ang=ga_results[opt_n]['angles_deg']
opt_r=np.radians(opt_ang)
ax5.scatter(opt_r,[1.0]*opt_n,s=300,color='#27AE60',zorder=6,label='GA optimal',edgecolors='black',linewidth=0.5)
even_r=np.radians(np.arange(opt_n)*360/opt_n)
ax5.scatter(even_r,[0.6]*opt_n,s=200,color='#95A5A6',alpha=0.7,marker='s',label='Even spacing')
for a in opt_r: ax5.plot([0,a],[0,1.0],'#27AE60',lw=1.5,alpha=0.4)
for a in even_r: ax5.plot([0,a],[0,0.6],'gray',lw=1,alpha=0.25)
ax5.set_rticks([]); ax5.legend(fontsize=8.5,loc='lower right')
ax5.set_title(f'Optimal {opt_n}-screw layout\n(green=GA optimal, gray=even)',fontweight='bold',pad=18)

ax6=fig.add_subplot(gs[1,2])
P_g=np.arange(-3.5,4.75,0.25); C_g=np.arange(-3.0,0.25,0.25)
cov_map=np.zeros((len(P_g),len(C_g)))
for i,P_ in enumerate(P_g):
    for j,C_ in enumerate(C_g):
        sf=P_to_sag(float(P_)); ss=P_to_sag(float(P_+abs(C_)))
        if sf and ss:
            vm=3*max(abs(64*D0*(sf-S_BASE)/LENS_R**4),abs(64*D0*(ss-S_BASE)/LENS_R**4))*LENS_R**2/(4*TC**2)
            if vm<YIELD: cov_map[i,j]=1
im=ax6.pcolormesh(C_g,P_g,cov_map,cmap='RdYlGn',shading='auto')
plt.colorbar(im,ax=ax6,shrink=0.85,label='Achievable')
ax6.set_xlabel('Cylinder (D)'); ax6.set_ylabel('Sphere (D)')
ax6.set_title('Achievable Prescription Space\n(green = covered by AdaptivEyes)',fontweight='bold')

ax7=fig.add_subplot(gs[2,0:2])
common=[(1.0,0.0,0),(2.0,0.0,0),(3.5,0.0,0),(4.0,0.0,0),(2.0,-0.75,90),(2.0,-1.5,45),(3.0,-1.0,90),(2.5,-0.5,0)]
even_s_d=[i*360/opt_n for i in range(opt_n)]

def eval_single(angles_deg,sph,cyl,ax_d):
    return float(evaluate_batch(np.radians([angles_deg]),np.array([sph]),np.array([cyl]),np.array([ax_d]))[0])
s_ev=[eval_single(even_s_d,s,c,a) for (s,c,a) in common]
s_op=[eval_single(ga_results[opt_n]['angles_deg'],s,c,a) for (s,c,a) in common]
w=0.35; xp=range(len(common))
ax7.bar([x-w/2 for x in xp],s_ev,w,label='Even spacing',color='#E74C3C',alpha=0.85,edgecolor='white')
ax7.bar([x+w/2 for x in xp],s_op,w,label='GA optimal', color='#27AE60',alpha=0.85,edgecolor='white')
ax7.set_xticks(xp)
ax7.set_xticklabels([f'S{s:+.1f}\nC{c:+.2f}' for (s,c,a) in common],fontsize=8.5)
ax7.set_ylabel('Fitness score (lower=better)'); ax7.legend(fontsize=9)
ax7.set_title('Even Spacing vs GA-Optimal — Per Prescription Comparison',fontweight='bold')
ax7.grid(True,alpha=0.25,axis='y')
for x,(se,so) in enumerate(zip(s_ev,s_op)):
    if se>0.001:
        pct=(se-so)/se*100
        ax7.text(x+w/2,so+0.001,f'{pct:.0f}%',ha='center',fontsize=7.5,color='#1A5276',fontweight='bold')

ax8=fig.add_subplot(gs[2,2]); ax8.axis('off')
rect=plt.Rectangle((0,0),1,1,facecolor='#EAF4FB',edgecolor='#2980B9',linewidth=2,transform=ax8.transAxes)
ax8.add_patch(rect)
avg_improv=np.mean([(se-so)/max(se,0.001)*100 for se,so in zip(s_ev,s_op)])
stats=[
    ('Optimal screw count',  f'{opt_n}'),
    ('Best angles (°)',      ', '.join(str(a) for a in ga_results[opt_n]['angles_deg'])),
    ('Population coverage',  f"{ga_results[opt_n]['coverage_pct']:.1f}%"),
    ('GA fitness score',     f"{ga_results[opt_n]['score']:.4f}"),
    ('Improvement vs even',  f"+{avg_improv:.1f}%"),
    ('People served (1B)',   f"~{ga_results[opt_n]['coverage_pct']/100*1e3:.0f}M"),
    ('Sphere range',         '-3.5D to +4.5D'),
    ('Cylinder range',       '0 to -3.0D'),
]
y=0.91
for lbl,val in stats:
    ax8.text(0.05,y,lbl+':',fontsize=8.5,fontweight='bold',transform=ax8.transAxes,color='#2C3E50')
    ax8.text(0.58,y,val,fontsize=8,transform=ax8.transAxes,color='#1A5276'); y-=0.11
ax8.set_title('Optimization Summary',fontweight='bold',fontsize=10,pad=8)

fig.suptitle('AdaptivEyes — Genetic Algorithm: Optimal Screw Configuration for Global Myopia Population',
             fontsize=13,fontweight='bold')
p=os.path.join(OUT,'genetic_optimizer_results.png')
fig.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print(f"  Saved → {p}")

def jfix(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,np.ndarray): return o.tolist()
    raise TypeError(type(o))
with open(os.path.join(OUT,'genetic_optimizer_results.json'),'w') as f:
    json.dump(ga_results,f,indent=2,default=jfix)
print("[Done]")
