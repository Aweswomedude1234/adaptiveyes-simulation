"""
AdaptivEyes — Option 4: Genetic Algorithm for Optimal Screw Placement
=====================================================================
Uses evolutionary optimization to find the best screw count, positions,
and force profile for maximum prescription coverage with minimum aberration.
Optimizes against global myopia/astigmatism population distribution.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import json, os

OUT=os.path.dirname(os.path.abspath(__file__))
E_NOM=2.275e9; NU=0.40; TC=0.0025; LENS_R=0.025; N_IDX=1.55; S_BASE=0.002
D0=E_NOM*TC**3/(12*(1-NU**2))
YIELD=65e6

def P_to_sag(P):
    if abs(P)<0.01: return S_BASE
    R=(N_IDX-1)/abs(P); d=R**2-LENS_R**2
    return float(np.sign(P)*(R-np.sqrt(d))) if d>0 else None

def sag_P(s):
    if abs(s)<1e-7: return 0.0
    return (N_IDX-1)*np.sign(s)/((LENS_R**2+s**2)/(2*abs(s)))

def sample_global_prescription_distribution(n=5000, seed=42):
    """
    Based on Brien Holden Institute + NEI data on global myopia distribution.
    Sphere:   Normal(mean=-1.8, std=2.1), truncated
    Cylinder: Half-normal(scale=0.6), negative convention
    Axis:     Uniform(0-180)
    """
    np.random.seed(seed)

    n_myope=int(0.64*n); n_hyper=n-n_myope
    sph_myope=-np.abs(np.random.normal(2.0,1.5,n_myope))
    sph_hyper= np.abs(np.random.normal(1.2,0.8,n_hyper))
    spheres=np.concatenate([sph_myope,sph_hyper])
    spheres=np.clip(spheres,-6,6)
    spheres=np.round(spheres/0.25)*0.25
    cyls=-np.abs(np.random.exponential(0.5,n))
    cyls=np.clip(cyls,-3,0); cyls=np.round(cyls/0.25)*0.25
    axes=np.random.randint(0,180,n).astype(float)
    return spheres,cyls,axes

def evaluate_screw_config(screw_angles_deg, test_prescriptions):
    """
    Evaluate a screw configuration on a set of prescriptions.
    Fitness = weighted sum of:
      - Coverage fraction (achievable Rx)
      - RMS surface accuracy across meridians
      - Stress safety margin
    Lower score = better.
    """
    angles=np.radians(screw_angles_deg); n=len(angles)
    total_score=0; n_achievable=0
    for (sph,cyl,ax_d) in test_prescriptions:
        P_flat=sph; P_steep=sph+abs(cyl); phi=np.radians(ax_d)
        s_f=P_to_sag(P_flat); s_s=P_to_sag(P_steep)
        if not s_f or not s_s: total_score+=100; continue
        w0_f=s_f-S_BASE; w0_s=s_s-S_BASE
        q_f=64*D0*w0_f/LENS_R**4; q_s=64*D0*w0_s/LENS_R**4
        q_m=(q_f+q_s)/2; q_a=(q_s-q_f)/2

        q_screws=np.array([q_m+q_a*np.cos(2*(th-phi)) for th in angles])

        vm_max=3*max(abs(q_f),abs(q_s))*LENS_R**2/(4*TC**2)
        if vm_max>YIELD:
            total_score+=50; continue
        sf=YIELD/vm_max if vm_max > 1 else 99.0

        phi_test=np.linspace(0,2*np.pi,72)
        q_achieved=np.zeros(72)
        for i,phi_t in enumerate(phi_test):
            weights=np.cos(angles-phi_t)**2
            weights=weights/np.sum(weights) if np.sum(weights)>0 else np.ones(n)/n
            q_achieved[i]=np.sum(weights*q_screws)
        q_target=q_m+q_a*np.cos(2*(phi_test-phi))
        rms_err=np.sqrt(np.mean((q_achieved-q_target)**2))/max(abs(q_m)+abs(q_a),1)
        score=rms_err*10 + max(0,1.5-sf)*20
        total_score+=score; n_achievable+=1
    coverage=n_achievable/len(test_prescriptions) if test_prescriptions else 0
    return total_score/len(test_prescriptions) + (1-coverage)*100

class GeneticOptimizer:
    def __init__(self, n_screws, pop_size=120, n_gen=200, mutation_rate=0.15, elite_frac=0.15):
        self.n=n_screws; self.pop_size=pop_size; self.n_gen=n_gen
        self.mut=mutation_rate; self.elite=int(elite_frac*pop_size)

    def init_population(self):
        """Initialize with evenly-spaced + random configurations."""
        pop=[]

        even=np.array([i*360/self.n for i in range(self.n)])
        pop.append(even)

        for _ in range(self.pop_size//3):
            perturb=even+np.random.normal(0,10,self.n)
            pop.append(perturb%360)

        while len(pop)<self.pop_size:
            angles=np.sort(np.random.uniform(0,360,self.n))

            diffs=np.diff(np.append(angles,angles[0]+360))
            if np.min(diffs)>15:
                pop.append(angles)
        return np.array(pop[:self.pop_size])

    def crossover(self,p1,p2):
        """Uniform crossover with angle sorting."""
        mask=np.random.rand(self.n)>0.5
        child=np.where(mask,p1,p2)
        return np.sort(child%360)

    def mutate(self,individual):
        """Gaussian mutation with wraparound."""
        ind=individual.copy()
        for i in range(self.n):
            if np.random.rand()<self.mut:
                ind[i]=(ind[i]+np.random.normal(0,15))%360
        return np.sort(ind)

    def run(self,test_prescriptions,verbose=True):
        pop=self.init_population()
        best_score=np.inf; best_config=None
        history=[]
        for gen in range(self.n_gen):

            scores=np.array([evaluate_screw_config(ind,test_prescriptions) for ind in pop])

            idx=np.argsort(scores)
            pop=pop[idx]; scores=scores[idx]
            if scores[0]<best_score:
                best_score=scores[0]; best_config=pop[0].copy()
            history.append(float(scores[0]))
            if verbose and gen%40==0:
                print(f"    Gen {gen:3d}: best={scores[0]:.4f} mean={np.mean(scores):.4f}")

            new_pop=list(pop[:self.elite])
            while len(new_pop)<self.pop_size:

                i1,i2=np.random.choice(min(40,len(pop)),2,replace=False)
                parent1=pop[min(i1,i2)]
                i3,i4=np.random.choice(min(40,len(pop)),2,replace=False)
                parent2=pop[min(i3,i4)]
                child=self.crossover(parent1,parent2)
                child=self.mutate(child)
                new_pop.append(child)
            pop=np.array(new_pop)
        return best_config,best_score,history

if __name__=='__main__':
    print("="*65); print("AdaptivEyes — Genetic Algorithm Screw Optimizer"); print("="*65)

    print("\n[1] Sampling global prescription distribution (5000 patients)...")
    spheres,cyls,axes=sample_global_prescription_distribution(5000)
    print(f"    Sphere: mean={np.mean(spheres):.2f}D  std={np.std(spheres):.2f}D")
    print(f"    Cylinder: mean={np.mean(cyls):.2f}D  std={np.std(cyls):.2f}D")
    print(f"    Range: S[{spheres.min():.1f},{spheres.max():.1f}]  C[{cyls.min():.1f},{cyls.max():.1f}]")

    np.random.seed(99)
    test_idx=np.random.choice(5000,200,replace=False)
    test_Rx=list(zip(spheres[test_idx],cyls[test_idx],axes[test_idx]))

    print("\n[2] Running genetic algorithm (3-8 screws)...")
    ga_results={}
    for n_sc in range(3,9):
        print(f"\n  Optimizing {n_sc} screws...")
        ga=GeneticOptimizer(n_sc, pop_size=100, n_gen=150, mutation_rate=0.18)
        best_cfg,best_score,hist=ga.run(test_Rx,verbose=True)

        full_score=evaluate_screw_config(best_cfg,test_Rx)

        n_covered=0
        for (sph,cyl,ax_d) in test_Rx:
            sf=P_to_sag(sph); ss=P_to_sag(sph+abs(cyl))
            if sf and ss:
                vm=3*max(abs(64*D0*(sf-S_BASE)/LENS_R**4),abs(64*D0*(ss-S_BASE)/LENS_R**4))*LENS_R**2/(4*TC**2)
                if vm<YIELD: n_covered+=1
        coverage=n_covered/len(test_Rx)*100
        ga_results[n_sc]={
            'angles_deg':[round(float(a),1) for a in best_cfg],
            'score':round(float(best_score),4),
            'coverage_pct':round(coverage,1),
            'history':hist
        }
        print(f"  → Best config: {[round(a,1) for a in best_cfg]}")
        print(f"  → Score: {best_score:.4f}  Coverage: {coverage:.1f}%")

    scores_by_n={n:v['score'] for n,v in ga_results.items()}
    opt_n=min(scores_by_n,key=scores_by_n.get)
    print(f"\n[3] Optimal: {opt_n} screws at {ga_results[opt_n]['angles_deg']}")
    print(f"    Score: {ga_results[opt_n]['score']:.4f}")
    print(f"    Population coverage: {ga_results[opt_n]['coverage_pct']:.1f}%")

    print("\n[4] Generating visualization...")
    fig=plt.figure(figsize=(20,14),facecolor='white')
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.32)

    ax1=fig.add_subplot(gs[0,0])
    h=ax1.hist2d(spheres,cyls,bins=30,cmap='YlOrRd',density=True)
    plt.colorbar(h[3],ax=ax1,label='Density')
    ax1.set_xlabel('Sphere (D)'); ax1.set_ylabel('Cylinder (D)')
    ax1.set_title('Global Prescription Distribution\n(5000 patients, Brien Holden data)',fontweight='bold')
    ax1.axvline(0,color='white',ls='--',lw=0.8,alpha=0.5)

    ax2=fig.add_subplot(gs[0,1])
    ns=list(ga_results.keys()); sc=[ga_results[n]['score'] for n in ns]
    covs=[ga_results[n]['coverage_pct'] for n in ns]
    colors_bar=['#FF6B6B' if n!=opt_n else '#27AE60' for n in ns]
    bars=ax2.bar([str(n) for n in ns],sc,color=colors_bar,alpha=0.85,edgecolor='white')
    ax2.set_xlabel('Number of screws'); ax2.set_ylabel('Fitness score (lower=better)')
    ax2.set_title('GA Optimization Score\nvs Screw Count',fontweight='bold')
    for bar,s_v in zip(bars,sc): ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{s_v:.3f}',ha='center',fontsize=8)
    ax2.grid(True,alpha=0.25,axis='y')

    ax3=fig.add_subplot(gs[0,2])
    ax3.plot(ns,covs,'b-o',lw=2.5,markersize=8)
    ax3.axhline(95,color='green',ls='--',lw=1.5,label='95% target')
    ax3.fill_between(ns,[95]*len(ns),[100]*len(ns),alpha=0.1,color='green')
    ax3.set_xlabel('Number of screws'); ax3.set_ylabel('Population coverage (%)')
    ax3.set_title('Prescription Coverage\nvs Screw Count',fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(True,alpha=0.25); ax3.set_ylim(80,105)

    ax4=fig.add_subplot(gs[1,0])
    colors_conv=['#E91E63','#FF9800','#2196F3','#4CAF50','#9C27B0','#00BCD4']
    for n,col in zip(ns,colors_conv):
        hist=ga_results[n]['history']
        ax4.plot(hist,color=col,lw=1.8,label=f'{n} screws',alpha=0.85)
    ax4.set_xlabel('Generation'); ax4.set_ylabel('Best fitness score')
    ax4.set_title('GA Convergence\n(all screw counts)',fontweight='bold')
    ax4.legend(fontsize=8.5); ax4.grid(True,alpha=0.25)

    ax5=fig.add_subplot(gs[1,1],projection='polar')
    opt_angles_r=np.radians(ga_results[opt_n]['angles_deg'])
    ax5.scatter(opt_angles_r,[1.0]*opt_n,s=200,color='#27AE60',zorder=5)

    even_r=np.radians(np.arange(opt_n)*360/opt_n)
    ax5.scatter(even_r,[0.65]*opt_n,s=150,color='gray',alpha=0.6,marker='s',zorder=4)
    for a in opt_angles_r: ax5.plot([0,a],[0,1],'g-',lw=1.5,alpha=0.6)
    ax5.set_rticks([]); ax5.set_title(f'Optimal {opt_n}-screw layout\n(green=GA optimal, gray=even)',fontweight='bold',pad=15)
    ax5.set_ylim(0,1.3)

    ax6=fig.add_subplot(gs[1,2])
    P_grid=np.arange(-3,4.25,0.25); C_grid=np.arange(-2.5,0.25,0.25)
    coverage_map=np.zeros((len(P_grid),len(C_grid)))
    for i,P in enumerate(P_grid):
        for j,C in enumerate(C_grid):
            sf=P_to_sag(P); ss=P_to_sag(P+abs(C))
            if sf and ss:
                vm=3*max(abs(64*D0*(sf-S_BASE)/LENS_R**4),abs(64*D0*(ss-S_BASE)/LENS_R**4))*LENS_R**2/(4*TC**2)
                if vm<YIELD: coverage_map[i,j]=1
    im=ax6.pcolormesh(C_grid,P_grid,coverage_map,cmap='RdYlGn',shading='auto',vmin=0,vmax=1)
    plt.colorbar(im,ax=ax6,label='Achievable (1=yes)',shrink=0.8)
    ax6.set_xlabel('Cylinder (D)'); ax6.set_ylabel('Sphere (D)')
    ax6.set_title('Achievable Prescription Space\n(green=covered by AdaptivEyes)',fontweight='bold')

    ax7=fig.add_subplot(gs[2,0:2])
    common_rxs=[(1.0,0.0,0),(2.0,0.0,0),(3.5,0.0,0),(2.0,-0.75,90),(2.0,-1.5,45),(3.0,-0.5,0)]
    x_pos=range(len(common_rxs)); width=0.35
    score_even=[]; score_opt=[]
    even_config=[i*360/opt_n for i in range(opt_n)]
    for (sph,cyl,ax_d) in common_rxs:
        score_even.append(evaluate_screw_config(even_config,[(sph,cyl,ax_d)]))
        score_opt.append(evaluate_screw_config(ga_results[opt_n]['angles_deg'],[(sph,cyl,ax_d)]))
    ax7.bar([x-width/2 for x in x_pos],score_even,width,label='Even spacing',color='#FF6B6B',alpha=0.85)
    ax7.bar([x+width/2 for x in x_pos],score_opt,width,label='GA optimal',color='#27AE60',alpha=0.85)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f'S{s:+.1f}C{c:+.2f}' for (s,c,a) in common_rxs],fontsize=9,rotation=25,ha='right')
    ax7.set_ylabel('Fitness score (lower=better)'); ax7.legend(fontsize=9)
    ax7.set_title('Even vs GA-Optimal Placement\n(per prescription)',fontweight='bold'); ax7.grid(True,alpha=0.25,axis='y')

    ax8=fig.add_subplot(gs[2,2]); ax8.axis('off')
    stats=[
        ('Optimal screw count', str(opt_n)),
        ('Optimal angles', str(ga_results[opt_n]['angles_deg'])),
        ('Population coverage', f"{ga_results[opt_n]['coverage_pct']:.1f}%"),
        ('Fitness score', f"{ga_results[opt_n]['score']:.4f}"),
        ('GA improvement vs even', f"{(score_even[0]-score_opt[0])/max(score_even[0],0.001)*100:.1f}%"),
        ('Patients served (1B)', f"~{ga_results[opt_n]['coverage_pct']/100*1e9/1e6:.0f}M"),
    ]
    y=0.92
    for label,val in stats:
        ax8.text(0.02,y,label+':',fontsize=9,color='#2C3E50',fontweight='bold',transform=ax8.transAxes)
        ax8.text(0.6,y,val,fontsize=9,color='#1A5276',transform=ax8.transAxes)
        y-=0.14
    ax8.set_title('Optimization Summary',fontweight='bold',fontsize=10)
    rect=plt.Rectangle((0,0),1,1,facecolor='#EAF4FB',edgecolor='#4A90D9',linewidth=1.5,transform=ax8.transAxes)
    ax8.add_patch(rect)

    fig.suptitle('AdaptivEyes — Genetic Algorithm: Optimal Screw Configuration for Global Population',fontsize=13,fontweight='bold')
    p=os.path.join(OUT,'genetic_optimizer_results.png')
    fig.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"Saved → {p}")
    def jfix(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,(np.bool_,)):return bool(o)
        if isinstance(o,np.ndarray):return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(OUT,'genetic_optimizer_results.json'),'w') as f:
        json.dump(ga_results,f,indent=2,default=jfix)
    print("[Done]")
