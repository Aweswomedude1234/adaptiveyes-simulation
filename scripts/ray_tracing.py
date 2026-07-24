"""
AdaptivEyes — Option 3: Optical Ray Tracing (corrected geometry)
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import os, json

OUT=os.path.dirname(os.path.abspath(__file__))
N_IDX=1.55; LENS_R=0.025; S_BASE=0.002; TC=0.0025; E_NOM=2.275e9; NU=0.40

def P_to_sag(P):
    if abs(P)<0.01: return S_BASE
    R=(N_IDX-1)/abs(P); d=R**2-LENS_R**2
    return float(np.sign(P)*(R-np.sqrt(d))) if d>0 else None

def lens_R_curv(s): return (LENS_R**2+s**2)/(2*abs(s)) if abs(s)>1e-7 else 1e9

def surface_z_sph(x,y,s):
    r=np.sqrt(x**2+y**2); Rc=lens_R_curv(s)
    return np.sign(s)*(Rc-np.sqrt(np.maximum(Rc**2-r**2,0)))

def surface_z_toric(x,y,s_flat,s_steep,axis_deg):
    phi=np.arctan2(y,x); phi_ax=np.radians(axis_deg)
    Rf=lens_R_curv(s_flat); Rs=lens_R_curv(s_steep)
    r=np.sqrt(x**2+y**2)
    c=np.cos(phi-phi_ax)**2; s_=np.sin(phi-phi_ax)**2
    R_phi=1/(c/Rf+s_/Rs) if (c/Rf+s_/Rs)>0 else 1e9
    return R_phi - np.sqrt(np.maximum(R_phi**2-r**2,0))

def surface_normal(x,y,s,toroidal=False,s_flat=None,s_steep=None,axis_deg=0):
    eps=1e-6
    if toroidal:
        z0=surface_z_toric(x,y,s_flat,s_steep,axis_deg)
        dzx=(surface_z_toric(x+eps,y,s_flat,s_steep,axis_deg)-z0)/eps
        dzy=(surface_z_toric(x,y+eps,s_flat,s_steep,axis_deg)-z0)/eps
    else:
        z0=surface_z_sph(x,y,s)
        dzx=(surface_z_sph(x+eps,y,s)-z0)/eps
        dzy=(surface_z_sph(x,y+eps,s)-z0)/eps
    n=np.array([-dzx,-dzy,1.0]); return n/np.linalg.norm(n)

def snell(d,n_hat,n1,n2):
    d=d/np.linalg.norm(d); n_hat=n_hat/np.linalg.norm(n_hat)
    cos_i=float(-np.dot(d,n_hat));
    if cos_i<0: n_hat=-n_hat; cos_i=-cos_i
    sin_i2=max(0,1-cos_i**2); ratio=n1/n2; sin_t2=ratio**2*sin_i2
    if sin_t2>=1: return d
    cos_t=np.sqrt(1-sin_t2)
    return ratio*d+(ratio*cos_i-cos_t)*n_hat

def trace_ray_full(px,py,P_lens,object_dist=0.5,toroidal=False,s_flat=None,s_steep=None,axis_deg=0):
    """
    Trace ray from point source at object_dist.
    px,py: position in pupil plane (meters)
    P_lens: lens power (D)
    Returns (retinal_x, retinal_y) in microns, or None
    """
    s=P_to_sag(P_lens) if not toroidal else (s_flat+s_steep)/2
    if s is None: return None

    obj=np.array([0.0,0.0,-object_dist])

    pupil_pt=np.array([px,py,0.0])
    d=(pupil_pt-obj); d=d/np.linalg.norm(d)
    pos=obj.copy()

    if abs(d[2])>1e-10:
        t=(0-pos[2])/d[2]; pos=pos+t*d
    r_hit=np.sqrt(pos[0]**2+pos[1]**2)
    if r_hit>LENS_R: return None

    if toroidal:
        n_hat=surface_normal(pos[0],pos[1],None,True,s_flat,s_steep,axis_deg)
    else:
        n_hat=surface_normal(pos[0],pos[1],s)
    d=snell(d,n_hat,1.0,N_IDX)

    if abs(d[2])>1e-10:
        t=(TC-pos[2])/d[2]; pos=pos+t*d

    n_back=np.array([0.0,0.0,1.0])
    d=snell(d,n_back,N_IDX,1.0)

    P_eff=P_lens
    v=1/(P_eff-1/object_dist) if abs(P_eff-1/object_dist)>0.01 else 10.0
    retina_z=TC+v

    if abs(d[2])>1e-10:
        t=(retina_z-pos[2])/d[2]; pos_ret=pos+t*d
    else: return None
    return (pos_ret[0]*1e6, pos_ret[1]*1e6)

def spot_diagram(P_lens, n_rings=10, n_per_ring=20, object_dist=0.5,
                 toroidal=False, s_flat=None, s_steep=None, axis_deg=0):
    hits=[]; pupil_r=1.5e-3
    for ring in range(1,n_rings+1):
        r=pupil_r*ring/n_rings
        for j in range(n_per_ring):
            phi=2*np.pi*j/n_per_ring
            px=r*np.cos(phi); py=r*np.sin(phi)
            result=trace_ray_full(px,py,P_lens,object_dist,toroidal,s_flat,s_steep,axis_deg)
            if result: hits.append(result)
    return np.array(hits) if hits else np.zeros((2,2))

def rms_spot(hits):
    if len(hits)<3: return 999.0
    cx=np.mean(hits[:,0]); cy=np.mean(hits[:,1])
    return float(np.sqrt(np.mean((hits[:,0]-cx)**2+(hits[:,1]-cy)**2)))

def mtf_from_spots(hits,max_freq=60,n_freqs=50):
    freqs=np.linspace(0,max_freq,n_freqs)
    if len(hits)<3: return freqs,np.ones(n_freqs)
    cx=np.mean(hits[:,0]); cy=np.mean(hits[:,1])
    mtf=[]
    for f in freqs:
        f_cyc_um=f/57.3e3
        phases=2*np.pi*f_cyc_um*(hits[:,0]-cx)
        mtf.append(abs(np.mean(np.exp(1j*phases))))
    return freqs,np.array(mtf)

def va_from_rms(rms):
    if rms<2: return "20/10+"
    elif rms<5: return "20/20"
    elif rms<10: return "20/30"
    elif rms<20: return "20/50"
    elif rms<40: return "20/100"
    else: return "20/200+"

if __name__=='__main__':
    print("="*65); print("AdaptivEyes — Optical Ray Tracing"); print("="*65)

    cases=[
        ('Perfect +3.5D',       3.5, False, None, None, 0,   '#2196F3'),
        ('Corrected +2.0D\n→ Patient needs +3.5D', 2.0, False, None, None, 0, '#FF9800'),
        ('Overcorrected +4.5D', 4.5, False, None, None, 0,   '#9C27B0'),
        ('Astigmatic\nS+2.0 C-0.75 x90°',2.0,True,P_to_sag(2.0),P_to_sag(2.75),90,'#E91E63'),
        ('Astigmatic corrected\nS+2.0 C-0.75 x90° (full)',2.375,True,P_to_sag(2.0),P_to_sag(2.75),90,'#4CAF50'),
    ]

    print("\n[1] Spot diagrams...")
    spots_all=[]; rms_all=[]
    for (label,P,tor,sf,ss,ax_d,col) in cases:
        sp=spot_diagram(P,toroidal=tor,s_flat=sf,s_steep=ss,axis_deg=ax_d)
        rms=rms_spot(sp); va=va_from_rms(rms)
        spots_all.append(sp); rms_all.append(rms)
        print(f"  {label.replace(chr(10),' ')}: RMS={rms:.2f}μm  VA≈{va}")

    print("\n[2] Plotting...")
    fig=plt.figure(figsize=(22,14),facecolor='white')
    gs=gridspec.GridSpec(3,5,figure=fig,hspace=0.5,wspace=0.3)

    max_rms=max(rms_all)*1.5+2
    for col_i,(label,P,tor,sf,ss,ax_d,color) in enumerate(cases):
        ax=fig.add_subplot(gs[0,col_i])
        sp=spots_all[col_i]; rms=rms_all[col_i]
        if len(sp)>2:
            cx=np.mean(sp[:,0]); cy=np.mean(sp[:,1])
            dist=np.sqrt((sp[:,0]-cx)**2+(sp[:,1]-cy)**2)
            ax.scatter(sp[:,0]-cx,sp[:,1]-cy,c=dist,cmap='plasma',s=12,alpha=0.75,vmin=0)

        th=np.linspace(0,2*np.pi,100)
        ax.plot(5*np.cos(th),5*np.sin(th),'g:',lw=1.5,alpha=0.8,label='20/20 (5μm)')
        ax.plot(2*np.cos(th),2*np.sin(th),'b:',lw=1,alpha=0.5,label='Airy')
        lim=max(max_rms,15)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
        ax.set_title(f'{label}\nRMS={rms:.1f}μm  {va_from_rms(rms)}',fontsize=8,fontweight='bold')
        ax.set_xlabel('x (μm)',fontsize=8); ax.tick_params(labelsize=7)
        if col_i==0: ax.set_ylabel('y (μm)',fontsize=8)
        ax.grid(True,alpha=0.2)

    ax_mtf=fig.add_subplot(gs[1,:])
    for (label,P,tor,sf,ss,ax_d,color),sp in zip(cases,spots_all):
        freqs,mtf=mtf_from_spots(sp)
        ax_mtf.plot(freqs,mtf,color=color,lw=2.5,label=label.replace('\n',' '))
    ax_mtf.axvline(30,color='black',ls='--',lw=1.5,alpha=0.6,label='30 c/deg = 20/20 threshold')
    ax_mtf.axhline(0.15,color='gray',ls=':',lw=1,alpha=0.5,label='MTF=0.15 (perception cutoff)')
    ax_mtf.set_xlabel('Spatial frequency (cycles/degree)',fontsize=11)
    ax_mtf.set_ylabel('MTF',fontsize=11)
    ax_mtf.set_title('Modulation Transfer Function — Optical Image Quality',fontweight='bold',fontsize=12)
    ax_mtf.legend(fontsize=8.5,loc='upper right'); ax_mtf.grid(True,alpha=0.25); ax_mtf.set_ylim(0,1.05)

    ax_rms=fig.add_subplot(gs[2,0:2])
    P_sw=np.arange(1.5,4.75,0.25); rms_sw=[]
    for P_ in P_sw:
        sp_=spot_diagram(P_); rms_sw.append(rms_spot(sp_))
    ax_rms.plot(P_sw,rms_sw,'b-o',lw=2.5,markersize=7)
    ax_rms.axhline(5,color='green',ls='--',lw=2,label='20/20 (5μm)')
    ax_rms.fill_between(P_sw,0,5,alpha=0.1,color='green')
    ax_rms.set_xlabel('Lens prescription (D)'); ax_rms.set_ylabel('RMS spot size (μm)')
    ax_rms.set_title('Optical Quality vs Prescription\n(ray tracing)',fontweight='bold')
    ax_rms.legend(fontsize=9); ax_rms.grid(True,alpha=0.25)

    ax_ast=fig.add_subplot(gs[2,2:4])
    for (cyl,col,lbl) in [(-0.5,'#2196F3','C-0.50'),(-1.0,'#E91E63','C-1.00'),(-1.5,'#FF5722','C-1.50')]:
        sf=P_to_sag(2.0); ss=P_to_sag(2.0+abs(cyl))
        if not sf or not ss: continue
        sp_a=spot_diagram(2.0+abs(cyl)/2,toroidal=True,s_flat=sf,s_steep=ss,axis_deg=90)
        if len(sp_a)>2:
            cx=np.mean(sp_a[:,0]); cy=np.mean(sp_a[:,1])
            ax_ast.scatter(sp_a[:,0]-cx,sp_a[:,1]-cy,color=col,s=10,alpha=0.7,label=lbl)
    ax_ast.set_xlabel('x (μm)'); ax_ast.set_ylabel('y (μm)')
    ax_ast.set_title('Astigmatic Spot Shapes\n(elongation = uncorrected cylinder)',fontweight='bold')
    ax_ast.set_aspect('equal'); ax_ast.legend(fontsize=9); ax_ast.grid(True,alpha=0.25)

    ax_tab=fig.add_subplot(gs[2,4]); ax_tab.axis('off')
    rows=[['Condition','RMS','VA']]
    for (label,P,tor,sf,ss,ax_d,color),rms in zip(cases,rms_all):
        rows.append([label.replace('\n',' ')[:18],f'{rms:.1f}μm',va_from_rms(rms)])
    tbl=ax_tab.table(cellText=rows[1:],colLabels=rows[0],loc='center',bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor('#2196F3'); cell.set_text_props(color='white',fontweight='bold')
        elif r%2==0: cell.set_facecolor('#EAF4FB')
    ax_tab.set_title('VA Summary',fontweight='bold',fontsize=9)

    fig.suptitle('AdaptivEyes — Optical Ray Tracing: Lens Geometry → Visual Acuity',fontsize=14,fontweight='bold')
    p=os.path.join(OUT,'ray_tracing_results.png')
    fig.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"Saved → {p}"); print("[Done]")
