import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

OUT = os.path.dirname(os.path.abspath(__file__))

E_NOM=2.275e9; NU=0.40; TC=0.0025; LENS_R=0.025
N_IDX=1.55; S_BASE=0.002; YIELD=65e6
D0 = E_NOM*TC**3/(12*(1-NU**2))

def smp_E(c): return float(2.16e9+(E_NOM-2.16e9)*np.exp(-0.002*c))
def sag_P(s):
    if abs(s)<1e-7: return 0.0
    return (N_IDX-1)*np.sign(s)/((LENS_R**2+s**2)/(2*abs(s)))

class PINN:
    def __init__(self, sizes=[6,64,64,64,1], lr=5e-4):
        self.sizes=sizes; self.lr=lr
        np.random.seed(42)
        self.W=[np.random.randn(sizes[i],sizes[i+1])*np.sqrt(2/sizes[i])
                for i in range(len(sizes)-1)]
        self.b=[np.zeros((1,sizes[i+1])) for i in range(len(sizes)-1)]
        self.mW=[np.zeros_like(w) for w in self.W]
        self.vW=[np.zeros_like(w) for w in self.W]
        self.mb=[np.zeros_like(b) for b in self.b]
        self.vb=[np.zeros_like(b) for b in self.b]
        self.t=0
        self.lam_data=5.0; self.lam_pde=1.0; self.lam_bc=10.0

    def _fwd(self, X):
        A=[X]; Z=[]
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            z=A[-1]@W+b; Z.append(z)
            A.append(z if i==len(self.W)-1 else np.tanh(z))
        return A[-1], Z, A

    def predict(self, X):
        a=X.copy()
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            z=a@W+b
            a=z if i==len(self.W)-1 else np.tanh(z)
        return a

    def _backprop(self, Z, A, grad_out):
        N=A[0].shape[0]
        dA=grad_out; gW=[]; gb=[]
        for i in reversed(range(len(self.W))):
            dZ=dA if i==len(self.W)-1 else dA*(1-np.tanh(Z[i])**2)
            gW.insert(0, A[i].T@dZ/N)
            gb.insert(0, np.mean(dZ,axis=0,keepdims=True))
            dA=dZ@self.W[i].T
        return gW, gb

    def _adam(self, gW, gb):
        self.t+=1; b1,b2,eps=0.9,0.999,1e-8
        for i in range(len(self.W)):
            self.mW[i]=b1*self.mW[i]+(1-b1)*gW[i]
            self.vW[i]=b2*self.vW[i]+(1-b2)*gW[i]**2
            self.W[i]-=self.lr*(self.mW[i]/(1-b1**self.t))/(np.sqrt(self.vW[i]/(1-b2**self.t))+eps)
            self.mb[i]=b1*self.mb[i]+(1-b1)*gb[i]
            self.vb[i]=b2*self.vb[i]+(1-b2)*gb[i]**2
            self.b[i]-=self.lr*(self.mb[i]/(1-b1**self.t))/(np.sqrt(self.vb[i]/(1-b2**self.t))+eps)

    def pde_loss_grad(self, X_pde):
        q_total = (X_pde[:,2]*1e4 +
                   X_pde[:,3]*1e4 * np.cos(2*(X_pde[:,1]*2*np.pi - X_pde[:,4]*np.pi)))
        E_vals  = X_pde[:,5]*1e9
        D_vals  = E_vals*TC**3/(12*(1-NU**2))
        r_norm  = X_pde[:,0]
        w_exact = (q_total*LENS_R**4/(64*D_vals))*(1-r_norm**2)**2 * 1e6
        w_pred, Z, A = self._fwd(X_pde)
        res  = w_pred.ravel() - w_exact
        L    = float(np.mean(res**2))
        grad = (2*res/len(res)).reshape(-1,1)
        gW, gb = self._backprop(Z, A, grad)
        return L, gW, gb

    def train(self, X_data, y_data, X_pde, X_bc, n_epochs=3000):
        history={'total':[],'data':[],'pde':[],'bc':[]}
        for ep in range(n_epochs):
            wd, Zd, Ad = self._fwd(X_data)
            res_d = wd - y_data
            L_d   = float(np.mean(res_d**2))
            gWd, gbd = self._backprop(Zd, Ad, 2*res_d/len(X_data))

            L_p, gWp, gbp = self.pde_loss_grad(X_pde)

            wb, Zb, Ab = self._fwd(X_bc)
            L_b = float(np.mean(wb**2))
            gWb, gbb = self._backprop(Zb, Ab, 2*wb/len(X_bc))

            L_tot = self.lam_data*L_d + self.lam_pde*L_p + self.lam_bc*L_b
            gW_tot=[self.lam_data*gWd[i]+self.lam_pde*gWp[i]+self.lam_bc*gWb[i]
                    for i in range(len(self.W))]
            gb_tot=[self.lam_data*gbd[i]+self.lam_pde*gbp[i]+self.lam_bc*gbb[i]
                    for i in range(len(self.b))]
            self._adam(gW_tot, gb_tot)

            history['total'].append(L_tot)
            history['data'].append(L_d)
            history['pde'].append(L_p)
            history['bc'].append(L_b)
            if ep%500==0:
                print(f"  ep{ep:4d}: tot={L_tot:.5f} dat={L_d:.5f} "
                      f"pde={L_p:.5f} bc={L_b:.5f}")
        return history

def make_data(n_data=2000, n_pde=3000, n_bc=400):
    np.random.seed(42)
    a=LENS_R
    X_d=[]; y_d=[]
    for _ in range(n_data):
        sph=np.random.uniform(1.5,4.5)
        cyl=np.random.uniform(-2.5,0)
        ax =np.random.uniform(0,np.pi)
        cyc=np.random.randint(0,501)
        r  =np.random.uniform(0,1)
        phi=np.random.uniform(0,2*np.pi)
        E_c=smp_E(cyc); D=E_c*TC**3/(12*(1-NU**2))
        P_f=sph; P_s=sph+abs(cyl)
        def ps(P):
            R=(N_IDX-1)/abs(P); d=R**2-a**2
            return np.sign(P)*(R-np.sqrt(d)) if d>0 else None
        s_f=ps(P_f); s_s=ps(P_s)
        if not s_f or not s_s: continue
        q_f=64*D*(s_f-S_BASE)/a**4; q_s=64*D*(s_s-S_BASE)/a**4
        q_m=(q_f+q_s)/2; q_a=(q_s-q_f)/2
        q_phi=q_m+q_a*np.cos(2*(phi-ax))
        w=(q_phi*a**4/(64*D))*(1-r**2)**2*1e6
        X_d.append([r,phi/(2*np.pi),q_m/1e4,q_a/1e4,ax/np.pi,E_c/1e9])
        y_d.append([w])
    X_data=np.array(X_d,dtype=np.float32)
    y_data=np.array(y_d,dtype=np.float32)

    r_p=np.random.uniform(0.05,0.95,n_pde)
    phi_p=np.random.uniform(0,2*np.pi,n_pde)
    q_m_p=np.random.uniform(-5,5,n_pde)*1e4
    q_a_p=np.random.uniform(-2,2,n_pde)*1e4
    ax_p=np.random.uniform(0,np.pi,n_pde)
    E_p=np.array([smp_E(c) for c in np.random.randint(0,501,n_pde)])
    X_pde=np.column_stack([r_p,phi_p/(2*np.pi),q_m_p/1e4,
                            q_a_p/1e4,ax_p/np.pi,E_p/1e9]).astype(np.float32)

    X_bc=np.column_stack([np.ones(n_bc),
                           np.random.uniform(0,1,n_bc),
                           np.random.uniform(-5,5,n_bc),
                           np.random.uniform(-2,2,n_bc),
                           np.random.uniform(0,1,n_bc),
                           np.random.uniform(2.1,2.3,n_bc)]).astype(np.float32)
    return X_data,y_data,X_pde,X_bc

def visualize(pinn, history):
    fig=plt.figure(figsize=(18,11),facecolor='white')
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.32)

    ax1=fig.add_subplot(gs[0,0])
    ep=range(len(history['total']))
    ax1.semilogy(ep,history['total'],'k-',lw=2,label='Total')
    ax1.semilogy(ep,history['data'],'b-',lw=1.5,label=f'Data (×{pinn.lam_data})')
    ax1.semilogy(ep,history['pde'],'r-',lw=1.5,label=f'PDE (×{pinn.lam_pde})')
    ax1.semilogy(ep,history['bc'],'g-',lw=1.5,label=f'BC (×{pinn.lam_bc})')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('PINN Training Loss\n(3-component)',fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True,alpha=0.3)

    ax2=fig.add_subplot(gs[0,1])
    r_t=np.linspace(0,1,100).astype(np.float32)
    for P,col,ls in [(2.0,'blue','-'),(3.5,'green','-'),(4.0,'orange','--')]:
        R=(N_IDX-1)/P; d=R**2-LENS_R**2
        if d<0: continue
        s=R-np.sqrt(d); q=64*D0*(s-S_BASE)/LENS_R**4
        Xt=np.column_stack([r_t,np.zeros(100),np.full(100,q/1e4),
                             np.zeros(100),np.zeros(100),
                             np.full(100,E_NOM/1e9)]).astype(np.float32)
        w_e=(q*LENS_R**4/(64*D0))*(1-r_t**2)**2*1e6
        w_p=pinn.predict(Xt).ravel()
        ax2.plot(r_t,w_e,color=col,ls=ls,lw=2.5,label=f'+{P:.1f}D exact')
        ax2.plot(r_t,w_p,color=col,ls=':',lw=1.8,alpha=0.8,label=f'+{P:.1f}D PINN')
    ax2.set_xlabel('r/a'); ax2.set_ylabel('w (μm)')
    ax2.set_title('PINN vs Kirchhoff\nDeflection profiles',fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)

    ax3=fig.add_subplot(gs[0,2],projection='polar')
    r2=np.linspace(0,1,40); phi2=np.linspace(0,2*np.pi,60)
    R2,P2=np.meshgrid(r2,phi2)
    q_m2=1500.0; q_a2=600.0; ax2_=np.pi/4
    Xg=np.column_stack([R2.ravel(),P2.ravel()/(2*np.pi),
                         np.full(R2.size,q_m2/1e4),np.full(R2.size,q_a2/1e4),
                         np.full(R2.size,ax2_/np.pi),
                         np.full(R2.size,E_NOM/1e9)]).astype(np.float32)
    Wg=pinn.predict(Xg).reshape(R2.shape)
    c3=ax3.pcolormesh(P2,R2,Wg,cmap='RdYlBu_r',shading='auto')
    plt.colorbar(c3,ax=ax3,label='w (μm)',shrink=0.8)
    ax3.set_title('PINN 2D deformation field\n(astigmatic)',fontweight='bold',pad=15)

    ax4=fig.add_subplot(gs[1,0])
    Xm=[]; ym=[]
    for P in np.arange(1.5,4.5,0.2):
        R=(N_IDX-1)/P; d=R**2-LENS_R**2
        if d<0: continue
        s=R-np.sqrt(d); q=64*D0*(s-S_BASE)/LENS_R**4
        for r_ in np.linspace(0.05,0.95,20):
            w=(q*LENS_R**4/(64*D0))*(1-r_**2)**2*1e6
            Xm.append([r_,0,q/1e4,0,0,E_NOM/1e9]); ym.append(w)
    Xm=np.array(Xm); ym=np.array(ym)
    sc=StandardScaler(); mlp=MLPRegressor((64,64),max_iter=800,random_state=42)
    mlp.fit(sc.fit_transform(Xm),ym)
    for P,col,lbl in [(0.8,'purple','0.8D (extrap)'),(4.9,'red','4.9D (extrap)')]:
        R=(N_IDX-1)/abs(P); d=R**2-LENS_R**2
        if d<0: continue
        s=R-np.sqrt(d); q=64*D0*(s-S_BASE)/LENS_R**4
        r_t=np.linspace(0,1,80)
        Xt=np.column_stack([r_t,np.zeros(80),np.full(80,q/1e4),
                             np.zeros(80),np.zeros(80),
                             np.full(80,E_NOM/1e9)]).astype(np.float32)
        w_e=(q*LENS_R**4/(64*D0))*(1-r_t**2)**2*1e6
        w_p=pinn.predict(Xt).ravel()
        w_m=mlp.predict(sc.transform(Xt.astype(float)))
        ax4.plot(r_t,w_e,color=col,lw=2.5,ls='-')
        ax4.plot(r_t,w_p,color=col,lw=2,ls='--',label=f'PINN {lbl}')
        ax4.plot(r_t,w_m,color=col,lw=1.5,ls=':',alpha=0.7,label=f'MLP {lbl}')
    ax4.set_xlabel('r/a'); ax4.set_ylabel('w (μm)')
    ax4.set_title('Extrapolation comparison\nPINN vs standard MLP',fontweight='bold')
    ax4.legend(fontsize=7.5); ax4.grid(True,alpha=0.3)
    errors={}
    for P in [0.8,4.9]:
        R=(N_IDX-1)/abs(P); d=R**2-LENS_R**2
        if d<0: continue
        s=R-np.sqrt(d); q=64*D0*(s-S_BASE)/LENS_R**4
        r_t=np.linspace(0,1,80)
        Xt=np.column_stack([r_t,np.zeros(80),np.full(80,q/1e4),
                             np.zeros(80),np.zeros(80),
                             np.full(80,E_NOM/1e9)]).astype(np.float32)
        w_e=(q*LENS_R**4/(64*D0))*(1-r_t**2)**2*1e6
        w_p=pinn.predict(Xt).ravel()
        w_m=mlp.predict(sc.transform(Xt.astype(float)))
        errors[P]={'pinn_mae':float(np.mean(np.abs(w_p-w_e))),
                   'mlp_mae': float(np.mean(np.abs(w_m-w_e)))}
        print(f"  {P:.1f}D extrap: PINN={errors[P]['pinn_mae']:.4f}μm "
              f"MLP={errors[P]['mlp_mae']:.4f}μm  "
              f"PINN {errors[P]['mlp_mae']/max(errors[P]['pinn_mae'],0.001):.1f}× better")

    ax5=fig.add_subplot(gs[1,1],projection='polar')
    phi_m=np.linspace(0,2*np.pi,180)
    for (sph,cyl,ax_d,col) in [(2.0,-0.75,90,'blue'),(2.0,-1.5,45,'red')]:
        a=LENS_R; D=D0
        P_f=sph; P_s=sph+abs(cyl); ax_r=np.radians(ax_d)
        def ps3(P):
            R=(N_IDX-1)/abs(P); d=R**2-a**2
            return np.sign(P)*(R-np.sqrt(d)) if d>0 else None
        s_f=ps3(P_f); s_s=ps3(P_s)
        q_f=64*D*(s_f-S_BASE)/a**4; q_s=64*D*(s_s-S_BASE)/a**4
        q_m=(q_f+q_s)/2; q_a=(q_s-q_f)/2
        q_phi_m=q_m+q_a*np.cos(2*(phi_m-ax_r))
        s_phi=S_BASE+(q_phi_m*a**4/(64*D))
        P_phi=np.array([sag_P(s) for s in s_phi])
        ax5.plot(phi_m,P_phi,color=col,lw=2,
                 label=f'S{sph:+.1f}C{cyl:+.2f}x{ax_d}°')
        ax5.fill(phi_m,np.maximum(P_phi,0),alpha=0.1,color=col)
    ax5.set_title('Meridional power maps\n(astigmatic cases)',fontweight='bold',pad=15)
    ax5.legend(fontsize=8,loc='lower left')

    ax6=fig.add_subplot(gs[1,2])
    r_t2=np.linspace(0,1,80).astype(np.float32)
    P_t2=3.5; R=(N_IDX-1)/P_t2; d=R**2-LENS_R**2
    s=R-np.sqrt(d)
    for cyc,col in [(0,'blue'),(250,'orange'),(500,'red')]:
        E_c=smp_E(cyc); D_c=E_c*TC**3/(12*(1-NU**2))
        q=64*D_c*(s-S_BASE)/LENS_R**4
        Xt=np.column_stack([r_t2,np.zeros(80),np.full(80,q/1e4),
                             np.zeros(80),np.zeros(80),
                             np.full(80,E_c/1e9)]).astype(np.float32)
        w_p=pinn.predict(Xt).ravel()
        w_e=(q*LENS_R**4/(64*D_c))*(1-r_t2**2)**2*1e6
        ax6.plot(r_t2,w_e,color=col,lw=2,ls='-',label=f'Cycle {cyc} (exact)')
        ax6.plot(r_t2,w_p,color=col,lw=1.5,ls='--',alpha=0.8)
    ax6.set_xlabel('r/a'); ax6.set_ylabel('w (μm)')
    ax6.set_title('Fatigue effect on deflection\nPINN tracks E(cycle)',fontweight='bold')
    ax6.legend(fontsize=9); ax6.grid(True,alpha=0.3)

    fig.suptitle('AdaptivEyes — Physics-Informed Neural Network (PINN)',
                 fontsize=14,fontweight='bold')
    p=os.path.join(OUT,'pinn_results.png')
    fig.savefig(p,dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(); print(f"  Saved → {p}")
    return errors

if __name__=='__main__':
    print("="*65); print("AdaptivEyes — PINN"); print("="*65)
    print("\n[1] Generating data..."); X_d,y_d,X_pde,X_bc=make_data()
    print(f"    Data:{X_d.shape} PDE:{X_pde.shape} BC:{X_bc.shape}")
    print("\n[2] Training (3000 epochs)...")
    pinn=PINN([6,64,64,64,1],lr=5e-4)
    hist=pinn.train(X_d,y_d,X_pde,X_bc,3000)
    print("\n[3] Visualizing...")
    errs=visualize(pinn,hist)
    def jfix(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,(np.bool_,)):return bool(o)
        raise TypeError(type(o))
    with open(os.path.join(OUT,'pinn_results.json'),'w') as f:
        json.dump({'final_loss':hist['total'][-1],
                   'extrapolation_errors':errs},f,default=jfix)
    print("[Done]")
