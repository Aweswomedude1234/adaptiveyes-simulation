import numpy as np
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from fea_astigmatism import toroidal_fea, astig_optimizer, gen_training_data, smp_modulus

OUT = os.path.dirname(os.path.abspath(__file__))

def train_forward_model(X, y):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    sx=StandardScaler(); sy=StandardScaler()
    Xtr_s=sx.fit_transform(Xtr); Xte_s=sx.transform(Xte)
    ytr_s=sy.fit_transform(ytr)
    mdl=MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',
                     solver='adam',max_iter=3000,random_state=42,
                     early_stopping=True,validation_fraction=0.15,
                     n_iter_no_change=40,verbose=False)
    mdl.fit(Xtr_s,ytr_s)
    ypr=sy.inverse_transform(mdl.predict(Xte_s))
    r2=r2_score(yte,ypr); mae=mean_absolute_error(yte,ypr)
    return mdl,sx,sy,{'r2':round(float(r2),4),'mae':round(float(mae),6),
                      'n_train':len(Xtr),'n_test':len(Xte),'n_iter':mdl.n_iter_}

def train_inverse_model(X, y):
    X_inv = np.column_stack([y[:,0], y[:,1], X[:,2], X[:,3], X[:,4]])
    y_inv = X[:,:2]

    Xtr,Xte,ytr,yte = train_test_split(X_inv,y_inv,test_size=0.2,random_state=42)
    sx=StandardScaler(); sy=StandardScaler()
    Xtr_s=sx.fit_transform(Xtr); Xte_s=sx.transform(Xte)
    ytr_s=sy.fit_transform(ytr)
    mdl=MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',
                     solver='adam',max_iter=3000,random_state=42,
                     early_stopping=True,validation_fraction=0.15,
                     n_iter_no_change=40,verbose=False)
    mdl.fit(Xtr_s,ytr_s)
    ypr=sy.inverse_transform(mdl.predict(Xte_s))
    r2=r2_score(yte,ypr); mae=mean_absolute_error(yte,ypr)
    return mdl,sx,sy,{'r2':round(float(r2),4),'mae':round(float(mae),6),
                      'n_train':len(Xtr),'n_test':len(Xte),'n_iter':mdl.n_iter_}

def ml_predict_screw_plan(P_sphere, P_cylinder, axis_deg, cycle,
                           inv_mdl, inv_sx, inv_sy, n_screws=5):
    E_c = smp_modulus(cycle)
    X_q = np.array([[P_sphere, abs(P_cylinder), axis_deg/180.0, E_c/1e9, float(cycle)]])
    q_pred = inv_sy.inverse_transform(inv_mdl.predict(inv_sx.transform(X_q)))[0]
    q_mean_kPa, q_amp_kPa = q_pred

    from numpy import radians, cos
    phi_axis = radians(axis_deg)
    screw_angles = np.arange(n_screws)*(360/n_screws)
    contact_area = np.pi*(0.5e-3)**2
    pitch=0.4e-3; eff=0.35; T_max=0.05

    screws=[]
    for ang in screw_angles:
        theta=radians(ang)
        q_s = (q_mean_kPa + q_amp_kPa*cos(2*(theta-phi_axis)))*1e3
        F_N = abs(q_s)*contact_area
        turns = F_N*pitch/(2*np.pi*eff*T_max)
        turns = max(0.05, round(turns/0.25)*0.25)
        direction='CW' if q_s>=0 else 'CCW'
        screws.append({'angle_deg':float(ang),'turns':float(turns),'direction':direction,
                       'q_Pa':round(float(q_s),2)})
    return {'q_mean_Pa':round(q_mean_kPa*1e3,2),'q_amp_Pa':round(q_amp_kPa*1e3,2),'screws':screws}

if __name__=='__main__':
    np.random.seed(42)
    print("="*65)
    print("AdaptivEyes — ML Model with Astigmatism")
    print("="*65)

    astig_path = os.path.join(OUT,'astig_fea_results.json')
    print("\n[1] Loading FEA training data...")
    with open(astig_path) as f: d=json.load(f)
    X=np.array(d['training_X']); y=np.array(d['training_y'])
    print(f"    {X.shape[0]} samples, {X.shape[1]} features → {y.shape[1]} outputs")

    print("\n[2] Training forward model (pressures → outcomes)...")
    fwd_mdl,fwd_sx,fwd_sy,fwd_m=train_forward_model(X,y)
    print(f"    R²={fwd_m['r2']}  MAE={fwd_m['mae']:.6f}  Epochs={fwd_m['n_iter']}")
    print(f"    Outputs: P_sphere, P_cylinder, VM_stress, w0_flat, w0_steep")

    print("\n[3] Training inverse model (prescription → pressures)...")
    inv_mdl,inv_sx,inv_sy,inv_m=train_inverse_model(X,y)
    print(f"    R²={inv_m['r2']}  MAE={inv_m['mae']:.6f}  Epochs={inv_m['n_iter']}")
    print(f"    Outputs: q_mean_kPa, q_amp_kPa")

    print("\n[4] End-to-end: target Rx → ML pressures → screw plan...")
    test_rxs=[(3.5,0.0,0,0),(2.0,-0.75,90,0),(1.5,-1.25,45,0),
              (2.5,-1.5,30,0),(2.0,-0.75,90,250),(2.0,-0.75,90,500)]
    ml_results=[]
    for sph,cyl,ax,cyc in test_rxs:
        plan=ml_predict_screw_plan(sph,cyl,ax,cyc,inv_mdl,inv_sx,inv_sy)
        ml_results.append({'rx':{'sphere':sph,'cylinder':cyl,'axis':ax,'cycle':cyc},**plan})
        print(f"\n  S{sph:+.2f}C{cyl:+.2f}x{ax:03d} cyc={cyc}:")
        print(f"    q_mean={plan['q_mean_Pa']:.1f}Pa  q_amp={plan['q_amp_Pa']:.1f}Pa")
        for s in plan['screws']:
            print(f"    Screw@{s['angle_deg']:3.0f}°: {s['turns']:.2f}t {s['direction']}")

    def jfix(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,(np.bool_,)):return bool(o)
        if isinstance(o,np.ndarray):return o.tolist()
        raise TypeError(type(o))

    path=os.path.join(OUT,'ml_astig_results.json')
    with open(path,'w') as f:
        json.dump({'forward_metrics':fwd_m,'inverse_metrics':inv_m,'predictions':ml_results},
                  f,indent=2,default=jfix)
    print(f"\n[Done] → {path}")
