import numpy as np
from itertools import combinations
import json

N_IDX =1.55
D_LENS =50e-3
R_LENS =D_LENS /2
S0 =2e-3
TC_MIN =1.0e-3
E_NOM =2.275e9
E_DEG =2.160e9
YIELD =65e6
P_APP =0.2e6

def sagitta_to_radius (s ,r =R_LENS ):
    return (r **2 +s **2 )/(2 *s )

def lensmaker_power (s_front ,s_back =None ,tc =None ):
    R1 =sagitta_to_radius (abs (s_front ))
    sign =1 if s_front >0 else -1
    if s_back is None :

        P =(N_IDX -1 )*sign /R1
    else :
        R2 =sagitta_to_radius (abs (s_back ))
        sign2 =-1 if s_back >0 else 1
        t =tc if tc else 3e-3
        P =(N_IDX -1 )*(sign /R1 +sign2 /R2 +(N_IDX -1 )*t /(N_IDX *R1 *R2 ))
    return P

def power_to_sagitta (P_target ,r =R_LENS ):
    if abs (P_target )<0.01 :
        return 0.001
    R =(N_IDX -1 )/abs (P_target )

    discriminant =R **2 -r **2
    if discriminant <0 :
        return None
    s =R -np .sqrt (discriminant )
    return s if P_target >0 else -s

def screw_influence_matrix (screw_angles ,r =R_LENS ):
    phi =np .linspace (0 ,np .pi ,180 )
    A =np .zeros ((len (phi ),len (screw_angles )))
    for j ,theta in enumerate (screw_angles ):
        A [:,j ]=np .cos (theta -phi )**2
    return A ,phi

def compute_actuation (screw_angles ,delta_sag_target ,astig_axis =None ,astig_mag =0 ):
    A ,phi =screw_influence_matrix (screw_angles )

    if astig_mag >0 and astig_axis is not None :

        ax_rad =np .radians (astig_axis )
        target =delta_sag_target +astig_mag *np .cos (2 *(phi -ax_rad ))
    else :
        target =np .full (len (phi ),delta_sag_target )

    forces ,residuals ,rank ,sv =np .linalg .lstsq (A ,target ,rcond =None )
    predicted =A @forces
    rms_err =np .sqrt (np .mean ((predicted -target )**2 ))
    return forces ,rms_err ,predicted ,phi ,target

def max_stress_estimate (force_N ,n_screws ,lens_thickness =3e-3 ):
    contact_area =np .pi *(0.5e-3 )**2 *n_screws
    stress =force_N /contact_area if contact_area >0 else 0
    return abs (stress )

def find_physical_range (n_screws ,screw_angles ,E =E_NOM ,verbose =False ):
    P_range =np .arange (-6.0 ,8.25 ,0.25 )
    achievable =[]

    for P in P_range :
        s_target =power_to_sagitta (P )
        if s_target is None :
            continue
        delta_s =s_target -S0 if P >0 else s_target -(-S0 )

        forces ,rms ,predicted ,phi ,target =compute_actuation (
        screw_angles ,delta_s if P >=0 else s_target
        )

        max_f =np .max (np .abs (forces ))
        stress =max_stress_estimate (max_f *1e6 *abs (delta_s )*E /E_NOM ,n_screws )

        s_abs =abs (s_target )if s_target else S0
        tc_est =3e-3 -0.5 *(s_abs -S0 )*1000

        ok =rms <0.5e-3 and tc_est >TC_MIN
        achievable .append ({
        'P':round (P ,2 ),
        's_target':round (s_target *1000 ,3 )if s_target else None ,
        'rms_mm':round (rms *1000 ,4 ),
        'achievable':ok ,
        'tc_mm':round (tc_est ,3 )
        })

    return achievable

def optimize_screw_config (max_screws =8 ,n_candidates =24 ):
    candidate_angles =np .linspace (0 ,2 *np .pi ,n_candidates ,endpoint =False )
    test_prescriptions =[
    (0.5 ,0 ,0 ),(1.0 ,0 ,0 ),(2.0 ,0 ,0 ),(3.0 ,0 ,0 ),
    (-1.0 ,0 ,0 ),(-2.0 ,0 ,0 ),
    (1.0 ,0.75 ,90 ),(2.0 ,1.25 ,45 ),
    ]

    results ={}

    for n in range (2 ,max_screws +1 ):
        best_score =np .inf
        best_angles =None

        if n <=5 :
            configs =list (combinations (range (n_candidates ),n ))

            if len (configs )>500 :
                idx =np .random .choice (len (configs ),500 ,replace =False )
                configs =[configs [i ]for i in idx ]
        else :
            configs =[]
            for _ in range (600 ):
                configs .append (tuple (np .random .choice (n_candidates ,n ,replace =False )))

        for cfg in configs :
            angles =candidate_angles [list (cfg )]
            total_err =0
            for (P_sph ,P_cyl ,ax )in test_prescriptions :
                s_t =power_to_sagitta (P_sph )
                if s_t is None :
                    continue
                delta_s =s_t -S0
                astig_s =power_to_sagitta (P_cyl )if P_cyl else 0
                astig_s =astig_s if astig_s else 0
                _ ,rms ,_ ,_ ,_ =compute_actuation (angles ,delta_s ,
                np .radians (ax )if P_cyl else None ,
                astig_s )
                total_err +=rms

            if total_err <best_score :
                best_score =total_err
                best_angles =angles

        achievable =find_physical_range (n ,best_angles )
        p_min =min ((a ['P']for a in achievable if a ['achievable']),default =0 )
        p_max =max ((a ['P']for a in achievable if a ['achievable']),default =0 )

        results [n ]={
        'n_screws':n ,
        'angles_deg':[round (np .degrees (a )%360 ,1 )for a in best_angles ],
        'total_rms_error':round (best_score *1000 ,4 ),
        'min_diopter':round (p_min ,2 ),
        'max_diopter':round (p_max ,2 ),
        'range_D':round (p_max -p_min ,2 ),
        'score':round (best_score *1000 ,4 )
        }
        print (f"  {n } screws: range {p_min :.2f} to {p_max :.2f}D, RMS={best_score *1000 :.4f}mm")

    return results

def fatigue_correction (cycle_count ,P_target ,screw_angles ):

    E_current =E_NOM -(E_NOM -E_DEG )*min (cycle_count /500 ,1.0 )
    compliance_ratio =E_NOM /E_current

    s_target =power_to_sagitta (P_target )
    if s_target is None :
        return None

    delta_s =s_target -S0
    forces_nom ,rms_nom ,_ ,_ ,_ =compute_actuation (screw_angles ,delta_s )

    correction_factor =1.0 /compliance_ratio
    forces_corrected =forces_nom *correction_factor

    A ,phi =screw_influence_matrix (screw_angles )
    predicted_uncorrected =A @forces_nom *compliance_ratio
    s_actual_uncorrected =S0 +np .mean (predicted_uncorrected )
    P_actual =lensmaker_power (s_actual_uncorrected )
    focal_drift =P_actual -lensmaker_power (s_target )

    return {
    'cycle_count':cycle_count ,
    'E_current_GPa':round (E_current /1e9 ,4 ),
    'compliance_ratio':round (compliance_ratio ,4 ),
    'correction_factor':round (correction_factor ,4 ),
    'P_target':P_target ,
    'focal_drift_D':round (focal_drift ,4 ),
    'within_tolerance':abs (focal_drift )<0.12 ,
    'forces_nominal':[round (f ,4 )for f in forces_nom ],
    'forces_corrected':[round (f ,4 )for f in forces_corrected ]
    }

def prescription_to_actuation (sphere ,cylinder ,axis_deg ,n_screws ,screw_angles ,cycle_count =0 ):

    s_sph =power_to_sagitta (sphere )
    if s_sph is None :
        return {'error':f'Sphere {sphere }D outside physical range'}

    delta_sph =s_sph -S0

    s_cyl =power_to_sagitta (abs (cylinder ))if cylinder else 0
    s_cyl =s_cyl if s_cyl else 0

    forces ,rms ,predicted ,phi ,target =compute_actuation (
    screw_angles ,delta_sph ,
    np .radians (axis_deg )if cylinder else None ,
    s_cyl
    )

    fat =fatigue_correction (cycle_count ,sphere ,screw_angles )
    cf =fat ['correction_factor']if fat else 1.0
    forces_final =forces *cf

    max_f =np .max (np .abs (forces_final ))
    stress =max_stress_estimate (max_f *0.1 ,n_screws )
    safety_factor =YIELD /max (stress ,1e3 )

    s_predicted =S0 +predicted
    P_predicted =[lensmaker_power (s )for s in s_predicted ]
    P_along_steep =max (P_predicted )
    P_along_flat =min (P_predicted )
    cyl_achieved =P_along_steep -P_along_flat

    return {
    'prescription':{'sphere':sphere ,'cylinder':cylinder ,'axis':axis_deg },
    'n_screws':n_screws ,
    'screw_angles_deg':[round (np .degrees (a )%360 ,1 )for a in screw_angles ],
    'screw_forces':[round (float (f ),5 )for f in forces_final ],
    'rms_surface_error_mm':round (rms *1000 ,4 ),
    'safety_factor':round (float (safety_factor ),2 ),
    'cycle_count':cycle_count ,
    'fatigue_correction':round (cf ,4 ),
    'P_sphere_achieved':round (float (lensmaker_power (S0 +delta_sph )),3 ),
    'P_cylinder_achieved':round (float (cyl_achieved ),3 ),
    'achievable':rms <0.5e-3 and float (safety_factor )>1.5
    }

if __name__ =='__main__':
    np .random .seed (42 )

    print ("="*60 )
    print ("AdaptivEyes ML Simulation System")
    print ("="*60 )

    print ("\n[1] Optimizing screw configurations (2-8 screws)...")
    opt_results =optimize_screw_config (max_screws =8 )

    print ("\n[2] Prescription-to-actuation mapping (5 screws)...")
    best_5 =opt_results [5 ]
    angles_5 =[np .radians (a )for a in best_5 ['angles_deg']]

    test_rxs =[
    (1.00 ,0.00 ,0 ,0 ),
    (2.50 ,0.00 ,0 ,0 ),
    (-1.50 ,0.00 ,0 ,0 ),
    (2.00 ,-0.75 ,90 ,0 ),
    (1.50 ,-1.25 ,45 ,0 ),
    (2.00 ,0.00 ,0 ,250 ),
    ]

    mappings =[]
    for (sph ,cyl ,ax ,cyc )in test_rxs :
        result =prescription_to_actuation (sph ,cyl ,ax ,5 ,angles_5 ,cyc )
        mappings .append (result )
        print (f"  Rx S{sph :+.2f} C{cyl :+.2f} x{ax :03d} cyc={cyc }: "
        f"achieved={result ['achievable']}, SF={result ['safety_factor']}, "
        f"RMS={result ['rms_surface_error_mm']:.4f}mm")

    print ("\n[3] Fatigue correction across 0-500 cycles (P=+2.00D)...")
    fatigue_data =[]
    for cyc in range (0 ,550 ,50 ):
        fd =fatigue_correction (cyc ,2.0 ,angles_5 )
        fatigue_data .append (fd )
        print (f"  Cycle {cyc :3d}: drift={fd ['focal_drift_D']:+.4f}D, "
        f"correction={fd ['correction_factor']:.4f}, "
        f"within_tol={fd ['within_tolerance']}")

    print ("\n[4] Physical range summary by screw count...")
    range_summary ={n :{
    'min_D':opt_results [n ]['min_diopter'],
    'max_D':opt_results [n ]['max_diopter'],
    'range_D':opt_results [n ]['range_D'],
    'angles':opt_results [n ]['angles_deg'],
    'rms':opt_results [n ]['score']
    }for n in range (2 ,9 )}

    import os
    output_path =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'results.json')

    def json_safe (o ):
        if isinstance (o ,np .integer ):return int (o )
        if isinstance (o ,np .floating ):return float (o )
        if isinstance (o ,np .bool_ ):return bool (o )
        if isinstance (o ,np .ndarray ):return o .tolist ()
        raise TypeError (f'Object of type {type (o ).__name__ } is not JSON serializable')

    output ={
    'optimizer':opt_results ,
    'mappings':mappings ,
    'fatigue':fatigue_data ,
    'range_summary':range_summary
    }
    with open (output_path ,'w')as f :
        json .dump (output ,f ,indent =2 ,default =json_safe )

    print (f"\n[Done] Results saved to {output_path }")
