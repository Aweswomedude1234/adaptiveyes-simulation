LENS_DIAMETER    = 50.0;
LENS_RADIUS      = LENS_DIAMETER / 2;
LENS_THICKNESS   = 2.5;
LENS_CLEARANCE   = 0.3;

INNER_RING_WIDTH = 3.5;
INNER_RING_HEIGHT= 5.0;
LIP_HEIGHT       = 1.2;
LIP_WIDTH        = 1.0;

OUTER_RING_WIDTH = 6.0;
OUTER_RING_HEIGHT= 8.0;

N_SCREWS         = 5;
SCREW_RADIUS     = LENS_RADIUS + INNER_RING_WIDTH + 3.0;
SCREW_HOLE_DIA   = 1.6;
SCREW_BOSS_DIA   = 5.5;
SCREW_BOSS_HEIGHT= 8.0;

HINGE_WIDTH      = 1.2;
HINGE_ARC        = 15;
HINGE_DEPTH      = 1.5;

THERMO_GROOVE_W  = 1.5;
THERMO_GROOVE_D  = 0.8;
THERMO_72_HEIGHT = 6.5;
THERMO_85_HEIGHT = 3.5;

DRAIN_WIDTH      = 3.0;
DRAIN_HEIGHT     = 2.0;

BRIDGE_WIDTH     = 18.0;
BRIDGE_HEIGHT    = 4.0;
BRIDGE_THICKNESS = 2.5;

TEMPLE_TAB_W     = 8.0;
TEMPLE_TAB_H     = 5.0;
TEMPLE_TAB_T     = 3.0;

SHOW_FRAME       = true;
SHOW_LENS_GHOST  = false;
SHOW_SCREWS      = false;
EXPLODE_Z        = 0;

$fn = 120;

INNER_RING_ID  = LENS_DIAMETER + 2*LENS_CLEARANCE;
INNER_RING_OD  = INNER_RING_ID + 2*INNER_RING_WIDTH;

OUTER_RING_ID  = INNER_RING_OD;
OUTER_RING_OD  = OUTER_RING_ID + 2*OUTER_RING_WIDTH;

function screw_angle(i) = i * (360 / N_SCREWS);

module tube(h,id,od,center=false){
    difference(){
        cylinder(h=h,d=od,center=center);
        cylinder(h=h+0.02,d=id,center=center);
    }
}

module screw_boss(height){
    difference(){
        cylinder(h=height,d=SCREW_BOSS_DIA);

        translate([0,0,-0.01])
            cylinder(h=height+0.02,d=SCREW_HOLE_DIA);

        translate([0,0,height-1.5])
            cylinder(h=2,d1=SCREW_HOLE_DIA,d2=SCREW_HOLE_DIA+1.5);
    }

    translate([-SCREW_BOSS_DIA/2,-1,0])
        cube([1.5,2,height]);
}

module living_hinge_cut(radius,height){
    rotate_extrude(angle=HINGE_ARC,$fn=60)
        translate([radius-HINGE_DEPTH,0,0])
            square([HINGE_DEPTH+0.1,height]);
}

module thermo_groove(z_pos,ring_od){
    translate([0,0,z_pos])
        rotate_extrude($fn=120)
            translate([ring_od/2-THERMO_GROOVE_D,-THERMO_GROOVE_W/2,0])
                square([THERMO_GROOVE_D+0.1,THERMO_GROOVE_W]);
}

module drain_slot(){
    translate([-DRAIN_WIDTH/2,-OUTER_RING_OD/2-0.1,0])
        cube([DRAIN_WIDTH,OUTER_RING_OD/2+0.2,DRAIN_HEIGHT]);
}

module nose_bridge_tab(){
    translate([0,
              -(INNER_RING_OD/2+BRIDGE_THICKNESS/2),
               OUTER_RING_HEIGHT/2-BRIDGE_HEIGHT/2])
        cube([BRIDGE_WIDTH/2,BRIDGE_THICKNESS,BRIDGE_HEIGHT]);
}

module temple_tab(side=1){
    translate([side*(OUTER_RING_OD/2),
               -TEMPLE_TAB_W/2,
               OUTER_RING_HEIGHT/2-TEMPLE_TAB_H/2])
        cube([TEMPLE_TAB_T,TEMPLE_TAB_W,TEMPLE_TAB_H]);
}

module adaptiveyes_frame(){

    difference(){

        union(){

            difference(){
                tube(h=INNER_RING_HEIGHT,
                     id=INNER_RING_ID,
                     od=INNER_RING_OD);

                for(i=[0:N_SCREWS-1]){
                    rotate([0,0,screw_angle(i)-HINGE_ARC/2])
                        living_hinge_cut(INNER_RING_OD/2,INNER_RING_HEIGHT);
                }
            }

            translate([0,0,INNER_RING_HEIGHT-LIP_HEIGHT])
                tube(h=LIP_HEIGHT,
                     id=INNER_RING_ID-2*LIP_WIDTH,
                     od=INNER_RING_ID);

            tube(h=OUTER_RING_HEIGHT,
                 id=OUTER_RING_ID,
                 od=OUTER_RING_OD);

            for(i=[0:N_SCREWS-1]){
                rotate([0,0,screw_angle(i)])
                    translate([SCREW_RADIUS,0,0])
                        screw_boss(SCREW_BOSS_HEIGHT);
            }

            nose_bridge_tab();
            temple_tab(+1);
            temple_tab(-1);
        }

        translate([0,0,-0.1])
            cylinder(h=OUTER_RING_HEIGHT+0.2,d=INNER_RING_ID);

        thermo_groove(THERMO_72_HEIGHT,OUTER_RING_OD+0.5);
        thermo_groove(THERMO_85_HEIGHT,OUTER_RING_OD+0.5);

        translate([OUTER_RING_OD/2-0.5,-3,THERMO_72_HEIGHT-0.5])
            rotate([0,90,0])
                linear_extrude(height=0.6)
                    text("72°",size=2.5,
                    font="Liberation Sans:style=Bold",
                    halign="center",valign="center");

        translate([OUTER_RING_OD/2-0.5,-3,THERMO_85_HEIGHT-0.5])
            rotate([0,90,0])
                linear_extrude(height=0.6)
                    text("85°",size=2.5,
                    font="Liberation Sans:style=Bold",
                    halign="center",valign="center");

        drain_slot();

        translate([0,OUTER_RING_OD/2-0.4,OUTER_RING_HEIGHT-2])
            rotate([90,0,0])
                linear_extrude(height=0.6)
                    text("AdaptivEyes",
                    size=2.8,
                    font="Liberation Sans:style=Bold",
                    halign="center",
                    valign="center");
    }
}

module lens_ghost(){
    color("lightblue",0.3)
        cylinder(h=LENS_THICKNESS,d=LENS_DIAMETER);
}

module screw_viz(){
    for(i=[0:N_SCREWS-1]){
        rotate([0,0,screw_angle(i)])
            translate([SCREW_RADIUS,0,SCREW_BOSS_HEIGHT-0.5])
                color("silver")
                    cylinder(h=4,d=2);
    }
}

if(SHOW_FRAME)
    color("SteelBlue")
        translate([0,0,EXPLODE_Z])
            adaptiveyes_frame();

if(SHOW_LENS_GHOST)
    translate([0,0,INNER_RING_HEIGHT-LENS_THICKNESS-LIP_HEIGHT])
        lens_ghost();

if(SHOW_SCREWS)
    screw_viz();
