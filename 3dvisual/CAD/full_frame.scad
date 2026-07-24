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
SCREW_HOLE_DIA   = 1.8;
SCREW_BOSS_DIA   = 5.5;
SCREW_BOSS_HEIGHT= 8.0;
SCREW_HEAD_DIA   = 3.4;
SCREW_HEAD_H     = 1.5;
SCREW_SHAFT_L    = 6.0;
COUNTERSINK_ANGLE= 90;

SCREW_RADIUS     = LENS_RADIUS + INNER_RING_WIDTH
                 + (OUTER_RING_WIDTH) / 2 + 0.5;

HINGE_ARC        = 12;
HINGE_DEPTH      = 1.2;

THERMO_GROOVE_W  = 1.5;
THERMO_GROOVE_D  = 0.8;
THERMO_72_HEIGHT = 6.5;
THERMO_85_HEIGHT = 3.5;

IPD              = 64.0;
BRIDGE_H         = 6.0;
BRIDGE_T         = 2.5;
BRIDGE_ARC_R     = 8.0;

TEMPLE_LENGTH    = 130.0;
TEMPLE_W         = 5.0;
TEMPLE_T         = 3.0;
TEMPLE_TAPER     = 1.5;

DRAIN_WIDTH      = 3.0;
DRAIN_HEIGHT     = 2.0;

SHOW_FRAME       = true;
SHOW_LENS_GHOST  = true;
SHOW_SCREWS      = true;
EXPLODE_Z        = 0;

$fn = 120;

INNER_RING_ID  = LENS_DIAMETER + 2*LENS_CLEARANCE;
INNER_RING_OD  = INNER_RING_ID + 2*INNER_RING_WIDTH;
OUTER_RING_ID  = INNER_RING_OD;
OUTER_RING_OD  = OUTER_RING_ID + 2*OUTER_RING_WIDTH;

LENS_CX = IPD / 2;

function screw_angle(i) = i * (360 / N_SCREWS) + 90;

module tube(h, id, od, center=false) {
    difference() {
        cylinder(h=h, d=od, center=center);
        translate([0, 0, -0.01])
            cylinder(h=h+0.02, d=id, center=center);
    }
}

module living_hinge_cut(radius, height) {
    rotate_extrude(angle=HINGE_ARC, $fn=80)
        translate([radius - HINGE_DEPTH, 0, 0])
            square([HINGE_DEPTH + 0.1, height]);
}

module thermo_groove(z_pos, ring_od) {
    translate([0, 0, z_pos])
        rotate_extrude($fn=120)
            translate([ring_od/2 - THERMO_GROOVE_D, -THERMO_GROOVE_W/2, 0])
                square([THERMO_GROOVE_D + 0.1, THERMO_GROOVE_W]);
}

module drain_slot() {
    translate([-DRAIN_WIDTH/2, -OUTER_RING_OD/2 - 0.1, 0])
        cube([DRAIN_WIDTH, OUTER_RING_OD/2 + 0.2, DRAIN_HEIGHT]);
}

module screw_boss() {
    difference() {

        cylinder(h=SCREW_BOSS_HEIGHT, d=SCREW_BOSS_DIA);

        translate([0, 0, -0.01])
            cylinder(h=SCREW_BOSS_HEIGHT + 0.02, d=SCREW_HOLE_DIA);

        translate([0, 0, SCREW_BOSS_HEIGHT - SCREW_HEAD_H])
            cylinder(h=SCREW_HEAD_H + 0.1,
                     d1=SCREW_HOLE_DIA,
                     d2=SCREW_HEAD_DIA + 0.4);
    }
}

module screw_model() {

    color("silver")
    union() {

        translate([0, 0, 0])
            cylinder(h=SCREW_HEAD_H,
                     d1=SCREW_HOLE_DIA,
                     d2=SCREW_HEAD_DIA,
                     $fn=30);

        translate([0, 0, SCREW_HEAD_H])
            cylinder(h=0.4, d=SCREW_HEAD_DIA, $fn=30);

        translate([0, 0, SCREW_HEAD_H + 0.05]) {
            cube([SCREW_HEAD_DIA*0.7, 0.4, 0.5], center=true);
            cube([0.4, SCREW_HEAD_DIA*0.7, 0.5], center=true);
        }

        translate([0, 0, -SCREW_SHAFT_L])
            cylinder(h=SCREW_SHAFT_L, d=SCREW_HOLE_DIA, $fn=20);
    }
}

module screw_ring() {
    for (i = [0 : N_SCREWS-1]) {
        rotate([0, 0, screw_angle(i)])
            translate([SCREW_RADIUS, 0,
                       SCREW_BOSS_HEIGHT + EXPLODE_Z])
                rotate([180, 0, 0])
                    screw_model();
    }
}

module lens_ring_assembly() {
    difference() {
        union() {

            difference() {
                tube(h=INNER_RING_HEIGHT,
                     id=INNER_RING_ID,
                     od=INNER_RING_OD);

                for (i = [0 : N_SCREWS-1]) {
                    rotate([0, 0, screw_angle(i) - HINGE_ARC/2])
                        living_hinge_cut(INNER_RING_OD/2, INNER_RING_HEIGHT);
                }
            }

            translate([0, 0, INNER_RING_HEIGHT - LIP_HEIGHT])
                tube(h=LIP_HEIGHT,
                     id=INNER_RING_ID - 2*LIP_WIDTH,
                     od=INNER_RING_ID);

            tube(h=OUTER_RING_HEIGHT,
                 id=OUTER_RING_ID,
                 od=OUTER_RING_OD);

            for (i = [0 : N_SCREWS-1]) {
                rotate([0, 0, screw_angle(i)])
                    translate([SCREW_RADIUS, 0, 0])
                        screw_boss();
            }
        }

        translate([0, 0, -0.1])
            cylinder(h=OUTER_RING_HEIGHT + 0.2, d=INNER_RING_ID);

        thermo_groove(THERMO_72_HEIGHT, OUTER_RING_OD + 0.5);
        thermo_groove(THERMO_85_HEIGHT, OUTER_RING_OD + 0.5);

        translate([OUTER_RING_OD/2 - 0.5, -3, THERMO_72_HEIGHT - 0.5])
            rotate([0, 90, 0])
                linear_extrude(height=0.6)
                    text("72°", size=2.5,
                         font="Liberation Sans:style=Bold",
                         halign="center", valign="center");

        translate([OUTER_RING_OD/2 - 0.5, -3, THERMO_85_HEIGHT - 0.5])
            rotate([0, 90, 0])
                linear_extrude(height=0.6)
                    text("85°", size=2.5,
                         font="Liberation Sans:style=Bold",
                         halign="center", valign="center");

        drain_slot();
    }
}

module nose_bridge() {

    gap   = IPD - INNER_RING_OD;
    half  = gap / 2;

    difference() {

        translate([-half, -BRIDGE_T/2, OUTER_RING_HEIGHT/2 - BRIDGE_H/2])
            cube([gap, BRIDGE_T, BRIDGE_H]);

        translate([0, 0, OUTER_RING_HEIGHT/2 - BRIDGE_H/2 - 0.1])
            cylinder(h=BRIDGE_H + 0.2, r=BRIDGE_ARC_R, $fn=80);
    }
}

module temple_arm(side=1) {

    tab_x = side * OUTER_RING_OD/2;
    arm_z = OUTER_RING_HEIGHT/2 - TEMPLE_W/2;

    translate([tab_x, -TEMPLE_W/2, arm_z]) {

        cube([side * TEMPLE_T, TEMPLE_W, TEMPLE_W]);

        hull() {
            cube([side * TEMPLE_T, TEMPLE_W, TEMPLE_W]);
            translate([side * TEMPLE_LENGTH, (TEMPLE_W - (TEMPLE_W - TEMPLE_TAPER))/2, 0])
                cube([side * 0.1, TEMPLE_W - TEMPLE_TAPER, TEMPLE_W - TEMPLE_TAPER]);
        }
    }
}

module full_eyeglass_frame() {

    translate([LENS_CX, 0, 0])
        lens_ring_assembly();

    translate([-LENS_CX, 0, 0])
        mirror([1, 0, 0])
            lens_ring_assembly();

    nose_bridge();

    translate([LENS_CX, 0, 0])
        temple_arm(side=+1);

    translate([-LENS_CX, 0, 0])
        mirror([1, 0, 0])
            temple_arm(side=+1);

    translate([LENS_CX + OUTER_RING_OD/2 - 0.5, -3,
               OUTER_RING_HEIGHT - 2])
        rotate([0, 90, 0])
            linear_extrude(height=0.6)
                text("AdaptivEyes", size=2.8,
                     font="Liberation Sans:style=Bold",
                     halign="center", valign="center");
}

module lens_ghost() {
    color("lightblue", 0.25)
        cylinder(h=LENS_THICKNESS, d=LENS_DIAMETER);
}

if (SHOW_FRAME) {
    color("SteelBlue")
        full_eyeglass_frame();
}

if (SHOW_SCREWS) {

    translate([LENS_CX, 0, 0])
        screw_ring();

    translate([-LENS_CX, 0, 0])
        mirror([1, 0, 0])
            screw_ring();
}

if (SHOW_LENS_GHOST) {
    z_lens = INNER_RING_HEIGHT - LENS_THICKNESS - LIP_HEIGHT;
    translate([ LENS_CX, 0, z_lens]) lens_ghost();
    translate([-LENS_CX, 0, z_lens]) lens_ghost();
}
