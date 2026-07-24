$fn = 80;

LENS_D        = 50.0;
LENS_R        = LENS_D / 2;
RIM_W         = 3.5;
RIM_H         = 8.0;
LIP_H         = 1.5;
LIP_W         = 1.2;

BRIDGE_MIN    = 14.0;
BRIDGE_MAX    = 24.0;
BRIDGE_H      = 7.0;
BRIDGE_W      = 5.0;
SLIDE_SLOT_W  = 2.2;
SLIDE_SLOT_D  = 1.5;
THUMB_SCREW_D = 3.5;

TEMPLE_MIN    = 110.0;
TEMPLE_MAX    = 145.0;
TEMPLE_W      = 7.0;
TEMPLE_H      = 4.5;
RAIL_W        = 3.0;
RAIL_H        = 2.5;
RATCHET_PITCH = 3.0;
RATCHET_TEETH = 10;
RATCHET_ANGLE = 25;
LOCK_SPRING_T = 0.9;
LOCK_SPRING_L = 12.0;

HINGE_D       = 5.0;
HINGE_L       = 8.0;
HINGE_PIN_D   = 2.2;

NOSEPAD_L     = 8.0;
NOSEPAD_W     = 5.0;
NOSEPAD_T     = 1.5;

SHOW_LEFT_RIM    = true;
SHOW_RIGHT_RIM   = true;
SHOW_BRIDGE      = true;
SHOW_LEFT_TEMPLE = true;
SHOW_RIGHT_TEMPLE= true;
SHOW_LENS_GHOST  = false;

EXPLODE = 0;

module lens_rim() {
    difference() {
        union() {

            difference() {
                cylinder(h=RIM_H, d=LENS_D + 2*RIM_W);
                translate([0,0,-0.1])
                    cylinder(h=RIM_H+0.2, d=LENS_D + 0.4);
            }

            translate([0,0,RIM_H - LIP_H])
                difference() {
                    cylinder(h=LIP_H, d=LENS_D + 0.4);
                    translate([0,0,-0.1])
                        cylinder(h=LIP_H+0.2, d=LENS_D - 2*LIP_W);
                }
        }

        translate([-2, -(LENS_R+RIM_W+0.1), 0])
            cube([4, RIM_W+0.2, 3.5]);

        translate([LENS_R-1, -BRIDGE_W/2, RIM_H/2 - BRIDGE_H/2])
            cube([RIM_W+2, BRIDGE_W, BRIDGE_H]);
    }

    translate([-(LENS_R + RIM_W), -HINGE_L/2, RIM_H/2 - HINGE_D/2])
        hinge_lug(female=true);

    translate([LENS_R, -NOSEPAD_W/2, -NOSEPAD_L])
        nose_pad_arm();
}

module bridge_body(length) {

    cube([length, BRIDGE_W, BRIDGE_H], center=true);
}

module bridge_slider_track() {

    difference() {
        bridge_body(BRIDGE_MAX + 10);

        translate([0, 0, BRIDGE_H/2 - SLIDE_SLOT_D])
            cube([BRIDGE_MAX + 12, SLIDE_SLOT_W, SLIDE_SLOT_D + 0.1], center=true);

        for (i = [-4:4]) {
            translate([i * 2.5, BRIDGE_W/2 + 0.1, 0])
                rotate([90, 0, 0])
                    cylinder(h=BRIDGE_W + 0.2, d=THUMB_SCREW_D, center=true);
        }

        translate([(BRIDGE_MAX)/2, 0, 0])
            cylinder(h=BRIDGE_H+0.2, d=2, center=true);
        translate([-(BRIDGE_MAX)/2, 0, 0])
            cylinder(h=BRIDGE_H+0.2, d=2, center=true);
    }

    for (i=[0:2]) {
        x_pos = -5 + i*5;
        translate([x_pos, BRIDGE_W/2 - 0.3, BRIDGE_H/2 - 1])
            rotate([90,0,0])
                linear_extrude(0.5)
                    text(str(14+i*5,"mm"), size=1.8, halign="center");
    }
}

module ratchet_tooth(width) {

    linear_extrude(height=width)
        polygon([
            [0, 0],
            [RATCHET_PITCH * 0.7, RATCHET_H],
            [RATCHET_PITCH, RATCHET_H],
            [RATCHET_PITCH, 0]
        ]);
}

RATCHET_H = 1.2;

module temple_rail() {

    rail_length = TEMPLE_MAX - TEMPLE_MIN + 20;
    difference() {

        cube([rail_length, RAIL_W, RAIL_H]);

        for (i=[0:3])
            translate([5+i*15, 0.5, -0.1])
                cube([8, RAIL_W-1, RAIL_H-0.8]);
    }

    for (i=[0:RATCHET_TEETH-1])
        translate([i*RATCHET_PITCH + 4, 0, RAIL_H])
            ratchet_tooth(RAIL_W);

    translate([rail_length - 2, 0, 0])
        cube([2, RAIL_W+2, RAIL_H+1]);
}

module temple_outer_sleeve() {

    sleeve_length = TEMPLE_MIN;
    difference() {

        cube([sleeve_length, TEMPLE_W, TEMPLE_H]);

        translate([5, (TEMPLE_W-RAIL_W)/2 - 0.2, (TEMPLE_H-RAIL_H)/2 - 0.2])
            cube([sleeve_length - 6, RAIL_W + 0.4, RAIL_H + 0.4]);

        translate([sleeve_length - LOCK_SPRING_L - 3,
                   (TEMPLE_W-RAIL_W)/2 - 0.1,
                   TEMPLE_H - LOCK_SPRING_T - 0.8])
            cube([LOCK_SPRING_L + 0.5, RAIL_W + 0.2, LOCK_SPRING_T + 1]);

        translate([-0.1, TEMPLE_W/2, TEMPLE_H/2])
            rotate([0,90,0])
                cylinder(h=6, d=HINGE_PIN_D + 0.3);
    }

    translate([sleeve_length - LOCK_SPRING_L - 2,
               (TEMPLE_W-RAIL_W)/2,
               TEMPLE_H - LOCK_SPRING_T])
        difference() {
            cube([LOCK_SPRING_L, RAIL_W, LOCK_SPRING_T]);

            translate([0, 0, 0])
                rotate([0, -3, 0])
                    cube([LOCK_SPRING_L, RAIL_W, LOCK_SPRING_T * 2]);
        }

    translate([sleeve_length - 4,
               (TEMPLE_W-RAIL_W)/2,
               TEMPLE_H - LOCK_SPRING_T])
        cube([3, RAIL_W, RATCHET_H + 0.5]);

    translate([sleeve_length - LOCK_SPRING_L,
               TEMPLE_W/2,
               TEMPLE_H - 0.3])
        rotate([0,0,0])
            linear_extrude(0.5)
                text("PUSH", size=2.2, halign="center", valign="center");

    for (i=[0:3]) {
        translate([10+i*8, TEMPLE_W - 0.3, TEMPLE_H/2])
            rotate([90,0,0])
                linear_extrude(0.4)
                    text(str(110+i*12,"mm"), size=1.6, halign="center");
    }
}

module hinge_lug(female=false) {
    if (female) {
        difference() {
            cube([HINGE_L, HINGE_D + 2, HINGE_D + 2]);
            translate([HINGE_L/4, (HINGE_D+2)/2, (HINGE_D+2)/2])
                rotate([0,90,0])
                    cylinder(h=HINGE_L/2+0.1, d=HINGE_D);
            translate([-0.1, (HINGE_D+2)/2, (HINGE_D+2)/2])
                rotate([0,90,0])
                    cylinder(h=HINGE_L+0.2, d=HINGE_PIN_D+0.3);
        }
    } else {
        difference() {
            cube([HINGE_L, HINGE_D + 2, HINGE_D + 2]);
            translate([-0.1, (HINGE_D+2)/2, (HINGE_D+2)/2])
                rotate([0,90,0])
                    cylinder(h=HINGE_L+0.2, d=HINGE_PIN_D+0.3);
        }
    }
}

module nose_pad_arm() {
    difference() {
        cube([NOSEPAD_W, NOSEPAD_T*3, NOSEPAD_L]);

        translate([NOSEPAD_W/2, -0.1, NOSEPAD_L/2])
            rotate([-90,0,0])
                cylinder(h=NOSEPAD_T*3+0.2, r=NOSEPAD_W*0.6);
    }

    translate([-1, 0, NOSEPAD_L - 3])
        cube([NOSEPAD_W+2, NOSEPAD_T*2, 3]);
}

module lens_ghost() {
    color("lightblue", 0.25)
        cylinder(h=2.5, d=LENS_D);
}

bridge_center_x = 0;
rim_offset_x    = BRIDGE_MIN/2 + LENS_R + RIM_W;

ex = EXPLODE * 15;

if (SHOW_RIGHT_RIM) {
    translate([rim_offset_x + ex, 0, 0])
        color("SteelBlue")
            lens_rim();
}

if (SHOW_LEFT_RIM) {
    translate([-(rim_offset_x + ex), 0, 0])
        mirror([1,0,0])
            color("SteelBlue")
                lens_rim();
}

if (SHOW_BRIDGE) {
    translate([0, LENS_R*0.3, RIM_H/2])
        color("DodgerBlue")
            bridge_slider_track();
}

if (SHOW_RIGHT_TEMPLE) {
    translate([rim_offset_x + 2*RIM_W + ex*2, -TEMPLE_W/2, RIM_H/2 - TEMPLE_H/2])
        color("SteelBlue") {
            temple_outer_sleeve();
            translate([TEMPLE_MIN - 5, (TEMPLE_W-RAIL_W)/2, (TEMPLE_H-RAIL_H)/2])
                color("LightSteelBlue")
                    temple_rail();
        }
}

if (SHOW_LEFT_TEMPLE) {
    translate([-(rim_offset_x + 2*RIM_W + ex*2), -TEMPLE_W/2, RIM_H/2 - TEMPLE_H/2])
        mirror([1,0,0])
            color("SteelBlue") {
                temple_outer_sleeve();
                translate([TEMPLE_MIN - 5, (TEMPLE_W-RAIL_W)/2, (TEMPLE_H-RAIL_H)/2])
                    color("LightSteelBlue")
                        temple_rail();
            }
}

if (SHOW_LENS_GHOST) {
    translate([rim_offset_x + ex, 0, RIM_H - 2.5 - LIP_H])
        lens_ghost();
    translate([-(rim_offset_x + ex), 0, RIM_H - 2.5 - LIP_H])
        lens_ghost();
}
