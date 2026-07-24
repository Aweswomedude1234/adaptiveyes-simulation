TOOTH_PITCH    = 3.0;
TOOTH_DEPTH    = 1.2;
TOOTH_ANGLE    = 20;
RAMP_ANGLE     = 55;

RAIL_W         = 6.0;
RAIL_H         = 3.5;
RAIL_LEN       = 38.0;

OUTER_W        = 10.0;
OUTER_H        = 5.5;
OUTER_LEN      = 70.0;

INNER_LEN      = 80.0;
INNER_W        = RAIL_W - 0.4;
INNER_H        = RAIL_H - 0.3;

PAWL_W         = 4.5;
PAWL_H         = 5.0;
PAWL_THICKNESS = 1.4;
RELEASE_TAB_L  = 8.0;
RELEASE_TAB_H  = 3.0;

PIN_D          = 1.8;
WALL           = 1.6;

$fn = 40;

module single_tooth(h=RAIL_H) {

    polygon([
        [0,            0],
        [TOOTH_PITCH,  0],
        [TOOTH_PITCH,  TOOTH_DEPTH * 0.25],
        [TOOTH_PITCH * 0.35, TOOTH_DEPTH],
        [0,            TOOTH_DEPTH * 0.1]
    ]);
}

module toothed_rail() {
    difference() {
        cube([RAIL_LEN, RAIL_W, RAIL_H]);

        translate([RAIL_LEN - RAIL_H/2, RAIL_W/2, -0.1])
            cylinder(h=RAIL_H+0.2, r=RAIL_H/2);
    }

    n_teeth = floor(RAIL_LEN / TOOTH_PITCH) - 2;
    for (i = [1 : n_teeth]) {
        translate([i * TOOTH_PITCH, 0, RAIL_H])
        linear_extrude(height=RAIL_W, center=false)
        rotate([0,0,0])
        single_tooth();
    }
}

module inner_arm() {
    color("LightGray")
    union() {

        cube([INNER_LEN, RAIL_W, RAIL_H]);

        translate([8, 0, 0])
            toothed_rail();

        translate([INNER_LEN - 4, RAIL_W/2 - 2, -3])
            cube([4, 4, 3 + RAIL_H]);

        translate([INNER_LEN - 2, RAIL_W/2, -3 + 2])
            rotate([0, 90, 0])
            cylinder(h=5, d=PIN_D, center=true);
    }
}

module pawl() {
    color("DimGray")
    union() {

        cube([PAWL_THICKNESS, PAWL_W, PAWL_H]);

        translate([PAWL_THICKNESS, 0, PAWL_H - TOOTH_DEPTH - 0.5])
        union() {
            cube([TOOTH_DEPTH + 0.5, PAWL_W, TOOTH_DEPTH + 0.5]);
        }

        translate([PAWL_THICKNESS, 0, PAWL_H])
            cube([RELEASE_TAB_L, PAWL_W, RELEASE_TAB_H]);

    }
}

module outer_sleeve() {
    color("SlateGray")
    difference() {

        cube([OUTER_LEN, OUTER_W, OUTER_H]);

        translate([WALL, (OUTER_W - RAIL_W)/2, WALL])
            cube([OUTER_LEN - WALL, RAIL_W + 0.4, RAIL_H + 0.3]);

        translate([15, (OUTER_W - PAWL_W)/2 - 0.5, OUTER_H - WALL - 0.1])
            cube([OUTER_LEN - 20, PAWL_W + 1, WALL + 0.2]);

        translate([-0.1, OUTER_W/2 - 2, OUTER_H/2 - 1.5])
            cube([WALL + 0.2, 4, 3]);
    }

    translate([18, (OUTER_W - PAWL_W)/2, OUTER_H])
        pawl();

    translate([OUTER_LEN - WALL - 1, (OUTER_W - RAIL_W)/2, WALL])
        cube([WALL, RAIL_W + 0.4, 2]);
}

module assembled(extension_mm = 15) {

    outer_sleeve();

    translate([WALL + extension_mm, (OUTER_W - RAIL_W)/2, WALL])
        inner_arm();
}

module exploded() {

    outer_sleeve();

    translate([WALL, (OUTER_W - RAIL_W)/2, 22])
        inner_arm();

    color("Red", 0.5) {
        translate([0, -3, OUTER_H/2])
            cube([OUTER_LEN, 0.5, 0.5]);
        translate([WALL, -3, OUTER_H/2])
            cube([35, 0.5, 0.5]);
    }
}

assembled(15);
