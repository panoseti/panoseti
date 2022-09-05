<?php

ini_set('display_errors', 1);

// show a PFF file as a grayscale image,
// with buttons for moving forward or back in time

require_once("panoseti.inc");

function arrows_str($run, $fname, $frame) {
    $url = "image.php?run=$run&fname=$fname&frame=";
    return sprintf(
        '<a class="btn btn-sm btn-primary" href=%s%d><< min</a>
        <a class="btn btn-sm btn-primary" href=%s%d><< sec</a>
        <a class="btn btn-sm btn-primary" href=%s%d><< frame</a>
        <a class="btn btn-sm btn-primary" href=%s%d> frame >></a>
        <a class="btn btn-sm btn-primary" href=%s%d> sec >></a>
        <a class="btn btn-sm btn-primary" href=%s%d> min >></a>',
        $url, $frame - 200*60,
        $url, $frame - 200,
        $url, $frame - 1,
        $url, $frame + 1,
        $url, $frame + 200,
        $url, $frame + 200*60
    );
}

function show_frame($data, $arrows) {
    echo "<table>";
    for ($i=0; $i<32; $i++) {
        echo "<tr>";
        for ($j=0; $j<32; $j++) {
            $v = $data[$i*32+$j];
            $v  >>= 8;
            if ($v > 255) $v = 255;
            $color = sprintf("#%02x%02x%02x", $v, $v, $v);
            echo sprintf(
                '<td width=%dpx height=%dpx bgcolor="%s"> </td>',
                16, 16, $color
            );
        }
        echo "</tr>\n";
    }
    echo "<tr><td colspan=32 align=center><br>$arrows</td></tr>\n";
    echo "</table>";
}

function get_frame($file, $frame) {
    $f = fopen($file, "r");
    if (fseek($f, $frame*1024*2) < 0) {
        die("no such frame");
    }
    $x = fread($f, 1024*2);
    if (strlen($x)==0 ) die("no such frame");
    $y = array();
    $y = array_merge(unpack("S1024", $x));
        // unpack returns 1-offset array - BOOOOOOO!!!!!!
    if (!$y) {
        die("unpack");
    }
    return $y;
}

function rand_frame() {
    $x = array();
    for ($i=0; $i<1024; $i++) {
        $x[] = rand(0, 255);
    }
}

function main($run, $fname, $frame) {
    page_head("Image");
    echo "<p>Run: <a href=run.php?name=$run>$run</a>\n";
    echo "<p>File: <a href=file.php?run=$run&fname=$fname>$fname</a>\n";
    $path = ANALYSIS_ROOT."/$run/$fname/images.bin";
    $t = $frame/200.;
    echo "<p>Frame: $frame ($t sec)\n";
    $x = get_frame($path, $frame);
    $as = arrows_str($run, $fname, $frame);
    show_frame($x, $as);
    page_tail();
}

$run = get_str("run");
$fname = get_str("fname");
check_filename($fname);
check_filename($run);
$frame = get_int("frame");

main($run, $fname, $frame);

?>
