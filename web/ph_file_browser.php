<?php

// show a pulse-height PFF file as a grayscale image,
// with buttons for moving forward or backwards in the file.

require_once("panoseti.inc");
require_once("analysis.inc");

function truemod($num, $mod) {
    return ($mod + ($num % $mod)) % $mod;
}

function arrows_str($vol, $run, $file, $frame) {
    $url = "ph_file_browser.php?vol=$vol&run=$run&file=$file&frame=";
    return sprintf(
        '<a class="btn btn-sm btn-primary" href=%s%d><< 100 </a>
        <a class="btn btn-sm btn-primary" href=%s%d><< 10 </a>
        <a class="btn btn-sm btn-primary" href=%s%d><< 5 </a>
        <a class="btn btn-sm btn-primary" href=%s%d><< 1 </a>
        <a class="btn btn-sm btn-primary" href=%s%d> 1 >></a>
        <a class="btn btn-sm btn-primary" href=%s%d> 5 >></a>
        <a class="btn btn-sm btn-primary" href=%s%d> 10 >></a>
        <a class="btn btn-sm btn-primary" href=%s%d> 100 >></a>',
        $url, $frame - 100,
        $url, $frame - 10,
        $url, $frame - 5,
        $url, $frame - 1,
        $url, $frame + 1,
        $url, $frame + 5,
        $url, $frame + 10,
        $url, $frame + 100
    );
}

function show_frame_ph1024($data, $arrows, $bytes_pix) {
    echo "<table>";
    for ($i=0; $i<32; $i++) {
        echo "<tr>";
        for ($j=0; $j<32; $j++) {
            $v = $data[$i*32+$j];
            if ($bytes_pix == 2) {
                $v  >>= 8;
            }
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

function get_frame($f, $hs, $frame, $bytes_pix) {
    $frame_size = $hs + 1024*$bytes_pix;
    $offset = $frame*$frame_size + $hs;
    if (fseek($f, $offset) < 0) {
        die("no such frame");
    }
    $x = fread($f, 1024*$bytes_pix);
    if (strlen($x)==0 ) die("no such frame");
 
    // unpack returns 1-offset array - WTF???
    // array_merge() changes it to 0-offset
    //
    if ($bytes_pix == 1) {
        $y = array_merge(unpack("C1024", $x));
    } else {
        $y = array_merge(unpack("S1024", $x));
    }
    if (!$y) {
        die("unpack");
    }
    return $y;
}

function main($vol, $run, $file, $frame) {
    $p = parse_pff_name($file);
    $bytes_pix = intval($p['bpp']);
    page_head("Pulse-Height");
    echo "<p>Run: <a href=run.php?vol=$vol&name=$run>$run</a>\n";
    echo "<p>File: <a href=file.php?vol=$vol&run=$run&fname=$file>$file</a>\n";
    $path = "$vol/data/$run/$file";
    $f = fopen($path, "r");
    $hs = header_size($f);
    echo "<p>Frame: $frame\n";
    $x = get_frame($f, $hs, $frame, $bytes_pix);
    $as = arrows_str($vol, $run, $file, $frame);
    show_frame($x, $as, $bytes_pix);
    page_tail();
}

$vol = get_filename("vol");
$run = get_filename("run");
$file = get_filename("file");
$frame = get_int("frame");

main($vol, $run, $file, $frame);

?>
