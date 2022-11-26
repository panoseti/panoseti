<?php

// show a PFF file as a grayscale image,
// with buttons for moving forward or back in time

require_once("panoseti.inc");
require_once("analysis.inc");

function arrows_str($vol, $run, $file, $usecs, $frame) {
    $url = "image.php?vol=$vol&run=$run&file=$file&frame=";
    $fps = 1e6/$usecs;
    return sprintf(
        '<a class="btn btn-sm btn-primary" href=%s%d><< min</a>
        <a class="btn btn-sm btn-primary" href=%s%d><< sec</a>
        <a class="btn btn-sm btn-primary" href=%s%d><< frame</a>
        <a class="btn btn-sm btn-primary" href=%s%d> frame >></a>
        <a class="btn btn-sm btn-primary" href=%s%d> sec >></a>
        <a class="btn btn-sm btn-primary" href=%s%d> min >></a>',
        $url, $frame - (int)($fps*60),
        $url, $frame - (int)($fps),
        $url, $frame - 1,
        $url, $frame + 1,
        $url, $frame + (int)($fps),
        $url, $frame + (int)($fps*60)
    );
}

function show_frame($data, $arrows, $bytes_pix) {
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
    $dc = json_decode(file_get_contents("$vol/data/$run/data_config.json"));
    $usecs = $dc->image->integration_time_usec;
    $bytes_pix = $dc->image->quabo_sample_size/8;
    page_head("Image");
    echo "<p>Run: <a href=run.php?vol=$vol&name=$run>$run</a>\n";
    echo "<p>File: <a href=file.php?vol=$vol&run=$run&fname=$file>$file</a>\n";
    $path = "$vol/data/$run/$file";
    $f = fopen($path, "r");
    $hs = header_size($f);
    $t = $frame/200.;
    echo "<p>Frame: $frame ($t sec)\n";
    $x = get_frame($f, $hs, $frame, $bytes_pix);
    $as = arrows_str($vol, $run, $file, $usecs, $frame);
    show_frame($x, $as, $bytes_pix);
    page_tail();
}

$vol = get_str("vol");
$run = get_str("run");
$file = get_str("file");
check_filename($vol);
check_filename($run);
check_filename($file);
$frame = get_int("frame");

main($vol, $run, $file, $frame);

?>
