<?php

ini_set('display_errors', 1);

// show links to whatever we have for a file
//

require_once("panoseti.inc");

function movie_form($run, $fname) {
    return sprintf("
        <form action=make_movie.php method=post>
        <input type=hidden name=run value=%s>
        <input type=hidden name=fname value=%s>
            min value: <input name=min size=6> max value: <input name=max size=6>
            #frames: <input name=nframes size=6>
            <input type=submit value=OK>
        <form>
        ",
        $run, $fname
    );
}

function movie_links($run, $fname) {
    $x = [];
    foreach (scandir("derived/$run/$fname") as $f) {
        if (strstr($f, '.mp4')) {
            $x[] = "<a href=derived/$run/$fname/$f>$f</a>";
        }
    }
    return implode('<br>', $x);
}

function do_pff($run, $fname) {
    page_head("PanoSETI derived data");

    echo "<font size=+1>";

    $dir = "derived/$run/$fname";
    if (!is_dir($dir)) {
        echo "<p>No data available - may need to run analysis scripts.\n";
        page_tail();
        return;
    }
    start_table();
    row2("Observing run", $run);
    row2("File", $fname);
    row2("Pixel value histogram", "<a href=derived/$run/$fname/pixel_histogram.dat>View</a>");
    row2("Frame browser",
        "<a href=image.php?run=$run&fname=$fname&frame=0>View</a>"
    );
    row2("Movies", movie_links($run, $fname));
    row2("Make new movie", movie_form($run, $fname));
    $x = 'Pixels: ';
    foreach (scandir("derived/$run/$fname") as $pixel) {
        if ($pixel[0] == ".") continue;
        if (!is_numeric($pixel)) continue;
        $url = "pulse.php?file=$run/$fname&pixel=$pixel";
        $x .= "&nbsp;&nbsp <a href=$url>$pixel</a>\n";
    }
    row2("Pulse info", $x);
    end_table();
    page_tail();
}

$run = get_str('run');
$fname = get_str('fname');
check_filename($fname);
check_filename($run);
do_pff($run, $fname);

?>
