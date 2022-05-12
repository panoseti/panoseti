<?php

// show links to whatever we have for a file
//

require_once("panoseti.inc");

function do_h5($name) {
    page_head("File: $name");

    $dir = "pulse_out/$name";
    if (!is_dir($dir)) {
        echo "No info available\n";
        return;
    }
    foreach (scandir($dir) as $module) {
        if ($module[0] == ".") continue;
        echo "<h2>Dome $module</h3>";

        echo "<p><a href=image.php?file=$name&module=$module&frame=0>Images</a>";
        echo "<p>Pulse info: pixel ";
        foreach (scandir("$dir/$module") as $pixel) {
            if ($pixel[0] == ".") continue;
            if (!is_numeric($pixel)) continue;
            $url = "pulse.php?file=$name&module=$module&pixel=$pixel";
            echo "&nbsp;&nbsp <a href=$url>$pixel</a>\n";
        }
    }
    page_tail();
}

function do_pff($name) {
    page_head("File: $name");

    $dir = "pulse_out/$name";
    if (!is_dir($dir)) {
        echo "No info available\n";
        return;
    }
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        echo "<h2>File $f</h3>";

        echo "<p><a href=image.php?file=$name/$f&frame=0>Images</a>";
        echo "<p><a href=pulse_out/$name/$f/images.mp4>Movie</a>";
        echo "<p>Pulse info: pixel ";
        foreach (scandir("$dir/$f") as $pixel) {
            if ($pixel[0] == ".") continue;
            if (!is_numeric($pixel)) continue;
            $url = "pulse.php?file=$name/$f&pixel=$pixel";
            echo "&nbsp;&nbsp <a href=$url>$pixel</a>\n";
        }
    }
    page_tail();
}

$name = get_str('name');
check_filename($name);
if (strstr($name, 'pffd')) {
    do_pff($name);
} else {
    do_h5($name);
}

?>
