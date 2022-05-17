<?php

ini_set('display_errors', 1);

// show links to whatever we have for a file
//

require_once("panoseti.inc");

function is_pff($f) {
    if ($f == 'hk.pff') return false;
    return strstr($f, '.pff');
}

function main($name) {
    page_head("Observing run: $name");

    $dir = "data/$name";

    echo "<h2>Data files</h2>";
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        if (!is_pff($f)) continue;
        $n = filesize("data/$name/$f");
        if (!$n) continue;
        $n = number_format($n/1e6, 2);
        echo "
            <br>
            <a href=file.php?run=$name&fname=$f>$f</a> ($n MB)
        ";
    }
    echo "<h2>Ancillary files</h2>";
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        if (is_pff($f)) continue;
        echo "<br>
            <a href=data/$name/$f>$f</a>
        ";
    }
    page_tail();
}

$name = get_str('name');
check_filename($name);
main($name);

?>
