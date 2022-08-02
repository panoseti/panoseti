<?php

ini_set('display_errors', 1);

// top-level page: show list of data files,
// with link to per-file pages

require_once("panoseti.inc");

function main() {
    page_head("PanoSETI");
    echo "
        <h2>Observing runs</h2>
        <p>
        Browse PanoSETI data and data analysis products.
    ";
    foreach (scandir("data") as $f) {
        if (!strstr($f, '.pffd')) continue;
        echo "<p><a href=run.php?name=$f>$f</a>\n";
    }
    echo "
        <h2>Graphical parameter logs</h2>
        <p>
        <a href=http://visigoth.ucolick.org:3000>View</a>
    ";

    page_tail();
}

main();

?>
