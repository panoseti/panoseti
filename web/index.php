<?php

ini_set('display_errors', 1);

// top-level page: show list of data files,
// with link to per-file pages

require_once("panoseti.inc");

function main() {
    page_head("PanoSETI");
    echo "
        <p>
        Browse PanoSETI data and data analysis products.
        <p>
        <h2>Observing runs</h2>
    ";
    foreach (scandir("data") as $f) {
        if (!strstr($f, '.pffd')) continue;
        echo "<p><a href=run.php?name=$f>$f</a>\n";
    }
    page_tail();
}

main();

?>
