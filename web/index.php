<?php

// top-level page: show list of data files,
// with link to per-file pages

require_once("panoseti.inc");

function main() {
    page_head("PanoSETI science portal");
    echo "
        <p>
        This site is intended to provide a way to access
        PanoSETI data and the products of data analysis.
        <p>
        Other PanoSETI web resources:
        <ul>
        <li> <a href=https://oirlab.ucsd.edu/PANOSETI.html>Public site</a>
        <li> <a href=https://github.com/panoseti>Github</a>
        </ul>
        <h2>Data files</h2>
    ";
    foreach (scandir("PANOSETI_DATA") as $f) {
        if (!strstr($f, '.pff')) continue;
        echo "<p><a href=data_file.php?name=$f>$f</a>\n";
    }
    page_tail();
}

main();

?>
