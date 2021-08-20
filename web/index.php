<?php

// top-level page: show list of data files,
// with link to per-file pages

require_once("panoseti.inc");

function main() {
    page_head("PanoSETI data files");
    foreach (scandir("PANOSETI_DATA") as $f) {
        if ($f[0] == ".") continue;
        echo "<p><a href=data_file.php?name=$f>$f</a>\n";
    }
    page_tail();
}

main();

?>
