<?php

// top-level page: show list of data files,
// with link to per-file pages

function main() {
    echo "<h2>PanoSETI data files</h2>\n";
    foreach (scandir("PANOSETI_DATA") as $f) {
        if ($f[0] == ".") continue;
        echo "<a href=data_file.php?name=$f>$f</a><p>\n";
    }
}

main();

?>
