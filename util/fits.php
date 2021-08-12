<?php

// PHP code showing (in principle) how to parse a FITS file

function main() {
    $f = fopen("foo.fits", "r");
    for ($i=0; $i<36; $i++) {
        $s = fread($f, 80);
        echo "$s\n";

    }
    $x = fread($f, 32);
    $y = unpack("n16", $x);
    print_r($y);
}

main();

?>
