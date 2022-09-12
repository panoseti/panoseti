<?php

require_once("panoseti.inc");

function do_pff($run, $fname) {
    page_head("PFF file $fname");

    echo "<font size=+1>";

    start_table();
    row2("Observing run", $run);
    row2("File", $fname);
    row2("Download",
        "<a href=data/$run/$fname>PFF</a> &middot; FITS"
    );
    end_table();
    page_tail();
}

$run = get_str('run');
$fname = get_str('fname');
check_filename($fname);
check_filename($run);
do_pff($run, $fname);

?>
