<?php

require_once("panoseti.inc");
require_once("ph_file_browser.php");
require_once("image.php");

function main($vol, $run, $fname) {
    $p = parse_pff_name($fname);
    page_head("PFF file $fname");

    echo "<font size=+1>";
    start_table();
    row2("Observing run", "<a href=run.php?vol=$vol&name=$run>$run</a>");
    row2("File", $fname);
    
    
    // check file type (img or ph) and link to correct frame browser.
    if (str_contains($p['dp'], "img")) {
        row2("Frame browser", "<a href=image.php?vol=$vol&run=$run&file=$fname&frame=0>View</a>");
    } else if (str_contains($p['dp'], "ph")) { 
        row2("Frame browser", "<a href=ph_file_browser.php?vol=$vol&run=$run&file=$fname&frame=0>View</a>");
    } else {
        sprintf("Invalid data product: %s", $p['dp']);
    }

    row2("Download",
        "<a href=$vol/data/$run/$fname>PFF</a> &middot; FITS"
    );
    end_table();
    page_tail();
}

$vol = get_filename('vol');
$run = get_filename('run');
$fname = get_filename('fname');
main($vol, $run, $fname);

?>
