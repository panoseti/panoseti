<?php

// show links to whatever we have for a file
//

function show_pulse_info($name) {
}

function main($name) {
    echo "<h2>$name</h2>\n";

    $dir = "pulse_out/$name";
    if (is_dir($dir)) {
        echo "<h3>Pulse info</h3>\n";
        foreach (scandir($dir) as $f) {
            if ($f[0] == ".") continue;
            echo "<p><a href=pulse.php?file=$name&pixel=$f>pixel $f</a>\n";
        }
    } else {
        echo "No pulse info available\n";
    }
}

$name = $_GET['name'];
main($name);

?>
