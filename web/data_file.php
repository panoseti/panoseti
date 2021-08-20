<?php

// show links to whatever we have for a file
//

require_once("panoseti.inc");

function main($name) {
    page_head("File: $name");

    $dir = "pulse_out/$name";
    if (!is_dir($dir)) {
        echo "No pulse info available\n";
        return;
    }
    echo "<h2>Pulse info</h2>\n";
    foreach (scandir($dir) as $module) {
        if ($module[0] == ".") continue;
        echo "<h3>Module $module</h3>";
        foreach (scandir("$dir/$module") as $pixel) {
            if ($pixel[0] == ".") continue;
            $url = "pulse.php?file=$name&module=$module&pixel=$pixel";
            echo "<p><a href=$url>pixel $pixel</a>\n";
        }
    }
    page_tail();
}

$name = $_GET['name'];
main($name);

?>
