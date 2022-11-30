<?php

// do a pulse height coincidence run;
// form handler for ph_coincidence.inc:ph_coincidence_form()

require_once("panoseti.inc");

function main() {
    $username = get_login();
    $run = get_str('run');
    $vol = get_str('vol');
    $max_time_diff = (double)get_str('max_time_diff');
    $threshold_max_adc = (double)get_str('threshold_max_adc');
    $modules = get_str('modules');
    $all_modules = get_str('all_modules', true)?1:0;

    if (!$all_modules && strlen($modules)==0) {
        error_page("no modules specified");
    }
    $cmd = sprintf(
        './ph_coincidence.py --vol %s --run %s --max_time_diff %f --threshold_max_adc %f',
        $vol, $run, $max_time_diff, $threshold_max_adc
    );
    if ($username) {
        $cmd .= " --username $username ";
    }
    if ($all_modules) {
        $cmd .= " --modules all_modules";
    } elseif (strlen($modules)) {
        $cmd .= " --modules $modules";
    }
    page_head("Pulse height coincidence analysis run");
    echo "Command: $cmd
        <p>Output:
        <p>
        <pre>
    ";
    system($cmd, $retval);
    echo "</pre>";
    if ($retval) {
        echo "$cmd returned $retval";
        return;
    }
    echo "<p>
        <a href=analysis_type.php?type=ph_coincidence&vol=$vol&run=$run>
            Pulse height coincidence analysis of this run
        </a>
        </p>
    ";
    page_tail();
}

main();

?>
