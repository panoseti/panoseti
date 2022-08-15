<?php

require_once("panoseti.inc");

function main() {
    setcookie('auth', '', time()-3600, '/');
    setcookie('name', '', time()-3600, '/');
    page_head("Logged out", LOGIN_NONE);
    echo "Click to <a href=login.php>log in</a>.";
    page_tail();
}

main();

?>
