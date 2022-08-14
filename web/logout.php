<?php

require_once("panoseti.inc");

function main() {
    setcookie('auth', '', time()-3600, '/');
    setcookie('name', '', time()-3600, '/');
    //setcookie('redirect_to', '', time()-3600, '/');
    Header("Location: index.php");
}

main();

?>
