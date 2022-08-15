#! /usr/bin/env php

<?php

require_once("panoseti.inc");

function add_account($name, $passwd) {
    $passwd_path = '/home/panosetigraph/web/passwd.json';
    $passwds = json_decode(file_get_contents($passwd_path));
    if (isset($passwds->$name)) {
        die("An account with that name already exists\n");
    }
    $u = new stdClass;
    $u->auth =  bin2hex(random_bytes(16));
    $u->passwd_hash = hash_passwd($passwd);
    $passwds->$name = $u;
    file_put_contents($passwd_path, json_encode($passwds, JSON_PRETTY_PRINT));

    echo "Added account with name '$name' and passwd '$passwd'\n";
}

$stdin = fopen('php://stdin', 'r');
echo "Create a PanoSETI account\n";
echo "User name: ";
$name = trim(fgets($stdin));
echo "Password: ";
$passwd = trim(fgets($stdin));

add_account($name, $passwd);

?>
