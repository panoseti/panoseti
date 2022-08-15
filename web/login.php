<?php

require_once("panoseti.inc");

function login_form() {
    page_head("Log in", LOGIN_NONE);
    echo "
        <form method=post action=login.php>
        Name: <input name=name>
        <p><p>
        Password: <input type=passwd name=passwd>
        <p>
        <input name=submit type=submit value=OK>
    ";
    page_tail();
}

function login_action() {
    $name = post_str('name');
    $passwd = post_str('passwd');
    $u = check_passwd_hash($name, hash_passwd($passwd));
    if ($u) {
        setcookie('auth', $u->auth, time()+1000000, '/');
        setcookie('name', $name, time()+1000000, '/');
        page_head("Logged in as $name", LOGIN_NONE);
        echo "Go to the <a href=index.php>PanoSETI home page</a>.";
        page_tail();
        return;
    }
    page_head("Login failed", LOGIN_NONE);
    page_tail();
}

$name = get_login(true);
if ($name) {
    Header("location: index.php");
    exit;
}
if (post_str('submit')) {
    login_action();
} else {
    login_form();
}

?>
