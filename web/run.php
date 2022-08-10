<?php

// main page for an observing run.
// show links to the component files,
// and show comments and tags

ini_set('display_errors', 1);

require_once("panoseti.inc");

function add_tag($run) {
    $t = @file_get_contents("data/$run/tags.json");
    if ($t) {
        $tags = json_decode($t);
    } else {
        $tags = [];
    }
    $tag = new stdClass;
    $tag->who = post_str('who');
    $tag->when = time();
    $tag->tag = post_str('tag');
    if (!$tag->who) {
        die("must give your name");
    }
    if (!$tag->tag) {
        die("must give a tag");
    }
    $tags[] = $tag;
    file_put_contents("data/$run/tags.json", json_encode($tags));
    header("Location: run.php?name=$run");
}

function tags_form($run) {
    echo "
        <p>
        Add tag:
        <p>
        <form method=post action=run.php>
        <input type=hidden name=name value=$run>
        Tag: <input name=tag>
        <p><p>
        Your name: <input name=who>
        <p>
        <input type=submit name=add_tag value=OK>
        </form>
    ";
}

function show_tags($run) {
    $t = @file_get_contents("data/$run/tags.json");
    if ($t) {
        $tags = json_decode($t);
    } else {
        $tags = [];
    }
    if ($tags) {
        start_table();
        table_header("Who", "When", "Tag");
        foreach ($tags as $tag) {
            table_row($tag->who, date_str($tag->when), $tag->tag);
        }
        end_table();
    } else {
        echo "no tags";
    }
}

function add_comment($run) {
    $t = @file_get_contents("data/$run/comments.json");
    if ($t) {
        $comments = json_decode($t);
    } else {
        $comments = [];
    }
    $c = new stdClass;
    $c->who = post_str('who');
    $c->when = time();
    $c->comment = post_str('comment');
    if (!$c->who) {
        die("must give your name");
    }
    if (!$c->comment) {
        die("must give a comment");
    }
    $comments[] = $c;
    file_put_contents("data/$run/comments.json", json_encode($comments));
    header("Location: run.php?name=$run");
}

function comments_form($run) {
    echo "
        <p>
        Add comment:
        <p>
        <form method=post action=run.php>
        <input type=hidden name=name value=$run>
        <textarea name=comment rows4 cols=40></textarea>
        <p>
        Your name: <input name=who>
        <p>
        <input type=submit name=add_comment value=OK>
        </form>
    ";
}

function show_comments($run) {
    $t = @file_get_contents("data/$run/comments.json");
    if ($t) {
        $comments = json_decode($t);
    } else {
        $comments = [];
    }
    if ($comments) {
        start_table();
        table_header("Who", "When", "Comment");
        foreach ($comments as $comment) {
            table_row($comment->who, date_str($comment->when), $comment->comment);
        }
        end_table();
    } else {
        echo "no comments";
    }
}

function main($name) {
    page_head("Observing run: $name");

    $dir = "data/$name";

    echo "<h2>Data files</h2>";
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        if (!is_pff($f)) continue;
        $n = filesize("data/$name/$f");
        if (!$n) continue;
        $n = number_format($n/1e6, 2);
        echo "
            <br>
            <a href=file.php?run=$name&fname=$f>$f</a> ($n MB)
        ";
    }
    echo "<h2>Ancillary files</h2>";
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        if (is_pff($f)) continue;
        if (in_array($f, ['comments.json', 'tags.json'])) continue;
        echo "<br>
            <a href=data/$name/$f>$f</a>
        ";
    }

    echo "<h2>Comments</h2>";
    show_comments($name);
    comments_form($name);

    echo "<h2>Tags</h2>";
    show_tags($name);
    tags_form($name);
    page_tail();
}

$name = get_str('name');
if (!$name) $name = post_str('name');
check_filename($name);

if (post_str('add_comment')) {
    add_comment($name);
} else if (post_str('add_tag')) {
    add_tag($name);
} else {
    main($name);
}

?>
