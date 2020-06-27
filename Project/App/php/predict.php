<?php

    include_once('functions.php');

    $fullpathscript = realpath("../python/model.py");

    $command = escapeshellcmd('python3 ' . $fullpathscript . ' "' . $_GET['model_type'] . '" "' . $_GET['model_path'] .'" "' . $_GET['img_path'] . '"');
    $output = shell_exec($command);

    $char = getCharacterByResult($output, $_GET['model_path']);

    $r = array('name' => $char['name_characters'], 'path' => $char['path_characters'], 'debug' => $_SESSION['toprint']);
    echo json_encode($r);
?>