<?php

    include_once('functions.php');

    $fullpathscript = realpath("../python/model.py");

    $command = escapeshellcmd('python3 ' . $fullpathscript . ' "' . $_GET['model_type'] . '" "' . $_GET['model_path'] .'" "' . $_GET['img_path'] . '"');
    $output = shell_exec($command);

    $result = explode(";", $_GET['result']);

    $char = getCharacterByResult($result[intval($output)], $_GET['model_id']);

    $r = array('name' => $char['name_characters'], 'path' => $char['path_characters'], 'debug' => $output);
    echo json_encode($r);
?>