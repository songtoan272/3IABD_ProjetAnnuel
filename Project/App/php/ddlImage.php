<?php

session_start();

include_once('functions.php');

$id = 'error';
$uploaded = false;

if($_FILES['fileToUpload']['name'] == "__BB__.jpg"){
    $_SESSION['easter_egg'] = "BB";
} else {
    $_SESSION['easter_egg'] = "not";
}

if($_FILES['fileToUpload']['name'] != ""){
    $extension = getExtension($_FILES['fileToUpload']['name']);
    if(isImage($extension)){
        $id = generateRandomString() . '.' . $extension;
        $saveto = "../images/ia/" . $id;
        if(!move_uploaded_file($_FILES['fileToUpload']["tmp_name"], $saveto)){
            $id = 'error';
            $uploaded = false;
        } else {
            $uploaded = true;
        }
    }
}

if($_POST['urlImage'] != "" && $uploaded == false){
    $url = strtolower($_POST['urlImage']);

    
    $ch = curl_init ($url);
    
    curl_setopt($ch, CURLOPT_HEADER, 0);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
    curl_setopt($ch, CURLOPT_BINARYTRANSFER,1);
    $raw=curl_exec($ch);
    curl_close ($ch);

    $extension = getExtension(curl_getinfo($ch, CURLINFO_CONTENT_TYPE));

    if($extension == ''){
        $extension = 'jpg';
    }

    $id = generateRandomString() . '.' . $extension;
    $saveto = "../images/ia/" . $id;

    if(file_exists($saveto)){
        unlink($saveto);
    }
    $fp = fopen($saveto,'x');
    fwrite($fp, $raw);
    fclose($fp);
}

$model = getModel($_POST['model']);
$img_resized = resize_image($saveto, $model['img_width_model'], $model['img_height_model'], $model['img_conserve_ratio']);

predict_image($model['id_model'], $model['type_model'], $model['path_model'], $img_resized, $model['result_classes']);

$_SESSION['id'] = $id;

header('Location: http://92.222.76.60/app.php');
die();

?>