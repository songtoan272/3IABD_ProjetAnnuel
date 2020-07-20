<?php

function generateRandomString($length = 10) {
    $characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_';
    $charactersLength = strlen($characters);
    $randomString = '';
    for ($i = 0; $i < $length; $i++) {
        $randomString .= $characters[rand(0, $charactersLength - 1)];
    }
    return $randomString;
}

function getExtension($ext){
    if(!strrpos($ext, '/', 0)){
        $ext = str_replace(' ', '_', $ext);
        $path = pathinfo($ext);
        return $path['extension'];
    } else {
        return substr($ext, strrpos($ext, '/', 0) + 1, strlen($ext) - strrpos($ext, '/', 0) - 1);
    }
}

function isImage($ext){
    if($ext == 'png' || $ext == 'jpg' || $ext == 'jpeg'){
        return true;
    } else {
        return false;
    }
}

function openBDD(){
    $hote = 'localhost';
	$log = 'root';
	$mdp = 'root';
    $name = 'characters';
	try
	{
        $bdd = new mysqli($hote, $log, $mdp, $name);
        $bdd->set_charset('utf8');
        return $bdd;
	}
	catch (Exception $e)
	{
        die('Erreur : ' . $e->getMessage());
	}
}

function getCharacters($stringid){
    $bdd = openBDD();
	$cursor = mysqli_query($bdd, 'SELECT * FROM characters WHERE stringid_characters = "' . $stringid . '"');
    $bdd->close();
		
	return $cursor->fetch_array();
}

function getModels(){
    $bdd = openBDD();
    $cursor = mysqli_query($bdd, 'SELECT * FROM model');
    $bdd->close();
		
	return $cursor;
}

function getModel($id){
    $bdd = openBDD();
    $cursor = mysqli_query($bdd, 'SELECT * FROM model WHERE id_model = ' . $id);
    $bdd->close();
		
	return $cursor->fetch_array();
}

function getCharacterByResult($result, $model_id){
    $bdd = openBDD();

    $perso = mysqli_query($bdd, 'SELECT * FROM characters WHERE id_characters = ' . $result);

    $bdd->close();

	return $perso->fetch_array();
}

function resize_image($path, $width, $height, $ratio) {

    $fullpathscript = realpath("../python/resize_image.py");
    $fullpathfile = realpath($path);
    
    $command = escapeshellcmd('python3 ' . $fullpathscript . ' "' . $fullpathfile . '" ' . $width . ' ' . $height .' ' . $ratio);
    $output = shell_exec($command);
    return $output;
}

function predict_image($model_id, $model_type, $model_path, $img_path, $result_classes){

    $_SESSION['model_type'] = trim($model_type);
    $_SESSION['model_path'] = trim(realpath($model_path));
    $_SESSION['img_path'] = trim($img_path);
    $_SESSION['model_id'] = trim($model_id);
    $_SESSION['result_classes'] = trim($result_classes);
}

function getGif($dir){
    $files = scandir(realpath($dir));
    return $dir . $files[rand(2, count($files) - 1)];
}

function getPathMetrics($id_model){
    $bdd = openBDD();

    $metrics = mysqli_query($bdd, 'SELECT path_metric, path_chart FROM model WHERE id_model = ' . $id_model);

    $bdd->close();

    return $metrics->fetch_array();
}

function GuessExtension($path)
{
    $type = GetMimType($path);

    $extension = "";
    switch($type)
    {
       case "image/png":
           $extension = ".png";
           break;
        case "image/jpeg":
        default:
           $extension = ".jpg";
           break;
    }

    return $extension;
}

?>