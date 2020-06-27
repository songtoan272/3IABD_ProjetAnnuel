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

function getCharacterByResult($result, $model_path){
    $bdd = openBDD();
    $cursor = mysqli_query($bdd, 'SELECT * FROM result_char_model INNER JOIN characters ON result_char_model.id_characters_result_char_model = characters.id_characters INNER JOIN model ON model.id_model = result_char_model.id_model_result_char_model WHERE result_result_char_model = '.$result.' AND path_model = "'.$model_path.'"');
    $_SESSION['toprint'] = $result;
    $bdd->close();

	return $cursor->fetch_array();
}

function resize_image($path, $width, $height) {

    $fullpathscript = realpath("../python/resize_image.py");
    $fullpathfile = realpath($path);
    
    $command = escapeshellcmd('python3 ' . $fullpathscript . ' "' . $fullpathfile . '" ' . $width . ' ' . $height);
    $output = shell_exec($command);
    return $output;
}

function predict_image($model_type, $model_path, $img_path, $reliability){

    $_SESSION['model_type'] = trim($model_type);
    $_SESSION['model_path'] = trim(realpath($model_path));
    $_SESSION['img_path'] = trim($img_path);
}

function getGif($dir){
    $files = scandir(realpath($dir));
    return $dir . $files[rand(2, count($files) - 1)];
}

?>