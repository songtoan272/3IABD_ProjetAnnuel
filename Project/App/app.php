<?php 
	session_start();
	include_once('php/functions.php');
	$id = NULL;
	$cla = 'Error';
	if(isset($_SESSION['id'])){
		$id = $_SESSION['id'];
	}
	if(isset($_SESSION['class'])){
		$cla = $_SESSION['class'];
	}
	if(isset($_SESSION['toprint'])){
		echo var_dump($_SESSION['toprint']);
	}
?>
<!DOCTYPE HTML>

<html>
	<head>
		<title>Guess my character UwU</title>
		<meta charset="ISO-8859-1" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />
		<!--[if lte IE 8]><script src="assets/js/ie/html5shiv.js"></script><![endif]-->
		<link rel="stylesheet" href="assets/css/main.css" />
		<!--[if lte IE 8]><link rel="stylesheet" href="assets/css/ie8.css" /><![endif]-->
		<!--[if lte IE 9]><link rel="stylesheet" href="assets/css/ie9.css" /><![endif]-->
	</head>
	<body>
		<div id="page-wrapper">

			<!-- Header -->
				<div id="header">

					<!-- Logo -->
						<h1><a href="index.php" id="logo">Projet Annuel 3IABD <em>| LECONTE Guillaume / PINTO Thomas / CAO Song Toan</em></a></h1>
						<span id='coucou'></span>

					<!-- Nav -->
						<nav id="nav">
							<ul>
								<li class="current"><a href="app.php">Guess my character</a></li>
							</ul>
						</nav>

				</div>
        </div>
		<?php
			if($id != NULL){
				$dir = "images/loading_gif/";
				$load = getGif($dir);
				echo '<div style="display:flex;width:100%;margin-left:20px;margin-right:10px;margin-top:60px;">' .
					'<div style="width:50%;text-align:center;">';
				if($id != 'error'){
					echo '<span id="char_name" style="font-size:50px;"><em>Loading ...</em></span>' .
						'<div style="display:flex;">' .
						'<div style="width:50%;text-align:center;padding-right:10px;">' .
						'<b>Your image</b></br></br>' .
						"<img class=" . '"' . "fit-picture" . '"' .
						"src=images/ia/" . $id . ' style="max-width:100%;max-height:100%;"></div>'.
						'<div style="width:50%;text-align:center;padding-left:10px;">' .
						'<b>I guess</b></br></br>' .
						'<img id="char_img" class="fit-picture" ' .
						'src="'.$load.'" style="max-width:100%;max-height:100%;"></div></div>';
				} else {
					echo "<span>Désolé impossible d'ouvrir l'image</span>";
				}
				echo '</div><div style="width:50%;text-align:left;">';
			}
		?>
		<form action="php/ddlImage.php" method="post" enctype="multipart/form-data" style="margin-top:50px;text-align:center;width:auto;">
		<label for="model">Choose a model :</label>
		<select name="model" id="model">
		<?php
			$models = getModels();
			$c = 0;
			while($data = $models->fetch_array())
			{
				if($c == 0){
					echo '<option value="' . $data['id_model'] . '" selected>' . $data['name_model'] . '</option>';
				} else {
					echo '<option value="' . $data['id_model'] . '">' . $data['name_model'] . '</option>';
				}
				$c += 1;
			}
		?>
		</select>
		&nbsp;&nbsp;
		<input type="button" id="metrics" onClick="getMetrics();" value="Look metrics">
			</br>
			</br>
			<input type="text" name="urlImage" id ="urlImage" placeholder="Url" style="display:inline;margin-right:50px;margin-left:50px;width:70%;">
			</br>
			</br>
			Ou
			</br>
			</br>
			<input type="file" name="fileToUpload" id="fileToUpload" accept=".png, .jpg, .jpeg"> </br></br>
    		<input type="submit" value="Upload Image" name="submit">
		</form>

		<?php
			if($id != NULL){
				echo '</div></div>';
			}
			$_SESSION['id'] = NULL;
			$_SESSION['class'] = NULL;
			session_destroy();
		?>

		<!-- Scripts -->
			<script src="assets/js/jquery.min.js"></script>
			<script src="assets/js/jquery.dropotron.min.js"></script>
			<script src="assets/js/skel.min.js"></script>
			<script src="assets/js/util.js"></script>
			<!--[if lte IE 8]><script src="assets/js/ie/respond.min.js"></script><![endif]-->
			<script src="assets/js/main.js"></script>
		<?php
			if(isset($_SESSION['model_path'])){
				echo
				'<script>
				$(document).ready(function(){
					$.ajax({
						url: "php/predict.php?model_id='.$_SESSION['model_id'].'&model_path='.$_SESSION['model_path'].'&img_path='.$_SESSION['img_path'].'&model_type='.$_SESSION['model_type'].'&result='.$_SESSION['result_classes'].'",
						type: "GET",
						dataType: "json",
						success: function(result){
								document.getElementById("char_img").setAttribute("src", result["path"]);
								document.getElementById("char_name").innerHTML = result["name"];
								//document.getElementById("coucou").innerHTML = result["debug"];
							},
						error : function() {
							alert("Une erreur est survenue");
						}
					});
				});
				</script>';
			}
		?>
		<script>
			function getMetrics(){
				id_model = document.getElementById('model').value;
				window.open('http://92.222.76.60/display_metrics.php?model_id=' + id_model, '_blank');
			}
		</script>
	</body>
</html>