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
		<meta charset="utf-8" />
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
				<div style="display:flex;width:100%;margin-left:20px;margin-right:10px;margin-top:60px;">
					<div width="50%" style="height:500px;width:50%;border:1px solid #ccc;font:16px/26px Georgia, Garamond, Serif;overflow:auto;">
					<?php
						$path_metrics = getPathMetrics(intval($_GET['model_id']));
						echo nl2br(file_get_contents($path_metrics['path_metric']));
					?>
					</div>
					<div style="width:50%;margin-left:10px;">
						<?php
							if($path_metrics['path_chart'] != ''){
								$currentDir = getcwd();
								$char_path = str_replace($currentDir, "", realpath($path_metrics['path_chart']));
								echo '<img src="' . $char_path . '" />';
							}
						?>
					</div>
				</div>
        </div>
	</body>
</html>