<?php 
	session_start();
	header('Location: app.php');
?>
<!DOCTYPE HTML>

<html>
	<head>
		<title>Portfolio | LECONTE Guillaume</title>
		<meta charset="UTF-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />
		<!--[if lte IE 8]><script src="assets/js/ie/html5shiv.js"></script><![endif]-->
		<link rel="stylesheet" href="assets/css/main.css" />
		<!--[if lte IE 8]><link rel="stylesheet" href="assets/css/ie8.css" /><![endif]-->
		<!--[if lte IE 9]><link rel="stylesheet" href="assets/css/ie9.css" /><![endif]-->
		
		<?php
		include_once('php/fonctionPHP.php');
		$table = openBDDprojects(3);
		?>
	</head>
	
	<body>
		
		<div id="page-wrapper">

			<!-- Header -->
				<div id="header">

					<!-- Logo -->
						<h1><a href="index.html" id="logo">Portfolio <em>| Leconte Guillaume</em></a></h1>

					<!-- Nav -->
						<nav id="nav">
							<ul>
								<li class="current"><a href="index.php">Accueil</a></li>
								<li><a href="projet.php">Projets</a></li>
								<li><a href="stage.php">Stages</a></li>
								<li><a href="veille.php">Veille Technologique</a></li>
								<li><a href="cv.php">Curriculum vitae</a></li>
								<?php if($_SESSION['type'] == 'admin') { ?>
								<li><a href="app.php">Guess my character</a></li>
								<li><a href="/admin/addArticle.php">Ajouter articles</a></li>
								<li><a href="/php/logout.php">Logout</a></li>
								<?php } else { ?>
								<li><a href="/admin/loging.php">Login</a></li>
								<?php } ?>
							</ul>
						</nav>

				</div>

			<!-- Banner -->
				<section id="banner">
					<header>
						<h2>Bienvenue <em>sur mon Portfolio</em></h2>
					</header>
				</section>

			<!-- Gigantic Heading -->
				<section class="wrapper style2">
					<div class="container">
						<header class="major">
							<h2>A propos de moi</h2>
							<p>Je suis élève de deuxième année en BTS SIO option SLAM au lycée Saint Adjutor à Vernon.
							</br>Passionné par l'informatique j'ai décidé d'en faire mon métier.
							</br>A la suite de ma formation actuelle, je souhaite poursuive mes études.
							</br>
							</br>J'ai était accepté en 3e année à l'ESGI, option Intelligence Artificielle et Big Data,
							</br>Je recherche donc une entreprise pour effectuer un contrat en alternance.
							</p>
						</header>
					</div>
				</section>
			
			<!-- Highlights -->
			
				<section class="wrapper style1">
					<div class="container">
						<h2 class="claPers2">Dernières nouveautés</h2>
						<div class="row 200%">
						<?php
						while($ligne = $table->fetch_array(MYSQLI_BOTH)) {
							$image = afficheAllImg();
							while(($ligneImg = $image->fetch_array(MYSQLI_BOTH)) and ($ligneImg['idProjets'] != $ligne['id'])) {}
						?>
							<section class="4u 12u(narrower)">
								<div class="box highlight">
									<a href="article.php?index=<?php printf($ligne['id']) ?>"><img src="<?php printf($ligneImg['cheminRes']) ?>" style="max-width: 160px; max-height: 160px;" alt="" /></a>
									<h3><a href="article.php?index=<?php printf($ligne['id']) ?>"><?php printf($ligne['nom']) ?> <em><?php printf($ligne['language']) ?></em></h3></a>
									<p><?php printf($ligne['resume']) ?></p>
								</div>
							</section>
						<?php
						}
						?>
						</div>
					</div>
				</section>

			<!-- Footer -->
				<div id="footer">
					<div class="container">
						<div class="row">
							<section class="container">
								<h3 class="claPers">Contactez moi</h3>
								<form method="post" action="/php/mail.php" enctype="multipart/form-data">
									<div class="row 50%">
										<div class="6u 12u(mobilep)">
											<input type="text" name="name" id="name" class="formContact" placeholder="Nom" />
										</div>
										<div class="6u 12u(mobilep)">
											<input type="email" name="email" id="email" class="formContact" placeholder="Email" />
										</div>
									</div>
									<div class="row 50%">
										<div class="12u">
											<textarea name="message" id="message" class="formContact" placeholder="Message" rows="8"></textarea>
										</div>
									</div>
									<div class="row 50%">
										<div class="12u">
											<ul class="btnSend">
												<li><input type="submit" class="button alt" value="Envoyer" /></li>
											</ul>
										</div>
									</div>
								</form>
							</section>
						</div>
					</div>

				</div>

		</div>

		<!-- Scripts -->
			<script src="assets/js/jquery.min.js"></script>
			<script src="assets/js/jquery.dropotron.min.js"></script>
			<script src="assets/js/skel.min.js"></script>
			<script src="assets/js/util.js"></script>
			<!--[if lte IE 8]><script src="assets/js/ie/respond.min.js"></script><![endif]-->
			<script src="assets/js/main.js"></script>

	</body>
</html>