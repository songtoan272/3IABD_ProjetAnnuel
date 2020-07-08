use ndarray::{arr1, Array, ArrayView1, Array1, Array2};
use rand::{ Rng, thread_rng};
use ndarray_linalg::solve::Inverse;
use std::collections::HashSet;


#[derive(Debug)]
#[repr(C)]
pub struct RBF {
    // loss: f64,
    weights: Array2<f64>,  
    n_centroids: usize,
    centroids: Array2<f64>,
    n_features: usize,    //X.ncols()
    n_outputs: usize,     //Y.ncols()
    kmeans_mode: bool,
    gamma: f64,
    classification_mode: bool
}

#[allow(dead_code)]
impl RBF{


    pub fn init_rbf(
        n_centroids: usize,
        n_samples: usize,
        n_features: usize,
        n_outputs: usize,
        kmeans_mode: bool,
        gamma: f64,
        classification_mode: bool
    )-> RBF
    {   
        if !kmeans_mode{
            //naif, n_centroids = # samples
            RBF{
                // loss: 
                weights: Array2::zeros((n_samples, n_outputs)),
                n_centroids: n_samples,
                centroids: Array2::zeros((n_samples, n_features)),
                n_features,
                n_outputs,
                kmeans_mode,
                gamma: gamma,
                classification_mode
            }
        }else{
            RBF{
                // loss: 
                weights: Array2::zeros((n_centroids, n_outputs)),
                n_centroids: n_centroids,
                centroids: Array2::zeros((n_centroids, n_features)),
                n_features,
                n_outputs,
                kmeans_mode,
                gamma,
                classification_mode
            }
        }
        
    }


    
    fn norm2_squared(
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>
    ) -> f64
    {
        if x.dim() != y.dim() {
            panic!("2 Vectors are not of the same dimension to calculate the eucledien distance.")
        }
        let _v = (x - y).mapv(|x| x*x);
        _v.sum()
    }

    fn assign_points(centroids: &Vec<Vec<f64>>, x: &Array2<f64>) -> Vec<HashSet<usize>>{
        let mut clusters:Vec<HashSet<usize>> = Vec::with_capacity(centroids.len()); 
        for _i in 0..clusters.capacity(){
            clusters.push(HashSet::new());
        }
        // iterate through all samples
        // assign each sample to the nearest cluster
        for i in 0..x.nrows(){
            //find the nearest centroid to sample i 
            let mut min_idx:usize = 0;
            let mut min_norm = std::f64::MAX;
            for k in 0..centroids.len(){
                let _norm = RBF::norm2_squared(&x.row(i), &(arr1(&centroids[k]).view()));
                if _norm < min_norm{
                    min_idx = k;
                    min_norm = _norm;
                }
            }
            clusters[min_idx].insert(i);
        }
        return clusters;
    }

    /// return true if the 2 clusters are the same 
    fn differ_cluster(cluster1: &Vec<HashSet<usize>>, cluster2: &Vec<HashSet<usize>>) -> bool{
        if cluster1.len() == 0 || cluster2.len() == 0 || cluster1.len() != cluster2.len(){
            return false;
        }
        let mut res: bool = true;
        for k in 0..cluster1.len(){
            let diff: HashSet<usize> = cluster1[k].difference(&cluster2[k]).cloned().collect();
            res = res && (diff.len() == 0);
        }
        return res;
    }

    fn reevaluate_centroids(centroids: &mut Vec<Vec<f64>>, clusters: &Vec<HashSet<usize>>, x: &Array2<f64>) {
        for c in 0..clusters.len(){
            let mut sum_vec: Array1<f64> = Array::zeros(centroids[0].len());
            for i in clusters[c].iter(){
                sum_vec = sum_vec + x.row(*i);
            }
            centroids[c] = (sum_vec / clusters[c].len() as f64).to_vec();
        }
    }

    /// Calculate the centroids of all clusters using Lloyd's Algo
    /// Each centroid has the same dimension as the input
    fn lloyd_algo(&mut self, x:&Array2<f64>) {
        let mut rng = thread_rng();
        let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(self.n_centroids);
        let mut samples_chosen: HashSet<usize> = HashSet::with_capacity(self.n_centroids);
        let mut clusters: Vec<HashSet<usize>> = Vec::with_capacity(self.n_centroids);

        // initiate k centroids by taking k different samples
        let mut k: usize = 0;
        while samples_chosen.len() < self.n_centroids{
            let i = rng.gen_range(0, x.nrows());
            if !samples_chosen.contains(&i){
                centroids.push(x.row(i).to_vec());
                clusters.push(HashSet::new());
                clusters[k].insert(i);
                k+=1; 
                samples_chosen.insert(i);
            }
        }

        let mut old_clusters: Vec<HashSet<usize>> = Vec::with_capacity(self.n_centroids);
        while !RBF::differ_cluster(&clusters, &old_clusters){
            old_clusters = clusters.to_owned();
            clusters = RBF::assign_points(&centroids, x);
            RBF::reevaluate_centroids(&mut centroids, &clusters, x);
        }
        
        self.centroids = Array2::from_shape_vec((self.n_centroids, self.n_features), centroids.into_iter().flatten().collect()).unwrap();
    }


    // fn apc_iii_algo()



    fn init_centroids(&mut self, x: &Array2<f64>){
        if self.kmeans_mode{
            RBF::lloyd_algo(self, x);
        }else{  //naif, # centroids = # samples
            self.centroids = x.to_owned();
        }
    }


    fn init_gamma(&mut self){
        let mut max_dist = 0.0;
        for i in 0..self.n_centroids{
            for j in 0..self.n_centroids{
                let dist_ij = RBF::norm2_squared(&self.centroids.row(i), &self.centroids.row(j));
                if dist_ij > max_dist{
                    max_dist = dist_ij;
                }
            }
        }
        self.gamma = (max_dist / self.n_centroids as f64).sqrt()
    }


    fn init_theta(
        &self,
        x: &Array2<f64>
    ) -> Array2<f64>
    {
        let mut res: Vec<f64> = Vec::with_capacity(x.nrows() * self.n_centroids);
        for i in 0..x.nrows(){
            for j in 0..self.n_centroids{
                res.push((-self.gamma * RBF::norm2_squared(&x.row(i), &self.centroids.row(j))).exp());
            }
        }
        return Array::from_shape_vec((x.nrows(), self.n_centroids), res).unwrap();
    }



    fn calc_weights_naif(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>
    ){
        let theta = self.init_theta(x);
        self.weights = theta.inv().unwrap().dot(y);
    }

    fn calc_weights_kmeans(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>
    ){
        let theta = self.init_theta(x);
        let theta_transpose = theta.to_shared().reversed_axes();
        self.weights = (theta_transpose.dot(&theta)).inv().unwrap().dot(&theta_transpose).dot(y);
    }


    pub fn train_rbf(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        norm_gamma: bool,
    ){
        if norm_gamma{
            self.init_gamma();
        }
        self.init_centroids(x);
        if self.kmeans_mode{
            self.calc_weights_kmeans(x, y);
        }else{
            self.calc_weights_naif(x, y);
        }
    }


    pub fn predict_rbf(
        &self, 
        x_test: &Array2<f64>) 
        -> Array2<f64> 
    {
        let theta = self.init_theta(x_test);
        let res = theta.dot(&self.weights);
        return if self.classification_mode {res.mapv(|x| return if x < 0.0 {-1.} else {1.})} else {res}
    }

    pub fn get_weights(&self) -> &Array2<f64>{
        &self.weights
    }

    pub fn get_centroids(&self) -> &Array2<f64>{
        &self.centroids
    }

    pub fn get_n_features(&self) -> usize {
        self.n_features
    }

    pub fn get_n_outputs(&self) -> usize {
        self.n_outputs
    }

    pub fn get_n_samples(&self) -> usize {
        self.weights.nrows()
    }

    pub fn get_n_centroids(&self) -> usize {
        self.n_centroids
    }

    pub fn set_n_centroids(&mut self, n_centroids: usize) {
        self.n_centroids = n_centroids;

        if !self.kmeans_mode{
            if n_centroids != self.get_n_samples(){
                *self = RBF::init_rbf(
                    self.n_centroids,
                    self.get_n_samples(),
                    self.get_n_features(),
                    self.get_n_outputs(),
                    true, 
                    self.gamma,
                    self.classification_mode);
            }
        }else{
            if n_centroids == self.get_n_samples(){
                *self = RBF::init_rbf(
                    self.n_centroids,
                    self.get_n_samples(),
                    self.get_n_features(),
                    self.get_n_outputs(),
                    false, 
                    self.gamma,
                    self.classification_mode);
            }
        }
    }

    pub fn get_gamma(&self) -> f64 {
        self.gamma
    }

    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }

    pub fn set_kmeans_mode(&mut self, mode: bool){
        self.kmeans_mode = mode
    }

    pub fn get_mode(&self) -> bool{
        self.kmeans_mode
    }

    pub fn del_rbf(ptr_model: *mut RBF){
        if ptr_model.is_null() {
            return;
        }

        unsafe { Box::from_raw(ptr_model);}
    }
}


