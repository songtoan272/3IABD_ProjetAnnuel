use ndarray::{Array, ArrayView1, Array2, stack, Axis};
use ndarray_rand::{RandomExt, rand_distr:: Uniform};
use rand::{ Rng, thread_rng};

#[derive(Debug)]
#[repr(C)]
pub struct ClassificationModel{
    theta: Array2<f64>,
    alpha: f64,
}

#[allow(dead_code)]
impl ClassificationModel{

    fn add_x0(x: &Array2<f64>) -> Array2<f64>{    
        let x0: Array2<f64> = Array::ones((x.len_of(Axis(0)), 1));
    
        stack(Axis(1), &[x0.view(), x.view()]).unwrap()
    }

    fn init_theta_random(&self) -> Array2<f64> {
        Array::random((self.theta.nrows(), 1), Uniform::new(-1., 1.))
    }

    pub fn nb_features(&self) -> usize {
        self.theta.nrows() - 1
    }

    
    pub fn init_classification_model(
        nb_features: usize,
        alpha: f64)
    -> ClassificationModel
    {
        ClassificationModel{
            theta: Array::zeros((nb_features + 1, 1)),
            alpha
        }
    }

    fn calculate_hypothesis(&self, xk: &ArrayView1<f64>) -> f64{
        let res = xk.dot(&self.theta)[0];
        if res >= 0.0 {1.} else {-1.}
    }
    
    pub fn train_classification_rosenblatt(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        nb_iter: u64)
    {
        //add bias x0 to X
        let x = ClassificationModel::add_x0(&x);

        //init theta random
        self.theta = self.init_theta_random();
        
        //set random generator
        let mut rng = thread_rng();
        for _i in 0..nb_iter {
            let k = rng.gen_range(0, x.nrows());
            let predicted_xk: f64 = self.calculate_hypothesis(&x.row(k));
            let semi_grad: f64 = self.alpha * (y[[k, 0]] - predicted_xk);
            for j in 0..self.theta.nrows() {
                self.theta[[j, 0]] = self.theta[[j, 0]] + semi_grad * x[[k, j]];
            }
            // self.theta = &self.theta + &(alpha_prod * &(x.slice(s![(k-1)..k, ..]).reversed_axes()));
        }
    }


    pub fn train_classification_pla(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        nb_iter: u64)
    {
        //add bias x0 to X
        let x = ClassificationModel::add_x0(&x);

        //init theta random
        // self.theta = self.init_theta_random();
        
        //set random generator
        let mut rng = thread_rng();
        let mut _i = 0;
        while _i < nb_iter {
            let k = rng.gen_range(0, x.nrows());
            if self.calculate_hypothesis(&x.row(k)) == y[[k, 0]]{
                continue;
            }
            let semi_grad: f64 = self.alpha * y[[k, 0]];
            for j in 0..self.theta.nrows() {
                self.theta[[j, 0]] = self.theta[[j, 0]] + semi_grad * x[[k, j]];
            }
            _i += 1;
            // self.theta = &self.theta + &(alpha_prod * &x.row(k));
        }
    }
    
    pub fn predict_classification_model(&self, x_test: &Array2<f64>) -> Array2<f64>{
        let _x_test = ClassificationModel::add_x0(&x_test);
        if _x_test.ncols() != self.theta.nrows(){
            panic!("Number of features in instances to be predicted
            is not compatible.")
        }else{
            let sigmoid_func = |v:f64| -> f64 {if v >= 0.0 {1.} else {-1.}};
            _x_test.dot(&self.theta).mapv(sigmoid_func)
        }
    }

    pub fn get_theta(&self) -> &Array2<f64>{
        &self.theta
    }

    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    pub fn del_classification_model(ptr_model: *mut ClassificationModel){
        if ptr_model.is_null() {
            return;
        }

        unsafe { Box::from_raw(ptr_model);}
    }
}


