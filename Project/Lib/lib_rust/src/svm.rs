use std::f64;
use osqp::{CscMatrix, Problem, Settings};

#[derive(Debug)]
#[repr(C)]
pub struct SVM {
    // loss: f64,
    weights: Vec<f64>,
    bias: f64, 
    alpha: Vec<f64>,
    n_features: usize
}

#[allow(dead_code)]
impl SVM{

    pub fn init_svm(
        n_features: usize
    )-> SVM
    {
        SVM{
            weights: Vec::with_capacity(n_features),
            bias: 0.0,
            alpha: Vec::with_capacity(0),
            n_features
        }
    }


    fn kernel_normal(x1: &Vec<f64>, x2: &Vec<f64>) -> f64 {
        if x1.len() != x2.len() {
            panic!("Inputs are not of the same length.")
        }else{
            let mut res: f64 = 0.0;
            for i in 0..x1.len(){
                res += x1[i] * x2[i];
            }
            return res;
        }
    }

    fn make_P(X: &Vec<Vec<f64>>, Y: &Vec<f64>) -> Vec<f64>{
        let n_samples = X.len();

        let mut P: Vec<f64> = Vec::with_capacity(n_samples * n_samples);
        for i in 0..n_samples{
            for j in 0..n_samples{
                P.push(Y[i] * Y[j] * SVM::kernel_normal(&X[i], &X[j]));
            }
        }
        // println!("P_vec = {:#?}", P);
        return P;
    }

    fn make_A(Y: &Vec<f64>) -> Vec<f64>{
        let n_samples = Y.len();
        let mut res: Vec<f64> = Vec::with_capacity((n_samples + 1) * n_samples);
        for i in 0..n_samples{
            res.push(Y[i]);
        }
        for i in 1..n_samples+1{
            for j in 1..n_samples+1{
                if i == j {
                    res.push(1.);
                }else{
                    res.push(0.);
                }
            }
        }
        // println!("nsamples={}", n_samples);
        // println!("A = {:#?}", res);
        return res;
    }

    
    fn calculate_alpha(X: &Vec<Vec<f64>>, Y: &Vec<f64>) -> Vec<f64>{
        let n_samples: usize = X.len();
        
        //Define problem data
        let P = CscMatrix::from_row_iter_dense(n_samples, n_samples, SVM::make_P(X, Y));
        println!("P = {:?}", P);
        let q = vec![-1.; n_samples];
        let A = CscMatrix::from_row_iter_dense(n_samples+1, n_samples, SVM::make_A(Y));
        let l = vec![0.0; n_samples+1];
        let mut u = vec![f64::MAX; n_samples+1];
        u[0] = 0.0;
        // Extract the upper triangular elements of `P`
        let P = P.into_upper_tri();

        println!("P_vec = {:?}", SVM::make_P(X, Y));
        println!("q = {:?}", q);
        println!("A_vec = {:?}", SVM::make_A(Y));
        println!("A = {:?}", A);
        println!("l = {:?}", l);
        println!("u = {:?}", u);

        
        // Change the default alpha and disable verbose output
        let settings = Settings::default().alpha(0.5).verbose(true);

        // Create an OSQP problem
        let mut prob = Problem::new(P,
                                    q.as_slice(), 
                                    A, 
                                    l.as_slice(), 
                                    u.as_slice(), 
                                    &settings)
                                    .expect("failed to setup problem");   
        
        let alpha = prob.solve().x().expect("failed to solve alpha").to_owned();
        //verify alpha
        for i in 0..n_samples{
            if alpha[i] < 0. {
                panic!("invalid alpha: alpha of {}th sample is {}", i, alpha[i]);
            } 
        }
        return alpha;

    }



    pub fn train_svm(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        outputs: &Vec<f64>)
    {
        if inputs[0].len() != self.get_n_features() {
            panic!("Number of features does not conform to attributs of the models.")
        }
        self.alpha = SVM::calculate_alpha(inputs, outputs);

        //calculate weights based on alpha
        let mut max_idx: usize = 0;
        let mut max_val: f64 = self.alpha[0];
        let n_samples = inputs.len();
        for k in 0..n_samples{
            for i in 0..self.n_features{
                self.weights[i] += self.alpha[k] * outputs[k] * inputs[k][i];
            }
            if self.alpha[k] > max_val{
                max_idx = k;
                max_val = self.alpha[k];
            }
        }

        //calculate bias w0
        let mut w0 = 0.0;
        for i in 0..self.n_features{
            w0 += self.weights[i] * inputs[max_idx][i];
        }
        self.bias = 1./outputs[max_idx] - w0;
    }



    pub fn predict_svm(
        &self, 
        inputs_test: &Vec<Vec<f64>>) 
        -> Vec<f64> 
    {
        if inputs_test[0].len() != self.get_n_features(){
            panic!("Number of features in instances to be predicted
            is not compatible.")
        }else{
            let n_samples = inputs_test.len();
            let mut pred: Vec<f64> = Vec::with_capacity(n_samples);
            for k in 0..n_samples{
                let mut pred_k = 0.0;
                for i in 0..self.n_features{
                    pred_k += self.weights[i] * inputs_test[k][i];
                }
                pred_k += self.bias;
                pred.push(pred_k);
            }
            return pred;
        }
    }

    pub fn get_weights(&self) -> &Vec<f64>{
        &self.weights
    }

    pub fn get_bias(&self) -> f64{
        self.bias
    }

    pub fn get_n_features(&self) -> usize {
        self.n_features
    }

    pub fn del_svm(ptr_model: *mut SVM){
        unsafe { 
            let model = ptr_model.as_mut().unwrap();
            Box::from_raw(ptr_model);
        }
    }
    
}


