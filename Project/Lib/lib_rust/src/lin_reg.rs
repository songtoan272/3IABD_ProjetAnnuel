use ndarray::{Array, Array2, stack, Axis, arr1};
use ndarray_linalg::solve::Inverse;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct LinearRegModel{
    theta: Array2<f64>,
    //can add more attributs to define behaviors of the models
}

#[allow(dead_code)]
impl LinearRegModel{
    pub fn nb_features(&self) -> usize {
        self.theta.nrows() - 1
    }

    fn add_x0(x: &Array2<f64>) -> Array2<f64>{    
        let x0: Array2<f64> = Array::ones((x.len_of(Axis(0)), 1));
    
        match stack(Axis(1), &[x0.view(), x.view()]){
            Result::Ok(arr) => arr,
            Result::Err(error) => panic!("{}", error)
        }
    }


    fn normalize_equation(&mut self, x: &Array2<f64>, y: &Array2<f64>) {
        let x_transpose = x.to_shared().reversed_axes();
        let x_dot: Array2<f64> = x_transpose.dot(x);
        let x_dot_reversed: Array2<f64> = match x_dot.inv(){
            Result::Ok(arr) => arr,
            Result::Err(err) => {
                let eps = Array2::from_diag(&arr1(vec![1.; x.ncols()].as_slice())) * 0.001;
                (x_dot + eps).inv().unwrap()
            },
        };
        self.theta = x_dot_reversed.dot(&x_transpose).dot(y)
    }

    pub fn init_linear_model(nb_features: usize) -> LinearRegModel{
        LinearRegModel{
            theta: Array::zeros((nb_features + 1, 1))
        }
    }
    

    pub fn train_linear_regression_model(
        &mut self, 
        x: &Array2<f64>, 
        y: &Array2<f64>
    ) 
    {
        let _x = LinearRegModel::add_x0(&x);
        
        if x.nrows() != y.nrows(){
            eprintln!("Number of samples of X and of Y are not compatible.")
        }
        else {self.normalize_equation(&_x, &y);}
    }
    

    pub fn predict_linear_regression_model(
        &self, 
        x_test: &Array2<f64>
    )  -> Array2<f64>
    {
        let _x_test = LinearRegModel::add_x0(&x_test);

        //predict 
        if _x_test.ncols() != self.theta.nrows(){
            panic!("Number of features in instances to be predicted
            is not compatible.")
        }else{
            _x_test.dot(&self.theta)
        }
    }

    
    
    pub fn get_theta(&self) -> &Array2<f64>{
        &self.theta
    }

    pub fn del_linear_regression_model(ptr_model: *mut LinearRegModel){
        if ptr_model.is_null() {
            return;
        }

        unsafe { Box::from_raw(ptr_model);}
    }
}
