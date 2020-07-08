use std::slice::from_raw_parts;
use ndarray::{Array, Array2};
use array2d::Array2D;


mod lin_reg;
mod classification;
mod rbf;
mod mlp_array2d;

pub use lin_reg::LinearRegModel;
pub use classification::ClassificationModel;
pub use rbf::RBF;
pub use mlp_array2d::MLP;


fn from_raw_to_arr2(
    raw_ptr: *mut f64, 
    nrows: usize, 
    ncols: usize) 
    -> Array2<f64> 
{
    assert!(!raw_ptr.is_null());
    
    let _arr = unsafe {
        let _slice = from_raw_parts(raw_ptr, nrows * ncols);
        match Array::from_shape_vec((nrows, ncols),_slice.to_vec()){
            Result::Ok(arr) => arr,
            Result::Err(e) => panic!("{}", e),
        }
    };
    _arr
}

fn from_raw_to_arr2d(
    raw_ptr: *mut f64, 
    nrows: usize, 
    ncols: usize) 
    -> Array2D<f64> 
{
    assert!(!raw_ptr.is_null());
    
    let _arr = unsafe{
        let _slice = from_raw_parts(raw_ptr, nrows * ncols);
        Array2D::from_row_major(_slice, nrows, ncols)
    };
    return _arr;
}


fn from_arr2_to_vec1(arr: &Array2<f64>) -> Vec<f64>{
    let mut res: Vec<f64> = Vec::with_capacity(arr.nrows() * arr.ncols());
    for i in 0..arr.nrows(){
        for j in 0..arr.ncols(){
            res.push(arr[[i, j]]);
        }
    }
    res
}



//Function for linear regression exposed to Python
#[no_mangle]
pub extern fn init_linear_regression_model(nb_features: u64) -> *mut LinearRegModel{
    let lin_reg = LinearRegModel::init_linear_model(nb_features as usize);
    let boxed = Box::new(lin_reg);
    Box::into_raw(boxed)
}


#[no_mangle]
pub extern fn train_linear_regression_model(
    model: &mut LinearRegModel,
    x: *mut f64,
    y: *mut f64,
    nb_inputs: u64)
{
    let _x_arr = from_raw_to_arr2(x, nb_inputs as usize, model.nb_features());
    let _y_arr = from_raw_to_arr2(y, nb_inputs as usize, 1);
    model.train_linear_regression_model(&_x_arr, &_y_arr)
}



#[no_mangle]
pub extern fn predict_linear_regression_model(
    model: &LinearRegModel, 
    x_test: *mut f64, 
    nb_inputs: u64)
    -> *const f64
{
    let _x_test_arr = from_raw_to_arr2(x_test, nb_inputs as usize, model.nb_features());
    let y_pred_arr = model.predict_linear_regression_model(&_x_test_arr);
    let mut y_pred_vec: Vec<f64> = from_arr2_to_vec1(&y_pred_arr);
    Box::leak(y_pred_vec.into_boxed_slice()).as_ptr()
}

#[no_mangle]
pub extern fn del_linear_regression_model(model: *mut LinearRegModel){
    LinearRegModel::del_linear_regression_model(model);
}


#[no_mangle]
pub extern fn get_theta_linreg_model(
    model: &LinearRegModel
) -> *const f64 {
    let theta_arr = model.get_theta();
    let mut theta_vec: Vec<f64> = from_arr2_to_vec1(theta_arr);
    Box::leak(theta_vec.into_boxed_slice()).as_ptr()
}








//Functions for classification exposed to Python
#[no_mangle]
pub extern fn init_classification_model(nb_features: u64, alpha: f64) -> *mut ClassificationModel{
    let classification = ClassificationModel::init_classification_model(nb_features as usize, alpha);
    let boxed = Box::new(classification);
    Box::into_raw(boxed)
}


#[no_mangle]
pub extern fn train_classification_rosenblatt(
    model: &mut ClassificationModel,
    x: *mut f64,
    y: *mut f64,
    nb_inputs: u64,
    nb_iters: u64)
{
    let _x_arr = from_raw_to_arr2(x, nb_inputs as usize, model.nb_features());
    let _y_arr = from_raw_to_arr2(y, nb_inputs as usize, 1);
    model.train_classification_rosenblatt(&_x_arr, &_y_arr, nb_iters);
}

#[no_mangle]
pub extern fn train_classification_pla(
    model: &mut ClassificationModel,
    x: *mut f64,
    y: *mut f64,
    nb_inputs: u64,
    nb_iters: u64)
{
    let _x_arr = from_raw_to_arr2(x, nb_inputs as usize, model.nb_features());
    let _y_arr = from_raw_to_arr2(y, nb_inputs as usize, 1);
    model.train_classification_pla(&_x_arr, &_y_arr, nb_iters);
}


#[no_mangle]
pub extern fn predict_classification_model(
    model: &ClassificationModel, 
    x_test: *mut f64, 
    nb_inputs: u64)
    -> *const f64
{
    let _x_test_arr = from_raw_to_arr2(x_test, nb_inputs as usize, model.nb_features());
    let y_pred_arr = model.predict_classification_model(&_x_test_arr);
    let mut y_pred_vec: Vec<f64> = from_arr2_to_vec1(&y_pred_arr);
    Box::leak(y_pred_vec.into_boxed_slice()).as_ptr()
}

#[no_mangle]
pub extern fn del_classification_model(model: *mut ClassificationModel){
    ClassificationModel::del_classification_model(model);
}


#[no_mangle]
pub extern fn get_theta_classification(
    model: &ClassificationModel
) -> *const f64 {
    let theta_arr = model.get_theta();
    let mut theta_vec: Vec<f64> = from_arr2_to_vec1(theta_arr);
    Box::leak(theta_vec.into_boxed_slice()).as_ptr()
}

#[no_mangle]
pub extern fn get_alpha_classification(
    model: &ClassificationModel
) -> f64 {
    model.get_alpha()
}

#[no_mangle]
pub extern fn set_alpha_classification(
    model: &mut ClassificationModel,
    alpha: f64
) {
    model.set_alpha(alpha)
}








//Functions for RBF exposed to Python
#[no_mangle]
pub extern fn init_rbf(
    n_centroids: u64,
    n_samples: u64,
    n_features: u64,
    n_outputs: u64,
    kmeans_mode: bool,
    gamma: f64,
    classification_mode: bool
    ) -> *mut RBF{
    let rbf = RBF::init_rbf(n_centroids as usize,
        n_samples as usize,
        n_features as usize,
        n_outputs as usize,
        kmeans_mode,
        gamma,
        classification_mode);
    let boxed = Box::new(rbf);
    Box::into_raw(boxed)
}


#[no_mangle]
pub extern fn train_rbf(
    model: &mut RBF,
    x: *mut f64,
    y: *mut f64,
    nb_inputs: u64,
    norm_gamma: bool)
{
    let _x_arr = from_raw_to_arr2(x, nb_inputs as usize, model.get_n_features());
    let _y_arr = from_raw_to_arr2(y, nb_inputs as usize, model.get_n_outputs());
    model.train_rbf(&_x_arr, &_y_arr, norm_gamma);
}


#[no_mangle]
pub extern fn predict_rbf(
    model: &RBF, 
    x_test: *mut f64, 
    nb_inputs: u64)
    -> *const f64
{
    let _x_test_arr = from_raw_to_arr2(x_test, nb_inputs as usize, model.get_n_features());
    let y_pred_arr = model.predict_rbf(&_x_test_arr);
    let mut res: Vec<f64> = from_arr2_to_vec1(&y_pred_arr);
    // println!("rust = {:#?}", &res);
    let slice = res.into_boxed_slice();
    // let ptr = slice.as_ptr();
    Box::leak(slice).as_ptr()
}

#[no_mangle]
pub extern fn del_rbf(model: *mut RBF){
    RBF::del_rbf(model);
}

#[no_mangle]
pub extern fn get_weights_rbf(model: &RBF) -> *const f64{
    let w_arr = model.get_weights();
    let mut w_vec = from_arr2_to_vec1(w_arr);
    Box::leak(w_vec.into_boxed_slice()).as_ptr()
}

#[no_mangle]
pub extern fn get_centroids_rbf(model: &RBF) -> *const f64{
    let centroids_arr = model.get_centroids();
    let mut centroids_vec = from_arr2_to_vec1(centroids_arr);
    Box::leak(centroids_vec.into_boxed_slice()).as_ptr()
}

#[no_mangle]
pub extern fn get_n_features(model: &RBF) -> u64 {
    model.get_n_features() as u64
}

#[no_mangle]
pub extern fn get_n_outputs(model: &RBF) -> u64 {
    model.get_n_outputs() as u64
}

#[no_mangle]
pub extern fn get_n_samples(model: &RBF) -> u64 {
    model.get_n_samples() as u64
}

#[no_mangle]
pub extern fn get_n_centroids(model: &RBF) -> u64 {
    model.get_n_centroids() as u64
}

#[no_mangle]
pub extern fn set_n_centroids(model: &mut RBF, n_centroids: u64) {
    model.set_n_centroids(n_centroids as usize)
}

#[no_mangle]
pub extern fn get_gamma(model: &RBF) -> f64 {
    model.get_gamma()
}

#[no_mangle]
pub extern fn set_gamma(model: &mut RBF, gamma: f64) {
    model.set_gamma(gamma)
}

#[no_mangle]
pub extern fn set_kmeans_mode(model: &mut RBF, mode: bool){
    model.set_kmeans_mode(mode)
}

#[no_mangle]
pub extern fn get_mode(model: &RBF) -> bool{
    model.get_mode()
}










//Functions for MLP exposed to Python
#[no_mangle]
pub extern fn init_mlp(
    layer_sizes: *mut u64, 
    n_layers: u64,
    learning_rate: f64,
    classification_mode: bool
)-> *mut MLP
{
    let layer_sizes_vec = unsafe{
        from_raw_parts(layer_sizes as *mut usize, n_layers as usize)
    };
    let mlp = MLP::init_mlp(
        Vec::from(layer_sizes_vec),
        learning_rate,
        classification_mode
);
    let boxed = Box::new(mlp);
    Box::into_raw(boxed)
}


#[no_mangle]
pub extern fn train_mlp(
    model: &mut MLP,
    x: *mut f64,
    y: *mut f64,
    nb_inputs: u64,
    nb_iter: u64)
{
    let _x_arr = from_raw_to_arr2d(x, nb_inputs as usize, model.get_n_features());
    let _y_arr = from_raw_to_arr2d(y, nb_inputs as usize, model.get_n_outputs());
    model.train_mlp(&_x_arr, &_y_arr, nb_iter);
}


#[no_mangle]
pub extern fn predict_mlp(
    model: &MLP, 
    x_test: *mut f64, 
    nb_inputs: u64)
    -> *const f64
{
    let _x_test_arr = from_raw_to_arr2d(x_test, nb_inputs as usize, model.get_n_features());
    let mut y_pred_arr = model.predict_mlp(&_x_test_arr).as_rows();
    let mut res: Vec<f64> = Vec::with_capacity((nb_inputs as usize) * model.get_n_outputs());
    for i in 0..nb_inputs as usize{
        res.append(&mut y_pred_arr[i]);
    }
    // println!("rust = {:#?}", &res);
    let slice = res.into_boxed_slice();
    // let ptr = slice.as_ptr();
    Box::leak(slice).as_ptr()
    // ptr
}

#[no_mangle]
pub extern fn del_mlp(model: *mut MLP){
    MLP::del_mlp(model);
}


#[no_mangle]
pub extern fn get_weights_mlp(model: &MLP) -> *const f64{
    let _weights = model.get_weights();
    let n_layers = model.get_n_layers();
    let layer_sizes = model.get_layer_sizes();

    // Calculate total number of weights
    let mut n_elems: usize = 0;
    for l in 0..n_layers - 1{
        n_elems += layer_sizes[l] * layer_sizes[l+1]
    }
    
    // Convert Vector of Array2D into a Vec of f64
    let mut weights: Vec<f64> = Vec::with_capacity(n_elems);
    for l in 0..n_layers-1{
        let mut w_l: Vec<f64> = _weights[l].as_rows().iter().flat_map(|v| v.iter()).cloned().collect();
        weights.append(&mut w_l);
    }
    // println!("{:#?}", &weights);
    Box::leak(weights.into_boxed_slice()).as_ptr()
}


#[no_mangle]
pub extern fn get_bias_mlp(model: &MLP) -> *const f64 {
    let _bias = model.get_bias();
    let n_layers = model.get_n_layers();
    let layer_sizes = model.get_layer_sizes();

    // Calculate total number of weights
    let mut n_elems: usize = 0;
    for l in 1..n_layers{
        n_elems += layer_sizes[l]
    }
    
    // Convert Vector of Array2D into a Vec of f64
    let mut bias: Vec<f64> = Vec::with_capacity(n_elems);
    for l in 0..n_layers-1{
        let mut b_l: Vec<f64> = _bias[l].as_rows().iter().flat_map(|v| v.iter()).cloned().collect();
        bias.append(&mut b_l);
    }
    Box::leak(bias.into_boxed_slice()).as_ptr()
}

#[no_mangle]
pub extern fn get_n_iters(model: &MLP) -> u64 {
    model.get_n_iters()
}

#[no_mangle]
pub extern fn set_learning_rate_mlp(model: &mut MLP, learning_rate: f64) {
    model.set_learning_rate(learning_rate)
}

#[no_mangle]
pub extern fn set_mode_mlp(model: &mut MLP, mode: bool){
    model.set_mode(mode)
}


#[no_mangle]
pub extern fn dispose_ptr(ptr: *mut f64){
    unsafe{
        let v = ptr.as_mut().unwrap();
        Box::from_raw(v);
    }
    // println!("disposed")
}