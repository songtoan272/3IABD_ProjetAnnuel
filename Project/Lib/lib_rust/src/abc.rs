use ndarray::{Array, arr2, Array2, stack, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

// use ndarray_linalg::solve::*;
mod lin_reg;
mod classification;

pub fn main() {
    let x_ones: Array2<f64> = Array::ones((50, 2));
    let x1: Array2<f64> = Array::random((50, 2), Uniform::new(0., 1.)) * 0.9 + &x_ones;
    let x2: Array2<f64> = Array::random((50, 2), Uniform::new(0., 1.)) * 0.9 + 2. * &x_ones;
    let x = match stack(Axis(0), &[x1.view(), x2.view()]){
        Result::Ok(arr) => arr,
        Result::Err(e) => panic!(e),
    };

    let y: Array2<f64> = match stack(Axis(0), &[x_ones.view(), (x_ones.to_shared() * -1.).view()]){
        Result::Ok(arr) => arr,
        Result::Err(e) => panic!(e),
    };

    let mut class_model = classification::ClassificationModel::init_classification_model(&x, &y, 0.01);
    class_model.train_classification_rosenblatt(1000);
    let y_predicted = class_model.predict_classification_model(&x);

    println!("theta = {:#?}", class_model.get_theta());
    println!("y_predicted = {:#?}", y_predicted);
    println!("y = {:#?}", y);

    // let x: ndarray::Array2<f64> = arr2(&[
    //     [1., 0.],
    //     [0., 1.],
    //     [1., 1.],
    //     [0., 0.]
    // ]);

    // //x = lin_reg::add_x0(x);

    // let y: ndarray::Array2<f64> = arr2(&[
    //     [2.],
    //     [1.],
    //     [-2.],
    //     [-1.]
    // ]);

    // let mut lin_reg = lin_reg::LinearRegModel::init_linear_model(&x, &y);
    // lin_reg.train_linear_regression_model();

    // let x_test = arr2(&[
    //     [2., 3.],
    //     [2., 1.],
    //     [3., 1.5]
    // ]);

    // let y_predicted = lin_reg.predict_linear_regression_model(&x_test);


    // println!("theta = {:#?}", lin_reg.get_theta());
    // println!("{:#?}", y_predicted);
    // // println!("{:#?}", x.len_of(ndarray::Axis(1)));
    // // println!("{:#?}", lin_reg::normalize_equation(&x, &y));
    

    // println!("{:#?}", x.row(1).t());
    

}