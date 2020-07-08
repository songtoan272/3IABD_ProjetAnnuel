
use array2d::Array2D;
use ndarray::{Array, arr2, arr1, Array1, Array2};
mod lin_reg;
mod classification;
mod mlp_array2d;
mod rbf;


pub fn main(){

    let x: Array2D<f64> = Array2D::from_rows(&vec![
        vec![1., 0.],
        vec![0., 1.],
        vec![0., 0.],
        vec![1., 1.]
    ]);

    
    let y: Array2D<f64> = Array2D::from_rows(&vec![
        vec![1.],
        vec![1.],
        vec![-1.],
        vec![-1.]
    ]);
    // let y: Array2D<f64> = Array2D::from_rows(&vec![
    //     vec![1., -1., -1.],
    //     vec![-1., 1., -1.],
    //     vec![-1., -1., -1.],
    //     vec![-1., -1., 1.]]);
    
    // let x: Array2<f64> = arr2(&[
    //     [1., 1.],
    //     [2., 3.],
    //     [3., 3.]
    // ]);

    
    // let y: Array2D<f64> = Array2D::from_rows(&vec![
    //     vec![1., -1., 3.],
    //     vec![2., 2., 4.],
    //     vec![8., -1.5, -1.],
    //     vec![-5., -1., 2.]]);

    // let mut model: rbf::RBF = rbf::RBF::init_rbf(3, x.nrows(), x.ncols(), y.ncols(), true, 0.1, true);
    // // model.train_rbf(&x, &y, true, "lloyd");
    // let y_pred = model.predict_rbf(&x);
    
    let hidden_layer_sizes: Vec<usize> = vec![x.num_columns(), 5, y.num_columns()];

    let lr = 0.1;

    let mut model: mlp_array2d::MLP = mlp_array2d::MLP::init_mlp(hidden_layer_sizes, lr, true);

    model.train_mlp(&x, &y, 100000);
    
    let y_pred: Array2D<f64> = model.predict_mlp(&x);

    println!("weights = {:#?}", model.get_weights());
    println!("y_pred = {:#?}", y_pred);



    //TEST CLASSIFICATION SIMPLE

    //Linear Simple
    // let x1: Array2<f64> = arr2(&[
    //     [1., 1.],
    //     [2., 3.],
    //     [3., 3.]
    // ]);

    // let y1: Array2<f64> = arr1(&[1., -1., -1.]).into_shape((3, 1)).unwrap();
    
    // let mut model1 = classification::ClassificationModel::init_classification_model(x1.ncols(), 0.01);
    // model1.train_classification_rosenblatt(&x1, &y1, 1000);
    // println!("Rosenblatt:");
    // println!("W1={:?}", model1.get_theta());
    // println!("y_pred1={:?}", model1.predict_classification_model(&x1));
    // model1.train_classification_pla(&x1, &y1, 10000);
    // model1.set_alpha(0.1);
    // println!("PLA:");
    // println!("W1={:?}", model1.get_theta());
    // println!("y_pred1={:?}", model1.predict_classification_model(&x1));
    // println!("\n\n");
    
    //Linear Multiple

    // let xones: Array2<f64> = Array::ones((50, 2));
    // let x2: Array2<f64> = stack(Axis(0), &[
    //     (Array::random((50, 2), Uniform::new(-1., 1.)) * 0.9 + &xones).view(), 
    //     (Array::random((50, 2), Uniform::new(-1., 1.)) * 0.9 + (&xones * 2.0)).view(), 
    //     ]).unwrap();

    // let yones: Array2<f64> = Array::ones((50, 1));
    // let y2: Array2<f64> = stack(Axis(0), &[
    //     (&yones).view(), 
    //     (&yones * -1.0).view() 
    //     ]).unwrap();
    
    
    // let mut model2 = classification::ClassificationModel::init_classification_model(x2.ncols(), 0.1);
    // // model2.train_classification_rosenblatt(&x2, &y2, 10000);
    // // println!("Rosenblatt:");
    // // println!("W2={:?}", (model2.get_theta()));
    // // println!("y_pred2={:?}", model2.predict_classification_model(&x2));
    // // println!("y_real - y_pred2={:?}", &model2.predict_classification_model(&x2) - &y2);

    // model2.train_classification_pla(&x2, &y2, 10000);
    // println!("PLA:");
    // println!("W2={:?}", model2.get_theta());
    // println!("y_pred2={:?}", model2.predict_classification_model(&x2));
    // println!("y_real - y_pred2={:?}", &model2.predict_classification_model(&x2) - &y2);

    

    //TEST REGRESSION SIMPLE ALGO

    // //Linear simple 2D
    // let x3: Array2<f64> = arr2(&[
    //     [1.],
    //     [2.]
    // ]);

    // let y3: Array2<f64> = arr1(&[2., 3.]).into_shape((2, 1)).unwrap();
    
    // let mut model3 = lin_reg::LinearRegModel::init_linear_model(x3.ncols());
    // model3.train_linear_regression_model(&x3, &y3);
    // println!("W3={:?}", model3.get_theta());
    // println!("y_pred3={:?}", model3.predict_linear_regression_model(&x3));



    //Non Linear Simple 2D
    // let x4: Array2<f64> = arr2(&[
    //     [1.],
    //     [2.],
    //     [3.]
    // ]);

    // let y4: Array2<f64> = arr1(&[2., 3., 2.5]).into_shape((3, 1)).unwrap();
    
    // let mut model4 = lin_reg::LinearRegModel::init_linear_model(x4.ncols());
    // model4.train_linear_regression_model(&x4, &y4);
    // println!("W4={:?}", model4.get_theta());
    // println!("y_pred4={:?}", model4.predict_linear_regression_model(&x4));



    //Linear Simple 3D
    // let x5: Array2<f64> = arr2(&[
    //     [1., 1.],
    //     [2., 2.],
    //     [3., 1.]
    // ]);

    // let y5: Array2<f64> = arr1(&[2., 3., 2.5]).into_shape((3, 1)).unwrap();
    
    // let mut model5 = lin_reg::LinearRegModel::init_linear_model(x5.ncols());
    // model5.train_linear_regression_model(&x5, &y5);
    // println!("W5={:?}", model5.get_theta());
    // println!("y_pred5={:?}", model5.predict_linear_regression_model(&x5));




    //Linear Tricky 3D
    // let x6: Array2<f64> = arr2(&[
    //     [1., 1.],
    //     [2., 2.],
    //     [3., 3.]
    // ]);

    // let y6: Array2<f64> = arr1(&[1., 2., 3.]).into_shape((3, 1)).unwrap();
    
    // let mut model6 = lin_reg::LinearRegModel::init_linear_model(x6.ncols());
    // model6.train_linear_regression_model(&x6, &y6);
    // println!("W6={:?}", model6.get_theta());
    // println!("y_pred6={:?}", model6.predict_linear_regression_model(&x6));



    //Non Linear 3D
    // let x7: Array2<f64> = arr2(&[
    //     [1., 0.],
    //     [0., 1.],
    //     [1., 1.],
    //     [0., 0.]
    // ]);

    // let y7: Array2<f64> = arr1(&[2., 1., -2., -1.]).into_shape((4, 1)).unwrap();
    
    // let mut model7 = lin_reg::LinearRegModel::init_linear_model(x7.ncols());
    // model7.train_linear_regression_model(&x7, &y7);
    // println!("W7={:?}", model7.get_theta());
    // println!("y_pred7={:?}", model7.predict_linear_regression_model(&x7));
}