use ndarray::{Array, ArrayView1, Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr:: Uniform};
use rand::{ Rng, thread_rng};

#[derive(Debug)]
#[repr(C)]
pub struct MLP {
    // loss: f64,
    weights: Vec<Array2<f64>>,
    bias: Vec<Array1<f64>>, 
    n_layers: usize,      //nb hidden layers + 2
    hidden_layer_sizes: Vec<usize>,
    n_features: usize,    //X.ncols()
    n_outputs: usize,     //Y.ncols()
    learning_rate: f64,
    n_iters: u64,
    classification_mode: bool
}

#[allow(dead_code)]
impl MLP{

    
    ///Initialize the weights of all layers in the network
    ///The returned weights will be a Vector containing
    ///(n_layers - 1) Array2<f64>. The ith element is the 
    ///corresponding weights between layer i and i+1
    fn init_weights(
        hidden_layer_sizes: &Vec<usize>
    ) -> Vec<Array2<f64>>
    {
        let n_hidden_layers = hidden_layer_sizes.len();
        let mut W: Vec<Array2<f64>> = Vec::with_capacity(n_hidden_layers+1);

        //insert weights between the input layer and the first hidden layer
        // W.push(Array::random(
        //     (n_features, hidden_layer_sizes[0]),
        //     Uniform::new(-1., 1.)));
        
        //insert weights between any two hidden layers    
        for l in 1..n_hidden_layers{
            W.push(Array::random(
                (hidden_layer_sizes[l-1], hidden_layer_sizes[l]),
                Uniform::new(-1. ,1.)));
        }

        //insert weights between the last hidden layer and the output layer
        // W.push(Array::random(
        //     (hidden_layer_sizes[n_hidden_layers-1], n_outputs),
        //     Uniform::new(-1., 1.)));
        return W;
    }


    ///Initialize the Vector bias of the entire network
    ///The returned bias is a Vector containing (n_layers - 1)
    ///Array1<f64>. The ith element is the corresponding bias
    ///between layer i and i+1
    fn init_bias(
        hidden_layer_sizes: &Vec<usize>, 
    )-> Vec<Array1<f64>>
    {        
        let n_hidden_layers = hidden_layer_sizes.len();
        let mut B: Vec<Array1<f64>> = Vec::with_capacity(n_hidden_layers+1);
        for l in 0..n_hidden_layers{
            B.push(Array::random(
                hidden_layer_sizes[l],
                Uniform::new(-1. ,1.))
            );
        }
        return B;
    }


    pub fn init_mlp(
        hidden_layer_sizes: Vec<usize>, 
        n_features: usize,
        n_outputs: usize,
        learning_rate: f64,
        classification_mode: bool
    )-> MLP
    {
        MLP{
            // loss: 
            weights: MLP::init_weights(&hidden_layer_sizes),
            bias: MLP::init_bias(&hidden_layer_sizes),
            n_layers: hidden_layer_sizes.len(),
            hidden_layer_sizes,
            n_features,
            n_outputs,
            learning_rate,
            classification_mode,
            n_iters: 0
        }
    }


    /// Calculate the weighted sum of every layer for one input sample.
    /// The returned Vector has n_layers Array1<f64>.
    /// The ith element is the array of weighted sum of layer i+1
    fn calc_weighted_sum(
        &self, 
        input_k: &ArrayView1<f64>
    ) -> Vec<Array1<f64>>
    {
        let mut output: Vec<Array1<f64>> = Vec::with_capacity(self.n_layers - 1);
        output.push(input_k.dot(&(self.weights[0])) + &(self.bias[0]));
        for l in 1..output.capacity(){
            output.push(&(output[l-1]).dot(&(self.weights[l])) + &(self.bias[l]))
        }
        return output;
    }


    /// Calculate the output of each neurone of every layer for one input sample.
    /// The returned Vector has n_layers Array1<f64>.
    /// We consider the output of the first layer is the input feed to the model.
    /// The ith element is the array of outputs of layer i (index from 0)
    /// For example, if the network has layers' dimensions [ [2], [4], [3] ]
    /// then the effective outputs of all layers have dimension accordingly [ [2], [4], [3] ]
    fn feed_forward(
        &self,
        input_k: &ArrayView1<f64>
    ) -> Vec<Array1<f64>>
    {
        let mut output: Vec<Array1<f64>> = Vec::with_capacity(self.n_layers);
        output.push(input_k.to_owned());
        // println!("{}", input_k);
        let weighted_sum = self.calc_weighted_sum(input_k);
        for l in 1..output.capacity(){
            if l != (self.n_layers-1) || self.classification_mode{
                output.push(weighted_sum[l-1].mapv(|x| x.tanh()));
            }else{
                output.push(weighted_sum[l-1].view().to_owned());
                println!("{:?}", weighted_sum[l-1]);
            }
        }
        return output;
    }



    /// Calculate the value of backward propagation of all layers
    /// The result will be a vector of (n_layers -1) Array1<f64>
    /// The ith array is the backward propagation value of layer i+1
    fn feed_backward(
        &self, 
        output_k: &ArrayView1<f64>,
        effective_outputs: &Vec<Array1<f64>>
    ) -> Vec<Array1<f64>>
    {   
        let mut res: Vec<Array1<f64>> = Vec::with_capacity(effective_outputs.capacity() - 1);
        //calculate backward propagation of the last layer
        //the output of the last hidden layer 
        //delta_L = ([1] - [X_L ^2]) * (X_L - Y)
        if self.classification_mode {
            res.push(&effective_outputs[self.n_layers - 1].mapv( |x| (1.0 - x*x)) * &((&effective_outputs[self.n_layers - 1]) - output_k) );
        }else{
            res.push((&effective_outputs[self.n_layers - 1]) - output_k );
        }
        //calculate backward propagation of all previous layers
        //delta_l = ([1] - [output_l ^2]) * (delta_(l+1) x transpose(W_l))
        for l in (0..self.n_layers-2).rev(){
            // output_one_layer = &effective_outputs[l];
            // println!("output_one_layer = {:?}", output_one_layer);
            // println!("W_l = {:?}", self.weights[l+1]);
            // println!("delta_l = {:?}", res[self.n_layers - 3 - l]);
            res.push(
                &effective_outputs[l+1].mapv(
                |x| (1.0 - x*x)) * (
                    &res[self.n_layers - 3 - l].dot(&(self.weights[l+1]).to_shared().reversed_axes())
                ) 
            );            
        }
        res.reverse();
        return res;
    }



    fn update_weights(
        &mut self,
        effective_outputs: &Vec<Array1<f64>>,
        delta_back: &Vec<Array1<f64>>)
    {
        for l in 0..self.weights.len(){
            // println!("shape effective output {} =  {}", l, effective_outputs[l].dim());
            // println!("shape delta_back {} =  {}", l, delta_back[l].dim());
            self.weights[l] = &self.weights[l] - &(self.learning_rate * (
                effective_outputs[l].to_shared().into_shape((effective_outputs[l].len(), 1)).unwrap().dot(
                    &(delta_back[l].to_shared().into_shape((1, delta_back[l].len())).unwrap())
                )
            ));
            self.bias[l] = &self.bias[l] - &(self.learning_rate * &delta_back[l]);
        }
    }



    pub fn train_mlp(
        &mut self,
        inputs: &Array2<f64>,
        outputs: &Array2<f64>,
        nb_iter: u64)
    {
        if inputs.ncols() != self.n_features || outputs.ncols() != self.n_outputs{
            panic!("Number of features and number of outputs do not conform 
            to attributs of the models.")
        }
        //set random generator
        let mut rng = thread_rng();
        for _i in 0..nb_iter {
            //Take a random example input_k
            let k = rng.gen_range(0, inputs.nrows());
            let input_k = inputs.row(k);
            let output_k = outputs.row(k);
            //Calculate outputs of forward propagation
            let outputs_forward: Vec<Array1<f64>> = self.feed_forward(&input_k);
            //Calculate delta of backward propagation
            let delta_back: Vec<Array1<f64>> = self.feed_backward(&output_k, &outputs_forward);
            //Update Weights
            self.update_weights(&outputs_forward, &delta_back);
        }
        self.n_iters += nb_iter;
    }



    pub fn predict_mlp(
        &self, 
        inputs_test: &Array2<f64>) 
        -> Array2<f64> 
    {
        if inputs_test.ncols() != self.n_features{
            panic!("Number of features in instances to be predicted
            is not compatible.")
        }else{
            let mut pred: Array2<f64> = inputs_test.clone();
            for l in 0..self.n_layers-1{
                pred = (pred.dot(&self.weights[l]) + &self.bias[l]).mapv(|x| x.tanh());
            }
            return if self.classification_mode {pred} else {pred.mapv(|x| x.atanh())};
        }
    }


    pub fn get_n_layers(&self) -> usize{
        self.n_layers
    }

    pub fn get_layer_sizes(&self) -> Vec<usize>{
        self.hidden_layer_sizes
    }

    pub fn get_weights(&self) -> &Vec<Array2<f64>>{
        &self.weights
    }

    pub fn get_bias(&self) -> &Vec<Array1<f64>>{
        &self.bias
    }

    pub fn get_n_features(&self) -> usize {
        self.n_features
    }

    pub fn get_n_outputs(&self) -> usize {
        self.n_outputs
    }

    pub fn get_n_iters(&self) -> u64{
        self.n_iters
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    pub fn set_mode(&mut self, mode: bool){
        self.classification_mode = mode
    }

    pub fn get_mode(&self) -> bool{
        self.classification_mode
    }

    pub fn del_mlp(ptr_model: *mut MLP){
        if ptr_model.is_null() {
            return;
        }

        unsafe { Box::from_raw(ptr_model);}
    }
}


