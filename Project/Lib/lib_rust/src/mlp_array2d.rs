use array2d::Array2D;
use rand::{ Rng, thread_rng};

#[derive(Debug)]
#[repr(C)]
pub struct MLP {
    // loss: f64,
    weights: Vec<Array2D<f64>>,
    bias: Vec<Array2D<f64>>, 
    n_layers: usize,      //nb hidden layers + 2
    layer_sizes: Vec<usize>,
    learning_rate: f64,
    n_iters: u64,
    classification_mode: bool
}

#[allow(dead_code)]
impl MLP{


    fn transpose(x: &Array2D<f64>) -> Array2D<f64>{
        let mut res = Array2D::filled_with(0.0, x.num_columns(), x.num_rows());
        for i in 0..x.num_rows(){
            for j in 0..x.num_columns(){
                res[(j, i)] = x[(i, j)];
            }
        }
        return res;
    }


    fn get_row(x: &Array2D<f64>, nrow: usize) -> Array2D<f64>{
        let mut res : Array2D<f64> = Array2D::filled_with(0.0, 1, x.num_columns());
        let mut i = 0;
        for elem in x.row_iter(nrow){
            res[(0, i)] = *elem;
            i+=1;
        }
        return res;
    }

    
    pub fn calculate_loss_mse(
        &self,
        outputs: &Array2D<f64>,
        pred: &Array2D<f64>
    ) -> f64{
        let nb_samples = outputs.num_rows();
        let nb_outputs = self.get_n_outputs();

        let mut se = 0.0;
        for i in 0..nb_samples{
            let mut moy_dist = 0.0;
            for j in 0..nb_outputs{
                let a = pred[(i, j)] - outputs[(i, j)];
                moy_dist += a*a;
            }
            se += moy_dist / (nb_outputs as f64);
        }
        return se / (nb_samples as f64);
    }

    
    pub fn calculate_accuracy(
        &self, 
        outputs: &Array2D<f64>,
        pred: &Array2D<f64>
    ) -> f64 
    {
        let nb_samples = outputs.num_rows();
        let nb_outputs = self.get_n_outputs();

        let mut nb_success = 0;
        for i in 0..nb_samples{
            let mut max_pred = std::f64::MIN;
            let mut idx_max_pred = nb_outputs+1;
            let mut idx_out = nb_outputs+1;
            for j in 0..nb_outputs{
                if outputs[(i, j)] == 1.{
                    idx_out = j;
                }
                if pred[(i, j)] > max_pred{
                    max_pred = pred[(i, j)];
                    idx_max_pred = j;
                }
                // is_true = is_true && ((outputs[(i, j)] * pred[(i, j)]) >= 0.0);
            }
            if idx_max_pred == idx_out {nb_success += 1};
        }
        return (nb_success as f64) / (nb_samples as f64)
    }



    ///Initialize the weights of all layers in the network
    ///The returned weights will be a Vector containing
    ///(n_layers - 1) Array2D<f64>. The ith element is the 
    ///corresponding weights between layer i and i+1
    fn init_weights(
        layer_sizes: &Vec<usize>
    ) -> Vec<Array2D<f64>>
    {
        let n_layers = layer_sizes.len();
        let mut w: Vec<Array2D<f64>> = Vec::with_capacity(n_layers - 1);
        let mut rng = thread_rng();

        for layer in 0..w.capacity(){
            w.push(Array2D::filled_by_row_major(|| rng.gen_range(-1., 1.), layer_sizes[layer], layer_sizes[layer+1]));
        }
        return w;
    }


    ///Initialize the Vector bias of the entire network. 
    ///The returned bias is a Vector containing (n_layers - 1)
    ///Array2D<f64>. The ith element is the corresponding bias
    ///between layer i and i+1 of dimension (1, d(l+1))
    fn init_bias(
        layer_sizes: &Vec<usize>) 
        -> Vec<Array2D<f64>>
    {        
        let n_layers = layer_sizes.len();
        let mut rng = thread_rng();
        let mut b: Vec<Array2D<f64>> = Vec::with_capacity(n_layers-1);
        for l in 0..b.capacity(){
            b.push(Array2D::filled_by_row_major(|| rng.gen_range(-1., 1.), 1, layer_sizes[l+1]));
        }
        return b;
    }


    pub fn init_mlp(
        layer_sizes: Vec<usize>, 
        learning_rate: f64,
        classification_mode: bool
    )-> MLP
    {
        MLP{
            // loss: 
            weights: MLP::init_weights(&layer_sizes),
            bias: MLP::init_bias(&layer_sizes),
            n_layers: layer_sizes.len(),
            layer_sizes: layer_sizes.clone(),
            learning_rate,
            classification_mode,
            n_iters: 0
        }
    }


    
    /// Calculate the output of each neurone of every layer for one input sample.
    /// The returned Vector has n_layers Array2D<f64>.
    /// We consider the output of the first layer is the input feed to the model.
    /// The ith element is the array of outputs of layer i (index from 0)
    /// For example, if the network has 3 layers of dimensions [ [2], [4], [3] ]
    /// then the effective outputs of all layers have dimension accordingly [ [2], [4], [3] ]
    fn feed_forward(
        &self,
        input_k: &Array2D<f64>
    ) -> Vec<Array2D<f64>>
    {
        let mut output: Vec<Array2D<f64>> = Vec::with_capacity(self.n_layers);
        output.push(input_k.to_owned());
        for l in 1..self.n_layers{
            output.push(Array2D::filled_with(0.0, 1, self.layer_sizes[l]));
            for j in 0..self.layer_sizes[l]{
                let mut _res = 0.0;
                for i in 0..self.layer_sizes[l-1]{
                    _res += output[l-1][(0, i)] * self.weights[l-1][(i, j)];
                }
                _res += self.bias[l-1][(0, j)];
                if l != self.n_layers-1 || self.classification_mode{
                    _res = _res.tanh();
                }
                output[l][(0, j)] = _res;
            }
        }
        return output;
    }



    /// Calculate the value of backward propagation of all layers
    /// The result will be a vector of (n_layers -1) Array2D<f64>
    /// The ith array is the backward propagation value of layer i+1
    fn feed_backward(
        &self, 
        output_k: &Array2D<f64>,
        effective_outputs: &Vec<Array2D<f64>>
    ) -> Vec<Array2D<f64>>
    {   
        let mut res: Vec<Array2D<f64>> = Vec::with_capacity(self.n_layers - 1);
        //calculate backward propagation of the last layer
        //the output of the last hidden layer 
        //delta_L = ([1] - [X_L ^2]) * (X_L - Y)
        res.push(Array2D::filled_with(0.0, 1, self.layer_sizes[self.n_layers-1]));
        for j in 0..self.layer_sizes[self.n_layers-1] {
            res[0][(0, j)] = effective_outputs[self.n_layers-1][(0, j)] - output_k[(0, j)];
            if self.classification_mode{
                res[0][(0, j)] *= 1. - effective_outputs[self.n_layers-1][(0, j)] * effective_outputs[self.n_layers-1][(0, j)];
            }
        }

        //calculate backward propagation of all previous layers
        //delta_l = ([1] - [output_l ^2]) * (delta_(l+1) x transpose(W_l))
        for l in (1..self.n_layers-1).rev(){
            res.push(Array2D::filled_with(0.0, 1, self.layer_sizes[l]));
            for i in 0..self.layer_sizes[l]{
                let mut _res = 0.0;
                for j in 0..self.layer_sizes[l+1]{
                    _res += self.weights[l][(i, j)] * res[self.n_layers-2-l][(0, j)];
                }
                _res *= 1.0 - effective_outputs[l][(0, i)] * effective_outputs[l][(0, i)];
                res[self.n_layers-1-l][(0, i)] = _res;
            }
        }
        res.reverse();
        return res;
    }


    fn update_weights(
        &mut self,
        effective_outputs: &Vec<Array2D<f64>>,
        delta_back: &Vec<Array2D<f64>>)
    {
        for l in 0..self.weights.len(){
            // println!("shape effective output {} =  {}", l, effective_outputs[l].dim());
            // println!("shape delta_back {} =  {}", l, delta_back[l].dim());
            for j in 0..self.weights[l].num_columns(){
                for i in 0..self.weights[l].num_rows(){
                    self.weights[l][(i, j)] -= self.learning_rate * effective_outputs[l][(0, i)] * delta_back[l][(0, j)];
                }
                self.bias[l][(0, j)] -= self.learning_rate * delta_back[l][(0, j)];
            }
        }
    }



    pub fn train_mlp(
        &mut self,
        inputs: &Array2D<f64>,
        outputs: &Array2D<f64>,
        nb_iters: u64)
    {
        if inputs.num_columns() != self.get_n_features() || outputs.num_columns() != self.get_n_outputs(){
            panic!("Number of features and number of outputs do not conform 
            to attributs of the models.")
        }
        //set random generator
        let mut rng = thread_rng();
        for _i in 0..nb_iters {
            //Take a random example input_k
            let k = rng.gen_range(0, inputs.num_rows());
            let input_k = MLP::get_row(inputs, k);
            let output_k = MLP::get_row(outputs, k);
            let outputs_forward: Vec<Array2D<f64>> = self.feed_forward(&input_k);
            let delta_back: Vec<Array2D<f64>> = self.feed_backward(&output_k, &outputs_forward);
            self.update_weights(&outputs_forward, &delta_back);
        }
        self.n_iters += nb_iters;
    }


    /// train self and also return loss and accuracy on train dataset
    /// and on validation dataset.
    /// Return: nb_iterss * [loss_train, acc_train, loss_val, acc_val]
    pub fn train_mlp_return_metrics(
        &mut self,
        inputs: &Array2D<f64>,
        outputs: &Array2D<f64>,
        inputs_val: &Array2D<f64>,
        outputs_val: &Array2D<f64>,
        nb_iters: u64)
        -> Array2D<f64>
    {
        if inputs.num_columns() != self.get_n_features() || outputs.num_columns() != self.get_n_outputs(){
            panic!("Number of features and number of outputs do not conform 
            to attributs of the models.")
        }
        if inputs_val.num_columns() != self.get_n_features() || outputs_val.num_columns() != self.get_n_outputs(){
            panic!("Number of features and number of outputs do not conform 
            to attributs of the models.")
        }
        let mut metrics: Array2D<f64> = Array2D::filled_with(0.0, nb_iters as usize, 4);
        //set random generator
        let mut rng = thread_rng();
        for _i in 0..nb_iters {
            //Take a random example input_k
            let k = rng.gen_range(0, inputs.num_rows());
            let input_k = MLP::get_row(inputs, k);
            let output_k = MLP::get_row(outputs, k);
            let outputs_forward: Vec<Array2D<f64>> = self.feed_forward(&input_k);
            let delta_back: Vec<Array2D<f64>> = self.feed_backward(&output_k, &outputs_forward);
            self.update_weights(&outputs_forward, &delta_back);
            let y_pred = self.predict_mlp(inputs);
            let y_val_pred = self.predict_mlp(inputs_val);
            metrics[(_i as usize, 0)] = self.calculate_loss_mse(outputs, &y_pred);
            metrics[(_i as usize, 1)] = self.calculate_accuracy(outputs, &y_pred);
            metrics[(_i as usize, 2)] = self.calculate_loss_mse(outputs_val, &y_val_pred);
            metrics[(_i as usize, 3)] = self.calculate_accuracy(outputs_val, &y_val_pred);
        }

        self.n_iters += nb_iters;
        return metrics;
    }



    /// train model with one epoch = run through all samples once 
    /// return loss and accuracy on train dataset and on validation dataset.
    /// Return: [loss_train, acc_train, loss_val, acc_val]
    pub fn train_mlp_epoch(
        &mut self,
        inputs: &Array2D<f64>,
        outputs: &Array2D<f64>,
        inputs_val: &Array2D<f64>,
        outputs_val: &Array2D<f64>)
        -> Vec<f64>
    {
        if inputs.num_columns() != self.get_n_features() || outputs.num_columns() != self.get_n_outputs(){
            panic!("Number of features and number of outputs do not conform 
            to attributs of the models.")
        }
        if inputs_val.num_columns() != self.get_n_features() || outputs_val.num_columns() != self.get_n_outputs(){
            panic!("Number of features and number of outputs do not conform 
            to attributs of the models.")
        }
        let mut metrics: Vec<f64> = vec![0.0; 4];
        for k in 0..inputs.num_rows() {
            let input_k = MLP::get_row(inputs, k);
            let output_k = MLP::get_row(outputs, k);
            let outputs_forward: Vec<Array2D<f64>> = self.feed_forward(&input_k);
            let delta_back: Vec<Array2D<f64>> = self.feed_backward(&output_k, &outputs_forward);
            self.update_weights(&outputs_forward, &delta_back);
        }
        let y_pred = self.predict_mlp(inputs);
        let y_val_pred = self.predict_mlp(inputs_val);
        metrics[0] = self.calculate_loss_mse(outputs, &y_pred);
        metrics[1] = self.calculate_accuracy(outputs, &y_pred);
        metrics[2] = self.calculate_loss_mse(outputs_val, &y_val_pred);
        metrics[3] = self.calculate_accuracy(outputs_val, &y_val_pred);

        self.n_iters += inputs.num_rows() as u64;
        return metrics;
    }



    pub fn predict_mlp(
        &self, 
        inputs_test: &Array2D<f64>
    )
        -> Array2D<f64> 
    {
        if inputs_test.num_columns() != self.get_n_features(){
            panic!("Number of features in instances to be predicted
            is not compatible.")
        }else{
            let mut pred: Array2D<f64> = Array2D::filled_with(0.0, inputs_test.num_rows(), self.get_n_outputs());
            for k in 0.. inputs_test.num_rows(){
                let _outputs = self.feed_forward(&MLP::get_row(inputs_test, k));
                // if convert {
                //     let mut max: f64 = _outputs[self.n_layers-1][(0, 0)];
                //     let mut idmax = 0;
                //     for j in 1.._outputs[self.n_layers-1].num_columns(){
                //         if _outputs[self.n_layers-1][(0, j)] > max{
                //             max = _outputs[self.n_layers-1][(0, j)];
                //             idmax = j;
                //         }
                //     }
                //     for j in 0.._outputs[self.n_layers-1].num_columns(){
                //         if j == idmax{
                //             pred[(k, j)] = 1.;
                            
                //         }else{
                //             pred[(k, j)] = -1.;
                //         }
                //     }
                // }else{
                //     for j in 0.._outputs[self.n_layers-1].num_columns(){
                //         pred[(k, j)] = _outputs[self.n_layers-1][(0, j)];
                //     }
                // }
                for j in 0.._outputs[self.n_layers-1].num_columns(){
                    pred[(k, j)] = _outputs[self.n_layers-1][(0, j)];
                }
            }
            return pred;
        }
    }

    pub fn get_n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn get_layer_sizes(&self) -> &Vec<usize>{
        &self.layer_sizes
    }

    pub fn get_weights(&self) -> &Vec<Array2D<f64>>{
        &self.weights
    }

    pub fn get_bias(&self) -> &Vec<Array2D<f64>>{
        &self.bias
    }

    pub fn get_n_features(&self) -> usize {
        self.layer_sizes[0]
    }

    pub fn get_n_outputs(&self) -> usize {
        self.layer_sizes[self.get_n_layers() - 1]
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


