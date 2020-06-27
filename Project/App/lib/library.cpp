#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <iomanip>
#include <thread>
#include <chrono>

using namespace Eigen;
using namespace std;

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// Struct MLP première version
struct perceptron{
    double* weight;
    double value;
    double gradient;
    int weight_size;
} typedef perceptron;

struct layer{
    perceptron* perceptrons;
    int perceptrons_size;
} typedef layer;

struct MLP{
    layer* layers;
    int layers_size;
} typedef MLP;

// Struct MLP tirée de la version Python de M.Vidal
struct My_MLP{
    int nb_layers;
    int* layers;
    double*** w;
    double** deltas;
    double** x;
    int L;
} typedef My_MLP;

extern "C" {
    //-- Debut Linear ----------------------------------------------------------------------------------------------------------------------------
    DLLEXPORT double *linear_create_model(int nb_features) {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        auto w = new double[nb_features + 1];
        for (auto i = 0; i < nb_features + 1; i++) {
            w[i] = dist(mt);
        }
        return w;
    }

    DLLEXPORT double linear_predict_model_regression(const double *model, const double *inputs, int inputs_size) {
        auto sum = model[0];
        for (auto i = 0; i < inputs_size; i++) {
            sum += model[i + 1] * inputs[i];
        }
        return sum;
    }

    DLLEXPORT void linear_train_model_regression(
            double *model,
            double *dataset_inputs,
            double *dataset_expected_outputs,
            int dataset_samples_count,
            int dataset_sample_features_count
    ) {
        auto inputs_matrix = Eigen::Map<MatrixXd>(dataset_inputs, dataset_samples_count, dataset_sample_features_count);

        MatrixXd mat(inputs_matrix.rows(),inputs_matrix.cols() + 1);

        for(auto i = 0; i < mat.rows(); i++){
            mat(i, 0) = 1;
            for(auto j = 1; j < mat.cols(); j++){
                mat(i, j) = inputs_matrix(i, j - 1);
            }
        }

        auto outputs_matix = Eigen::Map<MatrixXd>(dataset_expected_outputs, dataset_samples_count, 1);
        auto result = ((mat.transpose() * mat).inverse() * mat.transpose()) * outputs_matix;
        auto r = new double[dataset_sample_features_count + 1];

        Eigen::Map<MatrixXd>(r, result.rows(), result.cols()) = result;

        for(auto i = 0; i < dataset_sample_features_count + 1; i++){
            model[i] = r[i];
        }
        delete[] r;
    }

    DLLEXPORT double linear_predict_model_classification(const double *model, const double *inputs, int inputs_size) {
        auto sum = linear_predict_model_regression(model, inputs, inputs_size);
        auto return_val =  sum >= 0 ? 1.0 : -1.0;
        return return_val;
    }

    DLLEXPORT void linear_train_model_classification(
            double *model,
            const double *dataset_inputs,
            const double *dataset_expected_outputs,
            int dataset_samples_count,
            int dataset_sample_features_count,
            double alpha,
            int iteration_count
    ) {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, dataset_samples_count - 1);

        for (auto it = 0; it < iteration_count; it++) {
            auto k = dist(mt);
            auto inputs_k = dataset_inputs + k * dataset_sample_features_count;

            auto expected_output_k = dataset_expected_outputs[k];

            auto predicted_output_k = linear_predict_model_classification(model, inputs_k, dataset_sample_features_count);

            auto semi_grad = alpha * (expected_output_k - predicted_output_k);
            for (auto i = 0; i < dataset_sample_features_count; i++) {
                model[i + 1] += semi_grad * inputs_k[i];
            }
            model[0] += semi_grad * 1.0;
        }
    }

    DLLEXPORT void linear_dispose_model(const double *model) {
        delete[] model;
    }

    //-- Fin Linear ----------------------------------------------------------------------------------------------------------------------------------




    //-- Debut MyMLP --------------------------------------------------------------------------------------------------------------------------

    double* _mlp_predict_common(My_MLP* mlp, double* inputs, bool classification_mode){

        for(auto j = 1; j < mlp->layers[0] + 1; j++){
            mlp->x[0][j] = inputs[j-1];
        }
        for(auto l = 1; l < mlp->L + 1; l++){
            for(auto j = 1; j < mlp->layers[l] + 1; j++){
                auto sum = 0.0;
                for(auto i = 0; i < mlp->layers[l-1] + 1; i++){
                    sum += mlp->w[l][i][j] * mlp->x[l-1][i];
                }
                if(l != mlp->L || classification_mode){
                    sum = tanh(sum);
                }
                mlp->x[l][j] = sum;
            }
        }
        return mlp->x[mlp->L] + 1;
    }

    void _mlp_train_common(My_MLP* mlp,
            double* dataset_inputs,
            int nb_samples,
            int nb_features,
            double* dataset_expected,
            int nb_features_expected,
            int iterations,
            double alpha,
            bool classification_mode){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, nb_samples - 1);

        for(auto it = 0; it < iterations; it++){
            auto k = dist(mt);
            auto inputs_k = dataset_inputs + k * nb_features;
            auto expectes_k = dataset_expected + k * nb_features_expected;
            _mlp_predict_common(mlp, inputs_k, classification_mode);

            for(auto j = 1; j < mlp->layers[mlp->L] + 1; j++){
                mlp->deltas[mlp->L][j] = mlp->x[mlp->L][j] - expectes_k[j - 1];
                if(classification_mode){
                    mlp->deltas[mlp->L][j] *= 1 - mlp->x[mlp->L][j] * mlp->x[mlp->L][j];
                }
            }

            for(auto l = mlp->L; l > 1; l--){
                for(auto i = 1; i < mlp->layers[l - 1] + 1; i++){
                    auto sum = 0.0;
                    for(auto j = 1; j < mlp->layers[l] + 1; j++){
                        sum += mlp->w[l][i][j] * mlp->deltas[l][j];
                    }
                    sum *= 1 - mlp->x[l-1][i] * mlp->x[l-1][i];
                    mlp->deltas[l-1][i] = sum;
                }
            }

            for(auto l = 1; l < mlp->L + 1; l++){
                for(auto i = 0; i < mlp->layers[l-1] + 1; i++){
                    for(auto j = 1; j < mlp->layers[l] + 1; j++){
                        mlp->w[l][i][j] -= alpha * mlp->x[l-1][i] * mlp->deltas[l][j];
                    }
                }
            }
        }
    }

    DLLEXPORT double* mlp_predict_classification(My_MLP* mlp, double* inputs){
        return _mlp_predict_common(mlp, inputs, true);
    }


    DLLEXPORT double* mlp_predict_regression(My_MLP* mlp, double* inputs){
        return _mlp_predict_common(mlp, inputs, false);
    }

    DLLEXPORT void mlp_train_classification(My_MLP* mlp,
            double* dataset_inputs,
            int nb_samples,
            int nb_features,
            double* dataset_expected,
            int nb_features_expected,
            double alpha,
            int iterations){
        _mlp_train_common(mlp, dataset_inputs, nb_samples, nb_features, dataset_expected, nb_features_expected, iterations, alpha, true);
    }

    DLLEXPORT void mlp_train_regression(My_MLP* mlp,
            double* dataset_inputs,
            int nb_samples,
            int nb_features,
            double* dataset_expected,
            int nb_features_expected,
            double alpha,
            int iterations){
        _mlp_train_common(mlp, dataset_inputs, nb_samples, nb_features, dataset_expected, nb_features_expected, iterations, alpha, false);
    }

    DLLEXPORT My_MLP* mlp_create_model(int* layers, int nb_layer){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        auto mlp = new My_MLP;
        mlp->nb_layers = nb_layer;
        mlp->layers = new int[nb_layer];
        for(auto i = 0; i < nb_layer; i++){
            mlp->layers[i] = layers[i];
        }

        mlp->L = nb_layer - 1;
        mlp->w = new double**[mlp->L + 1];
        for(auto l = 1; l < mlp->L + 1; l++){
            mlp->w[l] = new double*[layers[l-1] + 1];
            for(auto i = 0; i < layers[l-1] + 1; i++){
                mlp->w[l][i] = new double[layers[l] + 1];
                for(auto j = 0; j < layers[l] + 1; j++){
                    mlp->w[l][i][j] = dist(mt);
                }
            }
        }
        mlp->deltas = new double*[mlp->L + 1];
        for(auto l = 1; l < mlp->L+1; l++){
            mlp->deltas[l] = new double[layers[l]+1];
            for(auto j = 0; j < layers[l] + 1; j++){
                mlp->deltas[l][j] = 0.0;
            }
        }
        mlp->x = new double*[mlp->L + 1];
        for(auto l = 0; l < mlp->L + 1; l++){
            mlp->x[l] = new double[layers[l] + 1];
            for(auto j = 0; j < layers[l] + 1; j++){
                mlp->x[l][j] = 1.0;
            }
        }
        return mlp;
    }

    DLLEXPORT void mlp_save_model(My_MLP* model, char* path_char){
        std::ofstream file;
        std::string path = path_char;

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        auto str = oss.str();

        std::string name = "model_mlp_" + str + ".txt";
        auto fullpath = path + "/" + name;
        file.open(fullpath);
        file << model->L + 1 << ";\n";
        // Save Layers
        for(auto l = 0; l < model->L + 1; l++){
            file << model->layers[l] << ";";
        }
        file << "\n";

        //Save deltas
        for(auto l = 1; l < model->L + 1; l++){
            for(auto i = 0; i < model->layers[l] + 1; i++){
                file << fixed << setprecision(15) << model->deltas[l][i] << ";";
            }
            file << "\n";
        }

        //Save weights
        for(auto l = 1; l < model->L + 1; l++){
            for(auto i = 0; i < model->layers[l-1] + 1; i++){
                for(auto j = 1; j < model->layers[l] + 1; j++){
                    file << fixed << setprecision(15) << model->w[l][i][j] << ";";
                }
                if(l == model->L && i == model->layers[l-1]){

                } else {
                    file << "\n";
                }
            }
        }
        file.close();

        std::this_thread::sleep_for (std::chrono::seconds(1));
    }

    DLLEXPORT My_MLP* mlp_load_model(char* path_char){
        std::string path = path_char;
        std::string line;
        ifstream file(path);
        My_MLP* mlp = nullptr;
        std::string delimiter = ";";
        size_t pos = 0;
        std::string token = "1";

        if(file.is_open()){
            if(getline(file, line)){
                token = line.substr(0, line.find(delimiter));
            }
            int layers_size = std::stoi(token);
            auto layers = new int[layers_size];
            layers[0] = 0;
            auto count = 0;
            if(getline(file, line)){
                while ((pos = line.find(delimiter)) != std::string::npos) {
                    token = line.substr(0, pos);
                    layers[count] = std::stoi(token);
                    line.erase(0, pos + delimiter.length());
                    count++;
                }
            }

            mlp = mlp_create_model(layers, layers_size);
            int l = 1;
            int i = 0;
            int j = 0;
            int deltas_layers = 1;
            while(getline(file, line)){
                j = 1;
                while ((pos = line.find(delimiter)) != std::string::npos) {
                    token = line.substr(0, pos);
                    if(deltas_layers < layers_size){
                        mlp->deltas[l][j-1] = std::stof(token);
                    } else {
                        mlp->w[l][i][j] = std::stof(token);
                    }
                    line.erase(0, pos + delimiter.length());
                    j++;
                }
                i++;
                if(i > mlp->layers[l-1]){
                    i = 0;
                    l++;
                }
                if(deltas_layers < layers_size){
                    l++;
                }
                deltas_layers++;
                if(deltas_layers == layers_size){
                    l = 1;
                    i = 0;
                    j = 1;
                }
            }
            file.close();
        }

        return mlp;
    }


    DLLEXPORT void mlp_dispose(My_MLP* mlp){
        for(auto l = 1; l < mlp->L + 1; l++){
            for(auto i = 0; i < mlp->layers[l-1] + 1; i++){
                delete[] mlp->w[l][i];
            }
            delete[] mlp->w[l];
            delete[] mlp->deltas[l];
            delete[] mlp->x[l-1];
        }
        delete[] mlp->x[mlp->L];
        delete[] mlp->w;
        delete[] mlp->deltas;
        delete[] mlp->x;
        delete mlp;
    }

    void mlp_print_result(My_MLP* mlp){
    printf("[");
        for(auto j = 1; j < mlp->layers[mlp->L] + 1; j++){
            printf("%f, ", mlp->x[mlp->L][j]);
        }
        printf("]");
    }
    //-- Fin MyMLP -------------------------------------------------------------------------------------------------
}

int main(){

    auto iteration = 100000;
    auto alpha = 0.001;

    auto nb_samples = 4;
    auto nb_features_sample = 2;
    auto nb_features_expected = 1;

    auto X = new double*[nb_samples];
    X[0] = new double[nb_features_sample];
    X[0][0] = 1.0;
    X[0][1] = 0.0;
    X[1] = new double[nb_features_sample];
    X[1][0] = 0.0;
    X[1][1] = 1.0;
    X[2] = new double[nb_features_sample];
    X[2][0] = 0.0;
    X[2][1] = 0.0;
    X[3] = new double[nb_features_sample];
    X[3][0] = 1.0;
    X[3][1] = 1.0;

    auto Y = new double*[nb_samples];
    Y[0] = new double[nb_features_expected];
    Y[0][0] = 1.0;
    Y[1] = new double[nb_features_expected];
    Y[1][0] = 1.0;
    Y[2] = new double[nb_features_expected];
    Y[2][0] = -1.0;
    Y[3] = new double[nb_features_expected];
    Y[3][0] = -1.0;

    auto X_flattened = new double[nb_samples * nb_features_sample];
    auto Y_flattened = new double[nb_samples * nb_features_expected];

    auto c = 0;
    for(auto i = 0; i < nb_samples; i++){
        for(auto j = 0; j < nb_features_sample; j++){
            X_flattened[c] = X[i][j];
            c++;
        }
    }

    c = 0;
    for(auto i = 0; i < nb_samples; i++){
        for(auto j = 0; j < nb_features_expected; j++){
            Y_flattened[c] = Y[i][j];
            c++;
        }
    }


    auto nb_layer = 3;
    auto layers = new int[nb_layer];
    layers[0] = 2;
    layers[1] = 2;
    layers[2] = 1;

    auto mlp = mlp_create_model(layers, nb_layer);
    auto linear_model = linear_create_model(nb_features_sample);

    linear_train_model_classification(linear_model, X_flattened, Y_flattened, nb_samples, nb_features_sample, alpha, iteration);
    mlp_train_classification(mlp, X_flattened, nb_samples, nb_features_sample, Y_flattened, nb_features_expected, alpha, iteration);


    printf("\n--After train\n");
    for(auto i = 0; i < nb_samples; i++){
        printf("[%f]   ", Y_flattened[i]);

        mlp_predict_classification(mlp, X[i]);
        mlp_print_result(mlp);
        printf("    [%f]\n", linear_predict_model_classification(linear_model, X[i], nb_features_sample));
    }

    mlp_dispose(mlp);
    linear_dispose_model(linear_model);

    return 0;
}
