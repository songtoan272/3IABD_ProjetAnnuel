#include <random>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <thread>
#include <chrono>
#include <math.h>

#include "src/alglibinternal.h"
#include "src/dataanalysis.h"

using namespace alglib;
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

    DLLEXPORT void linear_save_model(double* model, int nb_features, char* path_char){
        std::ofstream file;
        std::string path = path_char;

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        auto str = oss.str();

        std::string name = "model_linear_" + str + ".txt";
        auto fullpath = path + "/" + name;
        file.open(fullpath);
        file << nb_features << "\n";
        for(auto j = 0; j < nb_features + 1; j++){
            file << model[j] << ";";
        }

        file.close();

        std::this_thread::sleep_for (std::chrono::seconds(1));
    }

    DLLEXPORT double* linear_load_model(char* path_char){
        std::string path = path_char;
        std::string line;
        ifstream file(path);
        double* linear = nullptr;
        std::string delimiter = ";";
        size_t pos = 0;
        std::string token = "1";

        if(file.is_open()){
            int nb_features = 0;
            if(getline(file, line)){
                nb_features = std::stoi(line);
            }
            linear = linear_create_model(nb_features);
            if(getline(file, line)){
                int i = 0;
                while ((pos = line.find(delimiter)) != std::string::npos) {
                    token = line.substr(0, pos);
                    linear[i] = std::stof(token);
                    i++;
                    line.erase(0, pos + delimiter.length());
                }
            }
            file.close();
        }

        return linear;
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

    DLLEXPORT double* mlp_get_metrics(My_MLP* mlp, double* dataset, int nb_sample, int nb_features, double* expected_outputs, bool mode){
        int nb_sorti = mlp->layers[mlp->L];
        auto result = new double[2 + (nb_sorti * nb_sorti)];
        result[1] = 0;
        for(auto i = 0; i < 2 + (nb_sorti * nb_sorti); i++){
            result[i] = 0;
        }
        double s = 0.0;
        int my_model_work = 0;
        for (auto i = 0; i < nb_sample; i++){
            auto p = _mlp_predict_common(mlp, dataset + i * nb_features, mode);
            double f = p[0];
            int o = 0;
            auto a = 0.0;
            int o_p = 0;
            auto a_p = 0.0;
            for(auto j = 0; j < nb_sorti; j++){
                a += (p[j] - expected_outputs[i * nb_sorti + j]) * (p[j] - expected_outputs[i * nb_sorti + j]);
                if(p[j] > f){
                    o = j;
                    f = p[j];
                }
                if(expected_outputs[i * nb_sorti + j] > 0){
                    o_p = j;
                }
            }
            result[o_p * nb_sorti + o + 2]++;

            if(expected_outputs[i * nb_sorti + o] == 1.0){
                result[1]++;
            }
            s += a / nb_sorti;
        }
        result[0] = s / nb_sample;
        result[1] = result[1] / nb_sample;
        return result;
    }

    void _mlp_train_common(My_MLP* mlp,
            double* dataset_inputs,
            int nb_samples,
            int nb_features,
            double* dataset_expected,
            int nb_features_expected,
            int iterations,
            double alpha,
            bool classification_mode,
            bool like_keras=false){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, nb_samples - 1);
        for(auto it = 0; it < iterations; it++){
            int nb_it = 1;
            if(like_keras){
                nb_it = nb_samples;
            }
            for(auto p = 0; p < nb_it; p++){
                auto k = dist(mt);
                auto inputs_k = dataset_inputs + k * nb_features;
                auto expectes_k = dataset_expected + k * nb_features_expected;
                if(like_keras){
                    inputs_k = dataset_inputs + p * nb_features;
                    expectes_k = dataset_expected + p * nb_features_expected;
                }
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
    }

    DLLEXPORT double* train_with_retrieve_metrics(
            My_MLP* mlp,
            double* dataset_inputs,
            int nb_samples,
            int nb_features,
            double* dataset_expected,
            int nb_features_expected,
            double alpha,
            bool classification_mode,
            bool like_keras){
        _mlp_train_common(mlp,dataset_inputs,nb_samples,nb_features,dataset_expected,nb_features_expected,1,alpha,classification_mode,like_keras);
        auto loss = mlp_get_metrics(mlp, dataset_inputs, nb_samples, nb_features, dataset_expected, classification_mode);
        return loss;
    }

    DLLEXPORT double* mlp_predict_classification(My_MLP* mlp, double* inputs){
        return _mlp_predict_common(mlp, inputs, true);
    }


    DLLEXPORT double* mlp_predict_regression(My_MLP* mlp, double* inputs){
        return _mlp_predict_common(mlp, inputs, false);
    }

    DLLEXPORT void mlp_train_classification(
            My_MLP* mlp,
            double* dataset_inputs,
            int nb_samples,
            int nb_features,
            double* dataset_expected,
            int nb_features_expected,
            double alpha,
            int iterations,
            bool like_keras=false){
        _mlp_train_common(mlp, dataset_inputs, nb_samples, nb_features, dataset_expected, nb_features_expected, iterations, alpha, true, like_keras);
    }

    DLLEXPORT void mlp_train_regression(My_MLP* mlp,
            double* dataset_inputs,
            int nb_samples,
            int nb_features,
            double* dataset_expected,
            int nb_features_expected,
            double alpha,
            int iterations,
            bool like_keras=false){
        _mlp_train_common(mlp, dataset_inputs, nb_samples, nb_features, dataset_expected, nb_features_expected, iterations, alpha, false, like_keras);
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


    struct rbf{
        double** w;
        double* dataset;
        int nb_sample;
        int nb_feature;
        int nb_outputs;
        double gamma;
        double* result;
    } typedef rbf;


    //-- Debut RBF -------------------------------------------------------------------------------------------------

    DLLEXPORT rbf* rbf_create_model(){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        auto my_rbf = new rbf;
        my_rbf->w = new double*[1];
        my_rbf->gamma = 0.0;
        my_rbf->nb_sample = 1;
        my_rbf->nb_outputs = 1;
        my_rbf->result = new double[1];
        my_rbf->dataset = new double[1];

        for(auto i = 0; i < 1; i++){
            my_rbf->w[i] = new double[1];
        }

        return my_rbf;
    }

    double _get_distance_norme_two(int features, double* x, double* y){
        double s = 0;
        for(auto i = 0; i < features; i++){
            s += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return sqrt(abs(s));
    }

    double* _get_k_means(double* dataset, int nb_samples, int nb_features, int nb_clusters){

        auto representant = new double[nb_clusters * nb_features];
        auto representant_save = new double[nb_clusters * nb_features];
        auto sample_cluster = new int[nb_samples];
        auto count_cluster = new int[nb_clusters];

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, nb_samples);

        auto potential_points = new double[nb_samples];
        for(auto i = 0; i < nb_samples; i++){
            potential_points[i] = 0;
        }

        int c = 0;
        int o = 0;
        // Tirage au sort sur les points du dataset de representants
        for(auto i = 0; i < nb_clusters; i++){
            c = dist(mt);
            if(potential_points[c] == 0){
                potential_points[c] = 1;
            } else {
                while(potential_points[c] == 1){
                    c++;
                    c = c > nb_samples ? 0 : c;
                }
            }
            for(auto j = 0; j < nb_features; j++){
                representant[o] = (dataset + c * nb_features)[j];
                representant_save[o] = (dataset + c * nb_features)[j];
                o++;
            }
        }

        int _it = 0;
        bool same_state = false;

        // Tant que la situation n'est pas stable ou que + de 200 loops
        while(_it < 200 && !same_state){
            for(auto i = 0; i < nb_clusters; i++){
                count_cluster[i] = 0;
            }
            //Assignement de chaque points a son cluster le plus proche
            for(auto i = 0; i < nb_samples; i++){
                double min_distance = _get_distance_norme_two(nb_features, dataset + i * nb_features, representant);
                int cluster = 0;
                for(auto j = 0; j < nb_clusters; j++){
                    auto distance = _get_distance_norme_two(nb_features, dataset + i * nb_features, representant + j * nb_features);
                    if(distance < min_distance){
                        min_distance = distance;
                        cluster = j;
                    }
                }
                sample_cluster[i] = cluster;
                count_cluster[cluster]++;
            }

            //Reset des position des representatn
            for(auto i = 0; i < nb_clusters * nb_features; i++){
                representant[i] = 0;
            }

            //Somme des positions de chaque features des individus d'un meme cluster
            for(auto i = 0; i < nb_samples; i++){
                for(auto j = 0; j < nb_features; j++){
                    (representant + sample_cluster[i] * nb_features)[j] += (dataset + i * nb_features)[j];
                }
            }

            //Moyenne des positions
            for(auto i = 0; i < nb_clusters; i++){
                for(auto j = 0; j < nb_features; j++){
                    (representant + i * nb_features)[j] /= count_cluster[i];
                }
            }

            //Vérification que la situation à changé
            same_state = true;
            c = 0;
            for(auto i = 0; i < nb_clusters; i++){
                for(auto j = 0; j < nb_features; j++){
                    if(abs(representant[c] - representant_save[c]) > 0.001){
                        same_state = false;
                    }
                    representant_save[c] = representant[c];
                    c++;
                }
            }
            _it++;
        }

        delete[] representant_save;
        delete[] sample_cluster;
        delete[] count_cluster;

        return representant;
    }

    double* _rbf_copy_dataset(double* dataset, int nb_sample, int nb_feature){
        auto r = new double[nb_sample * nb_feature];
        for(auto i = 0; i < nb_sample * nb_feature; i++){
            r[i] = dataset[i];
        }
        return r;
    }

    DLLEXPORT void rbf_train(rbf* rbf,
        double* dataset_inputs,
        int nb_samples,
        int nb_features,
        double* dataset_expected,
        int nb_features_expected,
        int nb_clusters,
        double gamma
    ){
        rbf->gamma = gamma;
        rbf->nb_feature = nb_features;

        for(auto i = 0; i < rbf->nb_sample; i++){
            delete[] rbf->w[i];
        }
        delete[] rbf->w;
        delete[] rbf->dataset;

        rbf->nb_sample = nb_clusters > 1 ? nb_clusters : nb_samples;
        rbf->dataset = nb_clusters > 1 ? _get_k_means(dataset_inputs, nb_samples, nb_features, nb_clusters) : _rbf_copy_dataset(dataset_inputs, rbf->nb_sample, rbf->nb_feature);

        rbf->w = new double*[rbf->nb_sample];
        for(auto i = 0; i < rbf->nb_sample; i++){
            rbf->w[i] = new double[rbf->nb_outputs];
        }

        auto m = new double[nb_samples * rbf->nb_sample];

        int c = 0;
        for(auto i = 0; i < nb_samples; i++){
            auto x = dataset_inputs + i * nb_features;
            for(auto j = 0; j < rbf->nb_sample; j++){
                auto y = rbf->dataset + j * nb_features;
                auto distance = _get_distance_norme_two(nb_features, x, y);
                m[c] = exp((0 - rbf->gamma) *  pow(distance,2));
                c++;
            }
        }

        Eigen::MatrixXd m_matrix;
        Eigen::MatrixXd w;

        auto y_matrix = Eigen::Map<MatrixXd>(dataset_expected, nb_samples, nb_features_expected);

        if(nb_clusters > 1){
            m_matrix = Eigen::Map<MatrixXd>(m, nb_samples, nb_clusters);
            w = (m_matrix.transpose() * m_matrix).inverse() * m_matrix.transpose() * y_matrix;
        } else {
            m_matrix = Eigen::Map<MatrixXd>(m, nb_samples, nb_samples);
            w = m_matrix.inverse() * y_matrix;
        }

        for(auto i = 0; i < rbf->nb_sample; i++){
            for(auto j = 0; j < rbf->nb_outputs; j++){
                rbf->w[i][j] = w(i, j);
            }
        }
        delete[] m;
    }

    void _rbf_predict_common(rbf* rbf, double* input){
        for(auto i = 0; i < rbf->nb_outputs; i++){
            rbf->result[i] = 0.0;
        }

        for(auto i = 0; i < rbf->nb_sample; i++){
            auto distance = _get_distance_norme_two(rbf->nb_feature, input, rbf->dataset + i * rbf->nb_feature);
            for(auto j = 0; j < rbf->nb_outputs; j++){
                rbf->result[j] += rbf->w[i][j] * exp((0 - rbf->gamma) * pow(distance, 2));
            }
        }
    }

    DLLEXPORT double* rbf_predict_classification(rbf* rbf, double* input){
        _rbf_predict_common(rbf, input);
        if(rbf->nb_outputs == 1){
            rbf->result[0] = rbf->result[0] > 0.0 ? 1.0 : -1.0;
        } else {
            double max = rbf->result[0];
            int i = 0;
            for(auto j = 0; j < rbf->nb_outputs; j++){
                if(rbf->result[j] > max){
                    max = rbf->result[j];
                    i = j;
                }
            }
            for(auto j = 0; j < rbf->nb_outputs; j++){
                rbf->result[j] = i == j ? 1.0 : -1.0;
            }
        }

        return rbf->result;
    }

    DLLEXPORT double* rbf_predict_regression(rbf* rbf, double* input){
        _rbf_predict_common(rbf, input);
        return rbf->result;
    }

    DLLEXPORT double* rbf_get_clusters(rbf* rbf){
        return rbf->dataset;
    }

    DLLEXPORT void rbf_dispose(rbf* rbf){
        for(auto i = 0; i < rbf->nb_sample; i++){
            delete[] rbf->w[i];
        }
        delete[] rbf->w;
        delete[] rbf->result;
        delete[] rbf->dataset;
        delete rbf;
    }

    DLLEXPORT void rbf_save_model(rbf* rbf, char* path_char){
        std::ofstream file;
        std::string path = path_char;

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        auto str = oss.str();

        std::string name = "model_rbf_" + str + ".txt";
        auto fullpath = path + "/" + name;
        file.open(fullpath);


        file << rbf->nb_sample << ";" << rbf->nb_feature << ";" << rbf->nb_outputs << ";" << rbf->gamma << ";" << "\n";

        for(auto i = 0; i < rbf->nb_sample; i++){
            for(auto j = 0; j < rbf->nb_feature; j++){
                file << fixed << setprecision(15) << rbf->dataset[i * rbf->nb_feature + j] << ";";
            }
            file << "\n";
        }
        for(auto i = 0; i < rbf->nb_sample; i++){
            for(auto j = 0; j < rbf->nb_outputs; j++){
                file << fixed << setprecision(15) << rbf->w[i][j] << ";";
            }
            file << "\n";
        }
        file.close();

        std::this_thread::sleep_for (std::chrono::seconds(1));
    }

    DLLEXPORT rbf* rbf_load_model(char* path_char){
        std::string path = path_char;
        std::string line;
        ifstream file(path);
        rbf* rbf = nullptr;
        std::string delimiter = ";";
        size_t pos = 0;
        std::string token = "1";

        if(file.is_open()){
            auto data = new double[5];
            if(getline(file, line)){
                int count = 0;
                while ((pos = line.find(delimiter)) != std::string::npos) {
                    token = line.substr(0, pos);
                    data[count] = std::stof(token);
                    line.erase(0, pos + delimiter.length());
                    count++;
                }
            }

            rbf = rbf_create_model();
            rbf->nb_sample = data[0];
            rbf->nb_feature = data[1];
            rbf->nb_outputs = data[2];
            rbf->gamma = data[3];

            int count = 0;
            for(auto i = 0; i < rbf->nb_sample; i++){
                if(getline(file, line)){
                    while ((pos = line.find(delimiter)) != std::string::npos) {
                        token = line.substr(0, pos);
                        rbf->dataset[count] = std::stof(token);
                        line.erase(0, pos + delimiter.length());
                        count++;
                    }
                }
            }

            for(auto i = 0; i < rbf->nb_sample; i++){
                if(getline(file, line)){
                    count = 0;
                    while ((pos = line.find(delimiter)) != std::string::npos) {
                        token = line.substr(0, pos);
                        rbf->w[i][count] = std::stof(token);
                        line.erase(0, pos + delimiter.length());
                        count++;
                    }
                }
            }
            file.close();
        }
        return rbf;
    }


    //-- Fin RBF -------------------------------------------------------------------------------------------------



    //-- Debut SVM -----------------------------------------------------------------------------------------------

    struct SVM{
        double* W;
        int w_size;
    } typedef SVM;

    DLLEXPORT SVM* svm_create_model(int nb_features){
        auto model = new SVM;
        model->W = new double[nb_features + 1];
        return model;
    }

    double kernel_trick(double* X, double* Y, int nb_features){
        double x_scalaire = 0.0;
        double y_scalaire = 0.0;
        double x_y_scalaire = 0.0;
        for(auto i = 0; i < nb_features; i++){
            x_scalaire += pow(0-X[i], 2);
            y_scalaire += pow(0-Y[i], 2);
            x_y_scalaire += X[i] * Y[i];
        }
        double s = exp(x_scalaire) * exp(y_scalaire) * exp(2 * x_y_scalaire);
        return s;
    }

    DLLEXPORT void svm_train_model(SVM* model, double* inputs, int nb_inputs, int nb_features, double* labels, bool use_kernel_trick=false){

        auto linear = new double[nb_inputs];
        for(auto i = 0; i < nb_inputs; i++){
            linear[i] = -1;
        }

        real_1d_array lbnd;
        real_1d_array ubnd;
        integer_1d_array ct;
        ct.setlength(nb_inputs + 1);
        lbnd.setlength(nb_inputs);
        ubnd.setlength(nb_inputs);

        double ma = numeric_limits<float>::max();
        auto constraint = new double[(nb_inputs + 1) * (nb_inputs + 1)];
        int o = 0;
        for(auto i = 0; i < nb_inputs; i++){
            for(auto j = 0; j < nb_inputs; j++){
                if(i == j){
                    constraint[o] = 1.0;
                } else {
                    constraint[o] = 0.0;
                }
                o++;
            }
            constraint[o] = 0.0;
            ct[i] = 1;
            o++;
        }
        /*
        int o = 0;
        auto constraint = new double[nb_inputs + 1];*/
        for(int i = 0; i < nb_inputs; i++){
            lbnd[i] = 0.0;
            ubnd[i] = ma;
            constraint[o] = labels[i];
            o++;
        }
        constraint[o] = 0.0;
        ct[nb_inputs] = 0;

        auto kernel = new double[nb_inputs * nb_inputs];
        o = 0;
        for(auto i = 0; i < nb_inputs; i++){
            for(auto j = 0; j < nb_inputs; j++){
                auto s = 0.0;
                if(use_kernel_trick){
                    s = kernel_trick(inputs + i * nb_features, inputs + j * nb_features, nb_features);
                } else {
                    for(auto l = 0; l < nb_features; l++){
                        s += inputs[i * nb_features + l] * inputs[j * nb_features + l];
                    }
                }
                kernel[o] = labels[i] * labels[j] * s;
                o++;
            }
        }

        real_2d_array a;
        a.setcontent(nb_inputs, nb_inputs, kernel);
        real_1d_array b;
        b.setcontent(nb_inputs, linear);
        real_2d_array c;
        c.setcontent(nb_inputs + 1, nb_inputs + 1, constraint);

        real_1d_array x;
        minqpstate state;
        minqpreport rep;

        minqpcreate(nb_inputs, state);
        minqpsetquadraticterm(state, a);
        minqpsetlinearterm(state, b);
        //minqpsetbc(state, lbnd, ubnd);
        minqpsetlc(state, c, ct);

        minqpsetalgobleic(state, 0, 0, 0.01, 0);
        minqpoptimize(state);
        minqpresults(state, x, rep);

        int m = 0;
        auto W = new double[nb_features+1];
        for(auto i = 0; i < nb_features + 1; i++){
            W[i] = 0.0;
        }
        auto max = x[0];

        for(auto i = 0; i < nb_inputs; i++){
            if(x[i] > max){
                m = i;
                max = x[i];
            }
            x[i] = abs(x[i]) < 0.01 ? 0.0 : x[i];
            cout << x[i] << endl;
            for(auto j = 0; j < nb_features; j++){
                W[j+1] += x[i] * labels[i] * inputs[i * nb_features + j];
            }
        }
        cout << m << "   vecteur choose : " << x[m] << endl;
        W[0] = (1 / labels[m]);
        for(auto i = 0; i < nb_features; i++){
            W[0] -= W[i+1] * inputs[m * nb_features + i];
        }

        for(auto i = 0; i < nb_features + 1; i++){
            model->W[i] = W[i];
        }
        model->w_size = nb_features + 1;

        delete[] W;
        delete[] linear;
        delete[] constraint;
        delete[] kernel;
    }

    DLLEXPORT int svm_predict(SVM* model, double* input){
        double s = model->W[0];
        for(auto i = 0; i < model->w_size; i++){
            s += model->W[i+1] * input[i];
        }

        return s > 0.0 ? 1.0 : -1.0;
    }

    DLLEXPORT void svm_dispose(SVM* model){
        delete[] model->W;
    }

    DLLEXPORT void svm_save_model(SVM* svm, char* path_char){
        std::ofstream file;
        std::string path = path_char;

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        auto str = oss.str();

        std::string name = "model_svm_" + str + ".txt";
        auto fullpath = path + "/" + name;
        file.open(fullpath);

        file << svm->w_size << "\n";

        for(auto i = 0; i < svm->w_size; i++){
            file << fixed << setprecision(15) << svm->W[i] << ";";
        }

        file.close();

        std::this_thread::sleep_for (std::chrono::seconds(1));
    }

    DLLEXPORT SVM* svm_load_model(char* path_char){
        std::string path = path_char;
        std::string line;
        ifstream file(path);
        SVM* svm = nullptr;
        std::string delimiter = ";";
        size_t pos = 0;
        std::string token = "1";

        int nb_feature = 0;
        if(file.is_open()){
            if(getline(file, line)){
                nb_feature = std::stoi(line);
            }

            svm = svm_create_model(nb_feature - 1);
            svm->w_size = nb_feature;
            int count = 0;
            if(getline(file, line)){
                while ((pos = line.find(delimiter)) != std::string::npos) {
                    token = line.substr(0, pos);
                    svm->W[count] = std::stof(token);
                    line.erase(0, pos + delimiter.length());
                    count++;
                }
            }
            file.close();
        }
        return svm;
    }

    //-- Fin SVM -----------------------------------------------------------------------------------------------
}

int main(){


    int nb_sample = 3;
    int nb_features = 2;
    int nb_features_outputs = 1;
    double alpha = 0.01;
    int iteration = 100;
    int nb_clusters = 2;

    int nb_X = 10;
    int nb_Y = 10;

    nb_sample = nb_X * nb_Y;
    /*
    auto X = new double*[nb_sample];
    for(auto i = 0; i < nb_X; i++){
        for(auto j = 0; j < nb_Y; j++){
            X[i * nb_Y + j] = new double[nb_features];
            X[i * nb_Y + j][0] = i;
            X[i * nb_Y + j][1] = j;
        }
    }

    auto Y = new double*[nb_sample];
    for(auto i = 0; i < nb_sample; i++){
        Y[i] = new double[1];
        if(X[i][0] < 6){
            Y[i][0] = 1.0;
        } else {
            Y[i][0] = -1.0;
        }
    }
    */

    nb_sample = 40;
    auto X = new double*[40];
    for(auto i = 0; i < 40; i++){
        X[i] = new double[2];
    }
    X[0][0] = 1.60347094; X[0][1] = 1.16693448;
    X[1][0] = 1.30489739; X[1][1] = 1.62661505;
    X[2][0] = 1.16477863; X[2][1] = 1.44990294;
    X[3][0] = 1.37319578; X[3][1] = 1.0127223;
    X[4][0] = 1.65610174; X[4][1] = 1.47902849;
    X[5][0] = 1.36684749; X[5][1] = 1.77563461;
    X[6][0] = 1.02980506; X[6][1] = 1.56341694;
    X[7][0] = 1.02925712; X[7][1] = 1.60537159;
    X[8][0] = 1.76852481; X[8][1] = 1.13915218;
    X[9][0] = 1.74024371; X[9][1] = 1.01481541;
    X[10][0] = 1.78204282; X[10][1] = 1.34259493;
    X[11][0] = 1.60973192; X[11][1] = 1.70290689;
    X[12][0] = 1.57323083; X[12][1] = 1.0465343;
    X[13][0] = 1.27238007; X[13][1] = 1.76992059;
    X[14][0] = 1.8442367 ; X[14][1] = 1.40572677;
    X[15][0] = 1.75361148; X[15][1] = 1.32191531;
    X[16][0] = 1.07216853; X[16][1] = 1.78013893;
    X[17][0] = 1.27885133; X[17][1] = 1.85698043;
    X[18][0] = 1.89423394; X[18][1] = 1.70387675;
    X[19][0] = 1.03782436; X[19][1] = 1.80432737;

    X[20][0] = 2.27624217; X[20][1] = 2.58419508;
    X[21][0] = 2.64105432; X[21][1] = 2.66721891;
    X[22][0] = 2.10970416; X[22][1] = 2.58939382;
    X[23][0] = 2.57328797; X[23][1] = 2.02044592;
    X[24][0] = 2.29133622; X[24][1] = 2.73514272;
    X[25][0] = 2.63334756; X[25][1] = 2.33325562;
    X[26][0] = 2.82357141; X[26][1] = 2.640247;
    X[27][0] = 2.13019952; X[27][1] = 2.14812977;
    X[28][0] = 2.09006334; X[28][1] = 2.49612499;
    X[29][0] = 2.69743864; X[29][1] = 2.87647262;
    X[30][0] = 2.71201851; X[30][1] = 2.06131197;
    X[31][0] = 2.23386365; X[31][1] = 2.73238376;
    X[32][0] = 2.78201307; X[32][1] = 2.10719873;
    X[33][0] = 2.5250033 ; X[33][1] = 2.55400342;
    X[34][0] = 2.54833792; X[34][1] = 2.37994476;
    X[35][0] = 2.32866572; X[35][1] = 2.14322475;
    X[36][0] = 2.2579218 ; X[36][1] = 2.09528893;
    X[37][0] = 2.87172345; X[37][1] = 2.79329936;
    X[38][0] = 2.32268147; X[38][1] = 2.05962109;
    X[39][0] = 2.68723871; X[39][1] = 2.86047583;

    auto Y = new double*[40];
    for(auto i = 0; i < 40; i++){
        Y[i] = new double[1];
        Y[i][0] = i < 5 || i > 15 ? 1.0 : -1.0;
    }

    auto x_flattened = new double[nb_sample * nb_features];
    int c = 0;
    for(auto i = 0; i < nb_sample; i++){
        for(auto j = 0; j < nb_features; j++){
            x_flattened[c] = X[i][j];
            c++;
        }
    }

    auto y_flattened = new double[nb_sample * nb_features_outputs];
    c = 0;
    for(auto i = 0; i < nb_sample; i++){
        for(auto j = 0; j < nb_features_outputs; j++){
            y_flattened[c] = Y[i][j];
            c++;
        }
    }

    auto model = rbf_create_model();

    auto gamma = 5.0;

    rbf_train(model, x_flattened, nb_sample, nb_features, y_flattened, 1, 0, gamma);

    for(auto i = 0; i < nb_sample; i++){
        auto r = rbf_predict_classification(model, X[i]);
        printf("[%f, %f] : %f     %f)\n", X[i][0], X[i][1], Y[i][0], r[0]);
    }
    char* path_to_save = "D:/Utilisateurs/Bureau/projet_annuel";
    rbf_save_model(model, path_to_save);
    rbf_dispose(model);

    return 0;
}
