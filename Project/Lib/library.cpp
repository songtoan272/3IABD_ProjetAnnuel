#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <iomanip>
#include <thread>
#include <chrono>
#include <math.h>

#include "src/alglibinternal.h"
#include "src/alglibmisc.h"
#include "src/ap.h"
#include "src/dataanalysis.h"
#include "src/diffequations.h"
#include "src/fasttransforms.h"
#include "src/integration.h"
#include "src/interpolation.h"
#include "src/linalg.h"
#include "src/optimization.h"
#include "src/solvers.h"
#include "src/specialfunctions.h"
#include "src/statistics.h"

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


    struct rbf{
        double** w;
        double* dataset;
        int nb_sample;
        int nb_feature;
        int nb_outputs;
        double alpha;
        double* result;
        int nb_clusters;
    } typedef rbf;


    //-- Debut RBF -------------------------------------------------------------------------------------------------

    DLLEXPORT rbf* rbf_create_model(int nb_sample, int nb_outputs ,double alpha){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        auto my_rbf = new rbf;
        my_rbf->w = new double*[nb_sample];
        my_rbf->alpha = alpha;
        my_rbf->nb_sample = nb_sample;
        my_rbf->nb_outputs = nb_outputs;
        my_rbf->result = new double[nb_outputs];
        my_rbf->dataset = new double[nb_sample];

        for(auto i = 0; i < nb_sample; i++){
            my_rbf->w[i] = new double[nb_outputs];
            for(auto j = 0; j < nb_outputs; j++){
                my_rbf->w[i][j] = dist(mt);
            }
        }

        return my_rbf;
    }

    double _get_distance_norme_two(int features, double* x, double* y){
        double s = 0;
        for(auto i = 0; i < features; i++){
            s += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return sqrt(s);
    }

    double* _get_k_means(double* dataset, int nb_samples, int nb_features, int nb_clusters){
        auto representant = new double[nb_clusters * nb_features];
        auto representant_save = new double[nb_clusters * nb_features];
        auto sample_cluster = new int[nb_samples];
        auto count_cluster = new int[nb_clusters];

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, nb_samples);

        // Tirage au sort sur les points du dataset de representants
        for(auto i = 0; i < nb_clusters; i++){
            int check = 1;
            int loops = 0;
            int c = 0;
            while(check != 0 && loops < 1000){
                c = dist(mt);
                check = 0;
                for(auto y = 0; y < nb_clusters; y++){
                    if(count_cluster[y] == c){
                        check++;
                    }
                }
                loops++;
            }
            count_cluster[i] = c;
            for(auto j = 0; j < nb_features; j++){
                (representant + i * nb_features)[j] = (dataset + c * nb_features)[j];
                (representant_save + i * nb_features)[j] = (dataset + c * nb_features)[j];
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
            int c = 0;
            for(auto i = 0; i < nb_clusters; i++){
                for(auto j = 0; j < nb_features; j++){
                    if(abs(representant[c] - representant_save[c]) > 0.01){
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

    DLLEXPORT void rbf_train(rbf* rbf,
        double* dataset_inputs,
        int nb_samples,
        int nb_features,
        double* dataset_expected,
        int nb_features_expected,
        int nb_clusters
    ){
        rbf->nb_clusters = nb_clusters;
        rbf->nb_feature = nb_features;
        auto m = new double[nb_samples * nb_samples];
        double* representants;
        int nb_sample_representant;

        if(nb_clusters > 1){
            representants = _get_k_means(dataset_inputs, nb_samples, nb_features, nb_clusters);
            nb_sample_representant = nb_clusters;
        } else {
            representants = dataset_inputs;
            nb_sample_representant = nb_samples;
        }

        rbf->dataset = new double[nb_sample_representant * nb_features];
        for(auto i = 0; i < nb_sample_representant * nb_features; i++){
            rbf->dataset[i] = representants[i];
        }

        int c = 0;
        for(auto i = 0; i < nb_samples; i++){
            auto x = dataset_inputs + i * nb_features;
            for(auto j = 0; j < nb_sample_representant; j++){
                auto y = representants + j * nb_features;
                auto distance = _get_distance_norme_two(nb_features, x, y);
                m[c] = exp(0 - (rbf->alpha * pow(distance,2)));
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

        auto r = new double[nb_samples * nb_features_expected];
        Eigen::Map<MatrixXd>(r, w.rows(), w.cols()) = w;

        c = 0;
        for(auto i = 0; i < rbf->nb_sample; i++){
            for(auto j = 0; j < rbf->nb_outputs; j++){
                rbf->w[i][j] = r[c];
                c++;
            }
        }
        delete[] r;
        delete[] m;
    }

    void _rbf_predict_common(rbf* rbf, double* input){
        for(auto i = 0; i < rbf->nb_outputs; i++){
            rbf->result[i] = 0.0;
        }
        int samples = rbf->nb_sample;
        if(rbf->nb_clusters > 1){
            samples = rbf->nb_clusters;
        }
        for(auto i = 0; i < samples; i++){
            auto distance = _get_distance_norme_two(rbf->nb_feature, input, rbf->dataset + i * rbf->nb_feature);
            for(auto j = 0; j < rbf->nb_outputs; j++){
                rbf->result[j] += rbf->w[i][j] * exp(0 - (rbf->alpha * pow(distance, 2)));
            }
        }
    }

    DLLEXPORT double* rbf_predict_classification(rbf* rbf, double* input){
        _rbf_predict_common(rbf, input);
        for(auto i = 0; i < rbf->nb_outputs; i++){
            rbf->result[i] = rbf->result[i] < 0.0 ? -1.0 : 1.0;
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
        double s = 0.0;
        for(auto i = 0; i < nb_features; i++){
            s += exp(0-(X[i] * X[i])) * exp(0-(Y[i] * Y[i])) * exp(2 * X[i] * Y[i]);
        }
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
        minqpsetbc(state, lbnd, ubnd);
        minqpsetlc(state, c, ct);

        cout << c.tostring(1).c_str() << endl;
        cout << ct.tostring().c_str() << endl;

        o = 0;
        for(auto i = 0; i < nb_inputs; i++){
            for(auto j = 0; j < nb_inputs; j++){
                printf("%f  ", kernel[o]);
                o++;
            }
            printf("\n");
        }

        minqpsetalgobleic(state, 0, 0, 0.01, 0);
        minqpoptimize(state);
        minqpresults(state, x, rep);

        for(auto i = 0; i < nb_inputs; i++){
            printf("[");
            for(auto j = 0; j < nb_features; j++){
                printf("%f, ", inputs[i * nb_features + j]);
            }
            printf("]  :  %f\n", x[i]);
        }

        int m = -1;
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
            for(auto j = 0; j < nb_features; j++){
                W[j+1] += x[i] * labels[i] * inputs[i * nb_features + j];
            }
        }

        if(m != -1){
            W[0] = (1 / labels[m]);
            for(auto i = 0; i < nb_features; i++){
                W[0] -= W[i+1] * inputs[m * nb_features + i];
            }
        } else {
            W[0] = 0;
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

    //-- Fin SVM -----------------------------------------------------------------------------------------------
}

int main(){


    int nb_sample = 3;
    int nb_features = 2;
    int nb_features_outputs = 1;
    double alpha = 0.1;
    int nb_clusters = 2;
    auto X = new double*[nb_sample];
    X[0] = new double[nb_features];
    X[0][0] = 1.0;
    X[0][1] = 1.0;
    X[1] = new double[nb_features];
    X[1][0] = 2.0;
    X[1][1] = 3.0;
    X[2] = new double[nb_features];
    X[2][0] = 3.0;
    X[2][1] = 3.0;
/*
    X[3] = new double[nb_features];
    X[3][0] = 4.0;
    X[3][1] = 4.0;
    X[4] = new double[nb_features];
    X[4][0] = 3.0;
    X[4][1] = 1.0;*/

    auto Y = new double*[nb_sample];
    Y[0] = new double[nb_features_outputs];
    Y[0][0] = 1.0;
    Y[1] = new double[nb_features_outputs];
    Y[1][0] = -1.0;
    Y[2] = new double[nb_features_outputs];
    Y[2][0] = -1.0;
/*
    Y[3] = new double[nb_features_outputs];
    Y[3][0] = -1.0;
    Y[4] = new double[nb_features_outputs];
    Y[4][0] = 1.0;*/

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

    // Dataset Test
    int nb_sample_test = 8;
    auto X_test = new double*[nb_sample];
    X_test[0] = new double[nb_features];
    X_test[0][0] = 1.2;
    X_test[0][1] = 3.8;
    X_test[1] = new double[nb_features];
    X_test[1][0] = 1.2;
    X_test[1][1] = 4.2;
    X_test[2] = new double[nb_features];
    X_test[2][0] = 1.8;
    X_test[2][1] = 4.2;
    X_test[3] = new double[nb_features];
    X_test[3][0] = 2.4;
    X_test[3][1] = 4.45;
    X_test[4] = new double[nb_features];
    X_test[4][0] = 2.0;
    X_test[4][1] = 5.0;
    X_test[5] = new double[nb_features];
    X_test[5][0] = 1.0;
    X_test[5][1] = 5.0;
    X_test[6] = new double[nb_features];
    X_test[6][0] = 1.5;
    X_test[6][1] = 4.49;
    X_test[7] = new double[nb_features];
    X_test[7][0] = 1.5;
    X_test[7][1] = 4.51;

    auto model = svm_create_model(nb_features);
    auto model_t = svm_create_model(nb_features);

    svm_train_model(model, x_flattened, nb_sample, nb_features, y_flattened, false);
    svm_train_model(model_t, x_flattened, nb_sample, nb_features, y_flattened, true);

    for(auto i = 0; i < nb_sample; i++){
        auto k = svm_predict(model, X[i]);
        printf("[%f, %f] : %f  %d\n", X[i][0], X[i][1], Y[i][0], k);
    }
    printf("\n\n");
    for(auto i = 0; i < nb_sample; i++){
        auto k = svm_predict(model_t, X[i]);
        printf("[%f, %f] : %f  %d\n", X[i][0], X[i][1], Y[i][0], k);
    }

    svm_dispose(model);
    svm_dispose(model_t);

    /*
    auto x_test_flattened = new double[nb_sample_test * nb_features];
    c = 0;
    for(auto i = 0; i < nb_sample_test; i++){
        for(auto j = 0; j < nb_features; j++){
            x_test_flattened[c] = X_test[i][j];
            c++;
        }
    }

    auto rbf = rbf_create_model(nb_sample, nb_features_outputs, alpha);

    rbf_train(rbf, x_flattened, nb_sample, nb_features, y_flattened, nb_features_outputs, nb_clusters);

    for(auto i = 0; i < nb_sample; i++){
        auto r = rbf_predict_classification(rbf, X[i]);
        printf("(%f, %f)  =>  %f\n",X[i][0], X[i][1], r[0]);
    }

    rbf_dispose(rbf);
*/

    return 0;
}
