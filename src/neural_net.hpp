#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <vector>

using namespace std;

template< typename T >
using matrix = vector<vector<T>>;

float TOLERANCE = 0.00001;
bool operator==( const vector<float> &a, const vector<float> &b) {
    if ( a.size() != b.size() ) return false;
    for ( size_t i = 0; i < a.size(); i++) {
        if ( abs( a[i] - b[i] ) > TOLERANCE ) return false;
    }
    return true;
}

template< typename T >
void vec_div( vector<T> &vec, int d ) {
    for ( size_t v = 0; v < vec.size(); v++) {
        vec[v] = vec[v] / d;
    }
}

template< typename T >
void mat_div( matrix<T> &mat, int d ) {
    for ( size_t m = 0; m < mat.size(); m++) {
        vec_div( mat[m], d );
    }
}

template< typename T >
vector<T> get_column( const matrix<T> &mat, size_t col ) {
    vector<T> res( mat.size() );
    for (size_t i = 0; i < mat.size(); i++) {
        res[i] = mat[i][col];
    }
    return res;
}

template< typename T >
size_t get_max_idx( const vector<T> &vec ) {
    size_t max_idx = 0;
    for (size_t i = 1; i < vec.size(); i++) {
        if ( vec[i] > vec[max_idx] ) { max_idx = i; }
    }
    return max_idx;
}

template< typename T >
vector<int> get_max_idx( const matrix<T> &mat ) {
    vector<int> res( mat.size() );
    for (size_t i = 0; i < mat.size(); i++) {
        res[i] = get_max_idx( mat[i] );
    }
    return res;
}

// print functions
template< typename T >
void print_vec( const vector<T> &vec ) {
    std::cout << "< ";
    if ( !vec.empty() ) std::cout << vec[0];
    for ( size_t i = 1; i < vec.size(); i++ ) {
        std::cout << ", " << vec[i];
    }
    std::cout << " >";
}

template< typename T >
void print_matrix( const matrix<T> &mat ) {
    for ( size_t i = 0; i < mat.size(); i++ ) {
        print_vec( mat[i] );
        std::cout << endl;
    }
}

// activation functions and their derivatives
enum Activation{ STEP, RELU, LEAKY_RELU, SIGMOID, TANH, SOFTMAX };

// just for printing
string get_str( Activation a ) {
    switch (a) {
    case Activation::STEP:          return "Step";
    case Activation::RELU:          return "Relu";
    case Activation::LEAKY_RELU:    return "Leaky Relu";
    case Activation::SIGMOID:       return "Sigmoid";
    case Activation::TANH:          return "Tanh";
    default:                        return "Unknown";
    }
}

// activation functions and their derivatives
typedef float (*act_fun)(float);
float step( float x ) { return ( x < 0 ) ? 0 : 1; }
float relu( float x ) { return ( x < 0 ) ? 0 : x; }
float leaky_relu( float x ) { return ( x < 0 ) ? x/16 : x; }
float sigmoid( float x ) { return 1 / ( 1 + exp(-x) ); }  //{ return x / (1 + abs(x)); } // "fast" sigmoid
float tanh_fun( float x ) { return std::tanh(x); }
float step_diff( float ) { return 0; }
float relu_diff( float x ) { return ( x <= 0 ) ? 0 : 1; }
float leaky_relu_diff( float x ) { return ( x < 0 ) ? 1/16 : 1; }
float sigmoid_diff( float x ) { return sigmoid(x) * (1 - sigmoid(x) ); }
float tanh_diff( float x ) { return 1 - std::pow(std::tanh(x), 2); }
map< Activation, pair<act_fun, act_fun> > activ_functions = {
    { Activation::STEP,         make_pair(step, step_diff) },
    { Activation::RELU,         make_pair(relu, relu_diff) },
    { Activation::LEAKY_RELU,   make_pair(leaky_relu, leaky_relu_diff) },
    { Activation::SIGMOID,      make_pair(sigmoid, sigmoid_diff) },
    { Activation::TANH,         make_pair(tanh_fun, tanh_diff) }
};

// optimizers
enum Optimizer{ GRAD, MOMENTUM, ADAM };
// just for printing
string get_str( Optimizer opt ) {
    switch (opt) {
    case Optimizer::GRAD:          return "GRAD";
    case Optimizer::MOMENTUM:      return "MOMENTUM";
    case Optimizer::ADAM:          return "ADAM";
    default:                       return "Unknown";
    }
}

// structs representing the inner state of the network, same topology as Layer 
//            (row of neurons and inbound edges)
// used for storing all computations
struct layer_state {
    vector<float> potential;    // potential of each neuron
    vector<float> output;       // output of each neuron
    vector<float> derivative;   // derivative of sigma( potential )
    matrix<float> epsilon;      // gradient
    vector<float> epsilon_bias; // gradient for bias weights
    vector<float> err_output;   // (d Err / d output) for each neuron
    // optimizer computations
    matrix<float> m;    // used for MOMENTUM or first moment in ADAM
    matrix<float> v;    // used for second moment in ADAM
    vector<float> m_bias;
    vector<float> v_bias;

    layer_state( int n, int incoming ) {
        potential =     vector<float>(n);
        output =        vector<float>(n);
        derivative =    vector<float>(n);
        epsilon_bias =  vector<float>(n);
        epsilon =       matrix<float>( n, vector<float>(incoming) );
        err_output =    vector<float>(n);

        m =       matrix<float>( n, vector<float>(incoming) );
        v =       matrix<float>( n, vector<float>(incoming) );
        m_bias =  vector<float>(n);
        v_bias =  vector<float>(n);
    }
};

// struct for layer weights, activation function, weight initializers and other methods
// a Layer struct consists of a row of neurons and the weights of their inbound edges (coming from previous layer) 
struct Layer
{
    vector<float> bias;
    matrix<float> weights;
    float (*activation)(float);         // activation function
    float (*activ_derivative)(float);   // derivative of activation function

public:
    Layer( int neuron_count, int input_count, Activation act = Activation::RELU )
    : activation( activ_functions.at(act).first ), activ_derivative( activ_functions.at(act).second ) {
        bias =      vector<float>(neuron_count);
        weights =   matrix<float>( neuron_count, vector<float>(input_count) );
    }

    // uniform initialization
    // I found experimentally that it's better to initialize biases a bit higher
    void initialize_uniform( float min = 0, float max = 0.1 ) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(min, max);
        std::uniform_real_distribution<float> bias_distribution(min, 5*max);
        for ( size_t i = 0; i < weights.size(); i++ ) {
            for ( auto &w : weights[i] ) {
                w = distribution(generator);
            }
            bias[i] = bias_distribution(generator);
        }
    }
    // gaussian initialization
    void initialize_gauss( float mean = 0, float stddev = 1 ) {
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean, stddev);
        std::normal_distribution<float> bias_distribution(0.01, 0.01);
        for ( size_t i = 0; i < weights.size(); i++ ) {
            for ( auto &w : weights[i] ) {
                w = fabs( distribution(generator) ); // with negative weigths, RELU layers kept dying at the start
            }                                        // theoretically it should work but practically it didn't so YOLO, abs value :)
            bias[i] = fabs( bias_distribution(generator) );
        }
    }

    // the core of feed forward - computes potential and output for this layer
    void compute_potential( const vector<float> &input, layer_state &lay_state ) {
        float pot;
      #pragma omp parallel for num_threads(16)                    // multiprocessing 
        for ( size_t j = 0; j < bias.size(); j++ ) {
            pot = 0;
            for ( size_t i = 0; i < input.size(); i++ ) {
                pot += ( weights[j][i] * input[i] );
            }
            pot += bias[j];
            lay_state.potential[j] = pot;
            lay_state.output[j] = activation( pot );
        }
    }

    // computes the derivative of activation with current potential - used in backpropagation
    void compute_derivative( layer_state &lay_state ) {
        for ( size_t n = 0; n < bias.size(); n++ ) {
            lay_state.derivative[n] = activ_derivative( lay_state.potential[n] );
        }
    }
};

// Neural_net - the class for the whole neural network.
// Contains all layers and their states, parameter values,
//  feed forward function, functions needed for computing gradient,
//  train and predict functions and more!
class Neural_net
{
    float learning_rate, lr_decay;
    float momentum;
    size_t input_size;         // number of neurons in input layer
    vector<size_t> net_scheme; // network scheme excluding input layer for practical reasons
    vector<Activation> act_funs;
    vector<Layer> layers;
    vector<layer_state> lay_states;

    float beta1 = 0.9, beta2 = 0.999, eps = 0.00000001; // for ADAM optimizer

public:
    Neural_net( vector<size_t> scheme, vector<Activation> funs, float l_rate = 0.01, float l_decay = 0.001, float moment = 0.5 ) 
    : learning_rate( l_rate ), lr_decay( l_decay ), momentum(moment), input_size( scheme[0] ), 
      net_scheme( scheme.begin()+1, scheme.end() ), act_funs( funs ) {
        assert( scheme.size() > 1 );
        for ( int i = 1; i < scheme.size(); i++ ) {
            add_layer( scheme[i], scheme[i-1], act_funs[i-1] );
        }
    }

    void add_layer( size_t layer_size, size_t input_size, Activation a ) {
        layers.push_back(     Layer( layer_size, input_size, a ) );
        lay_states.push_back( layer_state( layer_size, input_size ) );
    }

    void init_unif( float min = 0, float max = 0.1 ) {
        for ( auto &lay : layers ) { lay.initialize_uniform(min, max); }
    }

    void init_gauss() {
        layers[0].initialize_gauss( 0, sqrt(2.0/input_size) );
        for (size_t i = 1; i < layers.size(); i++)
        {
            layers[i].initialize_gauss(0, sqrt(2.0/net_scheme[i-1]));
        }
    }

    // basic feed forward algorithm
    const vector<float> &feed_forward( const vector<float> &input ) {
        layers[0].compute_potential( input, lay_states[0] );
        for ( int i = 1; i < layers.size(); i++ ) {
            layers[i].compute_potential( lay_states[i-1].output, lay_states[i] );
        }
        return lay_states.back().output;
    }

    // computes error function output derivatives
    void backpropagation( const vector<float> &target_point ) {
        for ( size_t n = 0; n < net_scheme.back(); n++ ) {  // y_j - d_kj
            lay_states.back().err_output[n] = lay_states.back().output[n] - target_point[n];
        }
        for ( int lay = layers.size()-2; lay >= 0; --lay ) {
          #pragma omp parallel for num_threads(16)                    // multiprocessing 
            for ( size_t j = 0; j < net_scheme[lay]; j++ ) {
                float sum = 0;
                for ( size_t r = 0; r < net_scheme[lay+1]; r++ ) {
                    sum += lay_states[lay+1].err_output[r] 
                        * lay_states[lay+1].derivative[r] 
                        * layers[lay+1].weights[r][j];
                }
                lay_states[lay].err_output[j] = sum;
            }
        }        
    }

    // computes gradient for one layer
    void compute_epsilon_layer( layer_state &lay_state, const vector<float> &out_prev ) {
      #pragma omp parallel for num_threads(16)                    // multiprocessing 
        for ( size_t j = 0; j < lay_state.output.size(); j++ ) {
            for ( size_t i = 0; i < out_prev.size(); i++ ) {
                lay_state.epsilon[j][i] +=
                      lay_state.err_output[j] 
                    * lay_state.derivative[j] 
                    * out_prev[i]; 
            }
            lay_state.epsilon_bias[j] +=
                  lay_state.err_output[j] 
                * lay_state.derivative[j];
        }
    }

    // computes gradient for whole network, one training example
    void compute_epsilon( const vector<float> &data_row ) {
        compute_epsilon_layer( lay_states[0], data_row );
        for ( size_t lay = 1; lay < layers.size(); lay++ ) {
            compute_epsilon_layer( lay_states[lay], lay_states[lay-1].output );
        }        
    }

    // computes all activation functions derivatives
    void compute_derivatives() {
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
            layers[lay].compute_derivative( lay_states[lay] );
        }        
    }

    // computes gradient for given data batch
    void compute_gradient( const matrix<float> &data, const matrix<float> &labels, 
                            pair<size_t,size_t> batch_range ) {
        // initialize epsilon = 0;
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
            for ( auto &v : lay_states[lay].epsilon ) {
                fill( v.begin(), v.end(), 0 );
            }
            fill( lay_states[lay].epsilon_bias.begin(), 
                  lay_states[lay].epsilon_bias.end(), 0 );
        }
        // total squared error
        float err = 0;
        // go through training data
        for ( size_t k = batch_range.first; k < batch_range.second; k++ ) {
            // 1. forward pass
            feed_forward( data[k] );

            // compute derivatives
            compute_derivatives();
            
            // 2. backpropagation
            backpropagation( labels[k] );
            
            // compute gradient
            compute_epsilon( data[k] );
        }
        // average the gradient
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
            mat_div( lay_states[lay].epsilon, batch_range.second-batch_range.first );
        }
    }
    // just overload
    void compute_gradient( const matrix<float> &data, const matrix<float> &labels ) {
        compute_gradient( data, labels, make_pair(0, data.size()) );
    }

    void compute_single_adam( float &m, float &v, const float &epsilon, 
                              float beta1, float beta2, float eps ) {
                m = ( beta1*m + (1 - beta1)*epsilon );
                v = ( beta2*v + (1 - beta2)*epsilon*epsilon );   
    }

    void compute_adam() {
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
          #pragma omp parallel for num_threads(16)                    // multiprocessing 
            for ( size_t j = 0; j < net_scheme[lay]; j++ ) {
                for ( size_t i = 0; i < layers[lay].weights[0].size(); i++ ) {
                    compute_single_adam( lay_states[lay].m[j][i], lay_states[lay].v[j][i], 
                                         lay_states[lay].epsilon[j][i], beta1, beta2, eps );
                }
                compute_single_adam( lay_states[lay].m_bias[j], lay_states[lay].v_bias[j], 
                                         lay_states[lay].epsilon_bias[j], beta1, beta2, eps );
            }
        } 
    }

    // ---- single weight update functions for optimizers ---
    void update_gradient_descent( float &weight, const float &gradient ) {
        weight -= learning_rate*gradient;
    }

    void update_momentum( float &weight, const float &gradient, float &m) {
        m = ( momentum*m + learning_rate*gradient );
        weight -= m;
    }

    void update_adam( float &weight, const float &m, const float &v, const size_t &it ) {
        float mhat = m / (1 - powf(beta1, it) ), vhat = v / (1 - powf(beta2, it) );
        weight -= learning_rate * mhat / ( sqrt( vhat ) + eps );
    }

    // updates all weights
    void modify_weights( Optimizer opt, const size_t &it = 0 ) {
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
          #pragma omp parallel for num_threads(16)                    // multiprocessing 
            for ( size_t j = 0; j < net_scheme[lay]; j++ ) {
                for ( size_t i = 0; i < layers[lay].weights[0].size(); i++ ) {
                    switch (opt)
                    {
                    case Optimizer::GRAD:
                        update_gradient_descent( layers[lay].weights[j][i], lay_states[lay].epsilon[j][i] );
                        break;
                    case Optimizer::MOMENTUM:
                        update_momentum( layers[lay].weights[j][i], lay_states[lay].epsilon[j][i], lay_states[lay].m[j][i] );
                        break;
                    case Optimizer::ADAM:
                        update_adam( layers[lay].weights[j][i], lay_states[lay].m[j][i], lay_states[lay].v[j][i], it );
                        break;
                    }
                }
            }
        }
    }

    bool train( const matrix<float> &data, const matrix<float> &target, 
                size_t batch_size, Optimizer opt, float precision = 0.001, size_t epochs = 100000 )
    {
        auto lr_init = learning_rate;
        float err;
        size_t batch_start;
        size_t iter = 1;
        //auto rng = std::default_random_engine {};
        //std::shuffle(std::begin(cards_), std::end(cards_), rng);
        for ( size_t i = 0; i < epochs; i++ ) {
            batch_start = 0;
            while( batch_start+batch_size < data.size() ) {
                compute_gradient( data, target, make_pair(batch_start, batch_start+batch_size) );
                if (opt == Optimizer::ADAM) compute_adam();
                modify_weights(opt, iter);
                iter++;
                if ( learning_rate > 0.001 ) learning_rate = lr_init * ( 1 / (1+lr_decay*iter) ); // learning rate decay
                batch_start += batch_size;
            }
            // spaghetti code but whatever
            compute_gradient( data, target, make_pair( batch_start, data.size() ) );
            if (opt == Optimizer::ADAM) compute_adam();
            modify_weights(opt, iter);
            iter++;
            if ( learning_rate > 0.001 ) learning_rate = lr_init * ( 1 / (1+lr_decay*iter) ); // learning rate decay

            err = total_squared_error( data, target );
            //if ( i % 10 == 0 ) 
                //std::cout << " Epoch " << i << ", total error = " << err << ", lrate = " << learning_rate << endl;
            if ( err < precision ) return true;
        }
        return false;   
    }

    matrix<float> predict( const matrix<float> &data ) {
        matrix<float> pred;
        for (size_t k = 0; k < data.size(); k++) {
            pred.push_back( feed_forward( data[k] ) );
        }
        return pred;
    }

    //  ----------  UTILITY FUNCTIONS  ----------

    void set_weights( size_t lay, matrix<float> w ) {
        layers[ lay ].weights = w;
    }
    void set_biases( size_t lay, vector<float> b ) {
        layers[ lay ].bias = b;
    }
    void set_potential( size_t lay, vector<float> pot ) {
        lay_states[ lay ].potential = pot;
    }

    const layer_state &get_state( int lay ) {
        return lay_states[lay];
    }

    int get_layer_count() {
        return net_scheme.size();
    }

    float output_squared_error( const vector<float> &target ) {
        float err = 0;
        for ( size_t i = 0; i < net_scheme.back(); i++) {
            err += ( target[i] - lay_states.back().output[i] ) 
                 * ( target[i] - lay_states.back().output[i] );
        }
        return err;
    }

    float total_squared_error( const matrix<float> &data, const matrix<float> &target ) {
        float err = 0;
        for (size_t i = 0; i < data.size(); i++) {
            feed_forward( data[i] );
            err += output_squared_error( target[i] );
        }
        return err / data.size();
    }  

    //  ----------  PRINT FUNCTIONS  ----------
    void print() {
        std::cout << "Weights:" << endl;
        for ( size_t lay = 1; lay < layers.size(); lay++ ) {
            std::cout << "-------------" << endl;
            for ( size_t i = 0; i < net_scheme[lay]; i++ ) {
                print_vec( layers[lay].weights[i] );
                std::cout << "  [ " << layers[lay].bias[i] << " ]" << endl;
            }
        }
        std::cout << endl;
    }

    void print_gradient() {
        std::cout << "Gradient:" << endl;
        for ( size_t lay = 1; lay < layers.size(); lay++ ) {
            std::cout << "-------------" << endl;
            for ( size_t i = 0; i < net_scheme[lay]; i++ ) {
                print_vec( lay_states[lay].epsilon[i] );
                std::cout << "  [ " << lay_states[lay].epsilon_bias[i] << " ]" << endl;
            }
        }
        std::cout << endl;
    }

    void print_output() {
        std::cout << "Output:" << endl;
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
            std::cout << "-------------" << endl;
            print_vec( lay_states[lay].output );
            std::cout << endl;
        }
        std::cout << endl;
    }
};

