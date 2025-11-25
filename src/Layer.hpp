#include <map>
#include <vector>

using namespace std;

template< typename T >
using matrix = vector<vector<T>>;

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
};

enum LayerType{ DEEP, CONVOLUTIONAL };
// just for printing
string get_str( LayerType laytp ) {
    switch (laytp) {
    case LayerType::DEEP:          return "DEEP";
    case LayerType::CONVOLUTIONAL: return "CONVOLUTIONAL";
    default:                       return "Unknown";
    }
}

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