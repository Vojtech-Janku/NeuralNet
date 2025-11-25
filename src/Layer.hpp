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