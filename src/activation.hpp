#pragma once
#include <cmath>
#include <map>
#include <string>
using namespace std;

enum Activation{ STEP, RELU, LEAKY_RELU, SIGMOID, TANH, SOFTMAX };
// just for printing
string get_str( Activation act ) 
{
    switch (act) {
    case Activation::STEP:          return "Step";
    case Activation::RELU:          return "Relu";
    case Activation::LEAKY_RELU:    return "Leaky Relu";
    case Activation::SIGMOID:       return "Sigmoid";
    case Activation::TANH:          return "Tanh";
    case Activation::SOFTMAX:       return "Softmax";
    default:                        return "Unknown";
    }
}

//typedef std::vector<float> (*vector_function)(std::vector<float> &);

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