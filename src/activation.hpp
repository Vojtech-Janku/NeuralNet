#pragma once
#include <cmath>
#include <map>
#include <string>
using namespace std;

enum Activation{ STEP, RELU, LEAKY_RELU, SIGMOID, TANH, SOFTMAX };
// just for printing
inline string get_str( Activation act )
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

// activation functions and their derivatives
typedef float (*act_fun)(float);
inline float step( float x ) { return ( x < 0 ) ? 0 : 1; }
inline float relu( float x ) { return ( x < 0 ) ? 0 : x; }
inline float leaky_relu( float x ) { return ( x < 0 ) ? x/16 : x; }
inline float sigmoid( float x ) { return 1 / ( 1 + exp(-x) ); }
inline float tanh_fun( float x ) { return std::tanh(x); }
inline float step_diff( float ) { return 0; }
inline float relu_diff( float x ) { return ( x <= 0 ) ? 0 : 1; }
inline float leaky_relu_diff( float x ) { return ( x < 0 ) ? 1.0f/16 : 1; }
inline float sigmoid_diff( float x ) { return sigmoid(x) * (1 - sigmoid(x) ); }
inline float tanh_diff( float x ) { return 1 - std::pow(std::tanh(x), 2); }
// inline variable (C++17): one shared map, no matter how many .cpp files include this header
inline map< Activation, pair<act_fun, act_fun> > activ_functions = {
    { Activation::STEP,         make_pair(step, step_diff) },
    { Activation::RELU,         make_pair(relu, relu_diff) },
    { Activation::LEAKY_RELU,   make_pair(leaky_relu, leaky_relu_diff) },
    { Activation::SIGMOID,      make_pair(sigmoid, sigmoid_diff) },
    { Activation::TANH,         make_pair(tanh_fun, tanh_diff) }
};