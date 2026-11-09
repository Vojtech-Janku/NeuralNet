#pragma once
#include "Layer.hpp"


// Class representing individual layer of neurons with activation function, bias and weights, and last calculated state.
// Topologically, a DeepLayer object consists of a row of neurons and the weights of their inbound edges (coming from previous layer). 
class DeepLayer : public Layer
{

public:
    size_t size;
    size_t input_size;

    DeepLayer( size_t neuron_count, size_t input_count, Activation act = Activation::RELU )
    : Layer( { neuron_count, input_count}, {neuron_count}, state(neuron_count, input_count), act ),
      size(neuron_count), input_size(input_count)
    {}

    string getType()
    {
        return "DEEP";
    }

    size_t getSize()
    {
        return size;
    }

    size_t getInputSize()
    {
        return input_size;
    }

    Tensor &getWeights() {
        return weights;
    }

    Tensor &getBias() {
        return bias;
    }

    // the core of feed forward - computes potential and output for this layer
    // input can have any shape (e.g. a 2D conv output) - it is read as a flat vector via operator[]
    void compute_potential( const Tensor &input)
    {
      #pragma omp parallel for num_threads(16)                    // multiprocessing
        for ( size_t j = 0; j < getSize(); j++ )
        {
            float potential = 0;
            for ( size_t i = 0; i < input.getSize(); i++ ) 
            {
                potential += ( weights.at(j,i) * input[i] );
            }
            potential += bias[j];
            layState.potential[j] = potential;
        }
        if (act == SOFTMAX) {
            // subtract the max potential before exp - same result, but exp can't overflow
            float max_potential = layState.potential[0];
            for ( size_t j = 1; j < getSize(); j++ ) {
                if ( layState.potential[j] > max_potential ) max_potential = layState.potential[j];
            }
            float sum = 0;
            for ( size_t j = 0; j < getSize(); j++ ) {
                layState.output[j] = exp(layState.potential[j] - max_potential);
                sum += layState.output[j];
            }
            for ( size_t j = 0; j < getSize(); j++ ) {
                layState.output[j] /= sum;
            }
        } else {
            for ( size_t j = 0; j < getSize(); j++ ) {
                layState.output[j] = activation( layState.potential[j] );
            }
        }
    }

    // computes the derivative of activation with current potential - used in backpropagation
    void compute_derivative() {
        for ( size_t n = 0; n < getSize(); n++ ) 
        {
            if (act == SOFTMAX) {
                // softmax + cross-entropy: d Err / d potential = output - target, which backprop
                // already puts in err_output - so the derivative factor is just 1.
                // Only valid when softmax is the output layer (enforced in Neural_net::add_layer).
                layState.derivative[n] = 1;
            } else {
                layState.derivative[n] = activ_derivative( layState.potential[n] );
            }
        }
    }

    // accumulates this sample's weight and bias gradients into layState.epsilon / epsilon_bias
    void compute_epsilon( const Tensor &out_prev ) 
    {
      #pragma omp parallel for num_threads(16)                    // multiprocessing 
        for ( size_t j = 0; j < getSize(); j++ ) 
        {
            for ( size_t i = 0; i < out_prev.getSize(); i++ ) 
            {
                layState.epsilon.at(j,i) +=
                      layState.err_output[j] 
                    * layState.derivative[j] 
                    * out_prev[i]; 
            }
            layState.epsilon_bias[j] +=
                  layState.err_output[j] 
                * layState.derivative[j];
        }
    }
};