#pragma once
#include "Layer.hpp"

class ConvLayer : public Layer
{

    size_t input_height;
    size_t input_width;
    size_t kernel_size;
    //int C_in;
    //int C_out;
    size_t stride;
    bool padding;
    

    size_t output_height;
    size_t output_width;

public:
    ConvLayer( size_t input_height, size_t input_width, //int C_in, int C_out, 
        size_t kernel_size, size_t stride, bool padding, Activation act ) 
    : Layer( {kernel_size, kernel_size}, {1}, state(input_height-kernel_size+1, input_width-kernel_size+1, kernel_size), act ),
      input_height(input_height), input_width(input_width), //C_in(C_in), C_out(C_out),
      kernel_size(kernel_size), stride(stride), padding(padding),
      output_height(input_height-kernel_size+1), output_width(input_width-kernel_size+1)
    {}

    string getType() 
    {
        return "CONVOLUTIONAL";
    }

    size_t getSize() {
        return output_height*output_width;
    }

    size_t getInputSize()
    {
        return kernel_size*kernel_size;
    }

    Tensor &getWeights() {
        return weights;
    }

    Tensor &getBias() {
        return bias;
    }

    void compute_potential( const Tensor &input) 
    {
        //Tensor input = const_input;
        //input.reshape( {input_height,input_width} );

      #pragma omp parallel for num_threads(16)                    // multiprocessing 
        for ( size_t neuron_i = 0; neuron_i < output_height; neuron_i++ ) 
        {
            for (size_t neuron_j = 0; neuron_j < output_width; neuron_j++)
            {
                float potential = 0;
                for ( size_t kernel_i = 0; kernel_i < kernel_size; kernel_i++ ) {
                    for ( size_t kernel_j = 0; kernel_j < kernel_size; kernel_j++ ) {
                        potential += ( weights.at(kernel_i, kernel_j) * input.at(neuron_i+kernel_i, neuron_j+kernel_j) );
                    }
                }
                potential += bias[0];
                layState.potential.at(neuron_i, neuron_j) = potential;
                layState.output.at(neuron_i, neuron_j) = activation( potential );
            }
        }
    }

    void compute_derivative() {
        for ( size_t neuron_i = 0; neuron_i < output_height; neuron_i++ ) 
        {
            for (size_t neuron_j = 0; neuron_j < output_width; neuron_j++)
            {
                layState.derivative.at(neuron_i, neuron_j) = activ_derivative( layState.potential.at(neuron_i, neuron_j) );
            }
        }
    }

    // NOTE: this only computes the weight/bias gradient for THIS layer's kernel.
    // It does not compute the gradient w.r.t. this layer's input (d Err / d out_prev),
    // which is what Neural_net::backpropagation would need to keep propagating error
    // further back through another ConvLayer preceding this one. Backprop currently only
    // handles projecting error through a dense DeepLayer weight matrix (see
    // Neural_net::backpropagation's use of getWeights().at(r,j)), so a ConvLayer must be
    // followed by a DeepLayer, not another ConvLayer, until that's implemented.
    void compute_epsilon( const Tensor &out_prev )
    {
        //TODO: optimize mutliprocessing

        for (size_t i = 0; i < output_height; i++)
        {
            for (size_t j = 0; j < output_width; j++)
            {
                for ( size_t kernel_i = 0; kernel_i < kernel_size; kernel_i++ ) {
                  #pragma omp parallel for num_threads(16)                    // multiprocessing 
                    for ( size_t kernel_j = 0; kernel_j < kernel_size; kernel_j++ ) {
                        layState.epsilon.at(kernel_i, kernel_j) += layState.err_output.at(i,j)
                                    * layState.derivative.at(i,j)
                                    * out_prev.at( i+kernel_i, j+kernel_j );
                    }
                }
                layState.epsilon_bias[0] += layState.err_output.at(i,j)
                                    * layState.derivative.at(i,j);
            }
        }
    }

};