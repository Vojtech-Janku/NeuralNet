#include <map>
#include <random>
#include <vector>
#include "activation.hpp"
#include "Tensor.hpp"

using namespace std;

template< typename T >
using matrix = vector<vector<T>>;

class Layer
{
protected:
    Tensor weights;
    Tensor bias;
    Activation act;
    float (*activation)(float);         // activation function
    float (*activ_derivative)(float);   // derivative of activation function

    struct state 
    {
        Tensor potential;    // potential of each neuron
        Tensor output;       // output of each neuron
        Tensor derivative;   // derivative of sigma( potential )
        Tensor epsilon;      // gradient
        Tensor epsilon_bias; // gradient for bias weights
        Tensor err_output;   // (d Err / d output) for each neuron
        // TODO: optimizer computations

        state( size_t n, size_t incoming )
        : potential(    { n } ),
          output(       { n } ),
          derivative(   { n } ),
          epsilon(      { n, incoming } ),
          epsilon_bias( { n } ),
          err_output(   { n } )
          //m(            { n, incoming } ),
          //v(            { n, incoming } ),
          //m_bias(       { n } ),
          //v_bias(       { n } )
        {}

        state( size_t out_height, size_t out_width, size_t kernel_size )
        : potential(    { out_height, out_width } ),
          output(       { out_height, out_width } ),
          derivative(   { out_height, out_width } ),
          epsilon(      { kernel_size, kernel_size } ),
          epsilon_bias( { 1 } ),
          err_output(   { out_height, out_width } )
        {}
    };

    

public:
    Layer( vector<size_t> weight_shape, vector<size_t> bias_shape, state layState ) 
    : weights(weight_shape), bias(bias_shape), layState(layState) 
    {}

    state layState;

    virtual string getType() = 0;
    virtual size_t getSize() = 0;

    virtual Tensor &getWeights() = 0;
    virtual Tensor &getBias() = 0;

    
    void set_weights( const Tensor &w ) { weights = w; }
    void set_biases( const Tensor &b ) { bias = b; }

    //virtual void initialize_uniform( float min = 0, float max = 0.1 ) = 0;
    //virtual void initialize_gauss( float min = 0, float max = 0.1 ) = 0;

    virtual void compute_potential( const Tensor &input) = 0;
    virtual void compute_derivative() = 0;
    virtual void compute_epsilon( const Tensor &out_prev ) = 0;

    // TODO: add optimizers
    void modify_weights( float learning_rate ) {
        update_gradient_descent( learning_rate );
    }

    void update_gradient_descent( float learning_rate ) {
        for (size_t i = 0; i < weights.getSize(); i++)
        {
            weights[i] -= learning_rate*layState.epsilon[i];
        }
    }

    /*void update_momentum( float &weight, const float &gradient, float &m) {
        m = ( momentum*m + learning_rate*gradient );
        weight -= m;
    }

    void update_adam( float &weight, const float &m, const float &v, const size_t &it ) {
        float mhat = m / (1 - powf(beta1, it) ), vhat = v / (1 - powf(beta2, it) );
        weight -= learning_rate * mhat / ( sqrt( vhat ) + eps );
    }*/

    virtual ~Layer() {}
};

// Class representing individual layer of neurons with activation function, bias and weights, and last calculated state.
// Topologically, a DeepLayer object consists of a row of neurons and the weights of their inbound edges (coming from previous layer). 
class DeepLayer : public Layer
{
    // Struct representing the inner state of the layer.
    // Used for storing all computations.
    /*
    struct state 
    {
        Tensor potential;    // potential of each neuron
        Tensor output;       // output of each neuron
        Tensor derivative;   // derivative of sigma( potential )
        Tensor epsilon;      // gradient
        Tensor epsilon_bias; // gradient for bias weights
        Tensor err_output;   // (d Err / d output) for each neuron
        // optimizer computations
        Tensor m;    // used for MOMENTUM or first moment in ADAM
        Tensor v;    // used for second moment in ADAM
        Tensor m_bias;
        Tensor v_bias;


    };

    */

public:
    //vector<float> bias;
    //matrix<float> weights;
    size_t size;
    size_t input_size;
    Activation act;
    float (*activation)(float);         // activation function
    float (*activ_derivative)(float);   // derivative of activation function

    DeepLayer( int neuron_count, int input_count, Activation act = Activation::RELU )
    : Layer( { neuron_count, input_count}, {neuron_count}, state(neuron_count, input_count) ),
      size(neuron_count), input_size(input_count),
      act(act), activation( activ_functions.at(act).first ), activ_derivative( activ_functions.at(act).second )
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

    void set_potential( Tensor pot )
    {
        layState.potential = pot;
    }

    // uniform initialization
    // I found experimentally that it's better to initialize biases a bit higher
    void initialize_uniform( float min = 0, float max = 0.1 )
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(min, max);
        std::uniform_real_distribution<float> bias_distribution(min, 5*max);
        for ( size_t i = 0; i < getSize(); i++ ) 
        {
            for ( size_t j = 0; j < getInputSize(); j++  ) 
            {
                weights.at(i,j) = distribution(generator);
                //w = fabs( distribution(generator) ); // with negative weigths, RELU layers kept dying at the start
            }                                        // theoretically it should work but practically it didn't so YOLO, abs value :)
            bias[i] = bias_distribution(generator);
        }
    }
    // gaussian initialization //TODO: separate weights and biases initial distribution
    void initialize_gauss( float mean = 0, float stddev = 1 ) 
    {
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean, stddev);
        //std::normal_distribution<float> bias_distribution(0.01, 0.01);
        std::uniform_real_distribution<float> bias_distribution(0.01, 0.1);
        for ( size_t i = 0; i < getSize(); i++ ) 
        {
            for ( size_t j = 0; j < getInputSize(); j++  ) 
            {
                weights.at(i,j) = distribution(generator);
                //w = fabs( distribution(generator) ); // with negative weigths, RELU layers kept dying at the start
            }                                        // theoretically it should work but practically it didn't so YOLO, abs value :)
            bias[i] = bias_distribution(generator);
        }
    }

    // the core of feed forward - computes potential and output for this layer
    void compute_potential( const Tensor &input)
    {
        //input.flatten(); TODO: handle this in outside?

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
            float sum = 0;
            for ( size_t j = 0; j < getSize(); j++ ) {
                layState.output[j] = exp(layState.potential[j]);
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
                for ( size_t j = 0; j < getSize(); j++ ) {
                    // TODO: implement softmax derivative
                }
            } else {
                layState.derivative[n] = activ_derivative( layState.potential[n] );
            }
        }
    }

    // computes gradient    TODO: move to layer.state?
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

class ConvLayer : public Layer
{

    int input_height;
    int input_width;
    //int C_in;
    //int C_out;
    int stride;
    bool padding;
    int kernel_size;

    int output_height;
    int output_width;

public:
    ConvLayer( int input_height, int input_width, //int C_in, int C_out, 
        int kernel_size, int stride, bool padding, Activation act ) 
    : Layer( {kernel_size, kernel_size}, {1}, state(output_height, output_width, kernel_size) ),
      input_height(input_height), input_width(input_width), //C_in(C_in), C_out(C_out),
      kernel_size(kernel_size), stride(stride), padding(padding),
      output_height(input_height-kernel_size+1), output_width(input_width-kernel_size+1)
    {}

    string getType() 
    {
        return "CONVOLUTIONAL";
    }

    size_t getSize() {
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

        float potential;
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

class MaxPoolingLayer : public Layer {

    int pooling_size;

    matrix<pair<int,int>> mask;

};