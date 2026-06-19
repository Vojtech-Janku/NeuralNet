#include <map>
#include <random>
#include <vector>
#include "activation.hpp"

using namespace std;

template< typename T >
using matrix = vector<vector<T>>;



// Class representing individual layer of neurons with activation function, bias and weights, and last calculated state.
// Topologically, a Layer object consists of a row of neurons and the weights of their inbound edges (coming from previous layer). 
class Layer
{
    // Struct representing the inner state of the layer.
    // Used for storing all computations.
    struct state 
    {
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

        state( int n, int incoming ) 
        {
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

public:
    vector<float> bias;
    matrix<float> weights;
    Activation act;
    float (*activation)(float);         // activation function
    float (*activ_derivative)(float);   // derivative of activation function
    //vector_function activ_function;
    //vector_function deriv_function;
    state layState;

    Layer( int neuron_count, int input_count, Activation act = Activation::RELU )
    : act(act), activation( activ_functions.at(act).first ), activ_derivative( activ_functions.at(act).second ),
      layState( neuron_count, input_count ) 
    {
        bias =      vector<float>(neuron_count);
        weights =   matrix<float>( neuron_count, vector<float>(input_count) );
    }

    string getType()
    {
        return "DEEP";
    }

    size_t getSize()
    {
        return bias.size();
    }

    size_t getInputSize()
    {
        return weights[0].size();
    }

    void set_biases( vector<float> b )
    {
        bias = b;
    }
    void set_weights( matrix<float> w )
    {
        weights = w;
    }
    void set_potential( vector<float> pot )
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
        for ( size_t i = 0; i < weights.size(); i++ ) 
        {
            for ( auto &w : weights[i] ) 
            {
                w = distribution(generator);
            }
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
        for ( size_t i = 0; i < weights.size(); i++ ) 
        {
            for ( auto &w : weights[i] ) 
            {
                w = distribution(generator);
                //w = fabs( distribution(generator) ); // with negative weigths, RELU layers kept dying at the start
            }                                        // theoretically it should work but practically it didn't so YOLO, abs value :)
            bias[i] = bias_distribution(generator);
        }
    }

    // the core of feed forward - computes potential and output for this layer
    void compute_potential( const vector<float> &input)
    {
      #pragma omp parallel for num_threads(16)                    // multiprocessing
        for ( size_t j = 0; j < getSize(); j++ )
        {
            float potential = 0;
            for ( size_t i = 0; i < input.size(); i++ ) 
            {
                potential += ( weights[j][i] * input[i] );
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
    void compute_epsilon( const vector<float> &out_prev ) 
    {
      #pragma omp parallel for num_threads(16)                    // multiprocessing 
        for ( size_t j = 0; j < layState.output.size(); j++ ) 
        {
            for ( size_t i = 0; i < out_prev.size(); i++ ) 
            {
                layState.epsilon[j][i] +=
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

class ConvLayer : Layer
{
    int kernel_height;
    int kernel_width;
    bool cutoff;
public:
    string getType() 
    {
        return "CONVOLUTIONAL";
    }

    ConvLayer( int neuron_count, int input_count, Activation act, int kernel_height, int kernel_width, bool cutoff ) 
    : Layer( neuron_count, input_count, act ), kernel_height(kernel_height), kernel_width(kernel_width), cutoff(cutoff)
    {}

    // TODO: decide if I want to multiply sparse matrix or have a compact weight matrix with specific logic
    void compute_potential( const vector<float> &input) 
    {
        float potential;
      #pragma omp parallel for num_threads(16)                    // multiprocessing 
        for ( size_t neuron_idx = 0; neuron_idx < getSize(); neuron_idx++ ) 
        {
            potential = 0;
            for ( size_t i = 0; i < input.size(); i++ ) 
            {
                potential += ( weights[neuron_idx][i] * input[i] );
            }
            potential += bias[neuron_idx];
            layState.potential[neuron_idx] = potential;
            layState.output[neuron_idx] = activation( potential );
        }
    }
};