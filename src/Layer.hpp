#pragma once
#include <map>
#include <random>
#include <vector>
#include "activation.hpp"
#include "Tensor.hpp"
#include "optimizer.hpp"
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
    Layer( vector<size_t> weight_shape, vector<size_t> bias_shape, state layState, Activation act ) 
    : weights(weight_shape), bias(bias_shape), 
      act(act), activation( activ_functions.at(act).first ), 
      activ_derivative( activ_functions.at(act).second ),
      layState(layState)
    {}

    state layState;

    virtual string getType() = 0;
    virtual size_t getSize() = 0;
    virtual size_t getInputSize() = 0;

    virtual Tensor &getWeights() = 0;
    virtual Tensor &getBias() = 0;

    
    void set_weights( const Tensor &w ) { weights = w; }
    void set_biases( const Tensor &b ) { bias = b; }

    // uniform initialization
    // I found experimentally that it's better to initialize biases a bit higher
    void initialize_uniform( float min = 0, float max = 0.1 )
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(min, max);
        std::uniform_real_distribution<float> bias_distribution(min, 5*max);
        for ( size_t i = 0; i < weights.getSize(); i++ ) 
        {
            weights[i] = distribution(generator);
                //w = fabs( distribution(generator) ); // with negative weigths, RELU layers kept dying at the start                                    
                  // theoretically it should work but practically it didn't so YOLO, abs value :)
        }
        for (size_t i = 0; i < bias.getSize(); i++) { bias[i] = bias_distribution(generator); }
    }
    // gaussian initialization //TODO: separate weights and biases initial distribution
    void initialize_gauss( float mean = 0, float stddev = -1 )
    {
        if ( stddev < 0 ) stddev = sqrt( 2.0 / getInputSize() ); // He initialization by default
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean, stddev);
        //std::normal_distribution<float> bias_distribution(0.01, 0.01);
        std::uniform_real_distribution<float> bias_distribution(0.01, 0.1);
        for ( size_t i = 0; i < weights.getSize(); i++ ) 
        {
            weights[i] = distribution(generator);
                //w = fabs( distribution(generator) ); // with negative weigths, RELU layers kept dying at the start                                    
                  // theoretically it should work but practically it didn't so YOLO, abs value :)
        }
        for (size_t i = 0; i < bias.getSize(); i++) { bias[i] = bias_distribution(generator); }
    }

    virtual void compute_potential( const Tensor &input) = 0;
    virtual void compute_derivative() = 0;
    virtual void compute_epsilon( const Tensor &out_prev ) = 0;

    // TODO: add optimizers
    void modify_weights( float learning_rate ) {
        update_gradient_descent( learning_rate );
    }

    void update_gradient_descent( float learning_rate ) {
        for (size_t i = 0; i < weights.getSize(); i++) { weights[i] -= learning_rate*layState.epsilon[i]; }
        for (size_t i = 0; i < bias.getSize(); i++) { bias[i] -= learning_rate*layState.epsilon_bias[i]; }
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