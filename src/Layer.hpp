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

    // optimizer state, same shapes as weights / bias
    Tensor m_weights;   // MOMENTUM velocity, or first moment in ADAM
    Tensor v_weights;   // second moment in ADAM
    Tensor m_bias;
    Tensor v_bias;

    static constexpr float ADAM_BETA1 = 0.9f, ADAM_BETA2 = 0.999f, ADAM_EPS = 1e-8f;

    // one generator shared by all layers, so each layer gets different (but reproducible) values
    static std::mt19937 &rng() {
        static std::mt19937 generator( 42 );
        return generator;
    }

    // SOFTMAX is vector-valued, so it has no scalar function - DeepLayer handles it directly
    static act_fun scalar_activation( Activation a, bool derivative ) {
        if ( a == Activation::SOFTMAX ) return nullptr;
        const auto &funs = activ_functions.at(a);
        return derivative ? funs.second : funs.first;
    }

    // one optimizer step for a single parameter tensor
    static void update_params( Tensor &params, const Tensor &grad, Tensor &m, Tensor &v,
                               float learning_rate, Optimizer opt, size_t iter, float momentum ) {
        switch (opt) {
        case Optimizer::GRAD:
            for ( size_t i = 0; i < params.getSize(); i++ ) params[i] -= learning_rate*grad[i];
            break;
        case Optimizer::MOMENTUM:
            for ( size_t i = 0; i < params.getSize(); i++ ) {
                m[i] = momentum*m[i] + learning_rate*grad[i];
                params[i] -= m[i];
            }
            break;
        case Optimizer::ADAM: {
            float t = static_cast<float>( iter > 0 ? iter : 1 );
            float m_correction = 1 - std::pow( ADAM_BETA1, t );
            float v_correction = 1 - std::pow( ADAM_BETA2, t );
            for ( size_t i = 0; i < params.getSize(); i++ ) {
                m[i] = ADAM_BETA1*m[i] + (1 - ADAM_BETA1)*grad[i];
                v[i] = ADAM_BETA2*v[i] + (1 - ADAM_BETA2)*grad[i]*grad[i];
                float m_hat = m[i] / m_correction, v_hat = v[i] / v_correction;
                params[i] -= learning_rate * m_hat / ( std::sqrt(v_hat) + ADAM_EPS );
            }
            break;
        }
        }
    }

public:
    Layer( vector<size_t> weight_shape, vector<size_t> bias_shape, state layState, Activation act ) 
    : weights(weight_shape), bias(bias_shape), 
      act(act), activation( scalar_activation(act, false) ),
      activ_derivative( scalar_activation(act, true) ),
      m_weights(weight_shape), v_weights(weight_shape), m_bias(bias_shape), v_bias(bias_shape),
      layState(layState)
    {}

    state layState;

    // reseed the generator used by initialize_uniform / initialize_gauss
    static void seed( unsigned s ) { rng().seed( s ); }

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
        auto &generator = rng();
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
        auto &generator = rng();
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

    // applies the (already averaged) gradient in layState using the chosen optimizer
    // iter is the 1-based update count (used by ADAM's bias correction)
    void modify_weights( float learning_rate, Optimizer opt = Optimizer::GRAD,
                         size_t iter = 1, float momentum = 0.9f ) {
        update_params( weights, layState.epsilon,      m_weights, v_weights, learning_rate, opt, iter, momentum );
        update_params( bias,    layState.epsilon_bias, m_bias,    v_bias,    learning_rate, opt, iter, momentum );
    }

    virtual ~Layer() {}
};