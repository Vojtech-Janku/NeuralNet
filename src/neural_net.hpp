#include <cassert>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include "utils.hpp"
#include "Layer.hpp"

using namespace std;

enum LayerType{ DEEP, CONVOLUTIONAL, MAXPOOLING, NORMALIZATION, ATTENTION };

// optimizers
enum Optimizer{ GRAD, MOMENTUM, ADAM };
// just for printing
string get_str( Optimizer opt ) {
    switch (opt) {
    case Optimizer::GRAD:          return "GRAD";
    case Optimizer::MOMENTUM:      return "MOMENTUM";
    case Optimizer::ADAM:          return "ADAM";
    default:                       return "Unknown";
    }
}

// Neural_net - the class for the whole neural network.
// Contains all layers, feed forward function,
//  train and predict functions
class Neural_net
{
    float learning_rate, lr_decay;
    float momentum;
    size_t input_size;         // number of neurons in input layer
    vector<size_t> net_scheme; // network scheme excluding input layer for practical reasons
    vector<Activation> act_funs;
    vector<DeepLayer> layers;

    float beta1 = 0.9, beta2 = 0.999, eps = 0.00000001; // for ADAM optimizer

public:
    Neural_net( size_t input_size)
    : learning_rate(0.01), lr_decay(0.001), momentum(0.5), input_size(input_size) {}

    Neural_net( vector<size_t> scheme, vector<Activation> funs, float l_rate = 0.01, float l_decay = 0.001, float moment = 0.5 ) 
    : learning_rate( l_rate ), lr_decay( l_decay ), momentum(moment), input_size( scheme[0] ), 
      net_scheme( scheme.begin(), scheme.end() ), act_funs( funs ) {
        assert( scheme.size() > 1 );
        for ( size_t i = 1; i < scheme.size(); i++ ) {
            add_layer( LayerType::DEEP, act_funs[i-1], scheme[i] );
        }
    }

    vector<size_t> getScheme()
    {
        return net_scheme;
    }

    vector<DeepLayer> &getLayers()
    {
        return layers;
    }

    // TODO: add LayerType and a logic for Convolutional layer
    // TODO: dont need to specify input size every time, can get it from previous layer
    //void add_layer( size_t layer_size, size_t input_size, Activation a ) {
    //    layers.push_back(     DeepLayer( layer_size, input_size, a ) );
    //}

    void add_layer( LayerType layer_type, Activation a, size_t layer_size ) {

        size_t layer_input = getLayers().empty() ? getScheme().at(0) : getLayers().back().getSize();
        switch (layer_type)
        {
        case LayerType::DEEP:
            layers.push_back( DeepLayer( layer_size, layer_input, a ) );
            break;
        case LayerType::CONVOLUTIONAL:
            //layers.push_back( ConvLayer( layer_size, layer_input, a ) );
            break;        
        default:
            break;
        }
        
    }

    void init_unif( float min = 0, float max = 0.1 ) {
        for ( auto &lay : layers ) { lay.initialize_uniform(min, max); }
    }

    void init_gauss() {
        for ( auto &lay : layers )
        {
            lay.initialize_gauss( 0, sqrt( 2.0 / lay.getInputSize() ) );
        }
    }

    // basic feed forward algorithm
    const Tensor &feed_forward( const Tensor &input ) {
        layers[0].compute_potential( input );
        for ( size_t i = 1; i < layers.size(); i++ ) {
            layers[i].compute_potential( layers[i-1].layState.output );
        }
        return layers.back().layState.output;
    }

    // computes error function output derivatives
    void backpropagation( const Tensor &target_point ) {
        for ( size_t n = 0; n < layers.back().getSize(); n++ ) {  // y_j - d_kj
            layers.back().layState.err_output[n] = layers.back().layState.output[n] - target_point[n];
        }
        for ( int lay = layers.size()-2; lay >= 0; --lay ) {
          #pragma omp parallel for num_threads(16)                    // multiprocessing
            for ( size_t j = 0; j < layers[lay].getSize(); j++ ) {
                float sum = 0;
                for ( size_t r = 0; r < layers[lay+1].getSize(); r++ ) {
                    sum += layers[lay+1].layState.err_output[r] 
                        * layers[lay+1].layState.derivative[r] 
                        * layers[lay+1].getWeights().at(r,j);
                }
                layers[lay].layState.err_output[j] = sum;
            }
        }        
    }

    // computes gradient for whole network, one training example
    void compute_epsilon( const Tensor &data_row ) {
        layers[0].compute_epsilon( data_row );
        for ( size_t lay = 1; lay < layers.size(); lay++ ) {
            layers[lay].compute_epsilon( layers[lay-1].layState.output );
        }        
    }

    // computes all activation functions derivatives
    void compute_derivatives() {
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
            layers[lay].compute_derivative();
        }        
    }

    // computes gradient for given data batch
    void compute_gradient( const Tensor &data, const Tensor &labels, 
                            pair<size_t,size_t> batch_range ) {
        // initialize epsilon = 0;
        for ( DeepLayer &lay : layers ) {
            lay.layState.epsilon.clear();
            lay.layState.epsilon_bias.clear();
        }
        // total squared error
        //      float err = 0;
        // go through training data
        for ( size_t k = batch_range.first; k < batch_range.second; k++ ) {
            feed_forward( data.row(k) );
            compute_derivatives();
            backpropagation( labels.row(k) );
            compute_epsilon( data.row(k) );
        }
        // average the gradient
        for ( DeepLayer &lay : layers ) {
            lay.layState.epsilon / ( batch_range.second-batch_range.first );
        }
    }
    // just overload
    void compute_gradient( const Tensor &data, const Tensor &labels ) {
        compute_gradient( data, labels, make_pair(0, data.getShape()[1] ) );
    }

    void compute_single_adam( float &m, float &v, const float &epsilon, 
                              float beta1, float beta2, float eps ) {
                m = ( beta1*m + (1 - beta1)*epsilon );
                v = ( beta2*v + (1 - beta2)*epsilon*epsilon );   
    }

    /*
    void compute_adam() {
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
          #pragma omp parallel for num_threads(16)                    // multiprocessing 
            for ( size_t j = 0; j < layers[lay].getSize(); j++ ) {
                for ( size_t i = 0; i < layers[lay].getInputSize(); i++ ) {
                    compute_single_adam( layers[lay].layState.m.at(j,i), layers[lay].layState.v.at(j,i), 
                                         layers[lay].layState.epsilon.at(j,i), beta1, beta2, eps );
                }
                compute_single_adam( layers[lay].layState.m_bias[j], layers[lay].layState.v_bias[j], 
                                         layers[lay].layState.epsilon_bias[j], beta1, beta2, eps );
            }
        } 
    }
    */

    // ---- single weight update functions for optimizers ---
    void update_gradient_descent( float &weight, const float &gradient ) {
        weight -= learning_rate*gradient;
    }

    void update_momentum( float &weight, const float &gradient, float &m) {
        m = ( momentum*m + learning_rate*gradient );
        weight -= m;
    }

    void update_adam( float &weight, const float &m, const float &v, const size_t &it ) {
        float mhat = m / (1 - powf(beta1, it) ), vhat = v / (1 - powf(beta2, it) );
        weight -= learning_rate * mhat / ( sqrt( vhat ) + eps );
    }

    // updates all weights
    void modify_weights( Optimizer opt, const size_t &it = 0 ) {
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
          #pragma omp parallel for num_threads(16)                    // multiprocessing
            for ( size_t j = 0; j < layers[lay].getSize(); j++ ) {
                for ( size_t i = 0; i < layers[lay].getInputSize(); i++ ) {
                    switch (opt)
                    {
                    case Optimizer::GRAD:
                        update_gradient_descent( layers[lay].getWeights().at(j,i), layers[lay].layState.epsilon.at(j,i) );
                        break;
                    /*
                    case Optimizer::MOMENTUM:
                        update_momentum( layers[lay].getWeights().at(j,i), layers[lay].layState.epsilon.at(j,i), layers[lay].layState.m.at(j,i) );
                        break;
                    case Optimizer::ADAM:
                        update_adam( layers[lay].getWeights().at(j,i), layers[lay].layState.m.at(j,i), layers[lay].layState.v.at(j,i), it );
                        break;
                    */
                    }
                }
            }
        }
    }

    bool train( const Tensor &data, const Tensor &target, 
                size_t batch_size, Optimizer opt, float precision = 0.001, size_t epochs = 100000 )
    {
        auto lr_init = learning_rate;
        float err;
        size_t batch_start;
        size_t iter = 1;
        //auto rng = std::default_random_engine {};
        //std::shuffle(std::begin(cards_), std::end(cards_), rng);
        for ( size_t i = 0; i < epochs; i++ ) {
            batch_start = 0;
            while( batch_start+batch_size < data.getShape()[0] ) {
                compute_gradient( data, target, make_pair(batch_start, batch_start+batch_size) );
                //if (opt == Optimizer::ADAM) compute_adam();
                modify_weights(opt, iter);
                iter++;
                if ( learning_rate > 0.001 ) learning_rate = lr_init * ( 1 / (1+lr_decay*iter) ); // learning rate decay
                batch_start += batch_size;
            }
            // spaghetti code but whatever
            compute_gradient( data, target, make_pair( batch_start, data.getShape()[0] ) );
            //if (opt == Optimizer::ADAM) compute_adam();
            modify_weights(opt, iter);
            iter++;
            if ( learning_rate > 0.001 ) learning_rate = lr_init * ( 1 / (1+lr_decay*iter) ); // learning rate decay

            err = total_squared_error( data, target );
            //if ( i % 10 == 0 ) 
                //std::cout << " Epoch " << i << ", total error = " << err << ", lrate = " << learning_rate << endl;
            if ( err < precision ) return true;
        }
        return false;   
    }

    Tensor predict( const Tensor &data ) {
        size_t out_size = layers.back().getSize();
        Tensor pred( { data.getShape()[0], out_size } );
        for (size_t k = 0; k < data.getShape()[0]; k++) {
            const Tensor &out = feed_forward( data.row(k) );
            for (size_t j = 0; j < out_size; j++) pred.at(k,j) = out[j];
        }
        return pred;
    }

    int get_layer_count() {
        return layers.size();
    }

    float output_squared_error( const Tensor &target ) {
        float err = 0;
        for ( size_t i = 0; i < net_scheme.back(); i++) {
            err += ( target[i] - layers.back().layState.output[i] ) 
                 * ( target[i] - layers.back().layState.output[i] );
        }
        return err;
    }

    float total_squared_error( const Tensor &data, const Tensor &target ) {
        float err = 0;
        for (size_t i = 0; i < data.getShape()[0]; i++) {
            feed_forward( data.row(i) );
            err += output_squared_error( target.row(i) );
        }
        return err / data.getShape()[0];
    }  

    //  ----------  PRINT FUNCTIONS  ----------
    void print() {
        std::cout << "Weights:" << endl;
        for ( size_t lay = 1; lay < layers.size(); lay++ ) {
            std::cout << "-------------" << endl;
            for ( size_t i = 0; i < layers[lay].getSize(); i++ ) {
                print_vec( layers[lay].getWeights().row(i) );
                std::cout << "  [ " << layers[lay].getBias()[i] << " ]" << endl;
            }
        }
        std::cout << endl;
    }

    void print_gradient() {
        std::cout << "Gradient:" << endl;
        for ( size_t lay = 1; lay < layers.size(); lay++ ) {
            std::cout << "-------------" << endl;
            for ( size_t i = 0; i < layers[lay].getSize(); i++ ) {
                print_vec( layers[lay].layState.epsilon.row(i) );
                std::cout << "  [ " << layers[lay].layState.epsilon_bias[i] << " ]" << endl;
            }
        }
        std::cout << endl;
    }

    void print_output() {
        std::cout << "Output:" << endl;
        for ( size_t lay = 0; lay < layers.size(); lay++ ) {
            std::cout << "-------------" << endl;
            print_vec( layers[lay].layState.output );
            std::cout << endl;
        }
        std::cout << endl;
    }
};

