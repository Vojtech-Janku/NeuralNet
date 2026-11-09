// Tests ported from the old `./network test` mode (run_tests() in neural_net.cpp).
// Kept in one file for now: activation.hpp / utils.hpp still define non-inline functions,
// so including them from a second test .cpp would break linking. Split up after the header cleanup.
#include <gtest/gtest.h>
#include <cmath>
#include "neural_net.hpp"

// ---------- forward pass with hand-set weights ----------

// minimal XOR from the lecture: step activation, fixed weights
TEST( Network, MinimalXorForward ) {
    Neural_net net( 2 );
    net.add_layer( Activation::STEP, 2 );
    net.add_layer( Activation::STEP, 1 );
    net.getLayers().at(0)->set_weights( Tensor( { 2, 2 }, { 2, 2, -2, -2 } ) );
    net.getLayers().at(1)->set_weights( Tensor( { 1, 2 }, { 1, 1 } ) );
    net.getLayers().at(0)->set_biases( Tensor( { 2 }, { -1, 3 } ) );   // neuron 1 = OR, neuron 2 = NAND
    net.getLayers().at(1)->set_biases( Tensor( { 1 }, { -2 } ) );

    Tensor points = to_tensor( { {0,0}, {0,1}, {1,0}, {1,1} } );
    const float expected[] = { 0, 1, 1, 0 };
    for ( size_t i = 0; i < points.getShape()[0]; i++ ) {
        EXPECT_FLOAT_EQ( net.feed_forward( points.row(i) )[0], expected[i] ) << "input #" << i;
    }
}

// 3x3 input, 2x2 kernel, stride 1, no padding -> 2x2 output
TEST( ConvLayer, ForwardPass ) {
    Neural_net net( 9 );
    net.add_layer( Activation::RELU, 3, 3, 2 );
    // kernel [[1,0],[0,1]] sums each window's top-left and bottom-right pixel
    net.getLayers().at(0)->set_weights( Tensor( { 2, 2 }, { 1, 0, 0, 1 } ) );
    net.getLayers().at(0)->set_biases( Tensor( { 1 }, { -7 } ) );

    Tensor image( { 3, 3 }, { 1, 2, 3,
                              4, 5, 6,
                              7, 8, 9 } );
    // raw sums: 1+5=6, 2+6=8, 4+8=12, 5+9=14; minus bias 7 -> -1, 1, 5, 7; ReLU clips -1 to 0
    const float expected[] = { 0, 1, 5, 7 };
    const Tensor &out = net.feed_forward( image );
    ASSERT_EQ( out.getSize(), 4u );
    for ( size_t i = 0; i < 4; i++ ) EXPECT_FLOAT_EQ( out[i], expected[i] ) << "output #" << i;
}

// ---------- training ----------

// XOR learned by backprop (2-5-1 net, Adam), once per activation function
class XorBackprop : public ::testing::TestWithParam<Activation> {};

TEST_P( XorBackprop, Converges ) {
    Layer::seed( 42 );   // same starting weights regardless of test order
    Neural_net net( { 2, 5, 1 }, { GetParam(), GetParam() } );
    net.init_unif( 0, 1 );
    Tensor data     = to_tensor( { {0,0}, {0,1}, {1,0}, {1,1} } );
    Tensor expected = to_tensor( { {0},   {1},   {1},   {0} } );

    bool trained = net.train( data, expected, data.getShape()[0], 0.01, 0.001, Optimizer::ADAM, 0.01, 20000 );
    ASSERT_TRUE( trained ) << "did not reach precision within 20000 epochs";

    Tensor pred = net.predict( data );
    for ( size_t i = 0; i < 4; i++ ) {
        EXPECT_NEAR( pred.at(i,0), expected.at(i,0), 0.5 ) << "input #" << i;   // rounds to the right class
    }
}

INSTANTIATE_TEST_SUITE_P( Activations, XorBackprop,
    ::testing::Values( Activation::RELU, Activation::LEAKY_RELU, Activation::SIGMOID, Activation::TANH ),
    []( const ::testing::TestParamInfo<Activation> &info ) {
        switch ( info.param ) {
        case Activation::RELU:       return std::string( "Relu" );
        case Activation::LEAKY_RELU: return std::string( "LeakyRelu" );
        case Activation::SIGMOID:    return std::string( "Sigmoid" );
        case Activation::TANH:       return std::string( "Tanh" );
        default:                     return std::string( "Other" );
        }
    } );

// ---------- gradient check ----------

// cross-entropy loss of one sample
static float cross_entropy( Neural_net &net, const Tensor &x, const Tensor &target ) {
    const Tensor &y = net.feed_forward( x );
    float loss = 0;
    for ( size_t i = 0; i < y.getSize(); i++ ) loss -= target[i] * std::log( y[i] );
    return loss;
}

// backprop gradients of a ReLU -> Softmax net must match central finite differences of the CE loss
TEST( Gradient, SoftmaxCrossEntropyMatchesFiniteDifferences ) {
    Layer::seed( 42 );
    Neural_net net( 2 );
    net.add_layer( Activation::RELU, 4 );
    net.add_layer( Activation::SOFTMAX, 3 );
    net.init_gauss();
    Tensor data   = to_tensor( { {0.5f, -0.3f} } );
    Tensor labels = to_tensor( { {0, 1, 0} } );

    net.compute_gradient( data, labels );   // analytic gradient -> layState.epsilon

    const float h = 1e-3f;
    for ( size_t lay = 0; lay < net.getLayers().size(); lay++ ) {
        Layer &layer = *net.getLayers()[lay];
        Tensor &w = layer.getWeights();
        for ( size_t i = 0; i < w.getSize(); i++ ) {
            float original = w[i];
            w[i] = original + h;  float loss_plus  = cross_entropy( net, data.row(0), labels.row(0) );
            w[i] = original - h;  float loss_minus = cross_entropy( net, data.row(0), labels.row(0) );
            w[i] = original;
            float numeric  = ( loss_plus - loss_minus ) / ( 2*h );
            float analytic = layer.layState.epsilon[i];
            EXPECT_NEAR( analytic, numeric, 1e-3f + 1e-2f * std::fabs(numeric) )
                << "layer " << lay << ", weight " << i;
        }
    }
}
