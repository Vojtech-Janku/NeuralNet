#include "neural_net.hpp"
#include "data_io.hpp"
#include "preprocessing.hpp"
#include "metrics.hpp"
#include <chrono>
#include <string>

// creating neural net, with some prints
Neural_net make_model( vector<size_t> scheme, vector<Activation> act) {
    std::cout << "Creating neural net, scheme = ";
    print_vec( scheme );
    std::cout << ", activation = [ " << get_str( act[0] );
    for ( size_t i = 1; i < act.size(); i++ ) { std::cout << ", " << get_str( act[i] ); }
    std::cout << " ]";
    return Neural_net( scheme, act);
}

// training neural net, with some prints
void train_model( Neural_net &net, Tensor &train_data, Tensor &train_target, size_t batch_size, 
                  float learning_rate, float lr_decay, Optimizer opt, 
                  float prec, size_t epochs, float momentum ) {
    std::cout << "Training with params:     batch_size = " << batch_size << ", optimizer = " << get_str(opt)
              << ", momentum = " << momentum << ", epochs = " << epochs << ", precision = " << prec << "..." << endl;
    auto start = chrono::steady_clock::now();
    bool trained = net.train( train_data, train_target, batch_size, learning_rate, lr_decay, opt, prec, epochs, momentum );
    auto end = chrono::steady_clock::now();
    std::cout << "Computation stopped after model reached " 
              << ( (trained) ? "precision." : "maximum epochs." ) << endl;
    std::cout << "Training time: "
        << chrono::duration_cast<chrono::seconds>(end - start).count() / 60 << "min"
        << chrono::duration_cast<chrono::seconds>(end - start).count() % 60 << "sec" << endl;    
}

// original working config: 3 deep layers (MLP), no conv
void execute_mlp_workflow() {
   std::cout << "Neural network - feed-forward MLP" << endl;

    std::cout << "--- Neural net on fashion MNIST ---" << endl;
    // CONFIGURATION:
    // scheme is <input_size==784, 64, 30, 10>, activations RELU, RELU, SOFTMAX (cross-entropy gradient)
    // optimizer is momentum
    // batch size is 64, training for 20 epochs

    // READING DATA
    std::cout << "- Loading data..." << endl;
    auto train_vectors =    read_data("data/fashion_mnist_train_vectors.csv", ',');
    auto train_labels =     get_column( read_data("data/fashion_mnist_train_labels.csv", ','), 0 );

        // DATA TRANSFORMATIONS
    std::cout << "- Transforming data..." << endl;
    Tensor train_data = to_tensor( scale( train_vectors, 255 ) );
    Tensor train_target = to_tensor( transform_index( train_labels, 10 ) );

        //  CREATE NEW NEURAL NET
    std::cout << "- Neural Net" << endl;
    size_t input_size = train_vectors[0].size();
    size_t output_size = 10;
    vector<size_t> scheme = { input_size, 64, 30, output_size };
    vector<Activation> act = { Activation::RELU, Activation::RELU, Activation::SOFTMAX };
    float learning_rate = 0.01, lr_decay = 0.0002, moment = 0.9;
    Neural_net net = make_model( scheme, act);
    net.init_gauss();

        // LEARNING
    std::cout << "- Model Learning" << endl;
    size_t batch_size = 64;
    float prec = 0.1;
    size_t epochs = 20;
    Optimizer opt = Optimizer::MOMENTUM;
    train_model( net, train_data, train_target, batch_size, learning_rate, lr_decay, opt, prec, epochs, moment );

        // PREDICTION
    auto train_pred = get_max_idx( net.predict( train_data ) );
    export_data( "data/train_predictions.csv", train_pred );
    auto test_vectors = read_data("data/fashion_mnist_test_vectors.csv", ',');
    auto test_labels = get_column( read_data("data/fashion_mnist_test_labels.csv", ','), 0 );
    Tensor test_data = to_tensor( scale( test_vectors, 255 ) );
    Tensor test_target = to_tensor( transform_index( test_labels, 10 ) );

    auto test_pred = get_max_idx( net.predict( test_data ) );
    export_data( "data/test_predictions.csv", test_pred );

        // MODEL EVALUATION
    std::cout << "- Model Evaluation" << endl;
    std::cout << "Training set accuracy:   " << get_accuracy( train_pred, train_labels ) << endl;
    std::cout << "Test set accuracy:   " << get_accuracy( test_pred, test_labels ) << endl;

    std::cout << "DONE" << endl;
}

// conv layer feeding into a deep (output) layer
void execute_conv_workflow() {
   std::cout << "Neural network - conv + deep classifier" << endl;

    std::cout << "--- Neural net on fashion MNIST ---" << endl;
    // CONFIGURATION:
    // one conv layer (5x5 kernel, RELU) over the 28x28 images -> 24x24 feature map,
    //   flattened straight into one deep (output) layer, SIGMOID, 10-way classification
    //   (backprop through more than one conv layer isn't implemented yet - see note in ConvLayer)
    // batch size is 64, training for 20 epochs

    // READING DATA
    std::cout << "- Loading data..." << endl;
    auto train_vectors =    read_data("data/fashion_mnist_train_vectors.csv", ',');
    auto train_labels =     get_column( read_data("data/fashion_mnist_train_labels.csv", ','), 0 );

        // DATA TRANSFORMATIONS
    std::cout << "- Transforming data..." << endl;
    Tensor train_data = to_tensor( scale( train_vectors, 255 ) );
    train_data.reshape( { train_data.getShape()[0], 28, 28 } );   // conv layer needs 2D images, not flat pixel rows
    Tensor train_target = to_tensor( transform_index( train_labels, 10 ) );

        //  CREATE NEW NEURAL NET
    std::cout << "- Neural Net" << endl;
    size_t input_size = train_vectors[0].size();
    size_t output_size = 10;
    size_t kernel_size = 5;
    Neural_net net( input_size );
    net.add_layer( Activation::RELU, 28, 28, kernel_size );  // conv layer: 28x28 -> 24x24
    net.add_layer( Activation::SIGMOID, output_size );       // deep layer at the end (classification)
    net.init_gauss();

        // LEARNING
    std::cout << "- Model Learning" << endl;
    size_t batch_size = 64;
    float learning_rate = 0.01, lr_decay = 0.0002, moment = 0.9;
    float prec = 0.1;
    size_t epochs = 20;
    Optimizer opt = Optimizer::MOMENTUM;
    train_model( net, train_data, train_target, batch_size, learning_rate, lr_decay, opt, prec, epochs, moment );

        // PREDICTION
    auto train_pred = get_max_idx( net.predict( train_data ) );
    export_data( "data/train_predictions.csv", train_pred );
    auto test_vectors = read_data("data/fashion_mnist_test_vectors.csv", ',');
    auto test_labels = get_column( read_data("data/fashion_mnist_test_labels.csv", ','), 0 );
    Tensor test_data = to_tensor( scale( test_vectors, 255 ) );
    test_data.reshape( { test_data.getShape()[0], 28, 28 } );
    Tensor test_target = to_tensor( transform_index( test_labels, 10 ) );

    auto test_pred = get_max_idx( net.predict( test_data ) );
    export_data( "data/test_predictions.csv", test_pred );

        // MODEL EVALUATION
    std::cout << "- Model Evaluation" << endl;
    std::cout << "Training set accuracy:   " << get_accuracy( train_pred, train_labels ) << endl;
    std::cout << "Test set accuracy:   " << get_accuracy( test_pred, test_labels ) << endl;

    std::cout << "DONE" << endl;
}


// tests live in tests/ - run them with: ctest --test-dir build
int main() {
    try {
        //execute_conv_workflow();
        execute_mlp_workflow();
    } catch ( const std::exception &e ) {   // e.g. a missing data file or an invalid network setup
        std::cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}