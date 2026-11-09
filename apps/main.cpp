#include "neural_net.hpp"
#include "data_io.hpp"
#include "preprocessing.hpp"
#include "metrics.hpp"
#include <chrono>
#include <functional>
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

// ---------- Fashion MNIST experiments ----------

const size_t IMAGE_SIDE  = 28;   // Fashion MNIST images are 28x28 grayscale
const size_t NUM_CLASSES = 10;

// one split of the dataset (train or test), ready for the network
struct Dataset {
    Tensor inputs;        // {samples, 784}, or {samples, 28, 28} when loaded as images
    Tensor targets;       // one-hot labels, {samples, 10}
    vector<int> labels;   // class index of each sample
};

// loads a vectors + labels CSV pair, scales pixels to [0, 1] and one-hot encodes the labels
Dataset load_dataset( const string &vectors_path, const string &labels_path, bool as_images ) {
    auto vectors = read_data( vectors_path, ',' );
    auto labels  = get_column( read_data( labels_path, ',' ), 0 );
    Tensor inputs = to_tensor( scale( vectors, 255 ) );
    if ( as_images ) {
        inputs.reshape( { inputs.getShape()[0], IMAGE_SIDE, IMAGE_SIDE } );   // conv layers need 2D images
    }
    return { inputs, to_tensor( transform_index( labels, NUM_CLASSES ) ), labels };
}

// everything that differs between experiments
struct Experiment {
    string title;
    bool as_images;                                        // feed 28x28 images instead of flat 784 vectors
    std::function<Neural_net( size_t input_size )> build;  // creates the network (before weight init)
};

// MLP: <784, 64, 30, 10>, activations RELU, RELU, SOFTMAX (cross-entropy gradient)
Experiment mlp_experiment() {
    return { "feed-forward MLP", false, []( size_t input_size ) {
        vector<size_t> scheme = { input_size, 64, 30, NUM_CLASSES };
        vector<Activation> act = { Activation::RELU, Activation::RELU, Activation::SOFTMAX };
        return make_model( scheme, act );
    } };
}

// one conv layer (5x5 kernel, RELU) over the 28x28 images -> 24x24 feature map,
//   flattened straight into one deep (output) layer, SIGMOID, 10-way classification
//   (backprop through more than one conv layer isn't implemented yet - see note in ConvLayer)
Experiment conv_experiment() {
    return { "conv + deep classifier", true, []( size_t input_size ) {
        size_t kernel_size = 5;
        Neural_net net( input_size );
        net.add_layer( Activation::RELU, IMAGE_SIDE, IMAGE_SIDE, kernel_size );  // conv layer: 28x28 -> 24x24
        net.add_layer( Activation::SIGMOID, NUM_CLASSES );                       // deep layer at the end (classification)
        return net;
    } };
}

// trains on the train split, writes predictions for both splits to data/ and prints their accuracy
void run_experiment( const Experiment &exp ) {
    std::cout << "Neural network - " << exp.title << endl;
    std::cout << "--- Neural net on fashion MNIST ---" << endl;

    std::cout << "- Loading data..." << endl;
    Dataset train = load_dataset( "data/fashion_mnist_train_vectors.csv",
                                  "data/fashion_mnist_train_labels.csv", exp.as_images );

    std::cout << "- Neural Net" << endl;
    Neural_net net = exp.build( IMAGE_SIDE * IMAGE_SIDE );
    net.init_gauss();

    // training params - the same for every experiment for now (moving to a TrainConfig is planned)
    std::cout << "- Model Learning" << endl;
    size_t batch_size = 64, epochs = 20;
    float learning_rate = 0.01, lr_decay = 0.0002, moment = 0.9, prec = 0.1;
    Optimizer opt = Optimizer::MOMENTUM;
    train_model( net, train.inputs, train.targets, batch_size, learning_rate, lr_decay, opt, prec, epochs, moment );

    auto train_pred = get_max_idx( net.predict( train.inputs ) );
    export_data( "data/train_predictions.csv", train_pred );

    Dataset test = load_dataset( "data/fashion_mnist_test_vectors.csv",
                                 "data/fashion_mnist_test_labels.csv", exp.as_images );
    auto test_pred = get_max_idx( net.predict( test.inputs ) );
    export_data( "data/test_predictions.csv", test_pred );

    std::cout << "- Model Evaluation" << endl;
    std::cout << "Training set accuracy:   " << get_accuracy( train_pred, train.labels ) << endl;
    std::cout << "Test set accuracy:   " << get_accuracy( test_pred, test.labels ) << endl;

    std::cout << "DONE" << endl;
}


// tests live in tests/ - run them with: ctest --test-dir build
int main() {
    try {
        //run_experiment( conv_experiment() );
        run_experiment( mlp_experiment() );
    } catch ( const std::exception &e ) {   // e.g. a missing data file or an invalid network setup
        std::cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}