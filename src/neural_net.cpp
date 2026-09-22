#include "neural_net.hpp"
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>

pair<size_t, size_t> get_dimensions( string path, char del ) { 
    size_t row_count = 1, col_count = 1;
    string line;
    ifstream myfile( path );
    if ( !getline(myfile, line) ) return make_pair( 0, 0 );
    for ( char c : line ) { if ( c == del ) col_count++; }
    while ( getline(myfile, line) ) row_count++;
    return make_pair( row_count, col_count );
}

// data input / output functions
void SkipBOM(std::fstream &in)
{
    char test[3] = {0};
    in.read(test, 3);
    if ((unsigned char)test[0] == 0xEF && 
        (unsigned char)test[1] == 0xBB && 
        (unsigned char)test[2] == 0xBF)
    { return; }
    in.seekg(0);
}

auto read_data( string path, char del ) {
    string line, word;
    fstream fin( path, ios::in );
    assert( fin.is_open() );
    auto [rows, cols] = get_dimensions( path, del );
    matrix<int> table( rows, vector<int>( cols ) ); // initiate whole table at once

    SkipBOM( fin );
    size_t row = 0, col = 0;
    while ( getline( fin, line ) ) {
        col = 0;
        stringstream str(line);
        while( getline(str, word, ',') ) {
            table[row][col] = stoi(word);
            col++;
        }
        row++;
    }
    return table;
}

template<typename T>
void export_data( string path, const matrix<T> &data ) {
    ofstream output_file( path );
    size_t n_rows = data.size(), n_cols = data[0].size();
    for ( size_t r = 0; r < n_rows; r++ ) {
        output_file << data[r][0];
        for ( size_t c = 1; c < n_cols; c++ ) {
            output_file << "," ;
            output_file << data[r][c];
        }
        output_file << endl;
    }   
}

template<typename T>
void export_data( string path, const vector<T> &data ) {
    ofstream output_file( path );
    size_t n_rows = data.size();
    for ( size_t r = 0; r < n_rows; r++ ) {
        output_file << data[r] << endl;
    }   
}

// data transformation functions
auto scale( const matrix<int> &data_mat, float scale ) {
    size_t n_rows = data_mat.size(), n_cols = data_mat[0].size();
    matrix<float> res( n_rows, vector<float>( n_cols ) );
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            res[i][j] = data_mat[i][j] / scale;
        }
    }
    return res;
}

auto transform_index( const vector<int> &index_vec, size_t max ) {
    matrix<float> res( index_vec.size(), std::vector<float>(max, 0) );
    for (size_t i = 0; i < res.size(); i++) {
        res[i][ index_vec[i] ] = 1;
    }
    return res;
}

// functions for computing accuracy
size_t count_same( const vector<int> &pred, const vector<int> &target ) {
    assert( pred.size() == target.size() );
    size_t same = 0;
    for (size_t i = 0; i < pred.size(); i++) {
        if ( pred[i] == target[i] ) same++;
    }
    return same;
}

float get_accuracy( const vector<int> &pred, const vector<int> &target ) {
    return ( count_same( pred, target ) / (float) pred.size() );
}

// minimal XOR from lecture, with step activation function
void test_minimal_XOR() {
    std::cout << "Testing minimal (lecture) XOR:" << endl;
    // vector<size_t> scheme = {2, 2, 1};
    // vector<Activation> act = { Activation::STEP, Activation::STEP };
    // Neural_net net( scheme, act );
    Neural_net net( 2 );
    net.add_layer( Activation::STEP, 2 );
    net.add_layer( Activation::STEP, 1 );
        // SET WEIGHTS AND BIASES
    net.getLayers().at(0)->set_weights( Tensor( { 2, 2 }, { 2, 2, -2, -2 } ) );
    net.getLayers().at(1)->set_weights( Tensor( { 1, 2 }, { 1, 1 } ) );
    net.getLayers().at(0)->set_biases( Tensor( { 1 }, { 1, 3 } ) );
    net.getLayers().at(1)->set_biases( Tensor( { 1 }, { -2 } ) );
        // DATA
    Tensor points = to_tensor( { {0,0}, {0,1}, {1,0}, {1,1} } );
    Tensor expected = to_tensor( { {0}, {1}, {1}, {0} } );
        // RESULT
    for ( size_t i = 0; i < points.getShape()[0]; i++ ) {
        assert( net.feed_forward( points.row(i) ) == expected.row(i) );
    }
    std::cout << "PASSED" << endl;
}

// simple neural net with 1 conv layer
void test_simple_conv_layer() {
    std::cout << "Testing simple conv net:" << endl;
    // vector<size_t> scheme = {2, 3, 1};
    // vector<Activation> act = { Activation::STEP, Activation::STEP };
    // Neural_net net( scheme, act );

    Neural_net net( 2 );
    net.add_layer( Activation::RELU, 3 );
    net.add_layer( Activation::RELU, 1 );
        // SET WEIGHTS AND BIASES
    net.getLayers().at(0)->set_weights( Tensor( { 3, 2 }, { 2, 2, -2, -2, 1, -1 } ) );
    net.getLayers().at(1)->set_weights( Tensor( { 1, 3 }, { 1, 1, 1 } ) );
    net.getLayers().at(0)->set_biases( Tensor( { 3 }, { -1, 3, 0 } ) );
    net.getLayers().at(1)->set_biases( Tensor( { 1 }, { -2 } ) );
        // DATA
    Tensor points = to_tensor( { {0,0}, {0,1}, {1,0}, {1,1} } );
    Tensor expected = to_tensor( { {0}, {1}, {1}, {0} } );
        // RESULT
    for ( size_t i = 0; i < points.getShape()[0]; i++ ) {
        assert( net.feed_forward( points.row(i) ) == expected.row(i) );
    }
    std::cout << "PASSED" << endl;
}

// XOR solved by a more general network
void test_XOR_backprop( Activation a, float lr, size_t epochs = 10000000 ) {
    std::cout << "Testing XOR (activation = " << get_str( a ) 
              << ") with backpropagation:" << endl;
        //  CREATE NEW NEURAL NET
    vector<size_t> scheme = {2, 5, 1};
    vector<Activation> act = { a, a };
    Neural_net net( scheme, act);
    net.init_unif( 0, 1 );
        // DATA
    Tensor data = to_tensor({
        {0,0}, {0,1}, {1,0}, {1,1}
    });
    //Tensor expected = to_tensor( { {0.001}, {0.999}, {0.999}, {0.001} } );
    Tensor expected = to_tensor( { {0}, {1}, {1}, {0} } );
        // LEARNING
    bool trained = net.train( data, expected, data.getShape()[0], lr, 0.001, Optimizer::ADAM, 0.01, epochs );
        // RESULT
    Tensor pred = net.predict( data );
    assert( trained );
    std::cout << "PASSED" << endl;
}

// creating neural net, with some prints
Neural_net make_model( vector<size_t> scheme, vector<Activation> act) {
    std::cout << "Creating neural net, scheme = ";
    print_vec( scheme );
    std::cout << ", activation = [ " << get_str( act[0] );
    for ( size_t i = 1; i < act.size(); i++ ) { std::cout << ", " << get_str( act[i] ); }
    std::cout << " ]";
    return Neural_net( scheme, act);
}

// creating neural net, with some prints
Neural_net make_conv_model( vector<size_t> scheme, vector<Activation> act) {
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
                  float prec, size_t epochs) {
    std::cout << "Training with params:     batch_size = " << batch_size << ", optimizer = " << get_str(opt)
              << ", epochs = " << epochs << ", precision = " << prec << "..." << endl;
    auto start = chrono::steady_clock::now();
    bool trained = net.train( train_data, train_target, batch_size, learning_rate, lr_decay, opt, prec, epochs);
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
    // WORKING (final) CONFIGURATION:
    // scheme is <input_size==784, 64, 30, 10>, all activation is RELU
    //       (should probably use softmax for output layer but didn't implement it)
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
    vector<Activation> act = { Activation::RELU, Activation::RELU, Activation::SIGMOID };
    float learning_rate = 0.01, lr_decay = 0.0002, moment = 0.9;
    Neural_net net = make_model( scheme, act);
    net.init_gauss();

        // LEARNING
    std::cout << "- Model Learning" << endl;
    size_t batch_size = 64;
    float prec = 0.1;
    size_t epochs = 20;
    Optimizer opt = Optimizer::MOMENTUM;
    train_model( net, train_data, train_target, batch_size, learning_rate, lr_decay, opt, prec, epochs );

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
    // batch size is 20, training for 20 epochs

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
    train_model( net, train_data, train_target, batch_size, learning_rate, lr_decay, opt, prec, epochs );

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
    // using test data for the purpose of calculating and printing accuracy
    // - !!! COMMENT THIS SECTION BEFORE SUBMITTING !!!
    
    std::cout << "- Model Evaluation" << endl;
    // train pred accuracy
    std::cout << "Training set accuracy:   " << get_accuracy( train_pred, train_labels ) << endl;
    // test pred accuracy
    std::cout << "Test set accuracy:   " << get_accuracy( test_pred, test_labels ) << endl;
    
    std::cout << "DONE" << endl; 
}


int main() {
    execute_conv_workflow();
    //execute_mlp_workflow();
    return 0;
}