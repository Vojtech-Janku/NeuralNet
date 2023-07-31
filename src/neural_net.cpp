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
    vector<size_t> scheme = {2, 2, 1};
    vector<Activation> act = { Activation::STEP, Activation::STEP };
    Neural_net net( scheme, act );
        // SET WEIGHTS AND BIASES
    net.set_weights( 0, { { 2, 2 }, { -2, -2 } } );
    net.set_weights( 1, { { 1, 1 } } );
    net.set_biases( 0, { -1, 3 } );
    net.set_biases( 1, { -2 } );
        // DATA
    matrix<float> points = { {0,0}, {0,1}, {1,0}, {1,1} };
    matrix<float> expected = { {0}, {1}, {1}, {0} };
        // RESULT
    for ( size_t i = 0; i < points.size(); i++ ) {
        assert( net.feed_forward( points[i] ) == expected[i] );
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
    Neural_net net( scheme, act, lr );
    net.init_unif( 0, 1 );
        // DATA
    matrix<float> data = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    //matrix<float> expected = { {0.001}, {0.999}, {0.999}, {0.001} };
    matrix<float> expected = { {0}, {1}, {1}, {0} };
        // LEARNING
    bool trained = net.train( data, expected, data.size(), Optimizer::ADAM, 0.01, epochs );
        // RESULT
    matrix<float> pred = net.predict( data );
    assert( trained );
    std::cout << "PASSED" << endl;
}

// creating neural net, with some prints
Neural_net make_model( vector<size_t> scheme, vector<Activation> act, float learning_rate, float lr_decay, float momentum ) {
    std::cout << "Creating neural net, scheme = ";
    print_vec( scheme );
    std::cout << ", activation = [ " << get_str( act[0] );
    for ( size_t i = 1; i < act.size(); i++ ) { std::cout << ", " << get_str( act[i] ); }
    std::cout << " ], learning_rate = " << learning_rate << ", lr_decay = " << lr_decay << ", momentum = " << momentum << endl;
    return Neural_net( scheme, act, learning_rate, lr_decay, momentum );
}

// training neural net, with some prints
void train_model( Neural_net &net, matrix<float> &train_data, matrix<float> &train_target, 
                  size_t batch_size, Optimizer opt, float prec, size_t epochs ) {
    std::cout << "Training with params:     batch_size = " << batch_size << ", optimizer = " << get_str(opt)
              << ", epochs = " << epochs << ", precision = " << prec << "..." << endl;
    auto start = chrono::steady_clock::now();
    bool trained = net.train( train_data, train_target, batch_size, opt, prec, epochs );
    auto end = chrono::steady_clock::now();
    std::cout << "Computation stopped after model reached " 
              << ( (trained) ? "precision." : "maximum epochs." ) << endl;
    std::cout << "Training time: "
        << chrono::duration_cast<chrono::seconds>(end - start).count() / 60 << "min"
        << chrono::duration_cast<chrono::seconds>(end - start).count() % 60 << "sec" << endl;    
}

int main() {
    std::cout << "Neural network - feed-forward MLP" << endl;

    // TESTS
    //test_minimal_XOR();
    //test_XOR_backprop( Activation::SIGMOID, 0.01 );
    //test_XOR_backprop( Activation::RELU, 0.001 );

    std::cout << "--- Neural net on fashion MNIST ---" << endl;
    // WORKING (final) CONFIGURATION:
    // scheme is <input_size==784, 50, 30, 10>, all activation is RELU 
    //       (should probably use softmax for output layer but didn't implement it) 
    // learning rate is 0.1 with decay rate 0.0002 
    // optimizer is momentum, m = 0.5
    // weights initialized uniformly from 0 to 2/784, biases 5x higher
    // batch size is 20, training for 20 epochs

    // READING DATA
    std::cout << "- Loading data..." << endl;
    auto train_vectors =    read_data("../data/fashion_mnist_train_vectors.csv", ',');
    auto train_labels =     get_column( read_data("../data/fashion_mnist_train_labels.csv", ','), 0 );

        // DATA TRANSFORMATIONS
    std::cout << "- Transforming data..." << endl;
    auto train_data = scale( train_vectors, 255 ) ;
    auto train_target = transform_index( train_labels, 10 );

        //  CREATE NEW NEURAL NET
    std::cout << "- Neural Net" << endl;
    size_t input_size = train_vectors[0].size();
    vector<size_t> scheme = { input_size, 50, 30, 10 };
    vector<Activation> act = { Activation::RELU, Activation::RELU, Activation::RELU };
    // hyper parameters
    float learning_rate = 0.1, lr_decay = 0.0002, moment = 0.5;
    Neural_net net = make_model( scheme, act, learning_rate, lr_decay, moment );
    net.init_unif( 0, 2.0 / input_size );
    //net.init_gauss();

        // LEARNING
    std::cout << "- Model Learning" << endl;
    size_t batch_size = 20;
    float prec = 0.1;
    size_t epochs = 20;
    Optimizer opt = Optimizer::MOMENTUM;
    train_model( net, train_data, train_target, batch_size, opt, prec, epochs );

        // PREDICTION
    auto train_pred = get_max_idx( net.predict( train_data ) );
    export_data( "../train_predictions.csv", train_pred );
    auto test_data = scale( read_data("../data/fashion_mnist_test_vectors.csv", ','), 255 ) ;
    auto test_pred = get_max_idx( net.predict( test_data ) );
    export_data( "../test_predictions.csv", test_pred );

        // MODEL EVALUATION
    // using test data for the purpose of calculating and printing accuracy
    // - !!! COMMENT THIS SECTION BEFORE SUBMITTING !!!
    /*
    std::cout << "- Model Evaluation" << endl;
    // train pred accuracy
    std::cout << "Training set accuracy:   " << get_accuracy( train_pred, train_labels ) << endl;
    // test pred accuracy
    std::cout << "Test set accuracy:   " << get_accuracy( test_pred, test_labels ) << endl;
    */
    std::cout << "DONE" << endl;
    return 0;
}