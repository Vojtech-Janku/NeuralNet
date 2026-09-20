#include <cmath>
#include <vector>
#include "Tensor.hpp"

using namespace std;

template< typename T >
using matrix = vector<vector<T>>;

float TOLERANCE = 0.00001;

// vector and matrix util
bool operator==( const vector<float> &a, const vector<float> &b) {
    if ( a.size() != b.size() ) return false;
    for ( size_t i = 0; i < a.size(); i++) {
        if ( abs( a[i] - b[i] ) > TOLERANCE ) return false;
    }
    return true;
}

bool operator==( const Tensor &a, const Tensor &b) {
    if ( a.getSize() != b.getSize() ) return false;
    for ( size_t i = 0; i < a.getSize(); i++) {
        if ( abs( a[i] - b[i] ) > TOLERANCE ) return false;
    }
    return true;
}

// flattens a matrix<float> into a Tensor of shape {rows, cols}
Tensor to_tensor( const matrix<float> &m ) {
    size_t rows = m.size(), cols = rows ? m[0].size() : 0;
    vector<float> flat;
    flat.reserve( rows*cols );
    for ( auto &row : m ) {
        for ( auto v : row ) flat.push_back(v);
    }
    return Tensor( { rows, cols }, flat );
}

template< typename T >
void vec_div( vector<T> &vec, int d ) {
    for ( size_t v = 0; v < vec.size(); v++) {
        vec[v] = vec[v] / d;
    }
}

template< typename T >
void mat_div( matrix<T> &mat, int d ) {
    for ( size_t m = 0; m < mat.size(); m++) {
        vec_div( mat[m], d );
    }
}

template< typename T >
vector<T> get_column( const matrix<T> &mat, size_t col ) {
    vector<T> res( mat.size() );
    for (size_t i = 0; i < mat.size(); i++) {
        res[i] = mat[i][col];
    }
    return res;
}

template< typename T >
size_t get_max_idx( const vector<T> &vec ) {
    size_t max_idx = 0;
    for (size_t i = 1; i < vec.size(); i++) {
        if ( vec[i] > vec[max_idx] ) { max_idx = i; }
    }
    return max_idx;
}

template< typename T >
vector<int> get_max_idx( const matrix<T> &mat ) {
    vector<int> res( mat.size() );
    for (size_t i = 0; i < mat.size(); i++) {
        res[i] = get_max_idx( mat[i] );
    }
    return res;
}

// per-row argmax over a 2D Tensor
vector<int> get_max_idx( const Tensor &mat ) {
    size_t rows = mat.getShape()[0], cols = mat.getShape()[1];
    vector<int> res( rows );
    for ( size_t i = 0; i < rows; i++ ) {
        size_t max_idx = 0;
        for ( size_t j = 1; j < cols; j++ ) {
            if ( mat.at(i,j) > mat.at(i,max_idx) ) { max_idx = j; }
        }
        res[i] = max_idx;
    }
    return res;
}

// print functions
template< typename T >
void print_vec( const vector<T> &vec ) {
    std::cout << "< ";
    if ( !vec.empty() ) std::cout << vec[0];
    for ( size_t i = 1; i < vec.size(); i++ ) {
        std::cout << ", " << vec[i];
    }
    std::cout << " >";
}

void print_vec( const Tensor &t ) {
    std::cout << "< ";
    if ( t.getSize() > 0 ) std::cout << t[0];
    for ( size_t i = 1; i < t.getSize(); i++ ) {
        std::cout << ", " << t[i];
    }
    std::cout << " >";
}

template< typename T >
void print_matrix( const matrix<T> &mat ) {
    for ( size_t i = 0; i < mat.size(); i++ ) {
        print_vec( mat[i] );
        std::cout << endl;
    }
}