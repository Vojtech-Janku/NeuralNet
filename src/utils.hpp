#pragma once
#include <cmath>
#include <vector>
#include "Tensor.hpp"

using namespace std;

template< typename T >
using matrix = vector<vector<T>>;

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
vector<T> get_column( const matrix<T> &mat, size_t col ) {
    vector<T> res( mat.size() );
    for (size_t i = 0; i < mat.size(); i++) {
        res[i] = mat[i][col];
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
