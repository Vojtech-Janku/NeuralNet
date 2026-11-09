#pragma once
// turning network outputs into predictions and measuring how good they are
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "Tensor.hpp"

using namespace std;

// per-row argmax over a 2D Tensor - the predicted class of each sample
inline vector<int> get_max_idx( const Tensor &mat ) {
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

// number of positions where pred and target agree
inline size_t count_same( const vector<int> &pred, const vector<int> &target ) {
    if ( pred.size() != target.size() ) {
        throw std::invalid_argument( "count_same: " + to_string( pred.size() ) + " predictions but "
                                     + to_string( target.size() ) + " targets" );
    }
    size_t same = 0;
    for (size_t i = 0; i < pred.size(); i++) {
        if ( pred[i] == target[i] ) same++;
    }
    return same;
}

// fraction of correct predictions, in [0, 1]
inline float get_accuracy( const vector<int> &pred, const vector<int> &target ) {
    return ( count_same( pred, target ) / (float) pred.size() );
}
