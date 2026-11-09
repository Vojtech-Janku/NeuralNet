#pragma once
// turning raw data into network inputs and targets
#include <cstddef>
#include <vector>
#include "utils.hpp"   // matrix<T>

using namespace std;

// divides every value by `scale` (e.g. 255 to map pixels to [0, 1])
inline auto scale( const matrix<int> &data_mat, float scale ) {
    size_t n_rows = data_mat.size(), n_cols = data_mat[0].size();
    matrix<float> res( n_rows, vector<float>( n_cols ) );
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            res[i][j] = data_mat[i][j] / scale;
        }
    }
    return res;
}

// one-hot encoding: label k becomes a row of `max` zeros with a 1 at position k
inline auto transform_index( const vector<int> &index_vec, size_t max ) {
    matrix<float> res( index_vec.size(), std::vector<float>(max, 0) );
    for (size_t i = 0; i < res.size(); i++) {
        res[i][ index_vec[i] ] = 1;
    }
    return res;
}
