#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>
using namespace std;

struct Tensor 
{
    vector<float> data;
    vector<size_t> shape;

    Tensor( vector<size_t> shape ) : shape(shape) {
        size_t total = 1;
        for ( auto d : shape ) total *= d;
        data = vector<float>(total);
    }

    Tensor( vector<size_t> shape, vector<float> data ) : shape(shape), data(data) {
        reshape( shape );
    }

    float &operator[](size_t idx) {return data[idx];}
    const float &operator[](size_t idx) const {return data[idx];}

    float &at(size_t i, size_t j) {return data[i*shape[1]+j];}
    const float &at(size_t i, size_t j) const {return data[i*shape[1]+j];}

    Tensor &operator*(float n) {
        for (auto &e : data) e *= n;
        return *this;
    }

    Tensor &operator/(float n) {
        for (auto &e : data) e /= n;
        return *this;
    }

    Tensor operator*(Tensor &other) {
        return Tensor(vector<size_t>(0));   // TODO: fix this
    }

    size_t getSize() const { return data.size(); }
    size_t getDimension() const { return shape.size(); }
    vector<size_t> getShape() const { return shape; }

    void clear() { fill( data.begin(), data.end(), 0 ); }

    // strips the leading dimension, returning row i as its own (copied) Tensor
    // e.g. for a {samples, features} Tensor, row(k) gives the {features} Tensor for sample k
    Tensor row( size_t i ) const {
        vector<size_t> row_shape( shape.begin()+1, shape.end() );
        size_t row_size = 1;
        for ( auto d : row_shape ) row_size *= d;
        vector<float> row_data( data.begin() + i*row_size, data.begin() + (i+1)*row_size );
        return Tensor( row_shape, row_data );
    }

    void flatten() {
        shape = { data.size() };
    }

    void reshape( vector<size_t> newShape ){
        size_t total = 1;
        for ( auto d : newShape ) total *= d;
        if (total != data.size())
        {
            throw std::invalid_argument( "Tensor: shape does not match data size" );
        }
        shape = newShape;
    }
};