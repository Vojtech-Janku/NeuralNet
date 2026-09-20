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
        size_t total = 1;
        for ( auto d : shape ) total *= d;
        if (total != data.size())
        {
            throw std::invalid_argument( "Tensor: shape does not match data size" );
        }
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
};