#include <cstddef>
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

    float &operator[](size_t idx) {return data[idx];}
    float &at(size_t i, size_t j) {return data[i*shape[1]+j];}

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

    int getDimension() {
        return shape.size();
    }

    vector<size_t> getShape() {
        return shape;
    }
};