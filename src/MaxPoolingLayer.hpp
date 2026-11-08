#pragma once
#include "Layer.hpp"

class MaxPoolingLayer : public Layer {

    int pooling_size;

    matrix<pair<int,int>> mask;

};