#pragma once
#include <string>

// optimizers
enum Optimizer{ GRAD, MOMENTUM, ADAM };

// just for printing
inline std::string get_str( Optimizer opt ) {
    switch (opt) {
    case Optimizer::GRAD:          return "GRAD";
    case Optimizer::MOMENTUM:      return "MOMENTUM";
    case Optimizer::ADAM:          return "ADAM";
    default:                       return "Unknown";
    }
}
