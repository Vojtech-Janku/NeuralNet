#include <cmath>
#include <vector>

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

template< typename T >
void print_matrix( const matrix<T> &mat ) {
    for ( size_t i = 0; i < mat.size(); i++ ) {
        print_vec( mat[i] );
        std::cout << endl;
    }
}