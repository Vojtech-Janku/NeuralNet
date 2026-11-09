#pragma once
// reading and writing CSV data files
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "utils.hpp"   // matrix<T>

using namespace std;

// number of rows and columns of a delimited text file
inline pair<size_t, size_t> get_dimensions( string path, char del ) {
    size_t row_count = 1, col_count = 1;
    string line;
    ifstream myfile( path );
    if ( !getline(myfile, line) ) return make_pair( 0, 0 );
    for ( char c : line ) { if ( c == del ) col_count++; }
    while ( getline(myfile, line) ) row_count++;
    return make_pair( row_count, col_count );
}

// skips the UTF-8 byte order mark at the start of a file, if there is one
inline void SkipBOM(std::fstream &in)
{
    char test[3] = {0};
    in.read(test, 3);
    if ((unsigned char)test[0] == 0xEF &&
        (unsigned char)test[1] == 0xBB &&
        (unsigned char)test[2] == 0xBF)
    { return; }
    in.seekg(0);
}

// reads a CSV file of integers into a matrix
inline auto read_data( string path, char del ) {
    string line, word;
    fstream fin( path, ios::in );
    assert( fin.is_open() );
    auto [rows, cols] = get_dimensions( path, del );
    matrix<int> table( rows, vector<int>( cols ) ); // initiate whole table at once

    SkipBOM( fin );
    size_t row = 0, col = 0;
    while ( getline( fin, line ) ) {
        col = 0;
        stringstream str(line);
        while( getline(str, word, ',') ) {
            table[row][col] = stoi(word);
            col++;
        }
        row++;
    }
    return table;
}

// writes a matrix as CSV, one row per line
template<typename T>
void export_data( string path, const matrix<T> &data ) {
    ofstream output_file( path );
    size_t n_rows = data.size(), n_cols = data[0].size();
    for ( size_t r = 0; r < n_rows; r++ ) {
        output_file << data[r][0];
        for ( size_t c = 1; c < n_cols; c++ ) {
            output_file << "," ;
            output_file << data[r][c];
        }
        output_file << endl;
    }
}

// writes a vector, one value per line
template<typename T>
void export_data( string path, const vector<T> &data ) {
    ofstream output_file( path );
    size_t n_rows = data.size();
    for ( size_t r = 0; r < n_rows; r++ ) {
        output_file << data[r] << endl;
    }
}
