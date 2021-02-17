//
// Created by Aditya Dubey on 05/02/21.
//

#ifndef MATRIX_1_GMRES_IMP_H
#define MATRIX_1_GMRES_IMP_H

#include "NumCpp.hpp"
# include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <chrono>

namespace vars_ {

}
void gmres_algorithm (nc::NdArray <double> & A , nc::NdArray <double> & b , nc::NdArray <double> & x0 , double error ,long max_iter , std::vector <double> * g_vals) ;

void test_lstq() ;

void test_1_algo () ;

void test_matmul() ;

template<typename T>
void print_ (std::vector <T> * vec1) {
    for (auto x = vec1->begin() ; x!= vec1->end() ; x++) std::cout << (x - vec1->begin()) + 1 << " " << *x << std::endl ;
}

nc::NdArray <double> generate_mat ( int n , int m ) ;

void test_time_1 (int n) ;

void test_time_2 (int n) ;

#endif //MATRIX_1_GMRES_IMP_H

