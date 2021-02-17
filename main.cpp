#include <iostream>


#include "gmres_imp.h"

int main() {

    std::vector<double> * g_values = new std::vector<double>();

    std::cout << "Start the project ! " << std::endl;

    auto a = nc::random::randFloat  <double>  ({10, 10}, 0, 100);

    auto b = nc::random::randFloat <double> ({10, 1}, 0, 100);

    auto x0 = nc::random::randFloat <double> ({10, 1}, 0, 100);

    double error = std::pow(10, -7) ;

    long max_iter = 1000 ;

    std::cout << a.shape() << std::endl ;

   //  nc::NdArray <double> a_1 = { {1, 1 ,1 , 1,7}, {2, 3 , 0, 5,7}, {1, 2 , 1 , 8,5} , {1 , 2 , 7, 9,0} , {2 ,3,4,5,6} } ;

    nc::NdArray <double> a_1 = { {37 , 26 , 7  , 13 , 27} ,
                                 {3  , 46 , 45 , 27 , 3} ,
                                 {11 , 21 , 6  , 49 , 8} ,
                                 {31 , 9  , 8  , 8  , 5} ,
                                 {22 , 7  , 37 , 34 , 10} } ;

    nc::NdArray <double> b_1 =  { 6 , 0 , 11 ,  24 ,  11 } ;

    // nc::NdArray <double> b_1 = { {2 , 5, -1 , 7,6} };

    // nc::NdArray <double> x_1 = { {1 , 1, 1 ,0,0} };

    nc::NdArray <double> x_1 = {  5  , 11 ,  31 ,  27 ,  20  } ;

    b_1 = b_1.transpose() ;

    x_1 = x_1.transpose() ;

    a_1 = generate_mat(200 , 200) ;
    b_1 = generate_mat(200 ,1) ;
    x_1 = generate_mat(200 , 1) ;

    std::cout << "a_1 " <<  a_1 << std::endl;
    std::cout << "b_1 " << b_1 << std::endl;
    std::cout << "x_1 " << x_1 << std::endl;

    std::cout << "a1  : " << a_1.shape() << " b1 " << b_1.shape() << " " << std::endl ;

    auto start = std::chrono::high_resolution_clock::now();

    //gmres_algorithm(a_1 , b_1 , x_1  ,  error , max_iter , g_values) ;

    auto stop = std::chrono::high_resolution_clock::now();

        // test_time_1(200) ;
        test_time_2(200) ;

        //auto duration = duration_cast<std::chrono::seconds>(stop - start);

        auto micro_duration = duration_cast<std::chrono::microseconds>(stop - start);

        double total_time = static_cast<double>(micro_duration.count() * std::pow(10 , -6));

        //std::cout << " Time taken by function: " << micro_duration.count() << " microseconds " << std::endl;

        std::cout << " Total time : " << total_time << " seconds "<<  std::endl ;

    // test_matmul() ;
    // test_1_algo() ;

    //test_lstq() ;

    std::cout << " print g_values " << std::endl ;

    print_(g_values) ;

    delete [] g_values ;

    return 0;
}
