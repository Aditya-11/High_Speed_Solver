//
// Created by Aditya Dubey on 05/02/21.
//

#include "gmres_imp.h"

void gmres_algorithm (nc::NdArray <double> & A , nc::NdArray <double> & b , nc::NdArray <double> & x0 , double error ,long max_iter , std::vector <double > * g_val) {
    auto _a_shape = A.shape().rows;

    if (max_iter > _a_shape) max_iter = _a_shape;

    nc::NdArray <double> bb_ = nc::dot(A,x0);

    auto res = b - bb_;

    std::vector <nc::NdArray <double>> x_pred ;

    int l1_ = res.size() ;

    nc::NdArray <double> q_  = nc::zeros <double> (max_iter , _a_shape) ;

    double norm3 = static_cast<double>(nc::norm(res)[0]) ;

    double norm1 = 1 / norm3 ;

    auto res1 = res ;

    // std::cout << res << "  " << std::endl ;

    for (auto & x : res1)  x = x * norm1 ;

    for (int i = 0 ; i < q_.shape().cols ; i++)  q_ (0 , i) = res1 (i, 0) ;

    auto h_ = nc::zeros<double> (max_iter + 1,  max_iter) ;

    // std::cout << q_ << std::endl;

    for (int k = 0 ; k < max_iter ;  k ++) {

        nc::NdArray <double> cc_ = q_(k, q_.cSlice()) ;

        nc::NdArray<double> y_out = nc::dot(A, cc_.transpose()) ;

        for (int j = 0; j < k + 1; j++) {

            nc::NdArray<double> temp_h_ = nc::dot(q_(j, q_.cSlice()), y_out) ;
            h_(j, k) = static_cast <double> (temp_h_(0, 0)) ;
            nc::NdArray <double> temp_array = nc::zeros<double>(y_out.shape())  ;

            for (int z1 = 0 ; z1 < q_.shape().cols ; z1++ ) {
                double ss_  = static_cast<double>(q_(j , z1) * h_(j , k));
                if (std::fabs(ss_) > std::pow(10,-7) and ss_ < INT_MAX) temp_array(z1 , 0) = ss_;
            }

            y_out = y_out - temp_array ;
        }

        h_(k + 1 ,  k) = nc::norm(y_out)[0];

        // std::cout << y_out << std::endl ;

        if (h_(k + 1, k) != 0 and k != max_iter - 1)
        {
            for (int z1 = 0 ; z1 < _a_shape ; z1++) {
                double ss_ = (y_out(z1 , 0) / h_(k + 1, k));
                if (std::fabs(ss_) > std::pow(10,-7) && ss_ < INT_MAX) q_(k + 1, z1) = ss_ ; // donee
            }
        }

        // std::cout << " q_ " << q_ << std::endl ;

        // std::cout << " h_ " << h_ << std::endl ;

        // std::cout << " y_out " << y_out << std::endl ;

        nc::NdArray<double> b_ = nc::zeros<double>(max_iter + 1 , 1);

        b_[0] = static_cast<double>(norm3) ;

        nc::NdArray<double> c_ = nc::linalg::lstsq(h_, b_,std::pow(10,-6)) ;

       // std::cout << c_ << std::endl ;

        nc::NdArray<double> prod_ = nc::dot(q_.transpose() , c_.transpose()) ;

        double x_temp_ = static_cast <double> (nc::norm (b - nc::dot(A, (prod_ + x0)))[0] / nc::norm(b)[0])  ;

       // std::cout << " " << " error : " << x_temp_ << std::endl ;

        g_val->push_back(std::log10(x_temp_));

        if (x_temp_ < error) break;
    }
}


void test_lstq() {
    // 101 * 100 and 101 * 1

    auto aa_ = nc::random::randInt<int>({101 , 100}, 0, 100) ;

    auto bb_ = nc::random::randInt<int>({101 , 1}, 0, 100) ;

    auto cc_ = nc::linalg::lstsq(aa_, bb_);

    std::cout << cc_ << "  " << cc_.shape() << std::endl ;
}


void test_1_algo() {
    //
    auto aa_ = nc::random::randInt<int>({10 , 1}, 0, 10) ;

    auto bb_ = nc::random::randInt<int>({10 , 1}, 0, 10) ;

    std::cout << aa_.shape() << std::endl << std::endl ;

    auto cc_ = aa_ - bb_.transpose() ;

    std::cout << cc_ << std::endl ;
}


void test_matmul () {
    nc::NdArray <double> a_1 = {{37 , 26 , 7 , 13 , 27} ,
    {3 , 46 , 45 , 27 , 3} ,
    {11 , 21 , 6  , 49 , 8} ,
    {31 , 9 , 8 , 8 , 5} ,
    {22 , 7  , 37 ,  34 ,  10} } ;


            // { {1, 1 ,1,1}, {-1, -1 , -1, -1}, {2, 2 , 2 , 2} , { 0 , 1, 0 ,1 } } ;
            // nc::NdArray <double> b_1 = { {1,1,1,1} };
    nc::NdArray <double> b_1  =  { {6 , 0  , 11 , 24 , 11} } ;

    nc::NdArray <double> c_1 = { { 5  , 11 ,  31 ,  27 ,  20 } };


    for (int i = 0 ; i < a_1.shape().cols ; i ++ ) {
        auto cc1 =   nc::dot(a_1(i, a_1.cSlice()), b_1);

        std::cout << cc1 << " " << std::endl ;
    }
}

nc::NdArray <double> generate_mat ( int n , int m ) {
    return nc::random::randFloat  <double>  ({nc::uint32 (n) , nc::uint32 (m)}, 0, n+m );
}

void test_time_1 (int n) {
    auto a_1 = generate_mat(n , n);

    auto b_1 = generate_mat(n,1) ;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0 ; i < 200 ; i++) {
       auto c1 = nc::dot(a_1, b_1) ;
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto micro_duration = duration_cast<std::chrono::microseconds>(stop - start);

    double total_time = static_cast<double>(micro_duration.count() * std::pow(10 , -6));

    std::cout << " Total time Multiplication :  " << total_time << " seconds "<<  std::endl ;
}

void test_time_2 (int n) {
    auto a_1 = generate_mat(n+1 , n);
    auto b_1 = generate_mat(n+1 , 1);

    std::cout << "  a1  "<< a_1 << std::endl ;
    std::cout << "  b1  "<< b_1 << std::endl ;

    double val_tolerance = std::pow(10,-6) ;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0 ; i < n ; i++) {
        auto c1 = nc::linalg::lstsq(a_1,b_1,val_tolerance) ;
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto micro_duration = duration_cast<std::chrono::microseconds>(stop - start);

    double total_time = static_cast<double> (micro_duration.count() * std::pow(10 , -6));

    std::cout << " Total time LSTQ :  " << total_time << " seconds " << std::endl ;


    nc::NdArray<float> a1 ;

    a1.c

}