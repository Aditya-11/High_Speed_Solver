cmake_minimum_required(VERSION 3.17)
project(Matrix_1)

set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME}  main.cpp gmres_imp.cpp gmres_imp.h)

find_package(NumCpp 2.1.0 REQUIRED)
target_link_libraries(${PROJECT_NAME}
        NumCpp::NumCpp
        )