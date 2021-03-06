cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(unet)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

set(TORCH_PATH "/home/zheng/Desktop/libs/libtorch")

find_package(Torch REQUIRED 
             HINTS ${TORCH_PATH})

find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})

add_library(cnpy SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/cnpy/cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

set(CMAKE_CXX_FLAGS "-g -O0 -msse2 -msse3 -Wall -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -fopenmp")

add_library( scn SHARED IMPORTED )
set_target_properties( scn PROPERTIES IMPORTED_LOCATION 
        ${CMAKE_CURRENT_SOURCE_DIR}/sparseconvnet/build/libscn.so )

file(GLOB_RECURSE srcs 
        src/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${srcs})

target_link_libraries(${CMAKE_PROJECT_NAME}
                      ${TORCH_LIBRARIES}
                      cnpy
                      scn
)

include_directories($<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${TORCH_PATH}/include
)