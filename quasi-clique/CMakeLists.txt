cmake_minimum_required(VERSION 3.22)
project(QUASI-CLIQUE)

set(CMAKE_CXX_STANDARD 14)
find_package(PCL REQUIRED)
find_package(igraph 0.9.9 REQUIRED)
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#include_directories(
#        /opt/intel/oneapi/mkl/2023.0.0/include/
#        include  #-------即/usr/local/include路径
#)
include_directories(${PCL_INCLUDE_DIRS})

#link_directories(/opt/intel/oneapi/mkl/2023.0.0/lib/intel64
#        )

link_directories(${PCL_LIBRARY_DIRS})
source_group("Source Files" FILES src/*.cpp)
source_group("Header Files" FILES include/*.h)
add_definitions(${PCL_DEFINITIONS})
add_executable(QUASI-CLIQUE
    src/main.cpp
    src/desc_dec.cpp
    src/funcs.cpp
    src/PCR.cpp
    src/registration.cpp
    src/visualization.cpp
    src/Corepruning.cpp
    src/FastQC.cpp
    src/Util.cpp
    src/global_counter.cpp
)
# add_library(Mylib SHARED
#     include/args.hxx
#     include/Corepruning.h
#     include/Defines.h
#     include/Eva.h
#     include/FastQC.h
#     include/global_counter.h
#     include/RandList.h
#     include/Util.h

# )

# add_executable(QUASI-CLIQUE main.cpp desc_dec.cpp funcs.cpp PCR.cpp registration.cpp visualization.cpp Eva.h Corepruning.cpp FastQC.cpp Util.cpp)

target_link_libraries(QUASI-CLIQUE ${PCL_LIBRARIES} igraph::igraph)
