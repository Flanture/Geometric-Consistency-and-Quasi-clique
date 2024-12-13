#ifndef UTIL_H
#define UTIL_H

#include<fstream>
#include<iostream>
#include<string>
#include<list>
#include <vector>
#include <numeric>
#include<igraph/igraph.h>
#include "global_counter.h"
using namespace std;

class Util{
    public:
        int ReadGraph(string dataset_path,int **&Graph, int *&degree, int &bipartite);
        int ReadGraph(string dataset_path,int **&Graph, int *&degree); 
        int ReadGraph(const igraph_t &igraph, int **&Graph, int *&degree, int &bipartite);
        int ReadGraph(const igraph_t &igraph, int **&Graph, int *&degree);
        int ReadGraph(const igraph_t &igraph, int **&Graph, int *&degree, std::vector<int> &original_indices);

        static int ReadGraph(const igraph_t &igraph, std::vector<std::vector<int>> &Graph, std::vector<int> &degree);
};

#endif