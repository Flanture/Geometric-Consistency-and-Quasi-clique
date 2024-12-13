#ifndef COREPRUNING_H
#define COREPRUNING_H
#include<iostream>
#include<list>
#include<algorithm>
#include<vector>
#include <fstream>
#include"global_counter.h"
using std::cout;
using std::endl;
using std::sort;


class CoreLocate{

    public:
        // CoreLocate(int **Graph, int *degree, int graph_size, int K);
        CoreLocate(std::vector<std::vector<int>>& Graph, std::vector<int>& degree, int graph_size, int K);
        ~CoreLocate()=default;
        int CorePrune(int **&new_graph, int *&degree);
        int CorePrune(std::vector<std::vector<int>>& new_graph, std::vector<int>& new_degree);
        std::vector<int> original_indices; // 保存原始顶点编号
        int Coredecompose();
        int GetMaxcore();
        void CoreOrdering(int **&new_graph, int *&new_degree);
        void Bipartite_CoreOrdering(int **&new_graph, int *&new_degree);
        void Bipartite_R_CoreOrdering(int **&new_graph, int *&new_degree);
        int getGraphSize();
        std::vector<int>G_order;
        std::vector<int>G_index;
        std::vector<int>G_label;
        std::vector<int>order_G;
    private:
        std::vector<std::vector<int>>Graph;
        std::vector<int>degree;
        std::vector<int>G_temp;
        
        
        int K;
        int graph_size;
        int max_degree;

};

#endif
