#include<fstream>
#include<iostream>
#include<string>
#include<list>
#include<igraph/igraph.h>
#include "../include/global_counter.h"
#include "../include/Util.h"
using namespace std;

int Util::ReadGraph(string dataset_path,int **&Graph, int *&degree, int &bipartite){
    ifstream read;
    read.open(dataset_path);
    string temp;
    read>>temp;
    int graph_size=stoi(temp);
    // int *Graph1[graph_size];
    Graph=new int*[graph_size]; 
    delete []degree;
    degree=new int[graph_size];
    read>>temp;
    int B_index=stoi(temp);
    bipartite=B_index;
    read>>temp;
    int index=0;
    int *neg=new int[graph_size];
    char a;
    int temp_count=0;
    bool first=true;
    while(!read.eof()){
        if(first){
            read>>temp;
            first=false;
        }       
        read.get(a);
        // if(index==172862){
        //     cout<<temp_count<<endl;
        // }
        if(a=='\n'){
            if(index>=graph_size)
                break;
            degree[index]=temp_count;
            int *temp_array=new int[temp_count];
            for(int i=0;i<temp_count;++i){
                temp_array[i]=neg[i];
            }
            Graph[index]=temp_array;
            temp_count=0;
            index++;
            first=true;
            continue;
        }
        read>>temp;
        neg[temp_count]=stoi(temp);
        temp_count++;
        
    }
    delete []neg;
    return graph_size;
}

int Util::ReadGraph(string dataset_path,int **&Graph, int *&degree){
    ifstream read;
    read.open(dataset_path);
    string temp;
    read>>temp;
    int graph_size=stoi(temp);
    // int *Graph1[graph_size];
    Graph=new int*[graph_size]; 
    delete []degree;
    degree=new int[graph_size];
    read>>temp;
    int index=0;
    int *neg=new int[graph_size];
    char a;
    int temp_count=0;
    bool first=true;
    while(!read.eof()){
        if(first){
            read>>temp;
            first=false;
        }       
        read.get(a);
        // if(index==172862){
        //     cout<<temp_count<<endl;
        // }
        if(a=='\r' || a=='\n'){
            if(index>=graph_size)
                break;
            degree[index]=temp_count;
            int *temp_array=new int[temp_count];
            for(int i=0;i<temp_count;++i){
                temp_array[i]=neg[i];
            }
            Graph[index]=temp_array;
            temp_count=0;
            index++;
            first=true;
            continue;
        }
        read>>temp;
        neg[temp_count]=stoi(temp);
        temp_count++;
        
    }
    delete []neg;
    return graph_size;
}

int Util::ReadGraph(const igraph_t &igraph, int **&Graph, int *&degree, int &bipartite) {
    int n = igraph_vcount(&igraph); // 获取节点数量
    int m = igraph_ecount(&igraph); // 获取边数量
    
    // 动态分配二维数组和度数数组
    Graph = new int*[n];
    degree = new int[n];
    fill(degree, degree + n, 0);

    bipartite = 0; // 根据需要初始化 bipartite 的值

    std::string base_path = "/home/public/fyc/quasi-clique/visualization/";
    std::string filename = getFileName(base_path, "_graphs.txt");
    ofstream in_graphs;
    in_graphs.open(filename);

    in_graphs << n << " " << m << endl;
    // 遍历图的每一条边，填充 Graph 和 degree
    igraph_eit_t eit;
    igraph_eit_create(const_cast<igraph_t*>(&igraph), igraph_ess_all(IGRAPH_EDGEORDER_ID), &eit);
    while (!IGRAPH_EIT_END(eit)) {
        int from, to;
        igraph_edge(&igraph, IGRAPH_EIT_GET(eit), &from, &to);
        degree[from]++;
        degree[to]++;
        Graph[from][degree[from]++] = to;
        in_graphs << from << " " << to << endl;
        IGRAPH_EIT_NEXT(eit);
    }
    igraph_eit_destroy(&eit);

    for (int i = 0; i < n; ++i) {
        Graph[i] = new int[degree[i]];
        degree[i] = 0; // 重置 degree 数组以在下次循环中正确填充
    }


    igraph_eit_create(const_cast<igraph_t*>(&igraph), igraph_ess_all(IGRAPH_EDGEORDER_ID), &eit);
    while (!IGRAPH_EIT_END(eit)) {
        int from, to;
        igraph_edge(&igraph, IGRAPH_EIT_GET(eit), &from, &to);
        Graph[to][degree[to]++] = from; // 如果是无向图，需要添加反向边
        IGRAPH_EIT_NEXT(eit);
    }
    igraph_eit_destroy(&eit);
    in_graphs.close();
    return n;
}

int Util::ReadGraph(const igraph_t &igraph, int **&Graph, int *&degree, std::vector<int> &original_indices) {
    int n = igraph_vcount(&igraph); // 获取节点数量
    int m = igraph_ecount(&igraph); // 获取边数量
    
    // 动态分配二维数组和度数数组
    Graph = new int*[n];
    degree = new int[n];
    fill(degree, degree + n, 0);

    original_indices.resize(n);
    iota(original_indices.begin(), original_indices.end(), 0); // 保存初始编号

    // 遍历图的每一条边，填充 degree 数组
    igraph_eit_t eit;
    igraph_eit_create(const_cast<igraph_t*>(&igraph), igraph_ess_all(IGRAPH_EDGEORDER_ID), &eit);
    while (!IGRAPH_EIT_END(eit)) {
        int from, to;
        igraph_edge(&igraph, IGRAPH_EIT_GET(eit), &from, &to);
        degree[from]++;
        degree[to]++;
        IGRAPH_EIT_NEXT(eit);
    }
    igraph_eit_destroy(&eit);

    // 动态分配邻接表数组
    for (int i = 0; i < n; ++i) {
        Graph[i] = new int[degree[i]];
        degree[i] = 0; // 重置 degree 数组以在下次循环中正确填充
    }

    // 填充邻接表
    igraph_eit_create(const_cast<igraph_t*>(&igraph), igraph_ess_all(IGRAPH_EDGEORDER_ID), &eit);
    while (!IGRAPH_EIT_END(eit)) {
        int from, to;
        igraph_edge(&igraph, IGRAPH_EIT_GET(eit), &from, &to);
        Graph[from][degree[from]++] = to;
        Graph[to][degree[to]++] = from; // 如果是无向图，需要添加反向边
        IGRAPH_EIT_NEXT(eit);
    }
    igraph_eit_destroy(&eit);

    // 输出邻接表到文件
    // std::string base_path = "/home/public/fyc/quasi-clique/visualization/";
    // std::string filename = getFileName(base_path, "_adj_list.txt");
    // ofstream in_adj_list;
    // in_adj_list.open(filename);
    // in_adj_list << n << " " << m << endl;
    // for (int i = 0; i < n; ++i) {
    //     in_adj_list << i;
    //     for (int j = 0; j < degree[i]; ++j) {
    //         in_adj_list << " " << Graph[i][j];
    //     }
    //     in_adj_list << endl;
    // }
    // in_adj_list.close();

    return n;
}


int Util::ReadGraph(const igraph_t &igraph, std::vector<std::vector<int>> &Graph, std::vector<int> &degree) {
    int n = igraph_vcount(&igraph); // 获取节点数量
    int m = igraph_ecount(&igraph); // 获取边数量
    
    // 动态分配二维数组和度数数组
    Graph.resize(n);
    degree.resize(n, 0);


    // 遍历图的每一条边，填充 degree 数组
    igraph_eit_t eit;
    igraph_eit_create(const_cast<igraph_t*>(&igraph), igraph_ess_all(IGRAPH_EDGEORDER_ID), &eit);
    while (!IGRAPH_EIT_END(eit)) {
        int from, to;
        igraph_edge(&igraph, IGRAPH_EIT_GET(eit), &from, &to);
        degree[from]++;
        degree[to]++;
        IGRAPH_EIT_NEXT(eit);
    }
    igraph_eit_destroy(&eit);

    // 动态分配邻接表数组
    for (int i = 0; i < n; ++i) {
        Graph[i].resize(degree[i]);
        degree[i] = 0; // 重置 degree 数组以在下次循环中正确填充
    }

    // 填充邻接表
    igraph_eit_create(const_cast<igraph_t*>(&igraph), igraph_ess_all(IGRAPH_EDGEORDER_ID), &eit);
    while (!IGRAPH_EIT_END(eit)) {
        int from, to;
        igraph_edge(&igraph, IGRAPH_EIT_GET(eit), &from, &to);
        Graph[from][degree[from]++] = to;
        Graph[to][degree[to]++] = from; // 如果是无向图，需要添加反向边
        IGRAPH_EIT_NEXT(eit);
    }
    igraph_eit_destroy(&eit);

    // 输出邻接表到文件
    // std::string base_path = "/home/public/fyc/quasi-clique/visualization/";
    // std::string filename = getFileName(base_path, "_adj_list.txt");
    // std::ofstream in_adj_list;
    // in_adj_list.open(filename);
    // in_adj_list << n << " " << m << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     in_adj_list << i;
    //     for (int j = 0; j < degree[i]; ++j) {
    //         in_adj_list << " " << Graph[i][j];
    //     }
    //     in_adj_list << std::endl;
    // }
    // in_adj_list.close();

    return n;
}
