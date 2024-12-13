#include<iostream>
#include<list>
#include<algorithm>
#include<vector>
#include"../include/global_counter.h"
#include<fstream>
#include "../include/Corepruning.h"
#include<string>
using std::cout;
using std::endl;
using std::sort;
using std::ofstream;
using std::string;


int CoreLocate::getGraphSize() {
    return graph_size;
}
CoreLocate::CoreLocate(std::vector<std::vector<int>>& Graph, std::vector<int>& degree, int graph_size, int K)
    : Graph(Graph), degree(degree), graph_size(graph_size), K(K) {

    int max = 0;
    G_temp.resize(graph_size);
    G_label.resize(graph_size);
    G_order.resize(graph_size);
    order_G.resize(graph_size);
    G_index.resize(graph_size);
    original_indices.resize(graph_size);
    
    for (int i = 0; i < graph_size; ++i) {
        G_index[i] = -1;
        G_label[i] = 0;
        G_temp[i] = degree[i];
        original_indices[i] = i; // 初始化原始编号
        if (degree[i] > max) {
            max = degree[i];
        }
    }
    this->max_degree = max;
}
int CoreLocate::Coredecompose() {
    // int *bin = new int[max_degree + 1]();
    std::vector<int> bin(max_degree + 1);
    for (int i = 0; i <= max_degree; ++i) {
        bin[i] = 0;
    }
    for (int i = 0; i < graph_size; ++i) {
        bin[degree[i]] += 1;
    }

    int start = 0;
    for (int d = 0; d <= max_degree; ++d) {
        int num = bin[d];
        bin[d] = start;
        start += num;
    }
    // int *pos = new int[graph_size + 1]();
    // int *vert = new int[graph_size + 1]();
    std::vector<int>pos(graph_size + 1);
    std::vector<int>vert(graph_size + 1);
    for (int i = 0; i < graph_size; ++i) {
        pos[i] = bin[degree[i]];
        vert[pos[i]] = i;
        bin[degree[i]] += 1;
    }

    for (int i = max_degree; i >= 1; --i) {
        bin[i] = bin[i - 1];
    }
    bin[0] = 0;


     for (int i = 0; i < graph_size; ++i) {
        int node = vert[i];
        G_order[node] = i;
        order_G[i] = node;
        G_label[node] = G_temp[node];
    }
    return bin[K-1];
}

int CoreLocate::GetMaxcore(){
    int max=0;
    for(int i=0;i<graph_size;++i){
        if(G_label[i]>max){
            max=G_label[i];
        }
    }

    int temp_count=0;
    for(int i=0;i<graph_size;++i){
        if(G_label[i]>=max){
            temp_count++;
        }
    }
    //cout<<"------------ Statistics -------------"<<endl;
    //cout<<"MaxCore Num: "<<max<<endl;
    //cout<<"# of vertices in MaxCore: "<<temp_count<<endl;
    //cout<<"# of vertices in Graph: "<<graph_size<<endl;
    //cout<<"Per. of vertices in MaxCore: "<<(1.0*temp_count/graph_size)<<endl;
    //cout<<"-------------------------------------"<<endl;

    return max;
}



int CoreLocate::CorePrune(int **&new_graph, int *&new_degree) {
    int min_order = graph_size;
    for (int i = 0; i < graph_size; ++i) {
        if (G_label[i] >= K) {
            min_order = std::min(min_order, G_order[i]);
        }
    }

    int count = 0;
    for (int i = 0; i < graph_size; ++i) {
        if (G_label[i] >= K) {
            G_index[i] = G_order[i] - min_order;
            count++;
        } else {
            G_index[i] = -1;
        }
    }

    new_degree = new int[count];
    new_graph = new int*[count];
    for (int i = 0; i < graph_size; ++i) {
        if (G_index[i] >= 0) {
            int temp_count = 0;
            for (int j = 0; j < degree[i]; ++j) {
                if (G_index[Graph[i][j]] >= 0) {
                    temp_count++;
                }
            }
            int *neg = new int[temp_count];
            new_degree[G_index[i]] = temp_count;
            temp_count = 0;
            for (int j = 0; j < degree[i]; ++j) {
                if (G_index[Graph[i][j]] >= 0) {
                    neg[temp_count] = G_index[Graph[i][j]];
                    temp_count++;
                }
            }
            std::sort(neg, neg + new_degree[G_index[i]]);
            new_graph[G_index[i]] = neg;
        }
    }


    // std::string base_path = "/home/public/fyc/quasi-clique/visualization/";
    // std::string filename = getFileName(base_path, "_pruned_adj_list.txt");
    // std::ofstream out_adj_list;
    // out_adj_list.open(filename);
    // for (int i = 0; i < count; ++i) {
    //     out_adj_list << i;
    //     for (int j = 0; j < new_degree[i]; ++j) {
    //         out_adj_list << " " << new_graph[i][j];
    //     }
    //     out_adj_list << std::endl;
    // }
    // out_adj_list.close();

    return count;
}

int CoreLocate::CorePrune(std::vector<std::vector<int>>& new_graph, std::vector<int>& new_degree) {
    int min_order = graph_size;
    for (int i = 0; i < graph_size; ++i) {
        if (G_label[i] >= K) {
            min_order = std::min(min_order, G_order[i]);
        }
    }

    int count = 0;
    for (int i = 0; i < graph_size; ++i) {
        if (G_label[i] >= K) {
            G_index[i] = G_order[i] - min_order;
            count++;
        } else {
            G_index[i] = -1;
        }
    }

    new_degree.resize(count);
    new_graph.resize(count);

    for (int i = 0; i < graph_size; ++i) {
        if (G_index[i] >= 0) {
            int temp_count = 0;
            for (int j = 0; j < degree[i]; ++j) {
                if (G_index[Graph[i][j]] >= 0) {
                    temp_count++;
                }
            }
            std::vector<int> neg(temp_count);
            new_degree[G_index[i]] = temp_count;
            temp_count = 0;
            for (int j = 0; j < degree[i]; ++j) {
                if (G_index[Graph[i][j]] >= 0) {
                    neg[temp_count] = G_index[Graph[i][j]];
                    temp_count++;
                }
            }
            std::sort(neg.begin(), neg.end());
            new_graph[G_index[i]] = neg;
        }
    }

    // std::string base_path = "/home/public/fyc/quasi-clique/visualization/";
    // std::string filename = getFileName(base_path, "_pruned_adj_list.txt");
    // std::ofstream out_adj_list;
    // out_adj_list.open(filename);
    // out_adj_list<< count << " " << 0 << std::endl;
    // for (int i = 0; i < count; ++i) {
    //     out_adj_list << i;
    //     for (int j = 0; j < new_degree[i]; ++j) {
    //         out_adj_list << " " << new_graph[i][j];
    //     }
    //     out_adj_list << std::endl;
    // }
    // out_adj_list.close();

    return count;
}
void CoreLocate::CoreOrdering(int **&new_graph, int *&new_degree){
    new_degree=new int[graph_size];
    new_graph=new int*[graph_size];
    for(int i=0;i<graph_size;++i){    
        int *neg=new int[degree[i]];
        new_degree[G_order[i]]=degree[i];
        for(int j=0;j<degree[i];++j){   
            neg[j]=G_order[Graph[i][j]];  
        }
        sort(neg,neg+degree[i]);
        new_graph[G_order[i]]=neg;
    }
}




