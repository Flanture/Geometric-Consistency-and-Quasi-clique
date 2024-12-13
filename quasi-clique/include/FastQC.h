#ifndef FASTQC_H
#define FASTQC_H

#include<iostream>
#include<vector>
#include<set>
#include<time.h>
#include<math.h>
#include"RandList.h"
#include <igraph/igraph.h>
#include <fstream>
#include "global_counter.h"
#include <string>


using std::cout;
using std::vector;
using std::set;
using namespace std;


class FastQC{
    public:
        FastQC(int **Graph, int *degree, int graph_size, double gamma, int size_bound, set<int> *setG,int quiete);
        ~FastQC();
        void QCMiner();
        const igraph_vector_ptr_t& DCStrategy();
        const igraph_vector_ptr_t& DCStrategy(const std::vector<int>& original_indices,int num_of_nodes_lower_than_k);

        int res_num;
        int max_results;  // Add this line
    private:
        int **Graph;
        int *degree;
        int graph_size;
        double gamma;
        int size_bound;
        set<int> *setG;
        int quiete;
        
        RandList S, C, D;
        int *degInCS, *degInS;

        int *G_index;
        int valid;
        int *G_record;
        int *G_temp;

        vector<int> boundV;
        igraph_vector_ptr_t cliques;

        void KBranch_Rec(int k, int depth);

        /* Set of tool functions*/
        inline void RemoveFrC(int node);
        inline void RemoveFrS(int node);
        // Note that we skip RemoveFrD since it does not need to update either degInS or degInCS
        inline void AddToC(int node);
        inline void AddToS(int node);
        // Note that we skip AddToD since it does not need to update either degInS or degInCS
        inline void CToD(int node);
        inline void CToS(int node);
        inline void DToS(int node);
        inline void SToC(int node);
        /*----------------------*/

        void RefineC(int k, int node, vector<int> &ReC);
        void RefineD(int k, int node, vector<int> &ReD);
        void RefineCD(int k, int node, vector<int> &ReC, vector<int> &ReD, int low_bound);
        bool SIsPlex(int k);
        bool SCIsMaximal(int k);
        bool FastCDUpdate(int k);
        int EstimateK();
        bool IterRefineCD(int &k, vector<int> &ReC, vector<int> &ReD, int &low_bound);
        void RefineCD(int k, vector<int> &ReC, vector<int> &ReD, int low_bound);
        bool IsQC(int k);

        void OneHobP(int pivot);
        void TwoHobP(int pivot);
        string clique_address;
        string graph_address;

        /* Set of functions for debug*/
        void CheckDegree();
        bool CheckMaximal(int k);
        void Output();
        void OutputToFile();
        void OutputToVector();
        ofstream in;
        int size_p;
};

#endif