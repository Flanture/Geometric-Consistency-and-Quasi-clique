#include<iostream>
#include<vector>
#include<set>
#include<time.h>
#include<math.h>
#include "../include/RandList.h"
#include <igraph/igraph.h>
#include "../include/FastQC.h"
#include "../include/global_counter.h"
#include <string>
#include <unordered_map>

using std::cout;
using std::vector;
using std::set;
using std::ofstream;
using std::endl;



FastQC::FastQC(int **Graph, int *degree, int graph_size, double gamma, int size_bound, set<int> *setG, int quiete) {
    this->Graph = Graph;
    this->degree = degree;
    this->graph_size = graph_size;
    this->gamma = gamma;
    this->size_bound = size_bound;
    this->setG = setG;
    this->quiete = quiete;
    this->max_results = 0x3f3f3f;  // Add this line

    S.init(graph_size);
    C.init(graph_size);
    D.init(graph_size);

    res_num = 0;
    valid = 0;
    boundV.reserve(graph_size);

    degInCS = new int[graph_size];
    degInS = new int[graph_size];
    G_index = new int[graph_size];
    G_record = new int[graph_size];
    G_temp = new int[graph_size];
    for (int i = graph_size - 1; i >= 0; --i) {
        degInCS[i] = 0;
        degInS[i] = 0;
        G_index[i] = 0;
        G_record[i] = 0;
        G_temp[i] = 0;
    }
    igraph_vector_ptr_init(&cliques, 0);
    size_p = std::ceil(gamma * (size_bound - 1));
    // if (quiete != 0){
    //     std::string base_path = "/home/public/fyc/quasi-clique/visualization/";
    //     std::string filename = getFileName(base_path, "_cliques.txt");
    //     in.open(filename);
    // }
}

FastQC::~FastQC(){
    delete []degInCS;
    delete []degInS;
    in.close();
}

void FastQC::QCMiner(){
    for(int i=0;i<graph_size;++i){
        C.add(i);
        degInCS[i]=degree[i];
    }
    int ek=EstimateK();
    KBranch_Rec(ek,1);

}

std::unordered_map<int, bool> cache;

void FastQC::KBranch_Rec(int k, int depth){
    // if (res_num >= max_results) return;  // Add this line
    if (iteration_count.load() >= MAX_ITERATIONS) {
        return;
    }
    iteration_count.fetch_add(1);
    vector<int> RRC, RRD;
    int low_bound;
    bool brach_flag=IterRefineCD(k,RRC,RRD,low_bound);
    if(!brach_flag){
        for(int i:RRC) if(!C.contains(i)) AddToC(i);
        for(int i:RRD) if(!D.contains(i)) D.add(i);
        return;
    }
    if(S.vnum+C.vnum<low_bound) return;
    int temp_i, temp_j, temp_v; // temp varables used in each recursion    
    
    int pivot=-1;
    temp_v=graph_size+1;
    temp_j=S.vnum+C.vnum;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        pivot=degInCS[S.vlist[temp_i]]<temp_v?S.vlist[temp_i]:pivot; 
        temp_v=degInCS[pivot];
    }
    if(pivot<0||degInCS[pivot]>=temp_j-k){
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            pivot=degInCS[C.vlist[temp_i]]<temp_v?C.vlist[temp_i]:pivot;
            temp_v=degInCS[pivot];
        }
    }
    
    if(pivot<0||degInCS[pivot]>=temp_j-k){
        if(C.empty()){
            if(D.empty()&&S.vnum>=size_bound){
                OutputToFile();
                OutputToVector();
            }
            for(int i:RRC) if(!C.contains(i)) AddToC(i);
            for(int i:RRD) if(!D.contains(i)) D.add(i);
            return;
        }
        if((S.vnum+C.vnum>=size_bound)&&SCIsMaximal(k)){
            OutputToFile();
            OutputToVector();
        }
        for(int i:RRC) if(!C.contains(i)) AddToC(i);
        for(int i:RRD) if(!D.contains(i)) D.add(i);
        return;
    }
    
    // 通过缓存结果优化计算
    if (cache.find(pivot) != cache.end() && !cache[pivot]) {
        return;
    }

    /* Branching */
    // We remark that this implementation may be able to further refine by resuing some parts
    if(S.contains(pivot)){
        
        /* Case 1*/
        // Only Type 2 in this Case: we create those branches which include the pviot
        vector<int> nonNeg, ReC, ReD;
        ReD.reserve(D.vnum);
        ReC.reserve(C.vnum);
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            temp_j=C.vlist[temp_i];
            if(setG[pivot].find(temp_j)==setG[pivot].end()){
                nonNeg.push_back(temp_j);
            }
        }
        
        // branch 0
        int cur_node=nonNeg[0], pre_node;
        CToD(cur_node);
        KBranch_Rec(k,depth+1);
        // Now, cur_node is in D;
        pre_node=cur_node;
        int nodeID=1, branchID=0, branch_num=k-(S.vnum-degInS[pivot]);
        bool fast=false;
        while(branchID<branch_num){

            DToS(pre_node);
            if(!valid&&S.vnum==k&&C.vnum>graph_size/2){
                fast=true;
                valid=1;
                FastCDUpdate(k);
            }

            if(branchID==branch_num-1){
                for(auto i=nodeID+1;i<nonNeg.size();++i){
                    if(C.contains(nonNeg[i])){
                        RemoveFrC(nonNeg[i]);
                    }
                }
            }
            
            RefineCD(k,pre_node,ReC,ReD,low_bound);
            if(C.vnum==0) break;
            if(S.vnum+C.vnum<size_bound){
                break;
            }
            if(S.vnum+C.vnum<low_bound){
                break;
            }
            //for(;nodeID<nonNeg.size()&&!C.contains(nonNeg[nodeID]);++nodeID);
            for(;nodeID<nonNeg.size();++nodeID){
                if(C.contains(nonNeg[nodeID])&&(degInCS[nonNeg[nodeID]]<S.vnum+C.vnum-2||degInCS[pivot]<S.vnum+C.vnum-k)){
                    break;
                }
            }

            if(nodeID<nonNeg.size()){
                cur_node=nonNeg[nodeID];
                CToD(cur_node);
            }            
            KBranch_Rec(k,depth+1);
            if(nodeID>=nonNeg.size()) break;
            pre_node=cur_node;
            nodeID++;
            branchID++;
        }

        // Recovery
        fast=false;
        if(fast){
            for(int i:ReD) if(!D.contains(i)) D.add(i);
            for(int i:nonNeg){
                if(S.contains(i)) SToC(i);
                else if(D.contains(i)) D.remove(i);
            }
            C.clear();
            for(temp_i=0;temp_i<graph_size;++temp_i){
                if(!S.contains(temp_i)&&!D.contains(temp_i)){
                    C.add(temp_i);
                }
                degInCS[temp_i]=G_record[temp_i];
            }
            valid=0;

        }else{
            for(int i:ReC) if(!C.contains(i)) AddToC(i);
            for(int i:ReD) if(!D.contains(i)) D.add(i);
            for(int i:nonNeg){
                if(S.contains(i)) SToC(i);
                else if(D.contains(i)) D.remove(i);
                if(!C.contains(i)) AddToC(i);
            }
        }
        
        
        ReC.clear(); ReD.clear();
    }else{
        /* Case 2*/
        // Type 1: we first create branches which exclude the pivot
        // Type 2: we then create those branches which include the pivot
        vector<int> nonNeg, ReC, ReD;
        ReD.reserve(D.vnum);
        ReC.reserve(C.vnum);
        CToD(pivot); // do Type 1 branches which exclude the pivot
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            temp_j=C.vlist[temp_i];
            if(setG[pivot].find(temp_j)==setG[pivot].end()){
                nonNeg.push_back(temp_j);
            }
        }
        
        int nodeID=0, cur_node, branchID=0;
        bool rec_flag;
        while(true){
            for(;nodeID<nonNeg.size();++nodeID){
                if(degInCS[nonNeg[nodeID]]<C.vnum+S.vnum-k) break;
            }
            rec_flag=true;
            if(nodeID<nonNeg.size()){
                cur_node=nonNeg[nodeID];
                CToS(cur_node);
                RefineCD(k,cur_node,ReC,ReD,low_bound);
                for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
                    if(degInCS[D.vlist[temp_i]]==C.vnum+S.vnum){
                        rec_flag=false;
                    }
                }
            }
            
            if(rec_flag&&C.vnum+S.vnum>=size_bound&&C.vnum+S.vnum>=low_bound)
                KBranch_Rec(k,depth+1);

            if(nodeID<nonNeg.size()){
                RemoveFrS(cur_node);
                D.add(cur_node);
                for(int i:ReC) AddToC(i);
                for(int i:ReD) D.add(i);
                ReC.clear(); ReD.clear();
            }else{
                break;
            }
            nodeID++;
            branchID++;
            if(nodeID==branchID&&nodeID>=nonNeg.size()) break;
        }
        for(int node:nonNeg){
            if(D.contains(node)){
                D.remove(node);
                AddToC(node);
            }
        }
        //KBranch_Rec(k,depth+1);

        nodeID=0;branchID=0;
        int pre_node=pivot, branch_num=k-(S.vnum-degInS[pivot]+1)+1;
        while(branchID<branch_num){
            
            DToS(pre_node);
            
            if(branchID==branch_num-1){
                for(auto i=nodeID+1;i<nonNeg.size();++i){
                    if(C.contains(nonNeg[i])){
                        RemoveFrC(nonNeg[i]);
                    }
                }
            }
            RefineCD(k,pre_node,ReC,ReD,low_bound);
            if(C.vnum==0) break;
            if(S.vnum+C.vnum<size_bound){
                break;
            }
            if(S.vnum+C.vnum<low_bound){
                break;
            }
            for(;nodeID<nonNeg.size();++nodeID){
                if(C.contains(nonNeg[nodeID])&&(degInCS[nonNeg[nodeID]]<S.vnum+C.vnum-2||degInCS[pivot]<S.vnum+C.vnum-k)){
                    break;
                }
            }

            if(nodeID<nonNeg.size()){
                cur_node=nonNeg[nodeID];
                CToD(cur_node);
            }
            KBranch_Rec(k,depth+1);            
            
            
            if(nodeID>=nonNeg.size()) break; // End of the branch 

            pre_node=cur_node;
            nodeID++;
            branchID++;
        }

        // Recovery
        SToC(pivot);
        for(int i:ReC) if(!C.contains(i)) AddToC(i);
        for(int i:ReD) if(!D.contains(i)) D.add(i);
        for(int i:nonNeg){
            if(S.contains(i)) SToC(i);
            else if(D.contains(i)) D.remove(i);
            if(!C.contains(i)) AddToC(i);
        }
        ReC.clear(); ReD.clear();
    }
    for(int i:RRC) if(!C.contains(i)) AddToC(i);
    for(int i:RRD) if(!D.contains(i)) D.add(i);  
    cache[pivot] = true;
}


void FastQC::RefineC(int k, int node, vector<int> &ReC){
    // Find those boundary vertices in S that disconnect to `node'
    // Since S is a k-plex, # of those boundary vertices are bounded by O(k)
    vector<int> boundV;
    int temp_i, temp_j;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        temp_j=S.vlist[temp_i];
        if(degInS[temp_j]==S.vnum-k&&setG[node].find(temp_j)==setG[node].end()){
            boundV.push_back(temp_j);
        }
    }

    if(boundV.empty()){
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            temp_j=C.vlist[temp_i];
            if(degInS[temp_j]<S.vnum-k+1){
                ReC.push_back(temp_j);
                RemoveFrC(temp_j);
            }
        }
        return;
    }

    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_j=C.vlist[temp_i];
        if(degInS[temp_j]<S.vnum-k+1){
            RemoveFrC(temp_j);
            ReC.push_back(temp_j);
            continue;
        }
        for(int i:boundV){
            if(setG[temp_j].find(i)==setG[temp_j].end()){
                RemoveFrC(temp_j);
                ReC.push_back(temp_j);
                break;
            }
        }
    }
}

void FastQC::RefineD(int k, int node, vector<int> &ReD){
    // We refine D by following the similar procedure of refining C
    // We note that they can be combined to save some time, for which we shall do later
    vector<int> boundV;
    int temp_i, temp_j;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        temp_j=S.vlist[temp_i];
        if(degInS[temp_i]==S.vnum-k&&setG[node].find(temp_j)==setG[node].end()){
            boundV.push_back(temp_j);
        }
    }

    if(boundV.empty()){
        for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
            temp_j=D.vlist[temp_i];
            if(degInS[temp_j]<S.vnum-k+1){
                D.remove(temp_j);
                ReD.push_back(temp_j);
            }
        }
        return;
    }

    for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
        temp_j=D.vlist[temp_i];
        if(degInS[temp_j]<S.vnum-k+1){
            D.remove(temp_j);
            ReD.push_back(temp_j);
            continue;
        }
        for(int i:boundV){
            if(setG[temp_j].find(i)==setG[temp_j].end()){
                D.remove(temp_j);
                ReD.push_back(temp_j);
                break;
            }
        }
    }

}

void FastQC::RefineCD(int k, int node, vector<int> &ReC, vector<int> &ReD, int low_bound){
    // This is a simply combination of RefineC and RefineD
    //vector<int> boundV;
    if(S.vnum<k) return;
    boundV.clear();
    int temp_i, temp_j, low_p=std::ceil(gamma*(low_bound-1));
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        temp_j=S.vlist[temp_i];
        if(degInS[temp_j]==S.vnum-k&&setG[node].find(temp_j)==setG[node].end()){
            boundV.push_back(temp_j);
        }
    }
    
    if(boundV.empty()){
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            temp_j=C.vlist[temp_i];
            if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_p){
                RemoveFrC(temp_j);
                ReC.push_back(temp_j);
            }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
                // test!!!!!!
                RemoveFrC(temp_j);
                ReC.push_back(temp_j);
                //cout<<"ffff"<<endl;
            }
            if(C.vnum+S.vnum<size_bound||C.vnum+S.vnum<low_bound){
                return;
            }
        }

        if(C.vnum==0){
            for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
                temp_j=D.vlist[temp_i];
                if(degInS[temp_j]>=S.vnum-k+1){
                    return;
                }
            }
            if(IsQC(k)){
                OutputToFile();
                OutputToVector();
            }
            return;
        }

        for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
            temp_j=D.vlist[temp_i];
            if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_p){
                D.remove(temp_j);
                ReD.push_back(temp_j);
            }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
                D.remove(temp_j);
                ReD.push_back(temp_j);
            }
        }
        return;
    }
    
    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_j=C.vlist[temp_i];
        if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_p){
            RemoveFrC(temp_j);
            ReC.push_back(temp_j);
            continue;
        }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
            RemoveFrC(temp_j);
            ReC.push_back(temp_j);
            continue;
        }
        for(int i:boundV){
            if(setG[temp_j].find(i)==setG[temp_j].end()){
                RemoveFrC(temp_j);
                ReC.push_back(temp_j);
                break;
            }
        }
        if(C.vnum+S.vnum<size_bound||C.vnum+S.vnum<low_bound){
            return;
        }
    }
    
    if(C.vnum==0){
        int temp_node;
        bool nonmaximal;
        temp_i=boundV[0];
        for(temp_j=degree[temp_i]-1;temp_j>=0;--temp_j){
            temp_node=Graph[temp_i][temp_j];
            if(D.contains(temp_node)&&degInS[temp_node]>=S.vnum+1-k){
                if(degInS[temp_node]==S.vnum) return;
                nonmaximal=true;
                for(int i:boundV){
                    if(setG[temp_node].find(i)==setG[temp_node].end()){
                        nonmaximal=false;
                        break;
                    }
                }
                if(nonmaximal){
                    return;
                }
            }
        }
        if(IsQC(k)){
            OutputToFile();
            OutputToVector();
        }
        return;
    }

    for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
        temp_j=D.vlist[temp_i];
        if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_p){
            D.remove(temp_j);
            ReD.push_back(temp_j);
            continue;
        }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
            D.remove(temp_j);
            ReD.push_back(temp_j);
            continue;
        }
        for(int i:boundV){
            if(setG[temp_j].find(i)==setG[temp_j].end()){
                D.remove(temp_j);
                ReD.push_back(temp_j);
                break;
            }
        }
    }
}


inline void FastQC::RemoveFrC(int node){
    C.remove(node);
    for(int i=degree[node]-1;i>=0;--i)
        --degInCS[Graph[node][i]];
}

inline void FastQC::RemoveFrS(int node){
    S.remove(node);
    for(int i=degree[node]-1;i>=0;--i){
        --degInS[Graph[node][i]];
        --degInCS[Graph[node][i]];
    }
}

inline void FastQC::AddToC(int node){
    C.add(node);
    for(int i=degree[node]-1;i>=0;--i){
        ++degInCS[Graph[node][i]];
    }
}

inline void FastQC::AddToS(int node){
    S.add(node);
    for(int i=degree[node]-1;i>=0;--i){
        ++degInCS[Graph[node][i]];
        ++degInS[Graph[node][i]];
    }
}

inline void FastQC::CToS(int node){
    C.remove(node);
    S.add(node);
    for(int i=degree[node]-1;i>=0;--i){
        ++degInS[Graph[node][i]];
    }
}

inline void FastQC::CToD(int node){
    C.remove(node);
    D.add(node);
    for(int i=degree[node]-1;i>=0;--i){
        degInCS[Graph[node][i]]--;
    }
}

inline void FastQC::DToS(int node){
    D.remove(node);
    S.add(node);
    for(int i=degree[node]-1;i>=0;--i){
        ++degInCS[Graph[node][i]];
        ++degInS[Graph[node][i]];
    }
}

inline void FastQC::SToC(int node){
    S.remove(node);
    C.add(node);
    for(int i=degree[node]-1;i>=0;--i){
        degInS[Graph[node][i]]--;
    }
}

bool FastQC::SIsPlex(int k) {
    for (int i = S.vnum - 1; i >= 0; --i)
        if (degInS[S.vlist[i]] < S.vnum - k)
            return false;
    return true;
}

bool FastQC::SCIsMaximal(int k) {
    int temp_i, temp_j, temp_v = S.vnum + C.vnum;
    for (temp_i = D.vnum - 1; temp_i >= 0; --temp_i) {
        temp_j = D.vlist[temp_i];
        if (degInCS[temp_j] >= temp_v + 1 - k) break;
    }
    if (temp_i < 0) return true;

    boundV.clear();
    for (temp_i = S.vnum - 1; temp_i >= 0; --temp_i) {
        temp_j = S.vlist[temp_i];
        if (degInCS[temp_j] == temp_v - k) boundV.push_back(temp_j);
    }
    for (temp_i = C.vnum - 1; temp_i >= 0; --temp_i) {
        temp_j = C.vlist[temp_i];
        if (degInCS[temp_j] == temp_v - k) boundV.push_back(temp_j);
    }
    if (boundV.empty()) return false;

    bool check;
    for (temp_i = D.vnum - 1; temp_i >= 0; --temp_i) {
        temp_j = D.vlist[temp_i];
        if (degInCS[temp_j] >= temp_v + 1 - k) {
            check = true;
            for (int i : boundV) {
                if (setG[temp_j].find(i) == setG[temp_j].end()) {
                    check = false;
                    break;
                }
            }
            if (check) {
                return false;
            }
        }
    }

    return true;
}

void FastQC::CheckDegree(){
    int temp_i, temp_j, temp_node;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        temp_node=S.vlist[temp_i];
        for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
            G_index[Graph[temp_node][temp_j]]++;
        }
    }

    for(temp_i=graph_size-1;temp_i>=0;--temp_i){
        if(degInS[temp_i]!=G_index[temp_i]){
            cout<<"Degree error (S)"<<" "<<temp_i<<" "<<degInS[temp_i]<<" "<<G_index[temp_i]<<" "<<S.vnum<<std::endl;
            exit(0);
        }
    }

    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_node=C.vlist[temp_i];
        for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
            G_index[Graph[temp_node][temp_j]]++;
        }
    }

    for(temp_i=graph_size-1;temp_i>=0;--temp_i){
        if(degInCS[temp_i]!=G_index[temp_i]){
            cout<<"Degree error (CS)"<<std::endl;
            exit(0);
        }
        G_index[temp_i]=0;
    }
}

bool FastQC::CheckMaximal(int k){
    // Note that this function is used to check the maximality of S and C for debugging
    // Before invoking this function, S\cup C should be a valid k-plex
    if(S.empty()&&C.empty()){
        cout<<"S is empty"<<std::endl;
        exit(0);
    }
    
    vector<int> boundV;
    int temp_i, temp_j, temp_node, size=S.vnum+C.vnum;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        temp_node=S.vlist[temp_i];
        if(degInCS[temp_node]==size-k){
            boundV.push_back(temp_node);
        }
    }
    
    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_node=C.vlist[temp_i];
        if(degInCS[temp_node]==size-k){
            boundV.push_back(temp_node);
        }
    }
    
    if(boundV.empty()){
        for(temp_i=graph_size-1;temp_i>=0;--temp_i){
            if(degInCS[temp_i]>=size-k+1&&!C.contains(temp_i)&&!S.contains(temp_i)){
                // cout<<"not maximal"<<" "<<temp_i<<" "<<D.contains(temp_i)<<endl;
                // exit(0);
                return false;
            }
        }
        return true;
    }
    
    temp_i=boundV[0];
    for(temp_j=degree[temp_i]-1;temp_j>=0;--temp_j){
        temp_node=Graph[temp_i][temp_j];
        if(degInCS[temp_node]>=size-k+1&&!C.contains(temp_node)&&!S.contains(temp_node)){
            bool aa=true;
            for(int i:boundV){
                if(setG[temp_node].find(i)==setG[temp_node].end()){
                    aa=false;
                    break;
                }
            }
            if(aa){
                // cout<<"Not Maximal"<<" "<<temp_node<<endl;
                // Output();
                // exit(0);
                return false;
            }
        }
    }
    return true;

}

void FastQC::Output(){
    int temp_i;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        cout<<S.vlist[temp_i]<<" ";
    }
    cout<<std::endl;
    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        cout<<C.vlist[temp_i]<<" ";
    }
    cout<<std::endl;
}

bool FastQC::FastCDUpdate(int k){
    if(S.vnum!=k) return false;
    boundV.clear();
    int temp_i, temp_j, temp_node;
    for(temp_i=k-1;temp_i>=0;--temp_i){
        temp_node=S.vlist[temp_i];
        if(degInS[temp_node]==S.vnum-k){
            boundV.push_back(temp_node);
        }
    }
    if(boundV.empty()){
        for(temp_i=graph_size-1;temp_i>=0;--temp_i){
            G_record[temp_i]=degInCS[temp_i];
            degInCS[temp_i]=degInS[temp_i];
        }
        for(temp_i=k-1;temp_i>=0;--temp_i){
            temp_node=S.vlist[temp_i];
            for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
                G_index[Graph[temp_node][temp_j]]=1;
            }
        }
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            temp_node=C.vlist[temp_i];
            if(G_index[temp_node]==0){
                C.remove(temp_node);
            }else{
                for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
                    degInCS[Graph[temp_node][temp_j]]++;
                }
            }
        }
        for(temp_i=k-1;temp_i>=0;--temp_i){
            temp_node=S.vlist[temp_i];
            for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
                G_index[Graph[temp_node][temp_j]]=0;
            }
        }
        return true;
    }

    int temp_k=boundV[0];
    for(temp_i=graph_size-1;temp_i>=0;--temp_i){
        G_record[temp_i]=degInCS[temp_i];
        degInCS[temp_i]=degInS[temp_i];
    }
    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_node=C.vlist[temp_i];
        if(setG[temp_k].find(temp_node)==setG[temp_k].end()){
            C.remove(temp_node);
        }else{
            for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
                degInCS[Graph[temp_node][temp_j]]++;
            }
        }
    }
    return true;

}

int FastQC::EstimateK(){
    int d_min, temp_i,k;
    if(S.empty()){
        d_min=0;
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            d_min=d_min<degInCS[C.vlist[temp_i]]?degInCS[C.vlist[temp_i]]:d_min;
        }
        d_min=std::floor(d_min*(1-gamma)/gamma+0.000000001)+1;
        k=std::floor((1-gamma)*C.vnum+gamma+0.000000001);
        k=k<d_min? k:d_min;
        return k;
    }
    d_min=graph_size;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        d_min=d_min<degInCS[S.vlist[temp_i]]?d_min:degInCS[S.vlist[temp_i]];
    }
    d_min=std::floor(d_min*(1-gamma)/gamma+0.000000001)+1;
    k=std::floor((1-gamma)*(C.vnum+S.vnum)+gamma+0.000000001);
    k=k<d_min? k:d_min;
    return k;
}

bool FastQC::IterRefineCD(int &k, vector<int> &ReC, vector<int> &ReD, int &low_bound){
    int ek, temp_i, min_deg;
    low_bound=size_bound;
    while((ek=EstimateK())<k){
        k=ek;
        // Check feasibility
        if(S.vnum+C.vnum<size_bound) return false;
        min_deg=graph_size;
        for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
            if(degInS[S.vlist[temp_i]]<S.vnum-k||degInCS[S.vlist[temp_i]]<size_p){
                return false;
            }
            min_deg=min_deg>degInS[S.vlist[temp_i]]?degInS[S.vlist[temp_i]]:min_deg;
            //min_deg1=min_deg1>degInCS[S.vlist[temp_i]]?degInCS[S.vlist[temp_i]]:min_deg1;
        }
        temp_i=S.vnum+std::floor((gamma*(S.vnum-1)-min_deg)/(1-gamma)+0.00000001);        
        low_bound=low_bound>temp_i?low_bound:temp_i;

        // if(low_bound>size_bound){
        //     int aaa=0;
        //     for(temp_i=S.vnum-1;temp_i>=0;--temp_i) aaa+=degInS[S.vlist[temp_i]];
        //     for(temp_i=C.vnum-1;temp_i>=0;--temp_i) G_temp[temp_i]=degInS[C.vlist[temp_i]];
        //     sort(G_temp,G_temp+C.vnum,greater <>());
        //     int bbb=0;
        //     for(temp_i=0;temp_i<low_bound-S.vnum;++temp_i) bbb+=G_temp[temp_i];
        //     while(aaa+bbb<S.vnum*(ceil(gamma*(S.vnum+temp_i-1)))){
        //         bbb+=G_temp[temp_i];
        //         temp_i++;
        //         if(temp_i>=C.vnum) break;
        //     }
        //     if(temp_i+S.vnum>low_bound){
        //         low_bound=temp_i+S.vnum;
        //     }
        // }

        if(S.vnum+C.vnum<low_bound) return false;
        // if(std::ceil(min_deg1/gamma+0.00001)+1<=low_bound){
        //     cout<<"ffff "<<std::ceil(min_deg1/gamma+0.00001)+1<<" "<<low_bound<<endl;
        // }
        //if(low_bound>size_bound&&S.vlist[0]==18963) cout<<low_bound<<endl;
        // int tt=std::ceil(min_deg/gamma)+1;
        // for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        //     if(degInCS[S.vlist[temp_i]]+tt<std::ceil(gamma*(S.vnum+tt-1))){
        //         cout<<"ffff"<<endl;
        //     }
        // }
        
        // Update C and D below
        RefineCD(k,ReC,ReD,std::ceil(gamma*(low_bound-1)));
        if(C.empty()) return false;
        if(S.vnum+C.vnum<low_bound) return false;
    }
    return true;
}

void FastQC::RefineCD(int k, vector<int> &ReC, vector<int> &ReD, int low_bound){
    if(S.vnum<k) return;
    boundV.clear();
    int temp_i, temp_j;
    for(temp_i=S.vnum-1;temp_i>=0;--temp_i){
        temp_j=S.vlist[temp_i];
        if(degInS[temp_j]==S.vnum-k){
            boundV.push_back(temp_j);
        }
    }
    
    if(boundV.empty()){
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
            temp_j=C.vlist[temp_i];
            if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_bound){
                RemoveFrC(temp_j);
                ReC.push_back(temp_j);
            }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
                // test!!!!!!
                RemoveFrC(temp_j);
                ReC.push_back(temp_j);
            }
            if(C.vnum+S.vnum<low_bound){
                return;
            }
        }

        if(C.vnum==0){
            for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
                temp_j=D.vlist[temp_i];
                if(degInS[temp_j]>=S.vnum-k+1){
                    return;
                }
            }
            if(IsQC(k)){
                OutputToFile();
                OutputToVector();
            }
            return;
        }

        for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
            temp_j=D.vlist[temp_i];
            if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_bound){
                D.remove(temp_j);
                ReD.push_back(temp_j);
            }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
                // test!!!!!!
                D.remove(temp_j);
                ReD.push_back(temp_j);
            }
        }
        return;
    }
    
    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_j=C.vlist[temp_i];
        if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_bound){
            RemoveFrC(temp_j);
            ReC.push_back(temp_j);
            continue;
        }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
            // test!!!
            RemoveFrC(temp_j);
            ReC.push_back(temp_j);
            continue;
        }
        for(int i:boundV){
            if(setG[temp_j].find(i)==setG[temp_j].end()){
                RemoveFrC(temp_j);
                ReC.push_back(temp_j);
                break;
            }
        }
        if(C.vnum+S.vnum<low_bound){
            return;
        }
    }
    
    if(C.vnum==0){
        int temp_node;
        bool nonmaximal;
        temp_i=boundV[0];
        for(temp_j=degree[temp_i]-1;temp_j>=0;--temp_j){
            temp_node=Graph[temp_i][temp_j];
            if(D.contains(temp_node)&&degInS[temp_node]>=S.vnum+1-k){
                if(degInS[temp_node]==S.vnum) return;
                nonmaximal=true;
                for(int i:boundV){
                    if(setG[temp_node].find(i)==setG[temp_node].end()){
                        nonmaximal=false;
                        break;
                    }
                }
                if(nonmaximal){
                    return;
                }
            }
        }
        if(IsQC(k)){
            OutputToFile();
            OutputToVector();
        }
        return;
    }

    for(temp_i=D.vnum-1;temp_i>=0;--temp_i){
        temp_j=D.vlist[temp_i];
        if(degInS[temp_j]<S.vnum-k+1||degInCS[temp_j]<size_bound-k||degInCS[temp_j]<size_p||degInCS[temp_j]<low_bound){
            D.remove(temp_j);
            ReD.push_back(temp_j);
            continue;
        }else if(degInCS[temp_j]<std::ceil(gamma*(S.vnum+degInCS[temp_j]-degInS[temp_j]))){
            // test!!!!!!
            D.remove(temp_j);
            ReD.push_back(temp_j);
            continue;
        }
        for(int i:boundV){
            if(setG[temp_j].find(i)==setG[temp_j].end()){
                D.remove(temp_j);
                ReD.push_back(temp_j);
                break;
            }
        }
    }
}

void FastQC::OutputToFile(){
    // if(quiete==0) return;
    int temp_i;
    if(S.vnum+C.vnum<size_bound) return;
    if(!S.empty()){
        in<<(S.vnum+C.vnum)<<" "<<S.vlist[0];
        for(temp_i=S.vnum-1;temp_i>0;--temp_i)  in<<" "<<S.vlist[temp_i];
        for(temp_i=C.vnum-1;temp_i>=0;--temp_i) in<<" "<<C.vlist[temp_i];
        in<<endl;
    }else{
        in<<(S.vnum+C.vnum)<<" "<<C.vlist[0];
        for(temp_i=C.vnum-1;temp_i>0;--temp_i) in<<" "<<C.vlist[temp_i];
        for(temp_i=S.vnum-1;temp_i>=0;--temp_i) in<<" "<<S.vlist[temp_i];
        in<<endl;
    }
}


void FastQC::OutputToVector() {
    int temp_i;
    if (S.vnum + C.vnum < size_bound) return;

    igraph_vector_t* clique = new igraph_vector_t;
    igraph_vector_init(clique, S.vnum + C.vnum);

    int index = 0;
    if (!S.empty()) {
        VECTOR(*clique)[index++] = S.vlist[0];
        for (temp_i = S.vnum - 1; temp_i > 0; --temp_i) {
            VECTOR(*clique)[index++] = S.vlist[temp_i];
        }
        for (temp_i = C.vnum - 1; temp_i >= 0; --temp_i) {
            VECTOR(*clique)[index++] = C.vlist[temp_i];
        }
    } else {
        VECTOR(*clique)[index++] = C.vlist[0];
        for (temp_i = C.vnum - 1; temp_i > 0; --temp_i) {
            VECTOR(*clique)[index++] = C.vlist[temp_i];
        }
        for (temp_i = S.vnum - 1; temp_i >= 0; --temp_i) {
            VECTOR(*clique)[index++] = S.vlist[temp_i];
        }
    }

    igraph_vector_ptr_push_back(&cliques, clique);
    res_num++;
}

bool FastQC::IsQC(int k){
    if(C.vnum>0){
        cout<<"error"<<endl;
    }
    int ek=std::floor((1-gamma)*S.vnum+gamma);
    if(ek>=k) return true;
    for(int temp_i=S.vnum-1;temp_i>=0;--temp_i){
        if(degInS[temp_i]<S.vnum-ek){
            return false;
        }
    }
    return true;
}

const igraph_vector_ptr_t& FastQC::DCStrategy(){

    int temp_i, temp_j, temp_k, temp_node, temp_node2;
    int statis=0,ek;
    bool branch_flag=true;
    for(int pivot=0;pivot<graph_size-size_bound+1;++pivot){
        AddToS(pivot);
        for(temp_i=degree[pivot]-1;temp_i>=0;--temp_i){
            temp_node=Graph[pivot][temp_i];
            if(temp_node<=pivot) continue;
            //degInCS[temp_node]++;
            if(!C.contains(temp_node)){
                C.add(temp_node);
                for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
                    temp_node2=Graph[temp_node][temp_j];
                    degInCS[temp_node2]++;
                    if(temp_node2<=pivot) continue;
                    if(!C.contains(temp_node2)&&setG[pivot].find(temp_node2)==setG[pivot].end()){
                        C.add(temp_node2);
                        for(temp_k=degree[temp_node2]-1;temp_k>=0;--temp_k){
                            degInCS[Graph[temp_node2][temp_k]]++;
                        }
                    }
                }
            }
        } 

        OneHobP(pivot);
	    TwoHobP(pivot);
        OneHobP(pivot);
        TwoHobP(pivot);
        
        ek=EstimateK();
        if(ek==1){
            for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
                if(setG[pivot].find(C.vlist[temp_i])==setG[pivot].end()){
                    RemoveFrC(C.vlist[temp_i]);
                }
            }
        }

        
        if(C.vnum+1>=size_bound){
            branch_flag=true;
            for(temp_i=degree[pivot]-1;temp_i>=0;--temp_i){
                temp_node=Graph[pivot][temp_i];
                if(degInCS[temp_node]==C.vnum+1&&temp_node<pivot){
                    branch_flag=false;
                }
            }
            if(branch_flag)
                KBranch_Rec(ek,1);
        }

        C.clear();
        S.clear();
        D.clear();
        for(temp_i=graph_size-1;temp_i>=0;--temp_i){
            degInCS[temp_i]=0;
            degInS[temp_i]=0;
        }
            
    }
    //cout<<statis<<endl;
    return cliques;
}

const igraph_vector_ptr_t& FastQC::DCStrategy(const std::vector<int>& original_indices,int num_of_nodes_lower_than_k) {
    int temp_i, temp_j, temp_k, temp_node, temp_node2;
    int statis = 0, ek;
    bool branch_flag = true;
    for (int pivot = 0; pivot < graph_size - size_bound + 1; ++pivot) {
        if (res_num >= max_results) continue;  // Add this line
        AddToS(pivot);
        for (temp_i = degree[pivot] - 1; temp_i >= 0; --temp_i) {
            temp_node = Graph[pivot][temp_i];
            if (temp_node <= pivot) continue;
            if (!C.contains(temp_node)) {
                C.add(temp_node);
                for (temp_j = degree[temp_node] - 1; temp_j >= 0; --temp_j) {
                    temp_node2 = Graph[temp_node][temp_j];
                    degInCS[temp_node2]++;
                    if (temp_node2 <= pivot) continue;
                    if (!C.contains(temp_node2) && setG[pivot].find(temp_node2) == setG[pivot].end()) {
                        C.add(temp_node2);
                        for (temp_k = degree[temp_node2] - 1; temp_k >= 0; --temp_k) {
                            degInCS[Graph[temp_node2][temp_k]]++;
                        }
                    }
                }
            }
        }

        OneHobP(pivot);
        TwoHobP(pivot);
        OneHobP(pivot);
        TwoHobP(pivot);

        ek = EstimateK();
        if (ek == 1) {
            for (temp_i = C.vnum - 1; temp_i >= 0; --temp_i) {
                if (setG[pivot].find(C.vlist[temp_i]) == setG[pivot].end()) {
                    RemoveFrC(C.vlist[temp_i]);
                }
            }
        }

        if (C.vnum + 1 >= size_bound) {
            branch_flag = true;
            for (temp_i = degree[pivot] - 1; temp_i >= 0; --temp_i) {
                temp_node = Graph[pivot][temp_i];
                if (degInCS[temp_node] == C.vnum + 1 && temp_node < pivot) {
                    branch_flag = false;
                }
            }
            if (branch_flag)
                KBranch_Rec(ek, 1);
        }

        C.clear();
        S.clear();
        D.clear();
        for (temp_i = graph_size - 1; temp_i >= 0; --temp_i) {
            degInCS[temp_i] = 0;
            degInS[temp_i] = 0;
        }
    }

    // 映射回原始编号
    igraph_vector_ptr_t *mapped_cliques = new igraph_vector_ptr_t;
    igraph_vector_ptr_init(mapped_cliques, igraph_vector_ptr_size(&cliques));

    for (int i = 0; i < igraph_vector_ptr_size(&cliques); ++i) {
        igraph_vector_t *clique = (igraph_vector_t*)VECTOR(cliques)[i];
        igraph_vector_t *mapped_clique = new igraph_vector_t;
        igraph_vector_init(mapped_clique, igraph_vector_size(clique));
        for (int j = 0; j < igraph_vector_size(clique); ++j) {
            int new_position = (int)VECTOR(*clique)[j] + num_of_nodes_lower_than_k;
            // 找到原始节点编号
            // int original_node = std::find(original_indices.begin(), original_indices.end(), new_position) - original_indices.begin();
            int original_node = original_indices[new_position];
            VECTOR(*mapped_clique)[j] = original_node;
        }
        VECTOR(*mapped_cliques)[i] = mapped_clique;
    }

    return *mapped_cliques;
}
void FastQC::OneHobP(int pivot){
    int bound=std::ceil(gamma*(size_bound-1));
    int temp_i, temp_j, temp_node;
    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_node=C.vlist[temp_i];
        if(degInCS[temp_node]<bound){
            // C.remove(temp_node);
            // for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
            //     degInCS[Graph[temp_node][temp_j]]--;
            // }
            RemoveFrC(temp_node);
        }
    }
}

void FastQC::TwoHobP(int pivot){
    //int bound=size_bound-std::floor((1-gamma)*size_bound+gamma+0.00000001)-2.2*std::floor((1-gamma)*(size_bound+1)+gamma+0.00000001);
    int bound=size_bound-std::floor((1-gamma)*size_bound+gamma+0.00000001)-std::floor((1-gamma)*(size_bound+1)+gamma+0.00000001);
    int temp_i, temp_j, temp_node, count=0;
    for(temp_i=degree[pivot]-1;temp_i>=0;--temp_i){
        temp_node=Graph[pivot][temp_i];
        if(temp_node>pivot)
            G_index[temp_node]=1;
    }

    for(temp_i=C.vnum-1;temp_i>=0;--temp_i){
        temp_node=C.vlist[temp_i];
        count=0;
        for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
            if(G_index[Graph[temp_node][temp_j]]) count++;
        }
        if((G_index[temp_node]&&count<bound)||(!G_index[temp_node]&&count<bound+2)){
            C.remove(temp_node);
            for(temp_j=degree[temp_node]-1;temp_j>=0;--temp_j){
                degInCS[Graph[temp_node][temp_j]]--;
            }
        }
    }


    for(temp_i=degree[pivot]-1;temp_i>=0;--temp_i){
        G_index[Graph[pivot][temp_i]]=0;
    }
}
