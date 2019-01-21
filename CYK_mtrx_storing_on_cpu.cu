/*
**
**  Copy matrix from cpu on gpu each iteration
**
*/


#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <vector>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <bitset>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define BLOCK_SIZE 1024

int packedByBlocksNumber(int N, int size) {
  return (N / size + (N % size == 0 ? 0 : 1));
}

typedef struct mtrxStruct {
    int nrows; 
    int ncols; 
    uint32_t* bitArray;
} mtrx;

mtrx* mtrxInit(int rows, int cols) {
    mtrx* m = (mtrx*) malloc(sizeof(struct mtrxStruct));
    m->nrows = packedByBlocksNumber(rows, BLOCK_SIZE) * BLOCK_SIZE;
    m->ncols = packedByBlocksNumber(cols, 32);
    m->bitArray = reinterpret_cast<uint32_t *> (calloc(m->nrows * m->ncols, sizeof(uint32_t)));
    return m;
}

void mtrxWriteBit(mtrx* m, int i, int j){
     m->bitArray[i * m->ncols + (j / 32)] |= 1 << (31 - (j % 32));
}

__global__ void mulSqMtrxDevice(uint32_t *A, uint32_t *B, int rows, int cols, uint32_t *C) {
  
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_y = threadIdx.y;
  
    int aBegin = block_y * BLOCK_SIZE * cols + thread_y * cols;
    int aEnd = aBegin + cols - 1;
    int bBegin = block_x;
    uint32_t cElem = 0;
    int bInd = 0;
  
    for (int i = aBegin; i <= aEnd; i++){
      uint32_t mask = (1 << 31);
      uint32_t aCurr = A[i];
  
      for (int bit = 0; bit < 32; bit ++) {
        if (aCurr & mask) {
          cElem |= B[bInd * 32 * cols + bit * cols + bBegin ];
        } 
        aCurr <<= 1;
      }
  
      bInd ++;
    }
    C[block_y * BLOCK_SIZE * cols + thread_y * cols + bBegin] = cElem;
}

uint32_t* mulSqMtrxHost(mtrx* m1, mtrx* m2) {
    uint32_t *A = m1->bitArray;
    uint32_t *B = m2->bitArray;
    int num_rows = m1->nrows;
    int num_columns = m1->ncols;
  
    int ABC_size = num_rows * num_columns;
    int ABC_memsize = ABC_size * sizeof(uint32_t);
  
    uint32_t *d_A, *d_B, *d_C;
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_A), ABC_memsize));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_B), ABC_memsize));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_C), ABC_memsize));
  
    gpuErrchk(cudaMemcpy(d_A, A, ABC_memsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, ABC_memsize, cudaMemcpyHostToDevice));
  
    gpuErrchk(cudaMemset(d_C, 0, ABC_memsize));
  
    dim3 dimBlock (1, BLOCK_SIZE);
    dim3 dimGrid(num_columns, num_rows / BLOCK_SIZE); 
    
    mulSqMtrxDevice<<<dimGrid, dimBlock>>>(d_A, d_B, num_rows, num_columns, d_C);
    cudaDeviceSynchronize();
  
    uint32_t *g_C = reinterpret_cast<uint32_t *>(calloc(num_rows * num_columns, sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(g_C, d_C, ABC_memsize, cudaMemcpyDeviceToHost));
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
  
    return g_C;
}

bool MatrixAdd(uint32_t* A, uint32_t* B, int rows, int cols) {
  int num_rows = rows;
  int num_columns = cols;

  bool isDiff = false;

  for (int r = 0; r < num_rows; r++) {
    for (int c = 0; c < num_columns; c++) {
        uint32_t tmp = A[r * num_columns + c];
        uint32_t res = (tmp | B [r * num_columns + c])  ;     
        if ( res != tmp){
            A[r * num_columns + c] |= res;
            isDiff = true;
        }
    }
  }
  return isDiff;
}

bool * Decompress(uint32_t * c_arr, uint32_t N) {
  // int num_rows = N;
  int num_columns = packedByBlocksNumber(N, 32);
  bool * arr = reinterpret_cast<bool *>(calloc(N * N, sizeof(bool)));

  uint32_t el;
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      el = c_arr[r * num_columns + (c / 32)];
      if (el & (1 << (31 - (c % 32)))) {
        arr[r * N + c] = 1;
      }
    }
  }

  return arr;
}



typedef struct EdgeStruct {
    int startVertex;
    int endVertex;
    std::string label;
} EdgeType;

typedef std::vector<EdgeType> GraphType;

int readGraph(char* filename, GraphType& graph){
    int n = 0;
    EdgeType e;
    ifstream input(filename);

    string line ;
    while (getline(input, line)) {
            
        unsigned long deviderInd = line.find(' ');
        string startVertex = line.substr(0, deviderInd);

        line = line.substr(deviderInd+ 1, line.size() - (deviderInd + 1));
        deviderInd = line.find(' ');

        string term = line.substr(0, deviderInd);
        line = line.substr(deviderInd+ 1, line.size() - (deviderInd + 1));

        e.startVertex = stoi(startVertex);
        e.label = term;
        e.endVertex = stoi(line);

        n = max(n, e.startVertex);
        n = max(n, e.endVertex);
        graph.push_back(e);
    }
    input.close();
    return n + 1;
}

typedef std::vector<std::string> heads;
typedef std::pair<std::string, std::string> nonterminal_pair;

class Grammar {
    public:
        Grammar();
        ~Grammar();
        void read(const std::string &filename);
        void print();
    void IntersectionWithGraph(int n, GraphType graph, char* filename);

    private:
        std::map<std::string, mtrx*> nonterminals;
        std::map<nonterminal_pair, heads> rules;
        std::map<std::string, std::vector<std::string>> terminals;
};


Grammar::Grammar() = default;

void Grammar::read(const string &filename) {
    ifstream input(filename);

    string line;
    while (getline(input, line)) {

        unsigned long divider = line.find(' ');
        string head = line.substr(0, divider);
        string body = line.substr(divider + 1, line.size() - (divider + 1));
        nonterminals[head] = nullptr;

        divider = body.find(' ');
        if (divider == string::npos) {
            terminals[body].push_back(head);
            continue;
        }

        string fst = body.substr(0, divider);
        string snd = body.substr(divider + 1, body.size() - (divider + 1));

        nonterminals[fst] = nullptr;
        nonterminals[snd] = nullptr;

        rules[nonterminal_pair(fst, snd)].push_back(head);
    }

    input.close();
}

void Grammar::print() {
    cout << "Nonterminals: ";
    for (auto &nonterminal : nonterminals) {
        cout << nonterminal.first << ' ';
    }
    cout << '\n';
    cout << "Terminals: ";
    for (auto &terminal : terminals) {
        cout << terminal.first << ' ';
    }
    cout << '\n';
    cout << "Prods:\n";
    for (auto &prod : rules) {
        cout << prod.first.first << ' ' << prod.first.second << " <- ";
        for (auto &head : prod.second) {
            cout << head << ", ";
        }
        cout << '\n';
    }
    cout << '\n';
}

void Grammar::IntersectionWithGraph(int n, GraphType graph, char *filename) {
    
    for (auto &nonterminal: nonterminals) {
        nonterminal.second = mtrxInit(n, n);
    }
    
    for (auto &edge: graph) {
        for (auto &nonterm: terminals[edge.label]) {
            mtrxWriteBit(nonterminals[nonterm], edge.startVertex, edge.endVertex);
        }
    }
    bool finished = false;

    while (!finished) {
        finished = true;
        for (auto &rule: rules) {
            
                uint32_t* mul_result = mulSqMtrxHost(nonterminals[rule.first.first], nonterminals[rule.first.second]);
                for (auto &head: rule.second) {
                    if ( MatrixAdd(nonterminals[head]->bitArray, mul_result, nonterminals[head]->nrows, nonterminals[head]->ncols)) {
                        finished = false;
                    }
                }
                free(mul_result);
        }
    }

    ofstream outputfile;
    outputfile.open(filename);
    for (auto &nonterminal: nonterminals) {
        outputfile << nonterminal.first;
        bool *bitArray = Decompress(nonterminal.second->bitArray, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (bitArray[i * n + j] != 0) {
                    outputfile << ' ' << i << ' ' << j;
                }
            }
        }
        outputfile << endl;
    }
    outputfile.close();
}


int main(int argc, char* argv[]) {

    auto * grammar = new Grammar;
    grammar->read(argv[1]);

    GraphType graph = GraphType();
    int n = readGraph(argv[2], graph);

    grammar->IntersectionWithGraph(n, graph, argv[3]);

    return 0;
}