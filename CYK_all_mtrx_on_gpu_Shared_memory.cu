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

__device__ uint8_t is_changed;

uint32_t *mulMtrxTmp;

 
void setFlag() {
    uint8_t flag = 0;
    gpuErrchk(cudaMemcpyToSymbol(is_changed, &flag, sizeof(uint8_t)))
}
 
uint8_t getFlag() {
    uint8_t flag;
    gpuErrchk(cudaMemcpyFromSymbol(&flag, is_changed, sizeof(uint8_t)))
 
    return flag;
}

int packedByBlocksNumber(int N, int size) {
  return (N / size + (N % size == 0 ? 0 : 1));
}

typedef struct mtrxStruct {
    int nrows; 
    int ncols; 
    uint32_t* bitArray;
    uint32_t* gpuBitArray;
} mtrx;

mtrx* mtrxInit(int rows, int cols) {
    mtrx* m = (mtrx*) malloc(sizeof(struct mtrxStruct));
    m->nrows = packedByBlocksNumber(rows, BLOCK_SIZE) * BLOCK_SIZE;
    m->ncols = packedByBlocksNumber(cols, 32);
    m->bitArray = reinterpret_cast<uint32_t *> (calloc(m->nrows * m->ncols, sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&(m->gpuBitArray)), m->nrows * m->ncols * sizeof(uint32_t)));
    return m;
}

void copyMtrxFromCPUtoGPU(mtrx* m){
    gpuErrchk(cudaMemcpy(m->gpuBitArray,  m->bitArray, m->nrows * m->ncols * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void moveMtrxFromGPUtoCPU(mtrx* m){
    gpuErrchk(cudaMemcpy( m->bitArray, m->gpuBitArray, m->nrows * m->ncols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(m->gpuBitArray);
}

void mtrxWriteBit(mtrx* m, int i, int j){
     m->bitArray[i * m->ncols + (j / 32)] |= 1 << (31 - (j % 32));
}

__global__ void mulSqMtrxDevice(uint32_t *A, uint32_t *B, const int rows, int cols, uint32_t *C) {
  
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
  
    //int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
  
    int aBegin = block_y * BLOCK_SIZE * cols + thread_y * cols;
    int aEnd = aBegin + cols - 1;
    int bBegin = block_x;
    uint32_t cElem = 0;
    int bInd = 0;
    
    __shared__ uint32_t Bs[BLOCK_SIZE];

    for (int i = 0; i < rows; i++) {
        Bs[i] = B[i * cols + bBegin ];
    }
    __syncthreads();

  
    for (int i = aBegin; i <= aEnd; i++){
      uint32_t mask = (1 << 31);
      uint32_t aCurr = A[i];
  
      for (int bit = 0; bit < 32; bit ++) {
        if (aCurr & mask) {
        //   cElem |= B[bInd * 32 * cols + bit * cols + bBegin ];
          cElem |= Bs[bInd * 32 + bit];
        } 
        aCurr <<= 1;
      }
  
      bInd ++;
    }
    C[block_y * BLOCK_SIZE * cols + thread_y * cols + bBegin] = cElem;
}

void mulSqMtrxHost(mtrx* m1, mtrx* m2) {
    uint32_t *A_dev = m1->gpuBitArray;
    uint32_t *B_dev = m2->gpuBitArray;
    int num_rows = m1->nrows;
    int num_columns = m1->ncols;
  
    int ABC_size = num_rows * num_columns;
    int ABC_memsize = ABC_size * sizeof(uint32_t);
  
    //***
    gpuErrchk(cudaMemset(mulMtrxTmp, 0, ABC_memsize));
  
    dim3 dimBlock (1, BLOCK_SIZE);
    dim3 dimGrid(num_columns, num_rows / BLOCK_SIZE); 
    
    mulSqMtrxDevice<<<dimGrid, dimBlock>>>(A_dev, B_dev, num_rows, num_columns, mulMtrxTmp);
    cudaDeviceSynchronize();
  
}

__global__ void addToLeftSqMtrxDevice(uint32_t *A, uint32_t *B, int rows, int cols) {
  
    int block_y = blockIdx.y;
    int thread_y = threadIdx.y;
  
    int startInd= block_y * BLOCK_SIZE * cols + thread_y * cols;
    int endInd = startInd + cols - 1;
  
    for (int i = startInd; i <= endInd; i++){
        uint32_t tmp = A[i];
        A[i] |= B[i];
        if (A[i] != tmp){
            is_changed = 1;
        }
    }
}

void addToLeftMtrx(uint32_t* A_dev, int rows, int cols) {

    dim3 dimBlock (1, BLOCK_SIZE);
    dim3 dimGrid(1, rows / BLOCK_SIZE); 
    
    addToLeftSqMtrxDevice<<<dimGrid, dimBlock>>>(A_dev, mulMtrxTmp, rows, cols);
    cudaDeviceSynchronize();
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

    for (auto &nonterminal: nonterminals) {
        copyMtrxFromCPUtoGPU(nonterminal.second);
    }

    bool finished = false;

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&mulMtrxTmp), packedByBlocksNumber(n, 32) * packedByBlocksNumber(n, 1024) * 1024 * sizeof(uint32_t)));
    
    while (!finished) {
        setFlag();
        finished = true;
        for (auto &rule: rules) {
            mulSqMtrxHost(nonterminals[rule.first.first], nonterminals[rule.first.second]);
            for (auto &head: rule.second) {
                addToLeftMtrx(nonterminals[head]->gpuBitArray, nonterminals[head]->nrows, nonterminals[head]->ncols) ;  
            }
            
                
        }
        if (getFlag())
            finished = false;
    }
    cudaFree(mulMtrxTmp);
    
    for (auto &nonterminal: nonterminals) {
        moveMtrxFromGPUtoCPU(nonterminal.second);
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