#include <cstdlib>
#include <iostream>
#include <curand.h>
#include <math.h>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <map>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

using namespace std;

#include <sys/time.h>
typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp() {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

template <typename T>
struct constantMultiplier {
    const T a;
    constantMultiplier(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T &x) const { return a * x; }
};

template <typename T>
struct roundNumber {
    __host__ __device__ T operator()(const T &x) const { return T(int(x * 10)); }
};

template <typename T>
class ThrustMatrix {
public:
    thrust::device_vector<T> vec;
    
    ThrustMatrix() {}
    ThrustMatrix(unsigned int nRows, unsigned int nColumns) { resizeVector(nRows, nColumns); }
    ~ThrustMatrix() {
        vec.clear();  // empty the vector
        vec.shrink_to_fit(); // deallocate any capacity which may currently be associated with vec
    }


    unsigned int numRows() { return numberOfRows; }
    unsigned int numColumns() { return numberOfColumns; }
    T get(unsigned int row, unsigned int col) {
        if (row < numberOfRows and col < numberOfColumns) return vec[col * numberOfRows + row];
        else { cerr << "out of bounds index, exiting\n"; return 0; }
    }
    void set(unsigned int row, unsigned int col, T val) {
        if (row < numberOfRows and col < numberOfColumns) vec[col * numberOfRows + row] = val;
        else {
            cerr << "out of bounds index, exiting\n"; return;
        }    
    }
    T * devicePointer() { return thrust::raw_pointer_cast(&vec[0]); }
    void resetSize(unsigned int nRows, unsigned int nColumns) {
        if (nRows * nColumns != vec.size()) { cerr << "dimensions of matrix don't match vector size, exiting\n"; return; }
        numberOfRows = nRows; numberOfColumns = nColumns;
    }
    void resizeVector(unsigned int nRows, unsigned int nColumns) {
        vec.resize(nRows * nColumns);
        resetSize(nRows, nColumns);
    }

    void randomize() {
        // Create a pseudo-random number generator
        curandGenerator_t prng; curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
        // Set the seed for the random number generator using the system clock
        curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
        // Fill the array with random numbers on the device
        curandGenerateUniform(prng, devicePointer(), numberOfRows * numberOfColumns);
    }

    void fillSequence() {
        thrust::sequence(vec.begin(), vec.end());
    }

    void intRandomize() {
        randomize();
        thrust::transform(vec.begin(), vec.end(), vec.begin(), roundNumber<T>());
    }

    void setToIdentity() {
        if (numberOfRows != numberOfColumns) {
            cerr << "not a square matrix, exiting\n"; return;
        }
        thrust::fill(vec.begin(), vec.end(), 0);
        for(unsigned int i=0; i < vec.size(); i += numberOfRows + 1) vec[i] = 1;
    }

    void multiplyByConstant(T c) {
        thrust::transform(vec.begin(), vec.end(), vec.begin(), constantMultiplier<T>(c));
    }

    void printBlasMajor() {
        for(unsigned int i=0; i < numberOfRows; i++) {
            for(unsigned int j=0; j < numberOfColumns; j++) cout << vec[j * numberOfRows + i] << " ";
            cout << endl;
        }
        cout << "\n\n";
    }

    void printRowMajor() {
        for (unsigned int i=0; i < vec.size(); i++) {
            if (i % numberOfRows == 0 and i != 0) cout << endl;
            cout << vec[i] << " ";
        }
        cout << "\n\n";        
    }

    void printVector() {
        for (unsigned int i=0; i < vec.size(); i++) cout << vec[i] << " ";
        cout << "\n\n";
    }

private:
        unsigned int numberOfRows, numberOfColumns;
};

template <typename T>
void printDeviceVector(thrust::device_vector<T> &v) {
    for (int i=0; i<v.size(); i++) cout << v[i] << " ";
    cout << "\n\n";
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // cublasDgemm(); for doubles genetic vec multiply
    // cublasDgeam(); for matrix transpose, alpha=1, beta=0
}


template <typename T>
void printDeviceArrayAsMatrix(const T *A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; i++) {
        for(int j = 0; j < nr_cols_A; j++) cout << A[j * nr_rows_A + i] << " ";
        cout << endl;
    }
    cout << endl;
}

template <typename T>
void printDeviceArrayAsMatrix(thrust::device_vector<T> &A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; i++) {
        for(int j = 0; j < nr_cols_A; j++) cout << A[j * nr_rows_A + i] << " ";
        cout << endl;
    }
    cout << endl;
}

template <typename T>
void printDeviceArrayAsMatrix_rowMajor(thrust::device_vector<T> &A, int nr_rows_A) {
    for (unsigned int i=0; i < A.size(); i++) {
        if (i % nr_rows_A == 0 and i != 0) cout << endl;
        cout << A[i] << " ";
    }
    cout << "\n\n";        
}

struct KnotVectorGenerationFunctor {
    unsigned int R, k;
    KnotVectorGenerationFunctor(unsigned int _r, unsigned int _k) : R(_r), k(_k) {}
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t) {
        unsigned int i = thrust::get<0>(t);
        if (i < k) thrust::get<1>(t) = 0;
        else if (i < R) thrust::get<1>(t) = i - k + 1;
        else thrust::get<1>(t) =  R - k + 1;
    }
};

template <typename T>
void generateKnotVector(thrust::device_vector<T> &knotVector, unsigned int R, unsigned int k) {
    if (k < 1 or k >= R) { cerr << "k is not in range [1, R-1], exiting\n"; return; }
    knotVector.resize(R+k);

    thrust::counting_iterator<unsigned int> countBegin(0);
    thrust::counting_iterator<unsigned int> countEnd = countBegin + knotVector.size();

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, knotVector.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(countEnd,   knotVector.end())),
                     KnotVectorGenerationFunctor(R, k));
}

template<typename T>
struct NormalizationFunctor {
    T xmin, constXmaxRK;
    NormalizationFunctor(T _xmin, T _constxmaxrk) : xmin(_xmin), constXmaxRK(_constxmaxrk) {}
    __host__ __device__ T operator()(const T &x) const { return (x - xmin) * constXmaxRK; }
};

template <typename T>
void normalizeDataset(thrust::device_vector<T> &XTable, unsigned int R, unsigned int k) {
    T max = *thrust::max_element(XTable.begin(), XTable.end());
    T min = *thrust::min_element(XTable.begin(), XTable.end());
    T constMaxRK = (R - k + 1) / (max - min);
    //cerr << max << "    " << min << "    " << R << "     "  << k << "     " << constMaxRK << endl;
    thrust::transform(XTable.begin(), XTable.end(), XTable.begin(), NormalizationFunctor<T>(min, constMaxRK));
}

template <typename T>
void removeLastXElementsOfEveryChunk(cublasHandle_t &handle, thrust::device_vector<T> &W, thrust::device_vector<T> &oldW, 
                                unsigned int oldChunkSize, unsigned int newChunkSize) {
    if (oldW.size() % oldChunkSize != 0) {
        cerr << "improper matrix dimensions, neither m nor n can be " << oldChunkSize 
            << " when vector size is " << oldW.size() << ", exiting\n";
        return;
    }
    unsigned int numberOfChunks = oldW.size() / oldChunkSize;
    W.resize(numberOfChunks * newChunkSize);
    if (typeid(T) == typeid(double)) {
        double alpha = 1.0, beta = 0;
        cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, newChunkSize, numberOfChunks, &alpha, (double *)thrust::raw_pointer_cast(&oldW[0]), oldChunkSize, &beta,
                    (double *)thrust::raw_pointer_cast(&oldW[0]), oldChunkSize, (double *)thrust::raw_pointer_cast(&W[0]), newChunkSize);
    } else {
        float alpha = 1.0, beta = 0;
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, newChunkSize, numberOfChunks, &alpha, (float *)thrust::raw_pointer_cast(&oldW[0]), oldChunkSize, &beta,
                    (float *)thrust::raw_pointer_cast(&oldW[0]), oldChunkSize, (float *)thrust::raw_pointer_cast(&W[0]), newChunkSize);
    }
}

template <typename T>
struct RootCaseBSplineFunctor {
    const unsigned int chunkSize;
    T *TTable, *ZTable;
    RootCaseBSplineFunctor(thrust::device_vector<T> &_t, thrust::device_vector<T> &_z, unsigned int _s) : 
                        TTable(thrust::raw_pointer_cast(&_t[0])), ZTable(thrust::raw_pointer_cast(&_z[0])), chunkSize(_s) {}
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t) {
        unsigned int elementIndex = thrust::get<0>(t);
        unsigned int i = elementIndex / chunkSize, j = elementIndex % chunkSize;
        T zi = ZTable[i], tj = TTable[j], tjp1 = TTable[j+1];
        thrust::get<1>(t) = (tj <= zi and zi <= tjp1) ? 1 : 0; // is it zi <= tjp1 or zi < tjp1 ???? WTF
    }
};

template <typename T>
struct BSplineFunctor {
    const unsigned int chunkSize, k;
    T *TTable, *ZTable;
    BSplineFunctor(thrust::device_vector<T> &_t, thrust::device_vector<T> &_z, unsigned int _s, unsigned int _k) : 
                TTable(thrust::raw_pointer_cast(&_t[0])), ZTable(thrust::raw_pointer_cast(&_z[0])), chunkSize(_s), k(_k) {}
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t) {
        unsigned int elementIndex = thrust::get<0>(t);
        unsigned int i = elementIndex / chunkSize, j = elementIndex % chunkSize;
        T zi = ZTable[i], tj = TTable[j], tjpk = TTable[j+k], tjpkm1 = TTable[j+k-1], tjp1 = TTable[j+1];

        T d1 = tjpkm1 - tj, d2 = tjpk - tjp1;
        T s1 = (d1 == 0) ? 0.00f : (zi - tj) * thrust::get<1>(t) / d1;
        T s2 = (d2 == 0) ? 0.00f : (tjpk - zi) * thrust::get<2>(t) / d2;
        thrust::get<3>(t) = s1 + s2;
        //thrust::get<3>(t) = ((zi - tj) * thrust::get<1>(t) / (tjpkm1 - tj)) + ((tjpk - zi) * thrust::get<2>(t) / (tjpk - tjp1));
        //thrust::get<3>(t) = thrust::get<1>(t) +  thrust::get<2>(t);
    }
};


template <typename T>
void generateWeightingMatrix(cublasHandle_t &handle, ThrustMatrix<T> &WM, 
                            thrust::device_vector<T> &TTable, thrust::device_vector<T> &ZTable, 
                            const unsigned int R, const unsigned int K) {

    if (K < 1 or K >= R) { cerr << "k is not in range [1, R-1], exiting\n"; return; }

    thrust::counting_iterator<unsigned int> countBegin(0);
        
    if (K == 1) {
        thrust::counting_iterator<unsigned int> countEnd = countBegin + (R * ZTable.size());
        WM.resizeVector(R, ZTable.size()); // set the dimension attributes correctly for use    
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, WM.vec.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(countEnd,   WM.vec.end())),
                         RootCaseBSplineFunctor<T>(TTable, ZTable, R));
    } else {
        unsigned int oldchunkSize = R+K-1;
        thrust::device_vector<T> U(ZTable.size() * oldchunkSize); // M is by definition equal to ZTable, or the list of data values, normalized
        thrust::device_vector<T> V(U.size());
        thrust::counting_iterator<unsigned int> countEnd = countBegin + U.size();

        // generate B for k=1 (root case)
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, U.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(countEnd,   U.end())),
                         RootCaseBSplineFunctor<T>(TTable, ZTable, oldchunkSize));
        //printDeviceArrayAsMatrix<T>(U, oldchunkSize, ZTable.size());
        //printDeviceVector<T>(U);

        for (unsigned int tempK = 2;;) { // tempK MUST START AT 2 !!!
            // odd rounds
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, U.begin(), U.begin()+1, V.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(countEnd,   U.end()-1,   U.end(),   V.end())),
                             BSplineFunctor<T>(TTable, ZTable, oldchunkSize, tempK));
            //printDeviceArrayAsMatrix<T>(V, oldchunkSize, ZTable.size());
            //printDeviceVector<T>(V);
            if (++tempK > K) {
                removeLastXElementsOfEveryChunk<T>(handle, WM.vec, V, oldchunkSize, R); // copy to official W (will resize W), removing the gg blocks
                break;
            }

            // even rounds (to save memory, and not have to allocate new vectors in each round)
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, V.begin(), V.begin()+1, U.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(countEnd,   V.end()-1,   V.end(),   U.end())),
                             BSplineFunctor<T>(TTable, ZTable, oldchunkSize, tempK));
            //printDeviceArrayAsMatrix<T>(U, oldchunkSize, ZTable.size());
            //printDeviceVector<T>(U);

            if (++tempK > K) {
                removeLastXElementsOfEveryChunk<T>(handle, WM.vec, U, oldchunkSize, R); // copy to official W (will resize W), removing the gg blocks
                break;
            }
        }
        WM.resetSize(R, ZTable.size()); // set the dimension attributes correctly for use
    }
}

template <typename T>
void calculateJointProbabilityMatrix(cublasHandle_t &handle, ThrustMatrix<T> &C, ThrustMatrix<T> &A, ThrustMatrix<T> &B) {
    if (A.numRows() != B.numRows() or A.numColumns() != B.numColumns()) { cerr << "dimensions of A and B are not the same, exiting\n"; return; }
    C.resizeVector(A.numRows(), B.numRows());
    if (typeid(T) == typeid(double)) {
        double alpha = 1.0/A.numColumns(), beta = 0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, A.numRows(), B.numRows(), A.numColumns(), &alpha, 
                    (double *)A.devicePointer(), A.numRows(), (double *)B.devicePointer(), B.numRows(), &beta, (double *)C.devicePointer(), A.numRows());
    } else {
        float alpha = 1.0/A.numColumns(), beta = 0;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, A.numRows(), B.numRows(), A.numColumns(), &alpha, 
                    (float *)A.devicePointer(), A.numRows(), (float *)B.devicePointer(), B.numRows(), &beta, (float *)C.devicePointer(), A.numRows());
    }
}


template <typename T>
void sumReduceEachRowInMatrix(cublasHandle_t &handle, thrust::device_vector<T> &y, ThrustMatrix<T> &A) {
    thrust::device_vector<T> x(A.numColumns()); thrust::fill(x.begin(), x.end(), 1);  // as long x.size() > M, it is fine
    y.resize(A.numRows());
    if (typeid(T) == typeid(double)) {
        double alpha = 1.0/A.numColumns(), beta = 0;
        cublasDgemv(handle, CUBLAS_OP_N, A.numRows(), A.numColumns(), &alpha, (double *)A.devicePointer(), A.numRows(), 
                   (double *)thrust::raw_pointer_cast(&x[0]), 1, &beta, (double *)thrust::raw_pointer_cast(&y[0]), 1);
    } else {    
        float alpha = 1.0/A.numColumns(), beta = 0;
        cublasSgemv(handle, CUBLAS_OP_N, A.numRows(), A.numColumns(), &alpha, (float *)A.devicePointer(), A.numRows(), 
                   (float *)thrust::raw_pointer_cast(&x[0]), 1, &beta, (float *)thrust::raw_pointer_cast(&y[0]), 1);
    }
}

template<typename T>
struct shannonSum : public thrust::unary_function<T,T> {
    __host__ __device__ T operator()(const T &x) const { return (x > 0.0f) ? (-x * log2f(x)) : 0.0f; }
};

template<typename T>
T calculateShannonSumFromProbabilityVector(thrust::device_vector<T> &y) {
    return thrust::transform_reduce(y.begin(), y.end(), shannonSum<T>(), 0.0f, thrust::plus<T>());
}

template <typename T>
void testBSplineMainCode(cublasHandle_t &handle) {
    cout << "check example B-spline code:\n";

    thrust::device_vector<T> U(30); thrust::sequence(U.begin(), U.end());
    thrust::device_vector<T> V(U.size());
    unsigned int chunkSize = 10;
    unsigned int numberOfChunks = U.size() / chunkSize;

    ThrustMatrix<T> TTable(chunkSize, 1); thrust::sequence(TTable.vec.begin(), TTable.vec.end());
    ThrustMatrix<T> ZTable(numberOfChunks, 1); thrust::sequence(ZTable.vec.begin(), ZTable.vec.end());

    unsigned int K = 5;

    printDeviceVector<T>(U);

    // BSPLINE CODE - BE CAREUL!!

    thrust::counting_iterator<T> countBegin(0);
    thrust::counting_iterator<T> countEnd = countBegin + U.size();
    for (unsigned int tempK = 2;;) { // tempK MUST START AT 2 !!!
        // odd round
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, U.begin(), U.begin()+1, V.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(countEnd,   U.end()-1,   U.end(),   V.end())),
                         BSplineFunctor<T>(TTable.vec, ZTable.vec, chunkSize, tempK));
        printDeviceVector<T>(V);
        if (++tempK > K) {
            // copy to official W, removing the gg blocks
            break;
        }

        // even round (to save memory, and not have to allocate new vectors in each round)
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, V.begin(), V.begin()+1, U.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(countEnd,   V.end()-1,   V.end(),   U.end())),
                         BSplineFunctor<T>(TTable.vec, ZTable.vec, chunkSize, tempK));
        printDeviceVector<T>(U);
        if (++tempK > K) {
            // copy to official W, removing the gg blocks
            break;
        }
    }
}

template <typename T>
void testRemoveLastElement(cublasHandle_t &handle) {
    cout << "check remove last x elements of every chunk of vector:\n";
    thrust::device_vector<T> W1(30); thrust::sequence(W1.begin(), W1.end());
    thrust::device_vector<T> W2;
    printDeviceVector<T>(W1);
    removeLastXElementsOfEveryChunk<T>(handle, W2, W1, 6, 4);
    printDeviceVector<T>(W2);
}

template <typename T>
void testGenerateKnotVector() {
    cout << "check generate knot vector:\n";
    thrust::device_vector<T> knV;
    generateKnotVector<T>(knV, 10, 4);
    printDeviceVector<T>(knV);
}

template <typename T>
void testNormalizeDataset() {
    cout << "check normalize-transform vector:\n";
    thrust::device_vector<T> xta(80); thrust::sequence(xta.begin(), xta.end(), -20.0);
    printDeviceVector<T>(xta);
    normalizeDataset<T>(xta, 10, 3);
    printDeviceVector<T>(xta);
}

template <typename T>
void testMatrixMultiplicationAndOthers(cublasHandle_t &handle) {
    unsigned int R = 10, M = 60; // good at 2000x2000, R << M
    ThrustMatrix<T> tmA(R, M); tmA.fillSequence();
    ThrustMatrix<T> tmB(R, M); tmB.intRandomize();
    ThrustMatrix<T> tmC;


    // matrix multiplication (part of joint-entropy calculation)
    tmC.printVector();
    calculateJointProbabilityMatrix<T>(handle, tmC, tmA, tmB);
    tmC.printVector();


    // summing down each column (part of self-entropy calculation)
    thrust::device_vector<T> y;
    sumReduceEachRowInMatrix<T>(handle, y, tmB);
    cout << "summing down each column of matrix B (rows in Blas Major) produces vector :" << endl;
    printDeviceVector<T>(y);

    
    // shannon-sum all elements in a vector or ThrustMatrix
    T HValue = calculateShannonSumFromProbabilityVector<T>(y);
    cout << HValue << "\n\n\n\n";
}


template <typename T>
void testIntegration(cublasHandle_t &handle) {
    unsigned int R = 3, K = 2;

    // some dummy dataset in a device vector
    thrust::device_vector<T> ZTable(20); thrust::sequence(ZTable.begin(), ZTable.end());
    cout << "Original Dataset: \n"; printDeviceVector(ZTable);
    normalizeDataset<T>(ZTable, R, K);
    cout << "Normalized Dataset: \n"; printDeviceVector(ZTable);
    // the knot vector
    thrust::device_vector<T> TTable(R);
    generateKnotVector<T>(TTable, R, K);
    cout << "Knot Vector: \n"; printDeviceVector(TTable);

    // dump everything in, to compute the weight matrix
    ThrustMatrix<T> WM;
    generateWeightingMatrix<T>(handle, WM, TTable, ZTable, R, K);
    WM.printBlasMajor();

    // repeat for second the second random variable Y, to get WM2

    // joint entropy of X and Y
    // ThrustMatrix<T> P; // probability matrix
    // calculateJointProbabilityMatrix<T>(handle, P, WM, WM2);
    // T Hxy = calculateShannonSumFromProbabilityVector<T>(P.vec);

    // self entropy of random variable X
    // thrust::device_vector<T> x;
    // sumReduceEachRowInMatrix<T>(handle, x, WM);
    // T Hx = calculateShannonSumFromProbabilityVector<T>(x);


    // self entropy of random variable Y
    // thrust::device_vector<T> x;
    // sumReduceEachRowInMatrix<T>(handle, y, WM2);
    // T Hy = calculateShannonSumFromProbabilityVector<T>(y);

    // get MIValue
    // T MIValue = Hx + Hy - Hxy;
}









template <typename T>
void readInputFile(map<string, thrust::device_vector<T> > &listOfDeviceVectors, const string &filename) {
    ifstream inputFilestream(filename.c_str());
    if (not inputFilestream.is_open()) {
        cerr << "Could not open input configuration file " << filename << endl; exit(1);
    }
    string fileLine, lineKey;
    while (getline(inputFilestream, fileLine)) {
        istringstream iss(fileLine); T val;
        iss >> lineKey; 
        while (iss >> val)
            listOfDeviceVectors[lineKey].push_back(val);
    }
}







int main() {
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

/*
    testBSplineMainCode<float>(handle);
    testRemoveLastElement<float>(handle);
    testGenerateKnotVector<float>();
    testNormalizeDataset<float>();
    testMatrixMultiplicationAndOthers<float>(handle);

    //testIntegration<float>(handle);
*/

    map<string, thrust::device_vector<float> > listOfDeviceVectors;
    readInputFile<float>(listOfDeviceVectors, "eisen_2000.txt");


    // YPR193C  YPR190C

    timestamp_t t0 = get_timestamp();


    int R=10, K=3;
    cout << "R " << R << "; K " << K << endl;

    cout << "YPR193C:\n"; printDeviceVector<float>(listOfDeviceVectors["YPR193C"]);
    normalizeDataset<float>(listOfDeviceVectors["YPR193C"], R, K);
    cout << "YPR193C normalized:\n"; printDeviceVector<float>(listOfDeviceVectors["YPR193C"]);

    cout << "YPR190C:\n"; printDeviceVector<float>(listOfDeviceVectors["YPR190C"]);
    normalizeDataset<float>(listOfDeviceVectors["YPR190C"], R, K);
    cout << "YPR190C normalized:\n"; printDeviceVector<float>(listOfDeviceVectors["YPR190C"]);


    thrust::device_vector<float> TTable(R);
    generateKnotVector<float>(TTable, R, K);
    cout << "Knot Vector:\n"; printDeviceVector<float>(TTable);


    ThrustMatrix<float> WMX, WMY;
    generateWeightingMatrix<float>(handle, WMX, TTable, listOfDeviceVectors["YPR193C"], R, K);
    generateWeightingMatrix<float>(handle, WMY, TTable, listOfDeviceVectors["YPR190C"], R, K);
    cout << "Weight Matrix for YPR193C:\n"; WMX.printBlasMajor();
    cout << "Weight Matrix for YPR190C:\n"; WMY.printBlasMajor();


    ThrustMatrix<float> P; // probability matrix
    calculateJointProbabilityMatrix<float>(handle, P, WMX, WMY);
    cout << "Joint Probability Matrix (WMX x WMY):\n"; P.printBlasMajor();


    thrust::device_vector<float> X, Y;
    sumReduceEachRowInMatrix<float>(handle, X, WMX);
    sumReduceEachRowInMatrix<float>(handle, Y, WMY);
    cout << "Probability Vector for YPR193C\n"; printDeviceVector<float>(X);
    cout << "Probability Vector for YPR190C\n"; printDeviceVector<float>(Y);


    float Hx = calculateShannonSumFromProbabilityVector<float>(X);
    float Hy = calculateShannonSumFromProbabilityVector<float>(Y);
    float Hxy = calculateShannonSumFromProbabilityVector<float>(P.vec);


    float MIValue = Hx + Hy - Hxy;
    cout << "Hx " << Hx << "; Hy " << Hy << "; Hxy " << Hxy << endl;
    cout << "MIValue for YPR193C and YPR190C: " << MIValue << "\n\n";


    timestamp_t t1 = get_timestamp();
    double secs = (t1 - t0) / 1000000.0L;
    cout << "Compute time: " << secs << " seconds\n";


    // Destroy the handle
    cublasDestroy(handle);
    return 0;
}