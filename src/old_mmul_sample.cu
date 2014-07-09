/*
    thrust::device_vector<float> td_A(nr_rows_A * nr_cols_A), td_B(nr_rows_B * nr_cols_B), td_C(nr_rows_C * nr_cols_C);

    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float)); cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float)); cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
    // Fill the arrays A and B on GPU with random numbers
    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
    GPU_fill_rand(thrust::raw_pointer_cast(&td_A[0]), nr_rows_A, nr_cols_A);
    GPU_fill_rand(thrust::raw_pointer_cast(&td_B[0]), nr_rows_B, nr_cols_B);
*/


/*    
    // Optionally we can print the data
    std::cout << "A =" << std::endl;
    print_matrix(td_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(td_B, nr_rows_B, nr_cols_B);
    // Optionally we can copy the data back on CPU and print the arrays
    cudaMemcpy(h_A,thrust::raw_pointer_cast(&td_A[0]),nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B,thrust::raw_pointer_cast(&td_B[0]),nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "A =" << std::endl;
    print_matrix(h_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, nr_rows_B, nr_cols_B);
*/   


/*
    //Print the result
    std::cout << "C =" << std::endl;
    print_matrix(td_C, nr_rows_C, nr_cols_C);
    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,thrust::raw_pointer_cast(&td_C[0]),nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);
    //Free GPU memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);  
    // Free CPU memory
    free(h_A); free(h_B); free(h_C);  
*/  

    // Multiply A and B on GPU
    //gpu_blas_mmul(handle, tmA.devicePointer(), tmB.devicePointer(), tmC.devicePointer(), tmA.numRows(), tmA.numColumns(), tmB.numColumns());
    //gpu_blas_mmul(handle, thrust::raw_pointer_cast(&td_A[0]), thrust::raw_pointer_cast(&td_A[0]), thrust::raw_pointer_cast(&td_C[0]), nr_rows_A, nr_cols_A, nr_cols_B);


/*    float constalpha = 1;
    float constbeta = 0;
    unsigned int newChunk = 4, oldChunk = 6, size = 5;
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, newChunk, size, &constalpha,
                thrust::raw_pointer_cast(&W1[0]), oldChunk, &constbeta,
                thrust::raw_pointer_cast(&W1[0]), oldChunk,
                thrust::raw_pointer_cast(&W2[0]), newChunk);
*/


/* // summing up columns
    thrust::device_vector<float> x(M);
    thrust::fill(x.begin(), x.end(), 1);

    cublasSgemv(handle, CUBLAS_OP_N, tmB.numRows(), tmB.numColumns(), &alpha, tmB.devicePointer(), tmB.numRows(), 
                           thrust::raw_pointer_cast(&x[0]), 1, &beta,
                           thrust::raw_pointer_cast(&y[0]), 1);

*/

    //tmC.multiplyByConstant(10.0);
    //tmC.printBlasMajor(); // correct p
    //tmC.printRowMajor();


/*
//C = alpha*op(A)*op(B) + beta*C
void matrixMatrixMultiply( cublasHandle_t &handle, float alpha, cublasOperation_t operationOnA, ThrustMatrix &A, 
                    cublasOperation_t operationOnB, ThrustMatrix &B, float beta, ThrustMatrix &C ) {
    if (A.numColumns() != B.numRows()) {
        cout << "k does not match for matrix A and B, exiting\n"; return;
    }
    if (beta !=0 and !(C.numRows() == A.numRows() and C.numColumns() == B.numColumns())) {
        cout << "size mismatch in C, exiting\n"; return;
    }

    //    cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
 //       float *alpha, float *A, int lda, float *B, int ldb, float *beta, float *C, int ldc)

    unsigned int m = A.numRows(), n = B.numColumns(), k = A.numColumns();
    unsigned int &lda = m, &ldb = k, &ldc = m;

    if (operationOnA == CUBLAS_OP_T) {
        m = A.numColumns(); k = A.numRows();
    }
    if (operationOnB == CUBLAS_OP_T) {
        m = A.numRows(); n = B.numRows();
    }
    
    //if (beta == 0)
    C.resize(m, n);
    cublasSgemm(handle, operationOnA, operationOnB, m, n, k, &alpha, A.devicePointer(), lda, B.devicePointer(), ldb, &beta, C.devicePointer(), ldc);
}
*/

/*
void matrixVectorMultiply( cublasHandle_t &handle, float alpha, cublasOperation_t operationOnA, ThrustMatrix &A, 
                        float beta, thrust::device_vector<float> &x ) {

    cublasSgemv(handle, operationOnA, A.numRows(), A.numColumns(), &alpha, A.devicePointer(), A.numRows(), thrust::raw_pointer_cast(x.data()), 1, &beta,
                           float           *y, int incy)


}

cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)

    y = alpha*op(A)*x + beta*y;
*/

/*
    //cout << "A in row major =" << endl;
    //tmA.printRowMajor();
    //cout << "A in col major (what blas sees) =" << endl;
    //tmA.printBlasMajor();
    //cout << "B in col major =" << endl;
    //tmB.printBlasMajor();


    float alpha = 1.0/M, beta = 0;
    //int m = ; n; k; 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, tmA.numRows(), tmB.numRows(), tmA.numColumns(), &alpha, tmA.devicePointer(), tmA.numRows(), tmB.devicePointer(), tmB.numRows(), &beta, tmC.devicePointer(), tmA.numRows());
*/

/*    unsigned int oldchunkSize = R+K-1;
    thrust::device_vector<float> U(ZTable.size() * oldchunkSize); // M is by definition equal to ZTable, or the list of data values, normalized
    thrust::counting_iterator<unsigned int> countBegin(0); thrust::counting_iterator<unsigned int> countEnd = countBegin + U.size();

    // generate B for k=1 (root case)
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, U.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(countEnd,   U.end())),
                     RootCaseBSplineFunctor<float>(TTable, ZTable, oldchunkSize));
    //printDeviceVector(U);
    print_matrix_rowMajor<float>(TTable, R+K);
    print_matrix_rowMajor<float>(U, oldchunkSize);
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(countBegin, U.begin(), U.begin()+1, V.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(countEnd,   U.end()-1,   U.end(),   V.end())),
                         BSplineFunctor<T>(TTable, ZTable, oldchunkSize, tempK));
*/

/*
    for (map<string, thrust::device_vector<float> >::iterator it = listOfDeviceVectors.begin(); it != listOfDeviceVectors.end(); ++it) {
        cout << it->first << ": " << it->second.size() << endl;
    }
    cout << "values for YPR199C:\t\t"; printDeviceVector<float>(listOfDeviceVectors["YPR199C"]);
*/
