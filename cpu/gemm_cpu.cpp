#include <chrono>
#include "../include/utils.h"

#define NUM_RUNS 2

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  initialize(refC, Ref::M * Ref::N);				\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								
  
#define TIME(name) \
  for (int i = 0; i < 1; i++)						\
    {									\
      name(A, B, C, M, N, K);						\
    }									\
  std::chrono::duration<double, std::milli> time_ ## name;		\
  for (int i = 0; i < NUM_RUNS; i++)					\
    {									\
      initialize(C, M * N);						\
      auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K);						\
      auto end_time_ ## name = std::chrono::high_resolution_clock::now(); \
      time_ ## name += end_time_ ## name - start_time_ ## name;		\
    }									\
std::chrono::duration<double, std::milli> duration_ ## name = time_ ## name/float(NUM_RUNS); \
  std::cout << "Time taken for GEMM (CPU," << #name <<"): " << duration_ ## name.count() << "ms" << std::endl; 


// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
	C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler flags. So, you can use o3's code for that part
void gemm_cpu_o1(float* A, float* B, float *C, int M, int N, int K) {
	for (int i = 0; i < M; i++) {
		for (int k = 0; k < K; k++) {
			for (int j = 0; j < N; j++) {
				C[i * N + j]  += A[i * K + k]  * B[k * N + j];
		  	}
		}
	}

}

#define tileSize 32

void gemm_cpu_o2(float* A, float* B, float *C, int M, int N, int K) {
	//calculate the suitable tilesize
	//thinking L1 Cache as 32KB,
	// 4B*(64*64+2*(64*tS)) = 32kB
	//assume per cache line is 64b
	//
	int rowTile; 
	int columnTile;
	int innerTile;

	for (rowTile = 0; rowTile < M; rowTile += 64) {
		for (columnTile = 0; columnTile < N; columnTile += 64) {
			for (innerTile = 0; innerTile < K; innerTile += tileSize) {
				//do the work tile by tile
				//here, for simplification, row == i, column == j, inner index == k


				int rowTileEnd = 0;
				if (M<=rowTile + 64) rowTileEnd = M;
				else rowTileEnd = rowTile + 64;

				for (int row = rowTile; row < rowTileEnd; row++) {

					//for inner loop, we need to indicate the end point... cause not aligned prob
			  		int innerTileEnd = 0;
					if (K<=innerTile + tileSize) innerTileEnd = K;
					else innerTileEnd = innerTile + tileSize;

			  		for (int inner = innerTile; inner < innerTileEnd; inner++) {

						int coumnTileEnd = 0;
						if (N<=columnTile + 64) coumnTileEnd = N;
						else coumnTileEnd = coumnTileEnd + 64;

						for (int col = columnTile; col < coumnTileEnd; col++) {
				  			C[row * N + col] +=
						  	A[row * K + inner] * B[inner * N + col];
						} 
					} 
				} 
			} 
		} 
	}
}

void gemm_cpu_o3(float* A, float* B, float *C, int M, int N, int K) {
//same code as o3. Ref: https://siboehm.com/articles/22/Fast-MMM-on-CPU


	//default(none) forces the explict declare of shared var(So, the ohter will be private automatically)
	#pragma omp parallel for shared(A, B, C, M, N, K) default(none) collapse(2) num_threads(8)
	for ( int rowTile = 0; rowTile < M; rowTile += 64) {
		for ( int columnTile = 0; columnTile < N; columnTile += 64) {
			for ( int innerTile = 0; innerTile < K; innerTile += tileSize) {
				//do the work tile by tile
				//here, for simplification, row == i, column == j, inner index == k
				int rowTileEnd = 0;
				if (M<=rowTile + 64) rowTileEnd = M;
				else rowTileEnd = rowTile + 64;

				for (int row = rowTile; row < rowTileEnd; row++) {

					//for inner loop, we need to indicate the end point... cause not aligned prob
			  		int innerTileEnd = 0;
					if (K<=innerTile + tileSize) innerTileEnd = K;
					else innerTileEnd = innerTile + tileSize;

			  		for (int inner = innerTile; inner < innerTileEnd; inner++) {

						int coumnTileEnd = 0;
						if (N<=columnTile + 64) coumnTileEnd = N;
						else coumnTileEnd = coumnTileEnd + 64;

						for (int col = columnTile; col < coumnTileEnd; col++) {
				  			C[row * N + col] +=
						  	A[row * K + inner] * B[inner * N + col];
						} 
					} 
				} 
			} 
		} 
	}
}


int main(int argc, char* argv[]) {
	if (argc < 3) {
	  std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
	  return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	// Check if the kernel results are correct
	// note that even if the correctness check fails all optimized kernels will run.
	// We are not exiting the program at failure at this point.
	// It is a good idea to add more correctness checks to your code.
	// We may (at discretion) verify that your code is correct.
	float* refC = new float[Ref::M * Ref::N]();
	auto ref = Ref();
	CHECK(gemm_cpu_o0)
	CHECK(gemm_cpu_o1)
	CHECK(gemm_cpu_o2)
	CHECK(gemm_cpu_o3)
	delete[] refC;
	
	TIME(gemm_cpu_o0)
	TIME(gemm_cpu_o1)
	TIME(gemm_cpu_o2)
	TIME(gemm_cpu_o3)

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
