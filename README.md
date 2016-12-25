# parallel_mpi_gameoflife
This code was written as a part of Parallel computing YSDA course. Using the MPI the parallel version of Game Of Life was created. The square field is divided into N^2 smaller squares at every iteration and any process is responsible for its square. At the end of recomputation, processors share information about their boundaries with their neighbours.
