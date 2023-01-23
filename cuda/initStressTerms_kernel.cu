//
// auto-generated by op2.py
//

//user function
__device__ void initStressTerms_gpu( double *sigxx,double *sigyy,double *sigzz, const double *p, const double *q) {
    sigxx[0] = -p[0] - q[0];
    sigyy[0] = -p[0] - q[0];
    sigzz[0] = -p[0] - q[0];

}

// CUDA kernel function
__global__ void op_cuda_initStressTerms(
  double *arg0,
  double *arg1,
  double *arg2,
  const double *__restrict arg3,
  const double *__restrict arg4,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    initStressTerms_gpu(arg0+n*1,
                    arg1+n*1,
                    arg2+n*1,
                    arg3+n*1,
                    arg4+n*1);
  }
}


//host stub function
void op_par_loop_initStressTerms(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4){

  int nargs = 5;
  op_arg args[5];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[0].name      = name;
  OP_kernels[0].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  initStressTerms");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_0
      int nthread = OP_BLOCK_SIZE_0;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_initStressTerms<<<nblocks,nthread>>>(
      (double *) arg0.data_d,
      (double *) arg1.data_d,
      (double *) arg2.data_d,
      (double *) arg3.data_d,
      (double *) arg4.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].time     += wall_t2 - wall_t1;
  OP_kernels[0].transfer += (float)set->size * arg0.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg1.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg2.size * 2.0f;
  OP_kernels[0].transfer += (float)set->size * arg3.size;
  OP_kernels[0].transfer += (float)set->size * arg4.size;
}
