//
// auto-generated by op2.py
//

//user function
__device__ void CalcPosForNodes_gpu( 
                            double *x, double *y, double *z,
                            const double *xd, const double *yd, const double *zd,
                            const double *dt
) {
    x[0] += xd[0] * (*dt) ;
    y[0] += yd[0] * (*dt) ;
    z[0] += zd[0] * (*dt) ;

}

// CUDA kernel function
__global__ void op_cuda_CalcPosForNodes(
  double *arg0,
  double *arg1,
  double *arg2,
  const double *__restrict arg3,
  const double *__restrict arg4,
  const double *__restrict arg5,
  const double *arg6,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    CalcPosForNodes_gpu(arg0+n*1,
                    arg1+n*1,
                    arg2+n*1,
                    arg3+n*1,
                    arg4+n*1,
                    arg5+n*1,
                    arg6);
  }
}


//host stub function
void op_par_loop_CalcPosForNodes(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6){

  double*arg6h = (double *)arg6.data;
  int nargs = 7;
  op_arg args[7];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  args[6] = arg6;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(11);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[11].name      = name;
  OP_kernels[11].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  CalcPosForNodes");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //transfer constants to GPU
    int consts_bytes = 0;
    consts_bytes += ROUND_UP(1*sizeof(double));
    reallocConstArrays(consts_bytes);
    consts_bytes = 0;
    arg6.data   = OP_consts_h + consts_bytes;
    arg6.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<1; d++ ){
      ((double *)arg6.data)[d] = arg6h[d];
    }
    consts_bytes += ROUND_UP(1*sizeof(double));
    mvConstArraysToDevice(consts_bytes);

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_11
      int nthread = OP_BLOCK_SIZE_11;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_CalcPosForNodes<<<nblocks,nthread>>>(
      (double *) arg0.data_d,
      (double *) arg1.data_d,
      (double *) arg2.data_d,
      (double *) arg3.data_d,
      (double *) arg4.data_d,
      (double *) arg5.data_d,
      (double *) arg6.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[11].time     += wall_t2 - wall_t1;
  OP_kernels[11].transfer += (float)set->size * arg0.size * 2.0f;
  OP_kernels[11].transfer += (float)set->size * arg1.size * 2.0f;
  OP_kernels[11].transfer += (float)set->size * arg2.size * 2.0f;
  OP_kernels[11].transfer += (float)set->size * arg3.size;
  OP_kernels[11].transfer += (float)set->size * arg4.size;
  OP_kernels[11].transfer += (float)set->size * arg5.size;
}