//
// auto-generated by op2.py
//

//user function
__device__ void ApplyUpperBoundToVelocity_gpu( double *vnewc) {
    if (vnewc[0] > (m_eosvmax_cuda))
        vnewc[0] = (m_eosvmax_cuda) ;

}

// CUDA kernel function
__global__ void op_cuda_ApplyUpperBoundToVelocity(
  double *arg0,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    ApplyUpperBoundToVelocity_gpu(arg0+n*1);
  }
}


//host stub function
void op_par_loop_ApplyUpperBoundToVelocity(char const *name, op_set set,
  op_arg arg0){

  int nargs = 1;
  op_arg args[1];

  args[0] = arg0;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(35);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[35].name      = name;
  OP_kernels[35].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  ApplyUpperBoundToVelocity");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_35
      int nthread = OP_BLOCK_SIZE_35;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_ApplyUpperBoundToVelocity<<<nblocks,nthread>>>(
      (double *) arg0.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[35].time     += wall_t2 - wall_t1;
  OP_kernels[35].transfer += (float)set->size * arg0.size * 2.0f;
}
