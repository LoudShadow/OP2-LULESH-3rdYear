//
// auto-generated by op2.py
//

//user function
__device__ void CalcPHalfstep_gpu( double *p_new, const double *bvc,  const double *e_old, const double *vnewc) {
    p_new[0] = bvc[0] * e_old[0] ;

    if    (fabs(p_new[0]) <  m_p_cut_cuda   )
        p_new[0] = double(0.0) ;

    if    ( vnewc[0] >= m_eosvmax_cuda )
        p_new[0] = double(0.0) ;

    if    (p_new[0]       <  m_pmin_cuda)
        p_new[0]   = m_pmin_cuda ;

}

// CUDA kernel function
__global__ void op_cuda_CalcPHalfstep(
  double *arg0,
  const double *__restrict arg1,
  const double *__restrict arg2,
  const double *__restrict arg3,
  int   set_size ) {


  //process set elements
  for ( int n=threadIdx.x+blockIdx.x*blockDim.x; n<set_size; n+=blockDim.x*gridDim.x ){

    //user-supplied kernel call
    CalcPHalfstep_gpu(arg0+n*1,
                  arg1+n*1,
                  arg2+n*1,
                  arg3+n*1);
  }
}


//host stub function
void op_par_loop_CalcPHalfstep(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3){

  int nargs = 4;
  op_arg args[4];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(18);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[18].name      = name;
  OP_kernels[18].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  CalcPHalfstep");
  }

  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_18
      int nthread = OP_BLOCK_SIZE_18;
    #else
      int nthread = OP_block_size;
    #endif

    int nblocks = 200;

    op_cuda_CalcPHalfstep<<<nblocks,nthread>>>(
      (double *) arg0.data_d,
      (double *) arg1.data_d,
      (double *) arg2.data_d,
      (double *) arg3.data_d,
      set->size );
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[18].time     += wall_t2 - wall_t1;
  OP_kernels[18].transfer += (float)set->size * arg0.size * 2.0f;
  OP_kernels[18].transfer += (float)set->size * arg1.size;
  OP_kernels[18].transfer += (float)set->size * arg2.size;
  OP_kernels[18].transfer += (float)set->size * arg3.size;
}
