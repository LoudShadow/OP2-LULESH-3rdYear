//
// auto-generated by op2.py
//

//user function
inline void CalcPHalfstep(double *p_new, const double *bvc,  const double *e_old, const double *vnewc){
    p_new[0] = bvc[0] * e_old[0] ;

    if    (fabs(p_new[0]) <  m_p_cut   )
        p_new[0] = double(0.0) ;

    if    ( vnewc[0] >= m_eosvmax ) /* impossible condition here? */
        p_new[0] = double(0.0) ;

    if    (p_new[0]       <  m_pmin)
        p_new[0]   = m_pmin ;
}

// host stub function
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
  //create aligned pointers for dats
  ALIGNED_double       double * __restrict__ ptr0 = (double *) arg0.data;
  DECLARE_PTR_ALIGNED(ptr0,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr1 = (double *) arg1.data;
  DECLARE_PTR_ALIGNED(ptr1,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr2 = (double *) arg2.data;
  DECLARE_PTR_ALIGNED(ptr2,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr3 = (double *) arg3.data;
  DECLARE_PTR_ALIGNED(ptr3,double_ALIGN);

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(18);
  op_timers_core(&cpu_t1, &wall_t1);


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  CalcPHalfstep");
  }

  int exec_size = op_mpi_halo_exchanges(set, nargs, args);

  if (exec_size >0) {

    #ifdef VECTORIZE
    #pragma novector
    for ( int n=0; n<(exec_size/SIMD_VEC)*SIMD_VEC; n+=SIMD_VEC ){
      #pragma omp simd simdlen(SIMD_VEC)
      for ( int i=0; i<SIMD_VEC; i++ ){
        CalcPHalfstep(
          &(ptr0)[1 * (n+i)],
          &(ptr1)[1 * (n+i)],
          &(ptr2)[1 * (n+i)],
          &(ptr3)[1 * (n+i)]);
      }
    }
    //remainder
    for ( int n=(exec_size/SIMD_VEC)*SIMD_VEC; n<exec_size; n++ ){
    #else
    for ( int n=0; n<exec_size; n++ ){
    #endif
      CalcPHalfstep(
        &(ptr0)[1*n],
        &(ptr1)[1*n],
        &(ptr2)[1*n],
        &(ptr3)[1*n]);
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[18].name      = name;
  OP_kernels[18].count    += 1;
  OP_kernels[18].time     += wall_t2 - wall_t1;
  OP_kernels[18].transfer += (float)set->size * arg0.size * 2.0f;
  OP_kernels[18].transfer += (float)set->size * arg1.size;
  OP_kernels[18].transfer += (float)set->size * arg2.size;
  OP_kernels[18].transfer += (float)set->size * arg3.size;
}