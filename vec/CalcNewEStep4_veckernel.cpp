//
// auto-generated by op2.py
//

//user function
inline void CalcNewEStep4(const double *delvc, const double *pbvc, double *e_new, const double *vnewc, const double *bvc, const double *p_new,
                            const double *ql_old, const double *qq_old, const double *p_old, const double *q_old, const double *q_new,
                            const double *pHalfStep){
    // int ielem = regElemList[0];
    double q_tilde ;

    if (delvc[0] > double(0.)) {
        q_tilde = double(0.) ;
    }
    else {
        double ssc = ( pbvc[0] * e_new[0]
                + vnewc[0] * vnewc[0] * bvc[0] * p_new[0] ) / m_refdens ;

        if ( ssc <= m_ssc_thresh ) {
        ssc = m_ssc_low ;
        } else {
        ssc = sqrt(ssc) ;
        }

        q_tilde = (ssc*ql_old[0] + qq_old[0]) ;
    }

    e_new[0] = e_new[0] - (  double(7.0)*(p_old[0]     + q_old[0])
                            - double(8.0)*(pHalfStep[0] + q_new[0])
                            + (p_new[0] + q_tilde)) * delvc[0]*m_sixth ;

    if (fabs(e_new[0]) < m_e_cut) {
        e_new[0] = double(0.)  ;
    }
    if (     e_new[0]  < m_emin ) {
        e_new[0] = m_emin ;
    }   
}

// host stub function
void op_par_loop_CalcNewEStep4(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7,
  op_arg arg8,
  op_arg arg9,
  op_arg arg10,
  op_arg arg11){

  int nargs = 12;
  op_arg args[12];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  args[6] = arg6;
  args[7] = arg7;
  args[8] = arg8;
  args[9] = arg9;
  args[10] = arg10;
  args[11] = arg11;
  //create aligned pointers for dats
  ALIGNED_double const double * __restrict__ ptr0 = (double *) arg0.data;
  DECLARE_PTR_ALIGNED(ptr0,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr1 = (double *) arg1.data;
  DECLARE_PTR_ALIGNED(ptr1,double_ALIGN);
  ALIGNED_double       double * __restrict__ ptr2 = (double *) arg2.data;
  DECLARE_PTR_ALIGNED(ptr2,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr3 = (double *) arg3.data;
  DECLARE_PTR_ALIGNED(ptr3,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr4 = (double *) arg4.data;
  DECLARE_PTR_ALIGNED(ptr4,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr5 = (double *) arg5.data;
  DECLARE_PTR_ALIGNED(ptr5,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr6 = (double *) arg6.data;
  DECLARE_PTR_ALIGNED(ptr6,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr7 = (double *) arg7.data;
  DECLARE_PTR_ALIGNED(ptr7,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr8 = (double *) arg8.data;
  DECLARE_PTR_ALIGNED(ptr8,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr9 = (double *) arg9.data;
  DECLARE_PTR_ALIGNED(ptr9,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr10 = (double *) arg10.data;
  DECLARE_PTR_ALIGNED(ptr10,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr11 = (double *) arg11.data;
  DECLARE_PTR_ALIGNED(ptr11,double_ALIGN);

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(24);
  op_timers_core(&cpu_t1, &wall_t1);


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  CalcNewEStep4");
  }

  int exec_size = op_mpi_halo_exchanges(set, nargs, args);

  if (exec_size >0) {

    #ifdef VECTORIZE
    #pragma novector
    for ( int n=0; n<(exec_size/SIMD_VEC)*SIMD_VEC; n+=SIMD_VEC ){
      #pragma omp simd simdlen(SIMD_VEC)
      for ( int i=0; i<SIMD_VEC; i++ ){
        CalcNewEStep4(
          &(ptr0)[1 * (n+i)],
          &(ptr1)[1 * (n+i)],
          &(ptr2)[1 * (n+i)],
          &(ptr3)[1 * (n+i)],
          &(ptr4)[1 * (n+i)],
          &(ptr5)[1 * (n+i)],
          &(ptr6)[1 * (n+i)],
          &(ptr7)[1 * (n+i)],
          &(ptr8)[1 * (n+i)],
          &(ptr9)[1 * (n+i)],
          &(ptr10)[1 * (n+i)],
          &(ptr11)[1 * (n+i)]);
      }
    }
    //remainder
    for ( int n=(exec_size/SIMD_VEC)*SIMD_VEC; n<exec_size; n++ ){
    #else
    for ( int n=0; n<exec_size; n++ ){
    #endif
      CalcNewEStep4(
        &(ptr0)[1*n],
        &(ptr1)[1*n],
        &(ptr2)[1*n],
        &(ptr3)[1*n],
        &(ptr4)[1*n],
        &(ptr5)[1*n],
        &(ptr6)[1*n],
        &(ptr7)[1*n],
        &(ptr8)[1*n],
        &(ptr9)[1*n],
        &(ptr10)[1*n],
        &(ptr11)[1*n]);
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[24].name      = name;
  OP_kernels[24].count    += 1;
  OP_kernels[24].time     += wall_t2 - wall_t1;
  OP_kernels[24].transfer += (float)set->size * arg0.size;
  OP_kernels[24].transfer += (float)set->size * arg1.size;
  OP_kernels[24].transfer += (float)set->size * arg2.size * 2.0f;
  OP_kernels[24].transfer += (float)set->size * arg3.size;
  OP_kernels[24].transfer += (float)set->size * arg4.size;
  OP_kernels[24].transfer += (float)set->size * arg5.size;
  OP_kernels[24].transfer += (float)set->size * arg6.size;
  OP_kernels[24].transfer += (float)set->size * arg7.size;
  OP_kernels[24].transfer += (float)set->size * arg8.size;
  OP_kernels[24].transfer += (float)set->size * arg9.size;
  OP_kernels[24].transfer += (float)set->size * arg10.size;
  OP_kernels[24].transfer += (float)set->size * arg11.size;
}