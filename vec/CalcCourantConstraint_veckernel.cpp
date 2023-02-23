//
// auto-generated by op2.py
//

//user function
inline void CalcCourantConstraint(const double *ss, const double *vdov, const double *arealg, double *dtcourant){
    
    double dtf = ss[0] * ss[0];

    if ( vdov[0] < double(0.) ) {

        dtf = dtf
            + m_qqc2 * arealg[0] * arealg[0]
            * vdov[0] * vdov[0] ;
    }

    dtf = sqrt(dtf) ;

    dtf = arealg[0] / dtf ;

/* determine minimum timestep with its corresponding elem */
    if (vdov[0] != double(0.)) {
        if ( dtf < (*dtcourant) ) {
        (*dtcourant) = dtf ;
        // courant_elem = indx ;
        }
    }
}
#ifdef VECTORIZE
//user function -- modified for vectorisation
#if defined __clang__ || defined __GNUC__
__attribute__((always_inline))
#endif
inline void CalcCourantConstraint_vec( const double ss[][SIMD_VEC], const double vdov[][SIMD_VEC], const double arealg[][SIMD_VEC], double *dtcourant, int idx ) {

    double dtf = ss[0][idx] * ss[0][idx];

    if ( vdov[0][idx] < double(0.) ) {

        dtf = dtf
            + m_qqc2 * arealg[0][idx] * arealg[0][idx]
            * vdov[0][idx] * vdov[0][idx] ;
    }

    dtf = sqrt(dtf) ;

    dtf = arealg[0][idx] / dtf ;

    if (vdov[0][idx] != double(0.)) {
        if ( dtf < (*dtcourant) ) {
        (*dtcourant) = dtf ;

        }
    }

}
#endif

// host stub function
void op_par_loop_CalcCourantConstraint(char const *name, op_set set,
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
  ALIGNED_double const double * __restrict__ ptr0 = (double *) arg0.data;
  DECLARE_PTR_ALIGNED(ptr0,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr1 = (double *) arg1.data;
  DECLARE_PTR_ALIGNED(ptr1,double_ALIGN);
  ALIGNED_double const double * __restrict__ ptr2 = (double *) arg2.data;
  DECLARE_PTR_ALIGNED(ptr2,double_ALIGN);

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(38);
  op_timers_core(&cpu_t1, &wall_t1);

  if (OP_diags>2) {
    printf(" kernel routine with indirection: CalcCourantConstraint\n");
  }

  int exec_size = op_mpi_halo_exchanges(set, nargs, args);

  if (exec_size >0) {

    #ifdef VECTORIZE
    #pragma novector
    for ( int n=0; n<(exec_size/SIMD_VEC)*SIMD_VEC; n+=SIMD_VEC ){
      double dat3[SIMD_VEC];
      for ( int i=0; i<SIMD_VEC; i++ ){
        dat3[i] = INFINITY;
      }
      if (n<set->core_size && n>0 && n % OP_mpi_test_frequency == 0)
        op_mpi_test_all(nargs,args);
      if ((n+SIMD_VEC >= set->core_size) && (n+SIMD_VEC-set->core_size < SIMD_VEC)) {
        op_mpi_wait_all(nargs, args);
      }
      ALIGNED_double double dat0[1][SIMD_VEC];
      ALIGNED_double double dat1[1][SIMD_VEC];
      ALIGNED_double double dat2[1][SIMD_VEC];
      #pragma omp simd simdlen(SIMD_VEC)
      for ( int i=0; i<SIMD_VEC; i++ ){
        int idx0_1 = 1 * arg0.map_data[(n+i) * arg0.map->dim + 0];
        int idx1_1 = 1 * arg1.map_data[(n+i) * arg1.map->dim + 0];
        int idx2_1 = 1 * arg2.map_data[(n+i) * arg2.map->dim + 0];

        dat0[0][i] = (ptr0)[idx0_1 + 0];
        dat1[0][i] = (ptr1)[idx1_1 + 0];
        dat2[0][i] = (ptr2)[idx2_1 + 0];
      }
      #pragma omp simd simdlen(SIMD_VEC)
      for ( int i=0; i<SIMD_VEC; i++ ){
        CalcCourantConstraint_vec(
          dat0,
          dat1,
          dat2,
          dat3,
          i);
      }
      for ( int i=0; i<SIMD_VEC; i++ ){

      }
      for ( int i=0; i<SIMD_VEC; i++ ){
        *(double*)arg3.data = MIN(*(double*)arg3.data,dat3[i]);
      }
    }

    //remainder
    for ( int n=(exec_size/SIMD_VEC)*SIMD_VEC; n<exec_size; n++ ){
    #else
    for ( int n=0; n<exec_size; n++ ){
    #endif
      if (n==set->core_size) {
        op_mpi_wait_all(nargs, args);
      }
      int map0idx;
      map0idx = arg0.map_data[n * arg0.map->dim + 0];

      CalcCourantConstraint(
        &(ptr0)[1 * map0idx],
        &(ptr1)[1 * map0idx],
        &(ptr2)[1 * map0idx],
        (double*)arg3.data);
    }
  }

  if (exec_size == 0 || exec_size == set->core_size) {
    op_mpi_wait_all(nargs, args);
  }
  // combine reduction data
  op_mpi_reduce(&arg3,(double*)arg3.data);
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[38].name      = name;
  OP_kernels[38].count    += 1;
  OP_kernels[38].time     += wall_t2 - wall_t1;
  OP_kernels[38].transfer += (float)set->size * arg0.size;
  OP_kernels[38].transfer += (float)set->size * arg1.size;
  OP_kernels[38].transfer += (float)set->size * arg2.size;
  OP_kernels[38].transfer += (float)set->size * arg3.size * 2.0f;
  OP_kernels[38].transfer += (float)set->size * arg0.map->dim * 4.0f;
}
