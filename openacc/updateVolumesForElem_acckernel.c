//
// auto-generated by op2.py
//

//user function
//user function
//#pragma acc routine
inline void updateVolumesForElem_openacc( const double *vnew, double *v) {
    double tmpV = vnew[0] ;

    if ( fabs(tmpV - double(1.0)) < m_v_cut )
    tmpV = double(1.0) ;

    v[0] = tmpV ;
}

// host stub function
void op_par_loop_updateVolumesForElem(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1){

  int nargs = 2;
  op_arg args[2];

  args[0] = arg0;
  args[1] = arg1;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(37);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[37].name      = name;
  OP_kernels[37].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  updateVolumesForElem");
  }

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);


  if (set_size >0) {


    //Set up typed device pointers for OpenACC

    double* data0 = (double*)arg0.data_d;
    double* data1 = (double*)arg1.data_d;
    #pragma acc parallel loop independent deviceptr(data0,data1)
    for ( int n=0; n<set->size; n++ ){
      updateVolumesForElem_openacc(
        &data0[1*n],
        &data1[1*n]);
    }
  }

  // combine reduction data
  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[37].time     += wall_t2 - wall_t1;
  OP_kernels[37].transfer += (float)set->size * arg0.size;
  OP_kernels[37].transfer += (float)set->size * arg1.size * 2.0f;
}
