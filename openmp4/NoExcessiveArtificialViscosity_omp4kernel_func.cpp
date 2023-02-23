//
// auto-generated by op2.py
//

void NoExcessiveArtificialViscosity_omp4_kernel(
  double *data0,
  int dat0size,
  int count,
  int num_teams,
  int nthread){

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size]) \
    map(to: m_qstop_ompkernel)
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    const double *q = &data0[1*n_op];

    //inline function
    
      if ( q[0] > m_qstop_ompkernel ) {

      }
    //end inline func
  }

}
