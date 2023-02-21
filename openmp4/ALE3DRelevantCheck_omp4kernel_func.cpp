//
// auto-generated by op2.py
//

void ALE3DRelevantCheck_omp4_kernel(
  double *data0,
  int dat0size,
  int count,
  int num_teams,
  int nthread){

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size]) \
    map(to: m_eosvmax_ompkernel, m_eosvmin_ompkernel)
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    const double *v = &data0[1*n_op];

    //inline function
    
      double vc = v[0] ;

      if (m_eosvmin_ompkernel != double(0.)) {
          if (vc < m_eosvmin_ompkernel)
          vc = m_eosvmin_ompkernel ;
      }
      if (m_eosvmax_ompkernel != double(0.)) {
          if (vc > m_eosvmax_ompkernel)
          vc = m_eosvmax_ompkernel ;
      }
      if (vc <= 0.) {

      }
    //end inline func
  }

}
