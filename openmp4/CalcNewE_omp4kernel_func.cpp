//
// auto-generated by op2.py
//

void CalcNewE_omp4_kernel(
  double *data0,
  int dat0size,
  double *data1,
  int dat1size,
  double *data2,
  int dat2size,
  double *data3,
  int dat3size,
  double *data4,
  int dat4size,
  double *data5,
  int dat5size,
  int count,
  int num_teams,
  int nthread){

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size],data1[0:dat1size],data2[0:dat2size],data3[0:dat3size],data4[0:dat4size],data5[0:dat5size]) \
    map(to: m_emin_ompkernel)
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    double *e_new = &data0[1*n_op];
    const double *e_old = &data1[1*n_op];
    const double *delvc = &data2[1*n_op];
    const double *p_old = &data3[1*n_op];
    const double *q_old = &data4[1*n_op];
    const double *work = &data5[1*n_op];

    //inline function
    
      e_new[0] = e_old[0] - double(0.5) * delvc[0] * (p_old[0] + q_old[0])
          + double(0.5) * work[0];

      if (e_new[0]  < m_emin_ompkernel ) {
          e_new[0] = m_emin_ompkernel ;
      }
    //end inline func
  }

}
