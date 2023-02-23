//
// auto-generated by op2.py
//

void initStressTerms_omp4_kernel(
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
  int count,
  int num_teams,
  int nthread){

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size],data1[0:dat1size],data2[0:dat2size],data3[0:dat3size],data4[0:dat4size])
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    double *sigxx = &data0[1*n_op];
    double *sigyy = &data1[1*n_op];
    double *sigzz = &data2[1*n_op];
    const double *p = &data3[1*n_op];
    const double *q = &data4[1*n_op];

    //inline function
    
      sigxx[0] = -p[0] - q[0];
      sigyy[0] = -p[0] - q[0];
      sigzz[0] = -p[0] - q[0];
    //end inline func
  }

}