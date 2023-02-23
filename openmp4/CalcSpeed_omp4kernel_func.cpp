//
// auto-generated by op2.py
//

#include <math.h>

void CalcSpeed_omp4_kernel(
  double *data0,
  int dat0size,
  double *data1,
  int dat1size,
  double *data2,
  int dat2size,
  double *data3,
  int dat3size,
  int count,
  int num_teams,
  int nthread){

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size],data1[0:dat1size],data2[0:dat2size],data3[0:dat3size])
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    double *speed = &data0[1*n_op];
    const double *xd = &data1[1*n_op];
    const double *yd = &data2[1*n_op];
    const double *zd = &data3[1*n_op];

    //inline function
    
      speed[0] = sqrt((xd[0]*xd[0])+(yd[0]*yd[0])+(zd[0]*zd[0]));
    //end inline func
  }

}
