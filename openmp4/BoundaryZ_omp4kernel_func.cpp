//
// auto-generated by op2.py
//

void BoundaryZ_omp4_kernel(
  double *data0,
  int dat0size,
  int *data1,
  int dat1size,
  int count,
  int num_teams,
  int nthread){

  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size],data1[0:dat1size])
  #pragma omp distribute parallel for schedule(static,1)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    double *zdd = &data0[1*n_op];
    const int *symmZ = &data1[1*n_op];

    //inline function
    
      if(symmZ[0] & 0x01){
          zdd[0] = double(0.0);
      }
    //end inline func
  }

}
