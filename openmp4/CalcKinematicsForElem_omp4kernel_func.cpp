//
// auto-generated by op2.py
//

void CalcKinematicsForElem_omp4_kernel(
  int *map0,
  int map0size,
  double *data48,
  int dat48size,
  double *data49,
  int dat49size,
  double *data50,
  int dat50size,
  double *data51,
  int dat51size,
  double *data52,
  int dat52size,
  double *data53,
  int dat53size,
  double *data54,
  int dat54size,
  double *data55,
  int dat55size,
  double *arg56,
  double *data0,
  int dat0size,
  double *data8,
  int dat8size,
  double *data16,
  int dat16size,
  double *data24,
  int dat24size,
  double *data32,
  int dat32size,
  double *data40,
  int dat40size,
  int *col_reord,
  int set_size1,
  int start,
  int end,
  int num_teams,
  int nthread){

  double arg56_l = *arg56;
  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) map(to:data48[0:dat48size],data49[0:dat49size],data50[0:dat50size],data51[0:dat51size],data52[0:dat52size],data53[0:dat53size],data54[0:dat54size],data55[0:dat55size]) \
    map(to: m_twelfth_ompkernel)\
    map(to:col_reord[0:set_size1],map0[0:map0size],data0[0:dat0size],data8[0:dat8size],data16[0:dat16size],data24[0:dat24size],data32[0:dat32size],data40[0:dat40size])
  #pragma omp distribute parallel for schedule(static,1)
  for ( int e=start; e<end; e++ ){
    int n_op = col_reord[e];
    int map0idx;
    int map1idx;
    int map2idx;
    int map3idx;
    int map4idx;
    int map5idx;
    int map6idx;
    int map7idx;
    map0idx = map0[n_op + set_size1 * 0];
    map1idx = map0[n_op + set_size1 * 1];
    map2idx = map0[n_op + set_size1 * 2];
    map3idx = map0[n_op + set_size1 * 3];
    map4idx = map0[n_op + set_size1 * 4];
    map5idx = map0[n_op + set_size1 * 5];
    map6idx = map0[n_op + set_size1 * 6];
    map7idx = map0[n_op + set_size1 * 7];

    //variable mapping
    const double *p_x0 = &data0[1 * map0idx];
    const double *p_x1 = &data0[1 * map1idx];
    const double *p_x2 = &data0[1 * map2idx];
    const double *p_x3 = &data0[1 * map3idx];
    const double *p_x4 = &data0[1 * map4idx];
    const double *p_x5 = &data0[1 * map5idx];
    const double *p_x6 = &data0[1 * map6idx];
    const double *p_x7 = &data0[1 * map7idx];
    const double *p_y0 = &data8[1 * map0idx];
    const double *p_y1 = &data8[1 * map1idx];
    const double *p_y2 = &data8[1 * map2idx];
    const double *p_y3 = &data8[1 * map3idx];
    const double *p_y4 = &data8[1 * map4idx];
    const double *p_y5 = &data8[1 * map5idx];
    const double *p_y6 = &data8[1 * map6idx];
    const double *p_y7 = &data8[1 * map7idx];
    const double *p_z0 = &data16[1 * map0idx];
    const double *p_z1 = &data16[1 * map1idx];
    const double *p_z2 = &data16[1 * map2idx];
    const double *p_z3 = &data16[1 * map3idx];
    const double *p_z4 = &data16[1 * map4idx];
    const double *p_z5 = &data16[1 * map5idx];
    const double *p_z6 = &data16[1 * map6idx];
    const double *p_z7 = &data16[1 * map7idx];
    const double *p_xd0 = &data24[1 * map0idx];
    const double *p_xd1 = &data24[1 * map1idx];
    const double *p_xd2 = &data24[1 * map2idx];
    const double *p_xd3 = &data24[1 * map3idx];
    const double *p_xd4 = &data24[1 * map4idx];
    const double *p_xd5 = &data24[1 * map5idx];
    const double *p_xd6 = &data24[1 * map6idx];
    const double *p_xd7 = &data24[1 * map7idx];
    const double *p_yd0 = &data32[1 * map0idx];
    const double *p_yd1 = &data32[1 * map1idx];
    const double *p_yd2 = &data32[1 * map2idx];
    const double *p_yd3 = &data32[1 * map3idx];
    const double *p_yd4 = &data32[1 * map4idx];
    const double *p_yd5 = &data32[1 * map5idx];
    const double *p_yd6 = &data32[1 * map6idx];
    const double *p_yd7 = &data32[1 * map7idx];
    const double *p_zd0 = &data40[1 * map0idx];
    const double *p_zd1 = &data40[1 * map1idx];
    const double *p_zd2 = &data40[1 * map2idx];
    const double *p_zd3 = &data40[1 * map3idx];
    const double *p_zd4 = &data40[1 * map4idx];
    const double *p_zd5 = &data40[1 * map5idx];
    const double *p_zd6 = &data40[1 * map6idx];
    const double *p_zd7 = &data40[1 * map7idx];
    double *dxx = &data48[1*n_op];
    double *dyy = &data49[1*n_op];
    double *dzz = &data50[1*n_op];
    double *vnew = &data51[1*n_op];
    const double *volo = &data52[1*n_op];
    double *delv = &data53[1*n_op];
    const double *v = &data54[1*n_op];
    double *arealg = &data55[1*n_op];
    const double *deltaTime = &arg56_l;

    //inline function
    

     double B[3][8] ;
     double D[6] ;
     double x_local[8] ;
     double y_local[8] ;
     double z_local[8] ;
     double detJ = double(0.0) ;

     double volume ;
     double relativeVolume ;

     x_local[0] = p_x0[0];
     x_local[1] = p_x1[0];
     x_local[2] = p_x2[0];
     x_local[3] = p_x3[0];
     x_local[4] = p_x4[0];
     x_local[5] = p_x5[0];
     x_local[6] = p_x6[0];
     x_local[7] = p_x7[0];

     y_local[0] = p_y0[0];
     y_local[1] = p_y1[0];
     y_local[2] = p_y2[0];
     y_local[3] = p_y3[0];
     y_local[4] = p_y4[0];
     y_local[5] = p_y5[0];
     y_local[6] = p_y6[0];
     y_local[7] = p_y7[0];

     z_local[0] = p_z0[0];
     z_local[1] = p_z1[0];
     z_local[2] = p_z2[0];
     z_local[3] = p_z3[0];
     z_local[4] = p_z4[0];
     z_local[5] = p_z5[0];
     z_local[6] = p_z6[0];
     z_local[7] = p_z7[0];


     double dx61 = p_x6[0] - p_x1[0];
     double dy61 = p_y6[0] - p_y1[0];
     double dz61 = p_z6[0] - p_z1[0];

     double dx70 = p_x7[0] - p_x0[0];
     double dy70 = p_y7[0] - p_y0[0];
     double dz70 = p_z7[0] - p_z0[0];

     double dx63 = p_x6[0] - p_x3[0];
     double dy63 = p_y6[0] - p_y3[0];
     double dz63 = p_z6[0] - p_z3[0];

     double dx20 = p_x2[0] - p_x0[0];
     double dy20 = p_y2[0] - p_y0[0];
     double dz20 = p_z2[0] - p_z0[0];

     double dx50 = p_x5[0] - p_x0[0];
     double dy50 = p_y5[0] - p_y0[0];
     double dz50 = p_z5[0] - p_z0[0];

     double dx64 = p_x6[0] - p_x4[0];
     double dy64 = p_y6[0] - p_y4[0];
     double dz64 = p_z6[0] - p_z4[0];

     double dx31 = p_x3[0] - p_x1[0];
     double dy31 = p_y3[0] - p_y1[0];
     double dz31 = p_z3[0] - p_z1[0];

     double dx72 = p_x7[0] - p_x2[0];
     double dy72 = p_y7[0] - p_y2[0];
     double dz72 = p_z7[0] - p_z2[0];

     double dx43 = p_x4[0] - p_x3[0];
     double dy43 = p_y4[0] - p_y3[0];
     double dz43 = p_z4[0] - p_z3[0];

     double dx57 = p_x5[0] - p_x7[0];
     double dy57 = p_y5[0] - p_y7[0];
     double dz57 = p_z5[0] - p_z7[0];

     double dx14 = p_x1[0] - p_x4[0];
     double dy14 = p_y1[0] - p_y4[0];
     double dz14 = p_z1[0] - p_z4[0];

     double dx25 = p_x2[0] - p_x5[0];
     double dy25 = p_y2[0] - p_y5[0];
     double dz25 = p_z2[0] - p_z5[0];

     #define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
        ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

     double temp_volume =
        TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
           dy31 + dy72, dy63, dy20,
           dz31 + dz72, dz63, dz20) +
        TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
           dy43 + dy57, dy64, dy70,
           dz43 + dz57, dz64, dz70) +
        TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
           dy14 + dy25, dy61, dy50,
           dz14 + dz25, dz61, dz50);

        #undef TRIPLE_PRODUCT
     temp_volume *= m_twelfth_ompkernel;

     volume = temp_volume;



     relativeVolume = volume / volo[0] ;
     vnew[0] = relativeVolume ;
     delv[0] = relativeVolume - v[0] ;




     double a, charLength = double(0.0);
     double fx,fy,fz,gx,gy,gz,area;



     fx = (x_local[2] - x_local[0]) - (x_local[3] - x_local[1]);
     fy = (y_local[2] - y_local[0]) - (y_local[3] - y_local[1]);
     fz = (z_local[2] - z_local[0]) - (z_local[3] - z_local[1]);
     gx = (x_local[2] - x_local[0]) + (x_local[3] - x_local[1]);
     gy = (y_local[2] - y_local[0]) + (y_local[3] - y_local[1]);
     gz = (z_local[2] - z_local[0]) + (z_local[3] - z_local[1]);
     area =
        (fx * fx + fy * fy + fz * fz) *
        (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) *
        (fx * gx + fy * gy + fz * gz);
     a = area;
     charLength = std::fmax(a,charLength) ;



     fx = (x_local[6] - x_local[4]) - (x_local[7] - x_local[5]);
     fy = (y_local[6] - y_local[4]) - (y_local[7] - y_local[5]);
     fz = (z_local[6] - z_local[4]) - (z_local[7] - z_local[5]);
     gx = (x_local[6] - x_local[4]) + (x_local[7] - x_local[5]);
     gy = (y_local[6] - y_local[4]) + (y_local[7] - y_local[5]);
     gz = (z_local[6] - z_local[4]) + (z_local[7] - z_local[5]);
     area =
        (fx * fx + fy * fy + fz * fz) *
        (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) *
        (fx * gx + fy * gy + fz * gz);
     a = area;
     charLength = std::fmax(a,charLength) ;



     fx = (x_local[5] - x_local[0]) - (x_local[4] - x_local[1]);
     fy = (y_local[5] - y_local[0]) - (y_local[4] - y_local[1]);
     fz = (z_local[5] - z_local[0]) - (z_local[4] - z_local[1]);
     gx = (x_local[5] - x_local[0]) + (x_local[4] - x_local[1]);
     gy = (y_local[5] - y_local[0]) + (y_local[4] - y_local[1]);
     gz = (z_local[5] - z_local[0]) + (z_local[4] - z_local[1]);
     area =
        (fx * fx + fy * fy + fz * fz) *
        (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) *
        (fx * gx + fy * gy + fz * gz);
     a = area;
     charLength = std::fmax(a,charLength) ;



     fx = (x_local[6] - x_local[1]) - (x_local[5] - x_local[2]);
     fy = (y_local[6] - y_local[1]) - (y_local[5] - y_local[2]);
     fz = (z_local[6] - z_local[1]) - (z_local[5] - z_local[2]);
     gx = (x_local[6] - x_local[1]) + (x_local[5] - x_local[2]);
     gy = (y_local[6] - y_local[1]) + (y_local[5] - y_local[2]);
     gz = (z_local[6] - z_local[1]) + (z_local[5] - z_local[2]);
     area =
        (fx * fx + fy * fy + fz * fz) *
        (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) *
        (fx * gx + fy * gy + fz * gz);
     a = area;
     charLength = std::fmax(a,charLength) ;



     fx = (x_local[7] - x_local[2]) - (x_local[6] - x_local[3]);
     fy = (y_local[7] - y_local[2]) - (y_local[6] - y_local[3]);
     fz = (z_local[7] - z_local[2]) - (z_local[6] - z_local[3]);
     gx = (x_local[7] - x_local[2]) + (x_local[6] - x_local[3]);
     gy = (y_local[7] - y_local[2]) + (y_local[6] - y_local[3]);
     gz = (z_local[7] - z_local[2]) + (z_local[6] - z_local[3]);
     area =
        (fx * fx + fy * fy + fz * fz) *
        (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) *
        (fx * gx + fy * gy + fz * gz);
     a = area;
     charLength = std::fmax(a,charLength) ;



     fx = (x_local[4] - x_local[3]) - (x_local[7] - x_local[0]);
     fy = (y_local[4] - y_local[3]) - (y_local[7] - y_local[0]);
     fz = (z_local[4] - z_local[3]) - (z_local[7] - z_local[0]);
     gx = (x_local[4] - x_local[3]) + (x_local[7] - x_local[0]);
     gy = (y_local[4] - y_local[3]) + (y_local[7] - y_local[0]);
     gz = (z_local[4] - z_local[3]) + (z_local[7] - z_local[0]);
     area =
        (fx * fx + fy * fy + fz * fz) *
        (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) *
        (fx * gx + fy * gy + fz * gz);
     a = area;
     charLength = std::fmax(a,charLength) ;

     charLength = double(4.0) * volume / sqrt(charLength);

     arealg[0] = charLength;

     double dt2 = double(0.5) * (*deltaTime);

     x_local[0] -= dt2 * p_xd0[0];
     y_local[0] -= dt2 * p_yd0[0];
     z_local[0] -= dt2 * p_zd0[0];

     x_local[1] -= dt2 * p_xd1[0];
     y_local[1] -= dt2 * p_yd1[0];
     z_local[1] -= dt2 * p_zd1[0];

     x_local[2] -= dt2 * p_xd2[0];
     y_local[2] -= dt2 * p_yd2[0];
     z_local[2] -= dt2 * p_zd2[0];

     x_local[3] -= dt2 * p_xd3[0];
     y_local[3] -= dt2 * p_yd3[0];
     z_local[3] -= dt2 * p_zd3[0];

     x_local[4] -= dt2 * p_xd4[0];
     y_local[4] -= dt2 * p_yd4[0];
     z_local[4] -= dt2 * p_zd4[0];

     x_local[5] -= dt2 * p_xd5[0];
     y_local[5] -= dt2 * p_yd5[0];
     z_local[5] -= dt2 * p_zd5[0];

     x_local[6] -= dt2 * p_xd6[0];
     y_local[6] -= dt2 * p_yd6[0];
     z_local[6] -= dt2 * p_zd6[0];

     x_local[7] -= dt2 * p_xd7[0];
     y_local[7] -= dt2 * p_yd7[0];
     z_local[7] -= dt2 * p_zd7[0];











    double fjxxi, fjxet, fjxze;
    double fjyxi, fjyet, fjyze;
    double fjzxi, fjzet, fjzze;
    double cjxxi, cjxet, cjxze;
    double cjyxi, cjyet, cjyze;
    double cjzxi, cjzet, cjzze;

     fjxxi = double(.125) * ( (x_local[6]-x_local[0]) + (x_local[5]-x_local[3]) - (x_local[7]-x_local[1]) - (x_local[4]-x_local[2]) );
     fjxet = double(.125) * ( (x_local[6]-x_local[0]) - (x_local[5]-x_local[3]) + (x_local[7]-x_local[1]) - (x_local[4]-x_local[2]) );
     fjxze = double(.125) * ( (x_local[6]-x_local[0]) + (x_local[5]-x_local[3]) + (x_local[7]-x_local[1]) + (x_local[4]-x_local[2]) );

     fjyxi = double(.125) * ( (y_local[6]-y_local[0]) + (y_local[5]-y_local[3]) - (y_local[7]-y_local[1]) - (y_local[4]-y_local[2]) );
     fjyet = double(.125) * ( (y_local[6]-y_local[0]) - (y_local[5]-y_local[3]) + (y_local[7]-y_local[1]) - (y_local[4]-y_local[2]) );
     fjyze = double(.125) * ( (y_local[6]-y_local[0]) + (y_local[5]-y_local[3]) + (y_local[7]-y_local[1]) + (y_local[4]-y_local[2]) );

     fjzxi = double(.125) * ( (z_local[6]-z_local[0]) + (z_local[5]-z_local[3]) - (z_local[7]-z_local[1]) - (z_local[4]-z_local[2]) );
     fjzet = double(.125) * ( (z_local[6]-z_local[0]) - (z_local[5]-z_local[3]) + (z_local[7]-z_local[1]) - (z_local[4]-z_local[2]) );
     fjzze = double(.125) * ( (z_local[6]-z_local[0]) + (z_local[5]-z_local[3]) + (z_local[7]-z_local[1]) + (z_local[4]-z_local[2]) );


    cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
    cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
    cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

    cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
    cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
    cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

    cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
    cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
    cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

    B[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
    B[0][1] =      cjxxi  -  cjxet  -  cjxze;
    B[0][2] =      cjxxi  +  cjxet  -  cjxze;
    B[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
    B[0][4] = -B[0][2];
    B[0][5] = -B[0][3];
    B[0][6] = -B[0][0];
    B[0][7] = -B[0][1];

    B[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
    B[1][1] =      cjyxi  -  cjyet  -  cjyze;
    B[1][2] =      cjyxi  +  cjyet  -  cjyze;
    B[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
    B[1][4] = -B[1][2];
    B[1][5] = -B[1][3];
    B[1][6] = -B[1][0];
    B[1][7] = -B[1][1];

    B[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
    B[2][1] =      cjzxi  -  cjzet  -  cjzze;
    B[2][2] =      cjzxi  +  cjzet  -  cjzze;
    B[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
    B[2][4] = -B[2][2];
    B[2][5] = -B[2][3];
    B[2][6] = -B[2][0];
    B[2][7] = -B[2][1];

    detJ = double(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);



     const double inv_detJ = double(1.0) / detJ ;
     double dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
     const double* const pfx = B[0];
     const double* const pfy = B[1];
     const double* const pfz = B[2];

     D[0] = inv_detJ * ( pfx[0] * (p_xd0[0]-p_xd6[0])
                          + pfx[1] * (p_xd1[0]-p_xd7[0])
                          + pfx[2] * (p_xd2[0]-p_xd4[0])
                          + pfx[3] * (p_xd3[0]-p_xd5[0]) );

     D[1] = inv_detJ * ( pfy[0] * (p_yd0[0]-p_yd6[0])
                          + pfy[1] * (p_yd1[0]-p_yd7[0])
                          + pfy[2] * (p_yd2[0]-p_yd4[0])
                          + pfy[3] * (p_yd3[0]-p_yd5[0]) );

     D[2] = inv_detJ * ( pfz[0] * (p_zd0[0]-p_zd6[0])
                          + pfz[1] * (p_zd1[0]-p_zd7[0])
                          + pfz[2] * (p_zd2[0]-p_zd4[0])
                          + pfz[3] * (p_zd3[0]-p_zd5[0]) );

     dyddx  = inv_detJ * ( pfx[0] * (p_yd0[0]-p_yd6[0])
                          + pfx[1] * (p_yd1[0]-p_yd7[0])
                          + pfx[2] * (p_yd2[0]-p_yd4[0])
                          + pfx[3] * (p_yd3[0]-p_yd5[0]) );

     dxddy  = inv_detJ * ( pfy[0] * (p_xd0[0]-p_xd6[0])
                          + pfy[1] * (p_xd1[0]-p_xd7[0])
                          + pfy[2] * (p_xd2[0]-p_xd4[0])
                          + pfy[3] * (p_xd3[0]-p_xd5[0]) );

     dzddx  = inv_detJ * ( pfx[0] * (p_zd0[0]-p_zd6[0])
                          + pfx[1] * (p_zd1[0]-p_zd7[0])
                          + pfx[2] * (p_zd2[0]-p_zd4[0])
                          + pfx[3] * (p_zd3[0]-p_zd5[0]) );

     dxddz  = inv_detJ * ( pfz[0] * (p_xd0[0]-p_xd6[0])
                          + pfz[1] * (p_xd1[0]-p_xd7[0])
                          + pfz[2] * (p_xd2[0]-p_xd4[0])
                          + pfz[3] * (p_xd3[0]-p_xd5[0]) );

     dzddy  = inv_detJ * ( pfy[0] * (p_zd0[0]-p_zd6[0])
                          + pfy[1] * (p_zd1[0]-p_zd7[0])
                          + pfy[2] * (p_zd2[0]-p_zd4[0])
                          + pfy[3] * (p_zd3[0]-p_zd5[0]) );

     dyddz  = inv_detJ * ( pfz[0] * (p_yd0[0]-p_yd6[0])
                          + pfz[1] * (p_yd1[0]-p_yd7[0])
                          + pfz[2] * (p_yd2[0]-p_yd4[0])
                          + pfz[3] * (p_yd3[0]-p_yd5[0]) );
     D[5]  = double( .5) * ( dxddy + dyddx );
     D[4]  = double( .5) * ( dxddz + dzddx );
     D[3]  = double( .5) * ( dzddy + dyddz );

     dxx[0] = D[0];
     dyy[0] = D[1];
     dzz[0] = D[2];
    //end inline func
  }

  *arg56 = arg56_l;
}