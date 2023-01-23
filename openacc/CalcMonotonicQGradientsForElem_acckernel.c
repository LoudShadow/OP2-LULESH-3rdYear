//
// auto-generated by op2.py
//

//user function
//user function
//#pragma acc routine
inline void CalcMonotonicQGradientsForElem_openacc( 
    const double *p_x0, const double *p_x1, const double *p_x2, const double *p_x3, const double *p_x4, const double *p_x5, const double *p_x6, const double *p_x7,
    const double *p_y0, const double *p_y1, const double *p_y2, const double *p_y3, const double *p_y4, const double *p_y5, const double *p_y6, const double *p_y7,
    const double *p_z0, const double *p_z1, const double *p_z2, const double *p_z3, const double *p_z4, const double *p_z5, const double *p_z6, const double *p_z7,
    const double *p_xd0, const double *p_xd1, const double *p_xd2, const double *p_xd3, const double *p_xd4, const double *p_xd5, const double *p_xd6, const double *p_xd7,
    const double *p_yd0, const double *p_yd1, const double *p_yd2, const double *p_yd3, const double *p_yd4, const double *p_yd5, const double *p_yd6, const double *p_yd7,
    const double *p_zd0, const double *p_zd1, const double *p_zd2, const double *p_zd3, const double *p_zd4, const double *p_zd5, const double *p_zd6, const double *p_zd7,
    const double *volo,
    const double *vnew,
    double *delx_zeta,
    double *delv_zeta,
    double *delv_xi,
    double *delx_xi,
    double *delx_eta,
    double *delv_eta
) {
    double ax,ay,az;
    double dxv,dyv,dzv;

    double vol = volo[0]*vnew[0];
    double norm = double(1.0) / ( vol + m_ptiny );

    double dxj = double(-0.25)*((p_x0[0]+p_x1[0]+p_x5[0]+p_x4[0]) - (p_x3[0]+p_x2[0]+p_x6[0]+p_x7[0])) ;
    double dyj = double(-0.25)*((p_y0[0]+p_y1[0]+p_y5[0]+p_y4[0]) - (p_y3[0]+p_y2[0]+p_y6[0]+p_y7[0])) ;
    double dzj = double(-0.25)*((p_z0[0]+p_z1[0]+p_z5[0]+p_z4[0]) - (p_z3[0]+p_z2[0]+p_z6[0]+p_z7[0])) ;

    double dxi = double( 0.25)*((p_x1[0]+p_x2[0]+p_x6[0]+p_x5[0]) - (p_x0[0]+p_x3[0]+p_x7[0]+p_x4[0])) ;
    double dyi = double( 0.25)*((p_y1[0]+p_y2[0]+p_y6[0]+p_y5[0]) - (p_y0[0]+p_y3[0]+p_y7[0]+p_y4[0])) ;
    double dzi = double( 0.25)*((p_z1[0]+p_z2[0]+p_z6[0]+p_z5[0]) - (p_z0[0]+p_z3[0]+p_z7[0]+p_z4[0])) ;

    double dxk = double( 0.25)*((p_x4[0]+p_x5[0]+p_x6[0]+p_x7[0]) - (p_x0[0]+p_x1[0]+p_x2[0]+p_x3[0])) ;
    double dyk = double( 0.25)*((p_y4[0]+p_y5[0]+p_y6[0]+p_y7[0]) - (p_y0[0]+p_y1[0]+p_y2[0]+p_y3[0])) ;
    double dzk = double( 0.25)*((p_z4[0]+p_z5[0]+p_z6[0]+p_z7[0]) - (p_z0[0]+p_z1[0]+p_z2[0]+p_z3[0])) ;

    ax = dyi*dzj - dzi*dyj ;
    ay = dzi*dxj - dxi*dzj ;
    az = dxi*dyj - dyi*dxj ;

    delx_zeta[0] = vol / sqrt(ax*ax + ay*ay + az*az + m_ptiny) ;

    ax *= norm ;
    ay *= norm ;
    az *= norm ;

    dxv = double(0.25)*((p_xd4[0]+p_xd5[0]+p_xd6[0]+p_xd7[0]) - (p_xd0[0]+p_xd1[0]+p_xd2[0]+p_xd3[0])) ;
    dyv = double(0.25)*((p_yd4[0]+p_yd5[0]+p_yd6[0]+p_yd7[0]) - (p_yd0[0]+p_yd1[0]+p_yd2[0]+p_yd3[0])) ;
    dzv = double(0.25)*((p_zd4[0]+p_zd5[0]+p_zd6[0]+p_zd7[0]) - (p_zd0[0]+p_zd1[0]+p_zd2[0]+p_zd3[0])) ;

    delv_zeta[0] = ax*dxv + ay*dyv + az*dzv ;

    ax = dyj*dzk - dzj*dyk ;
    ay = dzj*dxk - dxj*dzk ;
    az = dxj*dyk - dyj*dxk ;

    delx_xi[0] = vol / sqrt(ax*ax + ay*ay + az*az + m_ptiny) ;

    ax *= norm ;
    ay *= norm ;
    az *= norm ;

    dxv = double(0.25)*((p_xd1[0]+p_xd2[0]+p_xd6[0]+p_xd5[0]) - (p_xd0[0]+p_xd3[0]+p_xd7[0]+p_xd4[0])) ;
    dyv = double(0.25)*((p_yd1[0]+p_yd2[0]+p_yd6[0]+p_yd5[0]) - (p_yd0[0]+p_yd3[0]+p_yd7[0]+p_yd4[0])) ;
    dzv = double(0.25)*((p_zd1[0]+p_zd2[0]+p_zd6[0]+p_zd5[0]) - (p_zd0[0]+p_zd3[0]+p_zd7[0]+p_zd4[0])) ;

    delv_xi[0] = ax*dxv + ay*dyv + az*dzv ;


    ax = dyk*dzi - dzk*dyi ;
    ay = dzk*dxi - dxk*dzi ;
    az = dxk*dyi - dyk*dxi ;

    delx_eta[0] = vol / sqrt(ax*ax + ay*ay + az*az + m_ptiny) ;

    ax *= norm ;
    ay *= norm ;
    az *= norm ;

    dxv = double(-0.25)*((p_xd0[0]+p_xd1[0]+p_xd5[0]+p_xd4[0]) - (p_xd3[0]+p_xd2[0]+p_xd6[0]+p_xd7[0])) ;
    dyv = double(-0.25)*((p_yd0[0]+p_yd1[0]+p_yd5[0]+p_yd4[0]) - (p_yd3[0]+p_yd2[0]+p_yd6[0]+p_yd7[0])) ;
    dzv = double(-0.25)*((p_zd0[0]+p_zd1[0]+p_zd5[0]+p_zd4[0]) - (p_zd3[0]+p_zd2[0]+p_zd6[0]+p_zd7[0])) ;

    delv_eta[0] = ax*dxv + ay*dyv + az*dzv ;

}

// host stub function
void op_par_loop_CalcMonotonicQGradientsForElem(char const *name, op_set set,
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
  op_arg arg11,
  op_arg arg12,
  op_arg arg13,
  op_arg arg14,
  op_arg arg15,
  op_arg arg16,
  op_arg arg17,
  op_arg arg18,
  op_arg arg19,
  op_arg arg20,
  op_arg arg21,
  op_arg arg22,
  op_arg arg23,
  op_arg arg24,
  op_arg arg25,
  op_arg arg26,
  op_arg arg27,
  op_arg arg28,
  op_arg arg29,
  op_arg arg30,
  op_arg arg31,
  op_arg arg32,
  op_arg arg33,
  op_arg arg34,
  op_arg arg35,
  op_arg arg36,
  op_arg arg37,
  op_arg arg38,
  op_arg arg39,
  op_arg arg40,
  op_arg arg41,
  op_arg arg42,
  op_arg arg43,
  op_arg arg44,
  op_arg arg45,
  op_arg arg46,
  op_arg arg47,
  op_arg arg48,
  op_arg arg49,
  op_arg arg50,
  op_arg arg51,
  op_arg arg52,
  op_arg arg53,
  op_arg arg54,
  op_arg arg55){

  int nargs = 56;
  op_arg args[56];

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
  args[12] = arg12;
  args[13] = arg13;
  args[14] = arg14;
  args[15] = arg15;
  args[16] = arg16;
  args[17] = arg17;
  args[18] = arg18;
  args[19] = arg19;
  args[20] = arg20;
  args[21] = arg21;
  args[22] = arg22;
  args[23] = arg23;
  args[24] = arg24;
  args[25] = arg25;
  args[26] = arg26;
  args[27] = arg27;
  args[28] = arg28;
  args[29] = arg29;
  args[30] = arg30;
  args[31] = arg31;
  args[32] = arg32;
  args[33] = arg33;
  args[34] = arg34;
  args[35] = arg35;
  args[36] = arg36;
  args[37] = arg37;
  args[38] = arg38;
  args[39] = arg39;
  args[40] = arg40;
  args[41] = arg41;
  args[42] = arg42;
  args[43] = arg43;
  args[44] = arg44;
  args[45] = arg45;
  args[46] = arg46;
  args[47] = arg47;
  args[48] = arg48;
  args[49] = arg49;
  args[50] = arg50;
  args[51] = arg51;
  args[52] = arg52;
  args[53] = arg53;
  args[54] = arg54;
  args[55] = arg55;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(14);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[14].name      = name;
  OP_kernels[14].count    += 1;

  int  ninds   = 6;
  int  inds[56] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,-1,-1,-1,-1,-1,-1,-1,-1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: CalcMonotonicQGradientsForElem\n");
  }

  // get plan
  #ifdef OP_PART_SIZE_14
    int part_size = OP_PART_SIZE_14;
  #else
    int part_size = OP_part_size;
  #endif

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);


  int ncolors = 0;

  if (set_size >0) {


    //Set up typed device pointers for OpenACC
    int *map0 = arg0.map_data_d;

    double* data48 = (double*)arg48.data_d;
    double* data49 = (double*)arg49.data_d;
    double* data50 = (double*)arg50.data_d;
    double* data51 = (double*)arg51.data_d;
    double* data52 = (double*)arg52.data_d;
    double* data53 = (double*)arg53.data_d;
    double* data54 = (double*)arg54.data_d;
    double* data55 = (double*)arg55.data_d;
    double *data0 = (double *)arg0.data_d;
    double *data8 = (double *)arg8.data_d;
    double *data16 = (double *)arg16.data_d;
    double *data24 = (double *)arg24.data_d;
    double *data32 = (double *)arg32.data_d;
    double *data40 = (double *)arg40.data_d;

    op_plan *Plan = op_plan_get_stage(name,set,part_size,nargs,args,ninds,inds,OP_COLOR2);
    ncolors = Plan->ncolors;
    int *col_reord = Plan->col_reord;
    int set_size1 = set->size + set->exec_size;

    // execute plan
    for ( int col=0; col<Plan->ncolors; col++ ){
      if (col==1) {
        op_mpi_wait_all_cuda(nargs, args);
      }
      int start = Plan->col_offsets[0][col];
      int end = Plan->col_offsets[0][col+1];

      #pragma acc parallel loop independent deviceptr(col_reord,map0,data48,data49,data50,data51,data52,data53,data54,data55,data0,data8,data16,data24,data32,data40)
      for ( int e=start; e<end; e++ ){
        int n = col_reord[e];
        int map0idx;
        int map1idx;
        int map2idx;
        int map3idx;
        int map4idx;
        int map5idx;
        int map6idx;
        int map7idx;
        map0idx = map0[n + set_size1 * 0];
        map1idx = map0[n + set_size1 * 1];
        map2idx = map0[n + set_size1 * 2];
        map3idx = map0[n + set_size1 * 3];
        map4idx = map0[n + set_size1 * 4];
        map5idx = map0[n + set_size1 * 5];
        map6idx = map0[n + set_size1 * 6];
        map7idx = map0[n + set_size1 * 7];


        CalcMonotonicQGradientsForElem_openacc(
          &data0[1 * map0idx],
          &data0[1 * map1idx],
          &data0[1 * map2idx],
          &data0[1 * map3idx],
          &data0[1 * map4idx],
          &data0[1 * map5idx],
          &data0[1 * map6idx],
          &data0[1 * map7idx],
          &data8[1 * map0idx],
          &data8[1 * map1idx],
          &data8[1 * map2idx],
          &data8[1 * map3idx],
          &data8[1 * map4idx],
          &data8[1 * map5idx],
          &data8[1 * map6idx],
          &data8[1 * map7idx],
          &data16[1 * map0idx],
          &data16[1 * map1idx],
          &data16[1 * map2idx],
          &data16[1 * map3idx],
          &data16[1 * map4idx],
          &data16[1 * map5idx],
          &data16[1 * map6idx],
          &data16[1 * map7idx],
          &data24[1 * map0idx],
          &data24[1 * map1idx],
          &data24[1 * map2idx],
          &data24[1 * map3idx],
          &data24[1 * map4idx],
          &data24[1 * map5idx],
          &data24[1 * map6idx],
          &data24[1 * map7idx],
          &data32[1 * map0idx],
          &data32[1 * map1idx],
          &data32[1 * map2idx],
          &data32[1 * map3idx],
          &data32[1 * map4idx],
          &data32[1 * map5idx],
          &data32[1 * map6idx],
          &data32[1 * map7idx],
          &data40[1 * map0idx],
          &data40[1 * map1idx],
          &data40[1 * map2idx],
          &data40[1 * map3idx],
          &data40[1 * map4idx],
          &data40[1 * map5idx],
          &data40[1 * map6idx],
          &data40[1 * map7idx],
          &data48[1 * n],
          &data49[1 * n],
          &data50[1 * n],
          &data51[1 * n],
          &data52[1 * n],
          &data53[1 * n],
          &data54[1 * n],
          &data55[1 * n]);
      }

    }
    OP_kernels[14].transfer  += Plan->transfer;
    OP_kernels[14].transfer2 += Plan->transfer2;
  }

  if (set_size == 0 || set_size == set->core_size || ncolors == 1) {
    op_mpi_wait_all_cuda(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[14].time     += wall_t2 - wall_t1;
}