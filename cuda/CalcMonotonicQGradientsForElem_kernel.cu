//
// auto-generated by op2.py
//

//user function
__device__ void CalcMonotonicQGradientsForElem_gpu( 
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
    double norm = double(1.0) / ( vol + m_ptiny_cuda );

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

    delx_zeta[0] = vol / sqrt(ax*ax + ay*ay + az*az + m_ptiny_cuda) ;

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

    delx_xi[0] = vol / sqrt(ax*ax + ay*ay + az*az + m_ptiny_cuda) ;

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

    delx_eta[0] = vol / sqrt(ax*ax + ay*ay + az*az + m_ptiny_cuda) ;

    ax *= norm ;
    ay *= norm ;
    az *= norm ;

    dxv = double(-0.25)*((p_xd0[0]+p_xd1[0]+p_xd5[0]+p_xd4[0]) - (p_xd3[0]+p_xd2[0]+p_xd6[0]+p_xd7[0])) ;
    dyv = double(-0.25)*((p_yd0[0]+p_yd1[0]+p_yd5[0]+p_yd4[0]) - (p_yd3[0]+p_yd2[0]+p_yd6[0]+p_yd7[0])) ;
    dzv = double(-0.25)*((p_zd0[0]+p_zd1[0]+p_zd5[0]+p_zd4[0]) - (p_zd3[0]+p_zd2[0]+p_zd6[0]+p_zd7[0])) ;

    delv_eta[0] = ax*dxv + ay*dyv + az*dzv ;


}

// CUDA kernel function
__global__ void op_cuda_CalcMonotonicQGradientsForElem(
  const double *__restrict ind_arg0,
  const double *__restrict ind_arg1,
  const double *__restrict ind_arg2,
  const double *__restrict ind_arg3,
  const double *__restrict ind_arg4,
  const double *__restrict ind_arg5,
  const int *__restrict opDat0Map,
  const double *__restrict arg48,
  const double *__restrict arg49,
  double *arg50,
  double *arg51,
  double *arg52,
  double *arg53,
  double *arg54,
  double *arg55,
  int start,
  int end,
  int   set_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid + start < end) {
    int n = tid + start;
    //initialise local variables
    int map0idx;
    int map1idx;
    int map2idx;
    int map3idx;
    int map4idx;
    int map5idx;
    int map6idx;
    int map7idx;
    map0idx = opDat0Map[n + set_size * 0];
    map1idx = opDat0Map[n + set_size * 1];
    map2idx = opDat0Map[n + set_size * 2];
    map3idx = opDat0Map[n + set_size * 3];
    map4idx = opDat0Map[n + set_size * 4];
    map5idx = opDat0Map[n + set_size * 5];
    map6idx = opDat0Map[n + set_size * 6];
    map7idx = opDat0Map[n + set_size * 7];

    //user-supplied kernel call
    CalcMonotonicQGradientsForElem_gpu(ind_arg0+map0idx*1,
                                   ind_arg0+map1idx*1,
                                   ind_arg0+map2idx*1,
                                   ind_arg0+map3idx*1,
                                   ind_arg0+map4idx*1,
                                   ind_arg0+map5idx*1,
                                   ind_arg0+map6idx*1,
                                   ind_arg0+map7idx*1,
                                   ind_arg1+map0idx*1,
                                   ind_arg1+map1idx*1,
                                   ind_arg1+map2idx*1,
                                   ind_arg1+map3idx*1,
                                   ind_arg1+map4idx*1,
                                   ind_arg1+map5idx*1,
                                   ind_arg1+map6idx*1,
                                   ind_arg1+map7idx*1,
                                   ind_arg2+map0idx*1,
                                   ind_arg2+map1idx*1,
                                   ind_arg2+map2idx*1,
                                   ind_arg2+map3idx*1,
                                   ind_arg2+map4idx*1,
                                   ind_arg2+map5idx*1,
                                   ind_arg2+map6idx*1,
                                   ind_arg2+map7idx*1,
                                   ind_arg3+map0idx*1,
                                   ind_arg3+map1idx*1,
                                   ind_arg3+map2idx*1,
                                   ind_arg3+map3idx*1,
                                   ind_arg3+map4idx*1,
                                   ind_arg3+map5idx*1,
                                   ind_arg3+map6idx*1,
                                   ind_arg3+map7idx*1,
                                   ind_arg4+map0idx*1,
                                   ind_arg4+map1idx*1,
                                   ind_arg4+map2idx*1,
                                   ind_arg4+map3idx*1,
                                   ind_arg4+map4idx*1,
                                   ind_arg4+map5idx*1,
                                   ind_arg4+map6idx*1,
                                   ind_arg4+map7idx*1,
                                   ind_arg5+map0idx*1,
                                   ind_arg5+map1idx*1,
                                   ind_arg5+map2idx*1,
                                   ind_arg5+map3idx*1,
                                   ind_arg5+map4idx*1,
                                   ind_arg5+map5idx*1,
                                   ind_arg5+map6idx*1,
                                   ind_arg5+map7idx*1,
                                   arg48+n*1,
                                   arg49+n*1,
                                   arg50+n*1,
                                   arg51+n*1,
                                   arg52+n*1,
                                   arg53+n*1,
                                   arg54+n*1,
                                   arg55+n*1);
  }
}


//host stub function
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


  int    ninds   = 6;
  int    inds[56] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,-1,-1,-1,-1,-1,-1,-1,-1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: CalcMonotonicQGradientsForElem\n");
  }
  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_14
      int nthread = OP_BLOCK_SIZE_14;
    #else
      int nthread = OP_block_size;
    #endif

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2);
      }
      int start = round==0 ? 0 : set->core_size;
      int end = round==0 ? set->core_size : set->size + set->exec_size;
      if (end-start>0) {
        int nblocks = (end-start-1)/nthread+1;
        op_cuda_CalcMonotonicQGradientsForElem<<<nblocks,nthread>>>(
        (double *)arg0.data_d,
        (double *)arg8.data_d,
        (double *)arg16.data_d,
        (double *)arg24.data_d,
        (double *)arg32.data_d,
        (double *)arg40.data_d,
        arg0.map_data_d,
        (double*)arg48.data_d,
        (double*)arg49.data_d,
        (double*)arg50.data_d,
        (double*)arg51.data_d,
        (double*)arg52.data_d,
        (double*)arg53.data_d,
        (double*)arg54.data_d,
        (double*)arg55.data_d,
        start,end,set->size+set->exec_size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[14].time     += wall_t2 - wall_t1;
}
