//
// auto-generated by op2.py
//

//user function
__device__ void CalcMonotonicQRegionForElem_gpu( 
    const double *delv_xi, const double *delv_xi_lxim, const double *delv_xi_lxip,
    const double *delv_eta, const double *delv_eta_letam, const double *delv_eta_letap,
    const double *delv_zeta, const double *delv_zeta_lzetam, const double *delv_zeta_lzetap,
    const double *delx_xi, const double *delx_eta, const double *delx_zeta,
    const int *elemBC,
    const double *m_vdov,
    double *qq, double *ql,
    const double *elemMass, const double *volo, const double *vnew
) {
    double qlin, qquad ;
    double phixi, phieta, phizeta ;
    double delvm = 0.0, delvp =0.0;
    int bcMask = elemBC[0] ;

    double norm = double(1.) / (delv_xi[0]+ m_ptiny_cuda ) ;

    switch (bcMask & XI_M) {
    case 0x00004:
    case 0:         delvm = delv_xi_lxim[0]; break ;
    case 0x00001: delvm = delv_xi[0] ;       break ;
    case 0x00002: delvm = double(0.0) ;      break ;
    default:
        delvm = 0;
        break;
    }
    switch (bcMask & 0x00038) {
    case 0x00020:
    case 0:         delvp = delv_xi_lxip[0] ; break ;
    case 0x00008: delvp = delv_xi[0] ;       break ;
    case 0x00010: delvp = double(0.0) ;      break ;
    default:
        delvp = 0;
        break;
    }

    delvm = delvm * norm ;
    delvp = delvp * norm ;

    phixi = double(.5) * ( delvm + delvp ) ;

    delvm *= m_monoq_limiter_mult_cuda ;
    delvp *= m_monoq_limiter_mult_cuda ;

    if ( delvm < phixi ) phixi = delvm ;
    if ( delvp < phixi ) phixi = delvp ;
    if ( phixi < double(0.)) phixi = double(0.) ;
    if ( phixi > m_monoq_max_slope_cuda) phixi = m_monoq_max_slope_cuda;

    norm = double(1.) / ( delv_eta[0] + m_ptiny_cuda ) ;

    switch (bcMask & 0x001c0) {
        case 0x00100:
        case 0:          delvm = delv_eta_letam[0] ; break ;
        case 0x00040: delvm = delv_eta[0] ;        break ;
        case 0x00080: delvm = double(0.0) ;        break ;
        default:
        delvm = 0;
        break;
    }
    switch (bcMask & 0x00e00) {
        case 0x00800:
        case 0:          delvp = delv_eta_letap[0] ; break ;
        case 0x00200: delvp = delv_eta[0] ;        break ;
        case 0x00400: delvp = double(0.0) ;        break ;
        default:
        delvp = 0;
        break;
    }

    delvm = delvm * norm ;
    delvp = delvp * norm ;

    phieta = double(.5) * ( delvm + delvp ) ;

    delvm *= m_monoq_limiter_mult_cuda ;
    delvp *= m_monoq_limiter_mult_cuda ;

    if ( delvm  < phieta ) phieta = delvm ;
    if ( delvp  < phieta ) phieta = delvp ;
    if ( phieta < double(0.)) phieta = double(0.) ;
    if ( phieta > m_monoq_max_slope_cuda)  phieta = m_monoq_max_slope_cuda;


    norm = double(1.) / ( delv_zeta[0] + m_ptiny_cuda ) ;

    switch (bcMask & 0x07000) {
        case 0x04000:
        case 0:           delvm = delv_zeta_lzetam[0] ; break ;
        case 0x01000: delvm = delv_zeta[0] ;         break ;
        case 0x02000: delvm = double(0.0) ;          break ;
        default:
        delvm = 0;
        break;
    }
    switch (bcMask & 0x38000) {
        case 0x20000:
        case 0:           delvp = delv_zeta_lzetap[0] ; break ;
        case 0x08000: delvp = delv_zeta[0] ;         break ;
        case 0x10000: delvp = double(0.0) ;          break ;
        default:
        delvp = 0;
        break;
    }

    delvm = delvm * norm ;
    delvp = delvp * norm ;

    phizeta = double(.5) * ( delvm + delvp ) ;

    delvm *= m_monoq_limiter_mult_cuda ;
    delvp *= m_monoq_limiter_mult_cuda ;

    if ( delvm   < phizeta ) phizeta = delvm ;
    if ( delvp   < phizeta ) phizeta = delvp ;
    if ( phizeta < double(0.)) phizeta = double(0.);
    if ( phizeta > m_monoq_max_slope_cuda  ) phizeta = m_monoq_max_slope_cuda;

    if ( m_vdov[0] > double(0.) )  {
        qlin  = double(0.) ;
        qquad = double(0.) ;
    }
    else {
        double delvxxi   = delv_xi[0]   * delx_xi[0]   ;
        double delvxeta  = delv_eta[0]  * delx_eta[0]  ;
        double delvxzeta = delv_zeta[0] * delx_zeta[0] ;

        if ( delvxxi   > double(0.) ) delvxxi   = double(0.) ;
        if ( delvxeta  > double(0.) ) delvxeta  = double(0.) ;
        if ( delvxzeta > double(0.) ) delvxzeta = double(0.) ;

        double rho = elemMass[0] / (volo[0] * vnew[0]) ;

        qlin = -m_qlc_monoq_cuda * rho *
        (  delvxxi   * (double(1.) - phixi) +
            delvxeta  * (double(1.) - phieta) +
            delvxzeta * (double(1.) - phizeta)  ) ;

        qquad = m_qqc_monoq_cuda * rho *
        (  delvxxi*delvxxi     * (double(1.) - phixi*phixi) +
            delvxeta*delvxeta   * (double(1.) - phieta*phieta) +
            delvxzeta*delvxzeta * (double(1.) - phizeta*phizeta)  ) ;
    }

    qq[0] = qquad ;
    ql[0] = qlin  ;


}

// CUDA kernel function
__global__ void op_cuda_CalcMonotonicQRegionForElem(
  const double *__restrict ind_arg0,
  const double *__restrict ind_arg1,
  const double *__restrict ind_arg2,
  const double *__restrict ind_arg3,
  const double *__restrict ind_arg4,
  const double *__restrict ind_arg5,
  const double *__restrict ind_arg6,
  const double *__restrict ind_arg7,
  const double *__restrict ind_arg8,
  const double *__restrict ind_arg9,
  const double *__restrict ind_arg10,
  const double *__restrict ind_arg11,
  const int *__restrict ind_arg12,
  const double *__restrict ind_arg13,
  double *__restrict ind_arg14,
  double *__restrict ind_arg15,
  const double *__restrict ind_arg16,
  const double *__restrict ind_arg17,
  const double *__restrict ind_arg18,
  const int *__restrict opDat0Map,
  const int *__restrict opDat1Map,
  const int *__restrict opDat2Map,
  const int *__restrict opDat4Map,
  const int *__restrict opDat5Map,
  const int *__restrict opDat7Map,
  const int *__restrict opDat8Map,
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
    int map4idx;
    int map5idx;
    int map7idx;
    int map8idx;
    map0idx = opDat0Map[n + set_size * 0];
    map1idx = opDat1Map[n + set_size * 0];
    map2idx = opDat2Map[n + set_size * 0];
    map4idx = opDat4Map[n + set_size * 0];
    map5idx = opDat5Map[n + set_size * 0];
    map7idx = opDat7Map[n + set_size * 0];
    map8idx = opDat8Map[n + set_size * 0];

    //user-supplied kernel call
    CalcMonotonicQRegionForElem_gpu(ind_arg0+map0idx*1,
                                ind_arg1+map1idx*1,
                                ind_arg2+map2idx*1,
                                ind_arg3+map0idx*1,
                                ind_arg4+map4idx*1,
                                ind_arg5+map5idx*1,
                                ind_arg6+map0idx*1,
                                ind_arg7+map7idx*1,
                                ind_arg8+map8idx*1,
                                ind_arg9+map0idx*1,
                                ind_arg10+map0idx*1,
                                ind_arg11+map0idx*1,
                                ind_arg12+map0idx*1,
                                ind_arg13+map0idx*1,
                                ind_arg14+map0idx*1,
                                ind_arg15+map0idx*1,
                                ind_arg16+map0idx*1,
                                ind_arg17+map0idx*1,
                                ind_arg18+map0idx*1);
  }
}


//host stub function
void op_par_loop_CalcMonotonicQRegionForElem(char const *name, op_set set,
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
  op_arg arg18){

  int nargs = 19;
  op_arg args[19];

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

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(15);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[15].name      = name;
  OP_kernels[15].count    += 1;


  int    ninds   = 19;
  int    inds[19] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: CalcMonotonicQRegionForElem\n");
  }
  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_15
      int nthread = OP_BLOCK_SIZE_15;
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
        op_cuda_CalcMonotonicQRegionForElem<<<nblocks,nthread>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        (double *)arg2.data_d,
        (double *)arg3.data_d,
        (double *)arg4.data_d,
        (double *)arg5.data_d,
        (double *)arg6.data_d,
        (double *)arg7.data_d,
        (double *)arg8.data_d,
        (double *)arg9.data_d,
        (double *)arg10.data_d,
        (double *)arg11.data_d,
        (int *)arg12.data_d,
        (double *)arg13.data_d,
        (double *)arg14.data_d,
        (double *)arg15.data_d,
        (double *)arg16.data_d,
        (double *)arg17.data_d,
        (double *)arg18.data_d,
        arg0.map_data_d,
        arg1.map_data_d,
        arg2.map_data_d,
        arg4.map_data_d,
        arg5.map_data_d,
        arg7.map_data_d,
        arg8.map_data_d,
        start,end,set->size+set->exec_size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[15].time     += wall_t2 - wall_t1;
}
