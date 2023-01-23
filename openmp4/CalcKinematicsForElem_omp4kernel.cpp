//
// auto-generated by op2.py
//

//user function
//user function

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
  int nthread);

// host stub function
void op_par_loop_CalcKinematicsForElem(char const *name, op_set set,
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
  op_arg arg55,
  op_arg arg56){

  double*arg56h = (double *)arg56.data;
  int nargs = 57;
  op_arg args[57];

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
  args[56] = arg56;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(12);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[12].name      = name;
  OP_kernels[12].count    += 1;

  int  ninds   = 6;
  int  inds[57] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,-1,-1,-1,-1,-1,-1,-1,-1,-1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: CalcKinematicsForElem\n");
  }

  // get plan
  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);

  #ifdef OP_PART_SIZE_12
    int part_size = OP_PART_SIZE_12;
  #else
    int part_size = OP_part_size;
  #endif
  #ifdef OP_BLOCK_SIZE_12
    int nthread = OP_BLOCK_SIZE_12;
  #else
    int nthread = OP_block_size;
  #endif

  double arg56_l = arg56h[0];

  int ncolors = 0;
  int set_size1 = set->size + set->exec_size;

  if (set_size >0) {

    //Set up typed device pointers for OpenMP
    int *map0 = arg0.map_data_d;
     int map0size = arg0.map->dim * set_size1;

    double* data48 = (double*)arg48.data_d;
    int dat48size = getSetSizeFromOpArg(&arg48) * arg48.dat->dim;
    double* data49 = (double*)arg49.data_d;
    int dat49size = getSetSizeFromOpArg(&arg49) * arg49.dat->dim;
    double* data50 = (double*)arg50.data_d;
    int dat50size = getSetSizeFromOpArg(&arg50) * arg50.dat->dim;
    double* data51 = (double*)arg51.data_d;
    int dat51size = getSetSizeFromOpArg(&arg51) * arg51.dat->dim;
    double* data52 = (double*)arg52.data_d;
    int dat52size = getSetSizeFromOpArg(&arg52) * arg52.dat->dim;
    double* data53 = (double*)arg53.data_d;
    int dat53size = getSetSizeFromOpArg(&arg53) * arg53.dat->dim;
    double* data54 = (double*)arg54.data_d;
    int dat54size = getSetSizeFromOpArg(&arg54) * arg54.dat->dim;
    double* data55 = (double*)arg55.data_d;
    int dat55size = getSetSizeFromOpArg(&arg55) * arg55.dat->dim;
    double *data0 = (double *)arg0.data_d;
    int dat0size = getSetSizeFromOpArg(&arg0) * arg0.dat->dim;
    double *data8 = (double *)arg8.data_d;
    int dat8size = getSetSizeFromOpArg(&arg8) * arg8.dat->dim;
    double *data16 = (double *)arg16.data_d;
    int dat16size = getSetSizeFromOpArg(&arg16) * arg16.dat->dim;
    double *data24 = (double *)arg24.data_d;
    int dat24size = getSetSizeFromOpArg(&arg24) * arg24.dat->dim;
    double *data32 = (double *)arg32.data_d;
    int dat32size = getSetSizeFromOpArg(&arg32) * arg32.dat->dim;
    double *data40 = (double *)arg40.data_d;
    int dat40size = getSetSizeFromOpArg(&arg40) * arg40.dat->dim;

    op_plan *Plan = op_plan_get_stage(name,set,part_size,nargs,args,ninds,inds,OP_COLOR2);
    ncolors = Plan->ncolors;
    int *col_reord = Plan->col_reord;

    // execute plan
    for ( int col=0; col<Plan->ncolors; col++ ){
      if (col==1) {
        op_mpi_wait_all_cuda(nargs, args);
      }
      int start = Plan->col_offsets[0][col];
      int end = Plan->col_offsets[0][col+1];

      CalcKinematicsForElem_omp4_kernel(
        map0,
        map0size,
        data48,
        dat48size,
        data49,
        dat49size,
        data50,
        dat50size,
        data51,
        dat51size,
        data52,
        dat52size,
        data53,
        dat53size,
        data54,
        dat54size,
        data55,
        dat55size,
        &arg56_l,
        data0,
        dat0size,
        data8,
        dat8size,
        data16,
        dat16size,
        data24,
        dat24size,
        data32,
        dat32size,
        data40,
        dat40size,
        col_reord,
        set_size1,
        start,
        end,
        part_size!=0?(end-start-1)/part_size+1:(end-start-1)/nthread,
        nthread);

    }
    OP_kernels[12].transfer  += Plan->transfer;
    OP_kernels[12].transfer2 += Plan->transfer2;
  }

  if (set_size == 0 || set_size == set->core_size || ncolors == 1) {
    op_mpi_wait_all_cuda(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit_cuda(nargs, args);

  if (OP_diags>1) deviceSync();
  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[12].time     += wall_t2 - wall_t1;
}
