//
// auto-generated by op2.py
//

//user function
#include "../IntegrateStressForElemsLoop.h"

// host stub function
void op_par_loop_IntegrateStressForElemsLoop(char const *name, op_set set,
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
  op_arg arg51){

  int nargs = 52;
  op_arg args[52];

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

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(1);
  op_timers_core(&cpu_t1, &wall_t1);

  if (OP_diags>2) {
    printf(" kernel routine with indirection: IntegrateStressForElemsLoop\n");
  }

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  if (set_size > 0) {

    for ( int n=0; n<set_size; n++ ){
      if (n<set->core_size && n>0 && n % OP_mpi_test_frequency == 0)
        op_mpi_test_all(nargs,args);
      if (n==set->core_size) {
        op_mpi_wait_all(nargs, args);
      }
      int map0idx;
      int map1idx;
      int map2idx;
      int map3idx;
      int map4idx;
      int map5idx;
      int map6idx;
      int map7idx;
      map0idx = arg0.map_data[n * arg0.map->dim + 0];
      map1idx = arg0.map_data[n * arg0.map->dim + 1];
      map2idx = arg0.map_data[n * arg0.map->dim + 2];
      map3idx = arg0.map_data[n * arg0.map->dim + 3];
      map4idx = arg0.map_data[n * arg0.map->dim + 4];
      map5idx = arg0.map_data[n * arg0.map->dim + 5];
      map6idx = arg0.map_data[n * arg0.map->dim + 6];
      map7idx = arg0.map_data[n * arg0.map->dim + 7];


      IntegrateStressForElemsLoop(
        &((double*)arg0.data)[1 * map0idx],
        &((double*)arg0.data)[1 * map1idx],
        &((double*)arg0.data)[1 * map2idx],
        &((double*)arg0.data)[1 * map3idx],
        &((double*)arg0.data)[1 * map4idx],
        &((double*)arg0.data)[1 * map5idx],
        &((double*)arg0.data)[1 * map6idx],
        &((double*)arg0.data)[1 * map7idx],
        &((double*)arg8.data)[1 * map0idx],
        &((double*)arg8.data)[1 * map1idx],
        &((double*)arg8.data)[1 * map2idx],
        &((double*)arg8.data)[1 * map3idx],
        &((double*)arg8.data)[1 * map4idx],
        &((double*)arg8.data)[1 * map5idx],
        &((double*)arg8.data)[1 * map6idx],
        &((double*)arg8.data)[1 * map7idx],
        &((double*)arg16.data)[1 * map0idx],
        &((double*)arg16.data)[1 * map1idx],
        &((double*)arg16.data)[1 * map2idx],
        &((double*)arg16.data)[1 * map3idx],
        &((double*)arg16.data)[1 * map4idx],
        &((double*)arg16.data)[1 * map5idx],
        &((double*)arg16.data)[1 * map6idx],
        &((double*)arg16.data)[1 * map7idx],
        &((double*)arg24.data)[1 * map0idx],
        &((double*)arg24.data)[1 * map1idx],
        &((double*)arg24.data)[1 * map2idx],
        &((double*)arg24.data)[1 * map3idx],
        &((double*)arg24.data)[1 * map4idx],
        &((double*)arg24.data)[1 * map5idx],
        &((double*)arg24.data)[1 * map6idx],
        &((double*)arg24.data)[1 * map7idx],
        &((double*)arg32.data)[1 * map0idx],
        &((double*)arg32.data)[1 * map1idx],
        &((double*)arg32.data)[1 * map2idx],
        &((double*)arg32.data)[1 * map3idx],
        &((double*)arg32.data)[1 * map4idx],
        &((double*)arg32.data)[1 * map5idx],
        &((double*)arg32.data)[1 * map6idx],
        &((double*)arg32.data)[1 * map7idx],
        &((double*)arg40.data)[1 * map0idx],
        &((double*)arg40.data)[1 * map1idx],
        &((double*)arg40.data)[1 * map2idx],
        &((double*)arg40.data)[1 * map3idx],
        &((double*)arg40.data)[1 * map4idx],
        &((double*)arg40.data)[1 * map5idx],
        &((double*)arg40.data)[1 * map6idx],
        &((double*)arg40.data)[1 * map7idx],
        &((double*)arg48.data)[1 * n],
        &((double*)arg49.data)[1 * n],
        &((double*)arg50.data)[1 * n],
        &((double*)arg51.data)[1 * n]);
    }
  }

  if (set_size == 0 || set_size == set->core_size) {
    op_mpi_wait_all(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[1].name      = name;
  OP_kernels[1].count    += 1;
  OP_kernels[1].time     += wall_t2 - wall_t1;
  OP_kernels[1].transfer += (float)set->size * arg0.size;
  OP_kernels[1].transfer += (float)set->size * arg8.size;
  OP_kernels[1].transfer += (float)set->size * arg16.size;
  OP_kernels[1].transfer += (float)set->size * arg24.size * 2.0f;
  OP_kernels[1].transfer += (float)set->size * arg32.size * 2.0f;
  OP_kernels[1].transfer += (float)set->size * arg40.size * 2.0f;
  OP_kernels[1].transfer += (float)set->size * arg48.size;
  OP_kernels[1].transfer += (float)set->size * arg49.size;
  OP_kernels[1].transfer += (float)set->size * arg50.size;
  OP_kernels[1].transfer += (float)set->size * arg51.size;
  OP_kernels[1].transfer += (float)set->size * arg0.map->dim * 4.0f;
}
