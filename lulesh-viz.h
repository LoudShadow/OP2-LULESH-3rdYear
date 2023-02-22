
#include <stdio.h>
#include <op_seq.h>

#include <mpi.h>

#if VIZ_MESH
extern "C" {
   #include "silo.h"
}
#endif

#if VIZ_MESH
int add_udc_mesh( DBfile *db,int g_numNode,int g_numElem,int m_numNode,op_dat p_x,op_dat p_y,op_dat p_z){
   
   int ok=0;
   const char* coordnames[3] = {"X", "Y", "Z"};

   double *coords[3] ;
   coords[0] = (double*)malloc( g_numNode * sizeof(double) );
   coords[1] = (double*)malloc( g_numNode * sizeof(double) );
   coords[2] = (double*)malloc( g_numNode * sizeof(double) );
   int* local_Coord_x = (int*) malloc(m_numNode * 8 * sizeof(int));
   int* local_Coord_y = (int*) malloc(m_numNode * 8 * sizeof(int));
   int* local_Coord_z = (int*) malloc(m_numNode * 8 * sizeof(int));
   op_fetch_data(p_x,local_Coord_x);
   op_fetch_data(p_y,local_Coord_y);
   op_fetch_data(p_z,local_Coord_z);
   MPI_Gather(local_Coord_x,m_numNode,MPI_DOUBLE,coords[0],g_numNode,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Gather(local_Coord_y,m_numNode,MPI_DOUBLE,coords[1],g_numNode,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Gather(local_Coord_z,m_numNode,MPI_DOUBLE,coords[2],g_numNode,MPI_DOUBLE,0,MPI_COMM_WORLD);

   ok += DBPutUcdmesh(db, "mesh", 3, (char**)&coordnames[0], (double**)coords,
                      g_numNode, g_numElem, "connectivity",
                      0, DB_DOUBLE, NULL);
   free(coords[0]);
   free(coords[1]);
   free(coords[2]);
   free(local_Coord_x);
   free(local_Coord_y);
   free(local_Coord_z);
   return ok;
}

int add_Zonelist2( DBfile *db,int myRank,int g_numElem,int m_numElem,int * nodeList){
   int shapeType[1] = {DB_ZONETYPE_HEX};
   int shapeSize[1] = {8};
   int shapeCnt[1] = {g_numElem};
   int ok=0;
   int* node_list_silo;
   if (myRank==0){
      node_list_silo = (int*) malloc(g_numElem * 8 * sizeof(int));
   }
   MPI_Gather(nodeList,m_numElem*8,MPI_INT,node_list_silo,g_numElem*8,MPI_INT,0,MPI_COMM_WORLD);
   if (myRank==0){
      int ok = DBPutZonelist2(db, "connectivity", g_numElem, 3,
                     node_list_silo,g_numElem*8,
                     0,0,0, /* Not carrying ghost zones */
                     shapeType, shapeSize, shapeCnt,
                     1, NULL);
      free(node_list_silo);
   }
   return ok;
}

int add_var_1(DBfile *db,int myRank,op_dat var,int flag){
   int g_length = op_get_size(var->set);
   int m_length = var->set->core_size;
   const char * name=var->name;

   double *g_var;
   int ok;
   if (myRank==0){
      g_var=(double*)malloc( g_length * sizeof(double) );
   }
   double *local_var = (double*) malloc(m_length * sizeof(double)); 
   op_fetch_data(var,local_var);
   MPI_Gather(local_var,m_length,MPI_DOUBLE,g_var,g_length,MPI_DOUBLE,0,MPI_COMM_WORLD);
   if (myRank==0){
      ok = DBPutUcdvar1(db, name, "mesh", g_var,
                      g_length, NULL, 0, DB_DOUBLE, flag,
                      NULL);
      free(g_var);
   }
   free(local_var);
   return ok;
}
int add_variables(DBfile *db,int myRank,int var_count,op_dat* var,int* flag){
   int ok=0;
   for (int i=0; i<var_count;i++){
      ok+=add_var_1(db,myRank,var[i],flag[i]);
   }
   return ok;
}
#endif

export void writeSiloFile(int myRank, int m_cycle,
   int g_numElem,
   int g_numNode,
   int m_numElem,
   int m_numNode,
   int * nodeList,
   op_dat p_x,
   op_dat p_y,
   op_dat p_z,

   op_dat p_e,
   op_dat p_p,
   op_dat p_v,
   op_dat p_q,
   op_dat p_xd,
   op_dat p_yd,
   op_dat p_zd,
   op_dat p_speed
   
   ){
   #if VIZ_MESH
   //SILO is done in serial only
   DBfile *db;
   char subdirName[32];
   char basename[32];
   sprintf(basename, "lulesh_plot_%d",m_cycle);
   db  = (DBfile*)DBCreate(basename, DB_CLOBBER, DB_LOCAL, NULL, DB_HDF5X);
   // DBMkDir(db, subdirName);
   DBSetDir(db, "/");
   int ok=0;
   ok +=add_Zonelist2(db,myRank,g_numElem,m_numElem,nodeList);
   ok +=add_udc_mesh(db,g_numNode,g_numElem,m_numNode,p_x,p_y,p_z);

   op_dat OP_vars[8]={p_e,p_p,p_v,p_q,p_xd,p_yd,p_zd,p_speed};
   int flags[8]={DB_ZONECENT,DB_ZONECENT,DB_ZONECENT,DB_ZONECENT,
            DB_NODECENT,DB_NODECENT,DB_NODECENT,DB_NODECENT};
   add_variables(db,myRank,8,OP_vars,flags);
   // ok +=add_var_1(db,myRank,p_e,DB_ZONECENT);
   // ok +=add_var_1(db,myRank,p_p,DB_ZONECENT);
   // ok +=add_var_1(db,myRank,p_v,DB_ZONECENT);
   // ok +=add_var_1(db,myRank,p_q,DB_ZONECENT);

   // ok +=add_var_1(db,myRank,p_xd,DB_NODECENT);
   // ok +=add_var_1(db,myRank,p_yd,DB_NODECENT);
   // ok +=add_var_1(db,myRank,p_zd,DB_NODECENT);
   // ok +=add_var_1(db,myRank,p_speed,DB_NODECENT);
   //ELEMETNS


   printf("OK value:%d\n",ok);
   DBClose(db);
   // free(nodelist_silo);
   #else
   printf("Must enable -DVIZ_MESH at compile time to vrite to visit\n");
   #endif
}