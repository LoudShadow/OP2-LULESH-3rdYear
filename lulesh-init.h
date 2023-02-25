
#include <op_seq.h>
#include <mpi.h>

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

#include "lulesh-util.h"

// Identifiers for free ndoes or boundary nodes
#define FREE_NODE 0x00
#define BOUNDARY_NODE 0x01

double *sigxx, *sigyy, *sigzz;
double *determ;

double* dvdx,*dvdy, *dvdz;
double* x8n, *y8n, *z8n;

double* vnewc;


//EOS Temp variables
double *e_old;
double *delvc;
double *p_old;
double *q_old;
double *compression;
double *compHalfStep;
double *qq_old;
double *ql_old;
double *work;
double *p_new;
double *e_new;
double *q_new;
double *bvc;
double *pbvc;

double *pHalfStep;

// Used in setup
int rowMin, rowMax;
int colMin, colMax;
int planeMin, planeMax ;

// Region information
int    m_numReg ;
int    m_cost; //imbalance cost
int *m_regElemSize ;   // Size of region sets
int *m_regNumList ;    // Region number per domain element
int **m_regElemlist ;  // region indexset 
int *m_regElemlist_2 ;  // region indexset 
int* localRegionSize;

int* m_region_i_to_lxim;
int* m_region_i_to_lxip;
int* m_region_i_to_letam;
int* m_region_i_to_letap;
int* m_region_i_to_lzetam;
int* m_region_i_to_lzetap;

int* nodelist;

int* lxim; /* element connectivity across each face */
int* lxip;
int* letam;
int* letap;
int* lzetam;
int* lzetap;

int local_total_region_size;

int* elemBC;            /* symmetry/free-surface flags for each elem face */

double* dxx,*dyy,*dzz ;  /* principal strains -- temporary */

double* delv_xi, *delv_eta, *delv_zeta ;    /* velocity gradient -- temporary */

double* delx_xi, *delx_eta, *delx_zeta ;    /* coordinate gradient -- temporary */

double* e;
double* p;

double* q;
double* ql;
double* qq;

double* v;     /* relative volume */        
double* vnew;  /* reference volume */
double* volo;  /* new relative volume -- temporary */
double* delv;  /* m_vnew - m_v */
double* m_vdov;  /* volume derivative over volume */

double* arealg; /* characteristic length of an element */

double* ss;

double* elemMass;

/* Node-centered */
double* x,*y,*z,*loc;                  /* coordinates */

double* xd,*yd,*zd;                 /* velocities */

double* xdd,*ydd,*zdd;              /* accelerations */

double* m_fx,*m_fy,*m_fz;                 /* forces */

double* nodalMass;          /* mass */

int* symmX,*symmY,*symmZ;              /* symmetry plane nodesets */

int* t_symmX,*t_symmY,*t_symmZ;

/* Other Constants */

// Variables to keep track of timestep, simulation time, and cycle
double  m_dtcourant ;         // courant constraint 
double  m_dthydro ;           // volume change constraint 
int   m_cycle ;             // iteration count for simulation 
double  m_dtfixed ;           // fixed time increment 
double  m_time ;              // current time 
double  m_deltatime ;         // variable time increment 
double  m_deltatimemultlb ;
double  m_deltatimemultub ;
double  m_dtmax ;             // maximum allowable time increment 
double  m_stoptime ;          // end time for simulation 

int sizeX ;
int sizeY ;
int sizeZ ;


//! global Data //
int g_sizeX ;
int g_sizeY ;
int g_sizeZ ;
int g_numElem ;
int g_numNode ;

// Region information
int *g_m_regElemSize ;   // Size of region sets
int *g_m_regNumList ;    // Region number per domain element
int **g_m_regElemlist ;  // region indexset 


int* g_nodelist;

int* g_lxim; /* element connectivity across each face */
int* g_lxip;
int* g_letam;
int* g_letap;
int* g_lzetam;
int* g_lzetap;

int* g_elemBC;            /* symmetry/free-surface flags for each elem face */

double* g_e;
double* g_p;

double* g_q;

double* g_v;     /* relative volume */        
double* g_volo;  /* new relative volume -- temporary */

double* g_ss;

double* g_elemMass;

/* Node-centered */
double* g_x,*g_y,*g_z,*g_loc;                  /* coordinates */

double* g_xd,*g_yd,*g_zd;                 /* velocities */

double* g_xdd,*g_ydd,*g_zdd;              /* accelerations */

double* g_nodalMass;          /* mass */

int* g_t_symmX,*g_t_symmY,*g_t_symmZ;

//! global Data end //

struct Domain{
   op_set nodes;
   op_set elems;
   op_set symmetry;
   op_set temp_vols;

   op_map p_nodelist;

   op_map p_symmX;
   op_map p_symmY;
   op_map p_symmZ;

   op_map p_lxim;
   op_map p_lxip;
   op_map p_letam;
   op_map p_letap;
   op_map p_lzetam;
   op_map p_lzetap;

   op_set* region_i;
   op_map* region_i_to_elems;
   op_map* region_i_to_lxim;
   op_map* region_i_to_lxip;
   op_map* region_i_to_letam;
   op_map* region_i_to_letap;
   op_map* region_i_to_lzetam;
   op_map* region_i_to_lzetap;

   op_dat p_x;
   op_dat p_y;
   op_dat p_z;

   op_dat p_fx;
   op_dat p_fy;
   op_dat p_fz;

   op_dat p_xd;
   op_dat p_yd;
   op_dat p_zd;

   op_dat p_xdd;
   op_dat p_ydd;
   op_dat p_zdd;

   op_dat p_qq;
   op_dat p_ql;
   op_dat p_qq_old;
   op_dat p_ql_old;

   op_dat p_e;
   op_dat p_e_old;
   op_dat p_e_new;
   op_dat p_elemMass;
   op_dat p_elemBC;

   op_dat p_p;
   op_dat p_p_old;
   op_dat p_p_new;

   op_dat p_q;
   op_dat p_q_old;
   op_dat p_q_new;

   op_dat p_v;
   op_dat p_volo;
   op_dat p_vnew;
   op_dat p_vdov;
   op_dat p_vnewc;

   op_dat p_delv;
   op_dat p_delv_xi ;    /* velocity gradient -- temporary */
   op_dat p_delv_eta ;
   op_dat p_delv_zeta ;
   op_dat p_delvc ;

   op_dat p_delx_xi ;    /* coordinate gradient -- temporary */
   op_dat p_delx_eta ;
   op_dat p_delx_zeta ;

   op_dat p_pHalfStep;
   op_dat p_pbvc;

   op_dat p_arealg;
   op_dat p_ss;

   op_dat p_nodalMass;

   op_dat p_sigxx;
   op_dat p_sigyy;
   op_dat p_sigzz;

   op_dat p_determ;

   op_dat p_dvdx;
   op_dat p_dvdy;
   op_dat p_dvdz;

   op_dat p_x8n;
   op_dat p_y8n;
   op_dat p_z8n;

   op_dat p_dxx;
   op_dat p_dyy;
   op_dat p_dzz;


   op_dat p_compression;
   op_dat p_compHalfStep;

   op_dat p_work;
   op_dat p_bvc;

   op_dat p_t_symmX;
   op_dat p_t_symmY;
   op_dat p_t_symmZ;

};
op_dat p_loc;
// ===================================================
// Allocate all standard Nodes
// ===================================================
static inline
void allocateElems(int m_numElem){

   nodelist = (int*) malloc(m_numElem*8 * sizeof(int));

   lxim = (int*) malloc(m_numElem * sizeof(int));
   lxip = (int*) malloc(m_numElem * sizeof(int));
   letam = (int*) malloc(m_numElem * sizeof(int));
   letap = (int*) malloc(m_numElem * sizeof(int));
   lzetam = (int*) malloc(m_numElem * sizeof(int));
   lzetap = (int*) malloc(m_numElem * sizeof(int));

   elemBC = (int*) malloc(m_numElem * sizeof(int));

   e = (double*) malloc(m_numElem * sizeof(double));
   p = (double*) malloc(m_numElem * sizeof(double));

   q = (double*) malloc(m_numElem * sizeof(double));
   ql = (double*) malloc(m_numElem * sizeof(double));
   qq = (double*) malloc(m_numElem * sizeof(double));

   v = (double*) malloc(m_numElem * sizeof(double));

   volo = (double*) malloc(m_numElem * sizeof(double));
   delv = (double*) malloc(m_numElem * sizeof(double));
   m_vdov = (double*) malloc(m_numElem * sizeof(double));

   arealg = (double*) malloc(m_numElem * sizeof(double));

   ss = (double*) malloc(m_numElem * sizeof(double));

   elemMass = (double*) malloc(m_numElem * sizeof(double));

   vnew = (double*) malloc(m_numElem * sizeof(double));
   vnewc = (double*) malloc(m_numElem * sizeof(double));

   // sigxx = (double*) malloc(m_numElem* 3 * sizeof(double));
   sigxx = (double*) malloc(m_numElem* sizeof(double));
   sigyy = (double*) malloc(m_numElem* sizeof(double));
   sigzz = (double*) malloc(m_numElem* sizeof(double));
   determ = (double*) malloc(m_numElem * sizeof(double));

   dvdx = (double*) malloc(m_numElem * 8 * sizeof(double));
   dvdy = (double*) malloc(m_numElem * 8 * sizeof(double));
   dvdz = (double*) malloc(m_numElem * 8 * sizeof(double));
   x8n = (double*) malloc(m_numElem * 8 * sizeof(double));
   y8n = (double*) malloc(m_numElem * 8 * sizeof(double));
   z8n = (double*) malloc(m_numElem * 8 * sizeof(double));

   dxx = (double*) malloc(m_numElem * sizeof(double));
   dyy = (double*) malloc(m_numElem * sizeof(double));
   dzz = (double*) malloc(m_numElem * sizeof(double));

   delx_xi = (double*) malloc(m_numElem * sizeof(double));
   delx_eta = (double*) malloc(m_numElem * sizeof(double));
   delx_zeta = (double*) malloc(m_numElem * sizeof(double));

   delv_xi = (double*) malloc(m_numElem * sizeof(double));
   delv_eta = (double*) malloc(m_numElem * sizeof(double));
   delv_zeta = (double*) malloc(m_numElem * sizeof(double));

   //EOS Temp Vars
   e_old = (double*) malloc(m_numElem *sizeof(double));
   delvc = (double*) malloc(m_numElem *sizeof(double));
   p_old = (double*) malloc(m_numElem *sizeof(double));
   q_old = (double*) malloc(m_numElem *sizeof(double));
   compression = (double*) malloc(m_numElem *sizeof(double));
   compHalfStep = (double*) malloc(m_numElem *sizeof(double));
   qq_old = (double*) malloc(m_numElem *sizeof(double));
   ql_old = (double*) malloc(m_numElem *sizeof(double));
   work = (double*) malloc(m_numElem *sizeof(double));
   p_new = (double*) malloc(m_numElem *sizeof(double));
   e_new = (double*) malloc(m_numElem *sizeof(double));
   q_new = (double*) malloc(m_numElem *sizeof(double));
   bvc = (double*) malloc(m_numElem *sizeof(double));
   pbvc = (double*) malloc(m_numElem *sizeof(double));
   pHalfStep = (double*) malloc(m_numElem *sizeof(double));
};

static inline
void allocateNodes(int m_numNode){

   x = (double*) malloc(m_numNode * sizeof(double)); // Coordinates
   y = (double*) malloc(m_numNode * sizeof(double));
   z = (double*) malloc(m_numNode * sizeof(double));
   loc = (double*) malloc(3 * m_numNode * sizeof(double)); // Coordinates


   xd = (double*) malloc(m_numNode * sizeof(double)); // Velocities
   yd = (double*) malloc(m_numNode * sizeof(double));
   zd = (double*) malloc(m_numNode * sizeof(double));

   xdd = (double*) malloc(m_numNode * sizeof(double)); //Accelerations
   // xdd = (double*) malloc(m_numNode * 3 * sizeof(double)); //Accelerations
   ydd = (double*) malloc(m_numNode * sizeof(double));
   zdd = (double*) malloc(m_numNode * sizeof(double));

   m_fx = (double*) malloc(m_numNode * sizeof(double)); // Forces
   // m_fx = (double*) malloc(m_numNode * 3 * sizeof(double)); // Forces
   m_fy = (double*) malloc(m_numNode * sizeof(double));
   m_fz = (double*) malloc(m_numNode * sizeof(double));

   t_symmX = (int*) malloc(m_numNode * sizeof(int));
   t_symmY = (int*) malloc(m_numNode * sizeof(int));
   t_symmZ = (int*) malloc(m_numNode * sizeof(int));

   nodalMass = (double*) malloc(m_numNode * sizeof(double));  // mass
}

static inline
void allocateVars(int m_numElem,int m_numNodes){
   allocateElems(m_numElem);
   allocateNodes(m_numNodes);
   m_regElemSize = (int*) malloc(m_numReg * sizeof(int));
}
// ===================================================
// Allocate all global Nodes
// ===================================================
static inline 
void allocateGlobalElems(int g_numElem){
   g_nodelist = (int*) malloc(g_numElem*8 * sizeof(int));

   g_lxim = (int*) malloc(g_numElem * sizeof(int));
   g_lxip = (int*) malloc(g_numElem * sizeof(int));
   g_letam = (int*) malloc(g_numElem * sizeof(int));
   g_letap = (int*) malloc(g_numElem * sizeof(int));
   g_lzetam = (int*) malloc(g_numElem * sizeof(int));
   g_lzetap = (int*) malloc(g_numElem * sizeof(int));

   g_elemBC = (int*) malloc(g_numElem * sizeof(int));

   g_e = (double*) malloc(g_numElem * sizeof(double));
   g_p = (double*) malloc(g_numElem * sizeof(double));

   g_q = (double*) malloc(g_numElem * sizeof(double));

   g_v = (double*) malloc(g_numElem * sizeof(double));

   g_volo = (double*) malloc(g_numElem * sizeof(double));

   g_ss = (double*) malloc(g_numElem * sizeof(double));

   g_elemMass = (double*) malloc(g_numElem * sizeof(double));
};

static inline
void allocateGlobalNodes(int g_numNode){

   g_x = (double*) malloc(g_numNode * sizeof(double)); // Coordinates
   g_y = (double*) malloc(g_numNode * sizeof(double));
   g_z = (double*) malloc(g_numNode * sizeof(double));
   g_loc = (double*) malloc(3 * g_numNode * sizeof(double));

   g_xd = (double*) malloc(g_numNode * sizeof(double)); // Velocities
   g_yd = (double*) malloc(g_numNode * sizeof(double));
   g_zd = (double*) malloc(g_numNode * sizeof(double));

   g_xdd = (double*) malloc(g_numNode * sizeof(double)); //Accelerations
   g_ydd = (double*) malloc(g_numNode * sizeof(double));
   g_zdd = (double*) malloc(g_numNode * sizeof(double));

   g_t_symmX = (int*) malloc(g_numNode * sizeof(int));
   g_t_symmY = (int*) malloc(g_numNode * sizeof(int));
   g_t_symmZ = (int*) malloc(g_numNode * sizeof(int));

   g_nodalMass = (double*) malloc(g_numNode * sizeof(double));  // mass
}

static inline
void allocateGlobalVars(int g_numElem,int g_numNodes){
  allocateGlobalElems(g_numElem);
  allocateGlobalNodes(g_numNodes);
}

static inline
void Distribute_sub_mapping(int myRank,int* regionSize,int **regions,int **local_mapping,int tag){   
   int maxSize=0;
   for (int i=0; i<m_numReg;i++){
      if (m_regElemSize[i]>maxSize){
         maxSize=m_regElemSize[i];
      }
   }
}

static inline 
int Convert_to_singular(int numReg,int** regions,int* regElemSize, int to_rank, int comm_size,int g_numElem,int tag,int **localValues){
   printf("HERE 1.2.2\n");
   int *localRegionSize =(int*)calloc(numReg,sizeof(int));
   int *localValues_Local=(int*)malloc(sizeof(int)*g_numElem);

   int start=0;
   for (int i = 0; i < to_rank; i++)
   {
      start+=compute_local_size(g_numElem,comm_size,i);
   }
   int end=start+compute_local_size(g_numElem,comm_size,to_rank);

   printf("HERE 1.2.3\n");
   int index=0;
   int totalSize=0;
   for(int r=0; r<numReg;r++){
      for(int e=0; e<regElemSize[r];e++){
         if (regions[r][e]>=start && regions[r][e]<end){
            localValues_Local[index++]  = regions[r][e];
            localRegionSize[r]++;
            totalSize++;
         }
      }
   }
   localValues[0]=localValues_Local;
   printf("SENDING %d size: %d to Rank: %d : %d %d\n",tag,numReg,to_rank,localRegionSize[0],localRegionSize[1]);
   MPI_Send(localRegionSize,numReg,MPI_INT,to_rank,tag,MPI_COMM_WORLD);
   printf("SENT TOTAL_SIZE: %d\n",totalSize);
   printf("localRegionSizes %d %d\n",localRegionSize[0],localRegionSize[1]);
   free(localRegionSize);
   localRegionSize=NULL;
   return totalSize;
}

static inline 
void map_and_send(int* the_map, int* values,int length,int rank,int tag){
   int *mapped_values=(int*)malloc(sizeof(int)*length);

   for (int i = 0; i < length; i++)
   {
      mapped_values[i]=the_map[values[i]];
   }
   MPI_Send(mapped_values,length,MPI_INT,rank,tag,MPI_COMM_WORLD);
   free(mapped_values);
   mapped_values=NULL;
}

static inline
void distrbute_Region_Information_seq(int myRank,int comm_size,int g_numElem){
   MPI_Barrier(MPI_COMM_WORLD);

   if (myRank==0){
      for (int j = 0; j < m_numReg; j++)
      {
         printf("START============================\n");
         for (int i = 0; i < g_m_regElemSize[j]; i++)
         {
            printf("%d\n",g_m_regElemlist[j][i]);
         }
         printf("END============================\n");

      }
   }
   MPI_Barrier(MPI_COMM_WORLD);
   printf("TOGETHER ALLL\n");
   MPI_Barrier(MPI_COMM_WORLD);
   
   



   int total_local_size;
   if (myRank==0){
      int start=0;
      int end=compute_local_size(g_numElem,comm_size,0);
      m_regElemlist_2 = (int*) malloc(g_numElem * sizeof(int));
      int index=0;
      local_total_region_size=0;
      for(int r=0; r<m_numReg;r++){
         m_regElemSize[r]=0;
         for(int e=0; e<g_m_regElemSize[r];e++){
            if (g_m_regElemlist[r][e]>=start && g_m_regElemlist[r][e]<end){
               m_regElemlist_2[index++]  = g_m_regElemlist[r][e];
               m_regElemSize[r]++;
               local_total_region_size++;
            }
         }
      }



      for (int rank=1; rank<comm_size;rank++){
         int *tmp;
         int **localValues= &tmp;
         int size=Convert_to_singular(m_numReg,g_m_regElemlist,g_m_regElemSize,rank,comm_size,g_numElem,0,localValues);
         MPI_Send(*localValues,size,MPI_INT,rank,1,MPI_COMM_WORLD);
         map_and_send(g_lxim,*localValues,size,rank,2);
         map_and_send(g_lxip,*localValues,size,rank,3);
         map_and_send(g_letam,*localValues,size,rank,4);
         map_and_send(g_letap,*localValues,size,rank,5);
         map_and_send(g_lzetam,*localValues,size,rank,6);
         map_and_send(g_lzetap,*localValues,size,rank,7);
         free(*localValues);
      }
   }else{

      MPI_Recv(m_regElemSize,m_numReg,MPI_INT,0,0,MPI_COMM_WORLD,NULL);
      local_total_region_size=0;
      for (int i = 0; i < m_numReg; i++)
      {
         local_total_region_size+=m_regElemSize[i];
      }
      m_regElemlist_2 = (int*)malloc(sizeof(int) * local_total_region_size);
   }
   
   m_region_i_to_lxim=(int*)malloc(sizeof(int)*local_total_region_size);
   m_region_i_to_lxip=(int*)malloc(sizeof(int)*local_total_region_size);
   m_region_i_to_letam=(int*)malloc(sizeof(int)*local_total_region_size);
   m_region_i_to_letap=(int*)malloc(sizeof(int)*local_total_region_size);
   m_region_i_to_lzetam=(int*)malloc(sizeof(int)*local_total_region_size);
   m_region_i_to_lzetap=(int*)malloc(sizeof(int)*local_total_region_size);
   if (myRank!=0){
      MPI_Recv(m_regElemlist_2,local_total_region_size,MPI_INT,0,1,MPI_COMM_WORLD,NULL);
      for (int i = 0; i < local_total_region_size; i++)
      {
         printf("OUT:%d %d\n",myRank,i,m_regElemlist_2[i]);
      }
      
      MPI_Recv(m_region_i_to_lxim,local_total_region_size,MPI_INT,0,2,MPI_COMM_WORLD,NULL);
      MPI_Recv(m_region_i_to_lxip,local_total_region_size,MPI_INT,0,3,MPI_COMM_WORLD,NULL);
      MPI_Recv(m_region_i_to_letam,local_total_region_size,MPI_INT,0,4,MPI_COMM_WORLD,NULL);
      MPI_Recv(m_region_i_to_letap,local_total_region_size,MPI_INT,0,5,MPI_COMM_WORLD,NULL);
      MPI_Recv(m_region_i_to_lzetam,local_total_region_size,MPI_INT,0,6,MPI_COMM_WORLD,NULL);
      MPI_Recv(m_region_i_to_lzetap,local_total_region_size,MPI_INT,0,7,MPI_COMM_WORLD,NULL);
   }else{
      printf("HERE 1.4.6\n");
      for (int i = 0; i < local_total_region_size; i++){m_region_i_to_lxim[i]=g_lxim[m_regElemlist_2[i]];}
      for (int i = 0; i < local_total_region_size; i++){m_region_i_to_lxip[i]=g_lxip[m_regElemlist_2[i]];}
      for (int i = 0; i < local_total_region_size; i++){m_region_i_to_letam[i]=g_letam[m_regElemlist_2[i]];}
      for (int i = 0; i < local_total_region_size; i++){m_region_i_to_letap[i]=g_letap[m_regElemlist_2[i]];}
      for (int i = 0; i < local_total_region_size; i++){m_region_i_to_lzetam[i]=g_lzetam[m_regElemlist_2[i]];}
      for (int i = 0; i < local_total_region_size; i++){m_region_i_to_lzetap[i]=g_lzetap[m_regElemlist_2[i]];}

   }
   printf("%d REACHED BARRIER %p\n",myRank,MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   printf("%d PASSED BARRIER\n",myRank);

}

static inline
void distributeGlobalElems(int myRank,int comm_size, int g_numElem , int numElem, int g_numNode, int numNode){
   distrbute_Region_Information_seq(myRank,comm_size,g_numElem);

  //Element Information

   // g_nodelist = (int*) malloc(g_numElem*8 * sizeof(int));
   // g_lxim = (int*) malloc(g_numElem * sizeof(int));
   // g_lxip = (int*) malloc(g_numElem * sizeof(int));
   // g_letam = (int*) malloc(g_numElem * sizeof(int));
   // g_letap = (int*) malloc(g_numElem * sizeof(int));
   // g_lzetam = (int*) malloc(g_numElem * sizeof(int));
   // g_lzetap = (int*) malloc(g_numElem * sizeof(int));
   // g_elemBC = (int*) malloc(g_numElem * sizeof(int));
   scatter_int_array(g_nodelist,nodelist,comm_size,g_numElem,numElem,8);
   scatter_int_array(g_lxim,lxim,comm_size,g_numElem,numElem,1);
   scatter_int_array(g_lxip,lxip,comm_size,g_numElem,numElem,1);
   scatter_int_array(g_letam,letam,comm_size,g_numElem,numElem,1);
   scatter_int_array(g_letap,letap,comm_size,g_numElem,numElem,1);
   scatter_int_array(g_lzetam,lzetam,comm_size,g_numElem,numElem,1);
   scatter_int_array(g_lzetap,lzetap,comm_size,g_numElem,numElem,1);
   scatter_int_array(g_elemBC,elemBC,comm_size,g_numElem,numElem,1);
   free(g_nodelist);
   free(g_lxim);
   free(g_lxip);
   free(g_letam);
   free(g_letap);
   free(g_lzetam);
   free(g_lzetap);
   free(g_elemBC);

   // g_e = (double*) malloc(g_numElem * sizeof(double));
   // g_p = (double*) malloc(g_numElem * sizeof(double));
   // g_q = (double*) malloc(g_numElem * sizeof(double));
   // g_v = (double*) malloc(g_numElem * sizeof(double));
   // g_volo = (double*) malloc(g_numElem * sizeof(double));
   // g_ss = (double*) malloc(g_numElem * sizeof(double));
   // g_elemMass = (double*) malloc(g_numElem * sizeof(double));
   scatter_double_array(g_e,e,comm_size,g_numElem,numElem,1);
   scatter_double_array(g_p,p,comm_size,g_numElem,numElem,1);
   scatter_double_array(g_q,q,comm_size,g_numElem,numElem,1);
   scatter_double_array(g_v,v,comm_size,g_numElem,numElem,1);
   scatter_double_array(g_volo,volo,comm_size,g_numElem,numElem,1);
   scatter_double_array(g_ss,ss,comm_size,g_numElem,numElem,1);
   scatter_double_array(g_elemMass,elemMass,comm_size,g_numElem,numElem,1);
   free(g_e);
   free(g_p);
   free(g_q);
   free(g_v);
   free(g_volo);
   free(g_ss);
   free(g_elemMass);

   //Node Information
   // g_t_symmX = (int*) malloc(g_numNode * sizeof(int));
   // g_t_symmY = (int*) malloc(g_numNode * sizeof(int));
   // g_t_symmZ = (int*) malloc(g_numNode * sizeof(int));

   scatter_int_array(g_t_symmX,t_symmX,comm_size,g_numNode,numNode,1);
   scatter_int_array(g_t_symmY,t_symmY,comm_size,g_numNode,numNode,1);
   scatter_int_array(g_t_symmZ,t_symmZ,comm_size,g_numNode,numNode,1);
   free(g_t_symmX);
   free(g_t_symmY);
   free(g_t_symmZ);

   // g_x = (double*) malloc(g_numNode * sizeof(double)); // Coordinates
   // g_y = (double*) malloc(g_numNode * sizeof(double));
   // g_z = (double*) malloc(g_numNode * sizeof(double));

   // g_xd = (double*) malloc(g_numNode * sizeof(double)); // Velocities
   // g_yd = (double*) malloc(g_numNode * sizeof(double));
   // g_zd = (double*) malloc(g_numNode * sizeof(double));

   // g_xdd = (double*) malloc(g_numNode * sizeof(double)); //Accelerations
   // g_ydd = (double*) malloc(g_numNode * sizeof(double));
   // g_zdd = (double*) malloc(g_numNode * sizeof(double));

   // g_nodalMass = (double*) malloc(g_numNode * sizeof(double));  // mass
   scatter_double_array(g_x,x,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_y,y,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_z,z,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_loc,loc,comm_size,g_numNode,numNode,3);
   free(g_x);
   free(g_y);
   free(g_z);
   free(g_loc);

   scatter_double_array(g_xd,xd,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_yd,yd,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_zd,zd,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_xdd,xdd,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_ydd,ydd,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_zdd,zdd,comm_size,g_numNode,numNode,1);
   scatter_double_array(g_nodalMass,nodalMass,comm_size,g_numNode,numNode,1);
   free(g_xd);
   free(g_yd);
   free(g_zd);
   free(g_xdd);
   free(g_ydd);
   free(g_zdd);
   free(g_nodalMass);
}

// ===================================================
// Init OP2 Vars
// ===================================================
static inline
Domain initOp2Vars(int myRank,int m_numElem,int m_numNode){
   struct Domain domain;
   domain.nodes = op_decl_set(m_numNode, "nodes");
   domain.elems = op_decl_set(m_numElem, "elems");
   // temp_vols = op_decl_set(m_numElem*8, "tempVols");

   domain.p_nodelist = op_decl_map(domain.elems, domain.nodes, 8, nodelist, "nodelist");
   
   domain.p_lxim = op_decl_map(domain.elems, domain.elems, 1, lxim, "lxim");
   domain.p_lxip = op_decl_map(domain.elems, domain.elems, 1, lxip, "lxip");
   domain.p_letam = op_decl_map(domain.elems, domain.elems, 1, letam, "letam");
   domain.p_letap = op_decl_map(domain.elems, domain.elems, 1, letap, "letap");
   domain.p_lzetam = op_decl_map(domain.elems, domain.elems, 1, lzetam, "lzetam");
   domain.p_lzetap = op_decl_map(domain.elems, domain.elems, 1, lzetap, "lzetap");

   domain.region_i = (op_set*)malloc(m_numReg* sizeof(op_set));
   domain.region_i_to_elems = (op_map*)malloc(m_numReg* sizeof(op_map));

   domain.region_i_to_lxim = (op_map*)malloc(m_numReg* sizeof(op_map));
   domain.region_i_to_lxip = (op_map*)malloc(m_numReg* sizeof(op_map));
   domain.region_i_to_letam = (op_map*)malloc(m_numReg* sizeof(op_map));
   domain.region_i_to_letap = (op_map*)malloc(m_numReg* sizeof(op_map));
   domain.region_i_to_lzetam = (op_map*)malloc(m_numReg* sizeof(op_map));
   domain.region_i_to_lzetap = (op_map*)malloc(m_numReg* sizeof(op_map));



   char regionName[20];
   char mapName[20];
   int offset=0;
   printf("STARTING maping\n");
   printf("Sizes at on rank: %d with sizes:",myRank);
   for (int i = 0; i < m_numReg; i++){
      printf(" %d ",m_regElemSize[i]);
      // if (m_regElemSize[i]==0){
      //    continue;
      // }

      sprintf(regionName,"region_%d",i);
      domain.region_i[i] = op_decl_set(m_regElemSize[i], regionName);

      sprintf(mapName,"region_%d_to_elems",i);
      domain.region_i_to_elems[i] = op_decl_map(domain.region_i[i],domain.elems,1,&(m_regElemlist_2[offset]),mapName);

      sprintf(mapName,"region_%d_to_lxim",i);
      domain.region_i_to_lxim[i] = op_decl_map(domain.region_i[i],domain.elems,1,&(m_region_i_to_lxim[offset]),mapName);
      sprintf(mapName,"region_%d_to_lxip",i);
      domain.region_i_to_lxip[i] = op_decl_map(domain.region_i[i],domain.elems,1,&(m_region_i_to_lxip[offset]),mapName);
      sprintf(mapName,"region_%d_to_letam",i);
      domain.region_i_to_letam[i] = op_decl_map(domain.region_i[i],domain.elems,1,&(m_region_i_to_letam[offset]),mapName);
      sprintf(mapName,"region_%d_to_letap",i);
      domain.region_i_to_letap[i] = op_decl_map(domain.region_i[i],domain.elems,1,&(m_region_i_to_letap[offset]),mapName);
      sprintf(mapName,"region_%d_to_lzetam",i);
      domain.region_i_to_lzetam[i] = op_decl_map(domain.region_i[i],domain.elems,1,&(m_region_i_to_lzetam[offset]),mapName);
      sprintf(mapName,"region_%d_to_lzetap",i);
      domain.region_i_to_lzetap[i] = op_decl_map(domain.region_i[i],domain.elems,1,&(m_region_i_to_lzetap[offset]),mapName);

      offset+=m_regElemSize[i];
   }
   printf("\n");
   printf("ENDING maping\n");

   printf("INIT Done\n");


   //Node Centred
   domain.p_t_symmX = op_decl_dat(domain.nodes, 1, "int", t_symmX, "t_symmX");
   domain.p_t_symmY = op_decl_dat(domain.nodes, 1, "int", t_symmY, "t_symmY");
   domain.p_t_symmZ = op_decl_dat(domain.nodes, 1, "int", t_symmZ, "t_symmZ");

   domain.p_x =op_decl_dat(domain.nodes, 1, "double", x, "p_x");
   domain.p_y =op_decl_dat(domain.nodes, 1, "double", y, "p_y");
   domain.p_z =op_decl_dat(domain.nodes, 1, "double", z, "p_z");
   p_loc = op_decl_dat(domain.nodes, 3, "double", loc, "p_loc");
   domain.p_xd = op_decl_dat(domain.nodes, 1, "double", xd, "p_xd");
   domain.p_yd = op_decl_dat(domain.nodes, 1, "double", yd, "p_yd");
   domain.p_zd = op_decl_dat(domain.nodes, 1, "double", zd, "p_zd");
   domain.p_xdd = op_decl_dat(domain.nodes, 1, "double", xdd, "p_xdd");
   domain.p_ydd = op_decl_dat(domain.nodes, 1, "double", ydd, "p_ydd");
   domain.p_zdd = op_decl_dat(domain.nodes, 1, "double", zdd, "p_zdd");
   domain.p_fx = op_decl_dat(domain.nodes, 1, "double", m_fx, "p_fx");
   domain.p_fy = op_decl_dat(domain.nodes, 1, "double", m_fy, "p_fy");
   domain.p_fz = op_decl_dat(domain.nodes, 1, "double", m_fz, "p_fz");

   domain.p_nodalMass = op_decl_dat(domain.nodes, 1, "double", nodalMass, "p_nodalMass");
   //Elem Centred
   domain.p_e = op_decl_dat(domain.elems, 1, "double", e, "p_e");
   domain.p_p = op_decl_dat(domain.elems, 1, "double", p, "p_p");
   domain.p_q = op_decl_dat(domain.elems, 1, "double", q, "p_q");
   domain.p_ql = op_decl_dat(domain.elems, 1, "double", ql, "p_ql");
   domain.p_qq = op_decl_dat(domain.elems, 1, "double", qq, "p_qq");
   domain.p_v = op_decl_dat(domain.elems, 1, "double", v, "p_v");
   domain.p_volo = op_decl_dat(domain.elems, 1, "double", volo, "p_volo");
   domain.p_delv = op_decl_dat(domain.elems, 1, "double", delv, "p_delv");
   domain.p_vdov = op_decl_dat(domain.elems, 1, "double", m_vdov, "p_vdov");
   domain.p_arealg = op_decl_dat(domain.elems, 1, "double", arealg, "p_arealg");

   domain.p_dxx = op_decl_dat(domain.elems, 1, "double", dxx, "p_dxx");
   domain.p_dyy = op_decl_dat(domain.elems, 1, "double", dyy, "p_dyy");
   domain.p_dzz = op_decl_dat(domain.elems, 1, "double", dzz, "p_dzz");
   
   domain.p_ss = op_decl_dat(domain.elems, 1, "double", ss, "p_ss");
   domain.p_elemMass = op_decl_dat(domain.elems, 1, "double", elemMass, "p_elemMass");
   domain.p_vnew = op_decl_dat(domain.elems, 1, "double", vnew, "p_vnew");
   domain.p_vnewc = op_decl_dat(domain.elems, 1, "double", vnewc, "p_vnewc");

   //Temporary
   domain.p_sigxx = op_decl_dat(domain.elems, 1, "double", sigxx, "p_sigxx");
   domain.p_sigyy = op_decl_dat(domain.elems, 1, "double", sigyy, "p_sigyy");
   domain.p_sigzz = op_decl_dat(domain.elems, 1, "double", sigzz, "p_sigzz");
   domain.p_determ = op_decl_dat(domain.elems, 1, "double", determ, "p_determ");

   domain.p_dvdx = op_decl_dat(domain.elems, 8, "double",dvdx, "dvdx");
   domain.p_dvdy = op_decl_dat(domain.elems, 8, "double",dvdy, "dvdy");
   domain.p_dvdz = op_decl_dat(domain.elems, 8, "double",dvdz, "dvdz");
   domain.p_x8n = op_decl_dat(domain.elems, 8, "double",x8n, "x8n");
   domain.p_y8n = op_decl_dat(domain.elems, 8, "double",y8n, "y8n");
   domain.p_z8n = op_decl_dat(domain.elems, 8, "double",z8n, "z8n");

   domain.p_delv_xi = op_decl_dat(domain.elems, 1, "double", delv_xi, "p_delv_xi"); 
   domain.p_delv_eta = op_decl_dat(domain.elems, 1, "double", delv_eta, "p_delv_eta"); 
   domain.p_delv_zeta = op_decl_dat(domain.elems, 1, "double", delv_zeta, "p_delv_zeta"); 

   domain.p_delx_xi = op_decl_dat(domain.elems, 1, "double", delx_xi, "p_delx_xi"); 
   domain.p_delx_eta = op_decl_dat(domain.elems, 1, "double", delx_eta, "p_delx_eta"); 
   domain.p_delx_zeta = op_decl_dat(domain.elems, 1, "double", delx_zeta, "p_delx_zeta"); 

   domain.p_elemBC = op_decl_dat(domain.elems, 1, "int", elemBC, "p_elemBC");

   //EOS temp variables
   domain.p_e_old = op_decl_dat(domain.elems, 1, "double", e_old, "e_old"); 
   domain.p_delvc = op_decl_dat(domain.elems, 1, "double", delvc, "delvc");
   domain.p_p_old = op_decl_dat(domain.elems, 1, "double", p_old, "p_old");
   domain.p_q_old = op_decl_dat(domain.elems, 1, "double", q_old, "q_old");
   domain.p_compression = op_decl_dat(domain.elems, 1, "double", compression, "compression");
   domain.p_compHalfStep = op_decl_dat(domain.elems, 1, "double", compHalfStep, "compHalfStep");
   domain.p_qq_old = op_decl_dat(domain.elems, 1, "double", qq_old, "qq_old");
   domain.p_ql_old = op_decl_dat(domain.elems, 1, "double", ql_old, "ql_old");
   domain.p_work = op_decl_dat(domain.elems, 1, "double", work, "work");
   domain.p_p_new = op_decl_dat(domain.elems, 1, "double", p_new, "p_new");
   domain.p_e_new = op_decl_dat(domain.elems, 1, "double", e_new, "e_new");
   domain.p_q_new = op_decl_dat(domain.elems, 1, "double", q_new, "q_new");
   domain.p_bvc = op_decl_dat(domain.elems, 1, "double", bvc, "bvc"); ;
   domain.p_pbvc = op_decl_dat(domain.elems, 1, "double", pbvc, "pbvc");
   domain.p_pHalfStep = op_decl_dat(domain.elems, 1, "double", pHalfStep, "pHalfStep");

   return domain;
}




static inline
Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
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

  volume *= twelveth;

  return volume ;
}

Real_t CalcElemVolume_Vec( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

static inline
Real_t volumeAt(int elemNumber,int meshEdgeElems,int edgeNodes,int edgeElems){
   int curr_z,curr_y,curr_x;

   get_loc_from_index(edgeElems,edgeElems,edgeElems,elemNumber, &curr_x, &curr_y, &curr_z);
   int nidx = (curr_z*edgeElems*edgeElems) 
         + (curr_y*edgeElems) 
         + (curr_x) 

         + (curr_y)
         + (curr_z*edgeElems)

         + (curr_z*edgeNodes);
   int nodeNumbers[8];
   nodeNumbers[0] = nidx                                       ;
   nodeNumbers[1] = nidx                                   + 1 ;
   nodeNumbers[2] = nidx                       + edgeNodes + 1 ;
   nodeNumbers[3] = nidx                       + edgeNodes     ;
   nodeNumbers[4] = nidx + edgeNodes*edgeNodes                 ;
   nodeNumbers[5] = nidx + edgeNodes*edgeNodes             + 1 ;
   nodeNumbers[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
   nodeNumbers[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;

   double x_local[8], y_local[8], z_local[8] ;

   for( int lnode=0 ; lnode<8 ; ++lnode )
   {
      int gnode = nodeNumbers[lnode];
      get_loc_from_index(edgeNodes,edgeNodes,edgeNodes,gnode, &curr_x, &curr_y, &curr_z);

      x_local[lnode] = double(1.125)*double(curr_x)/double(meshEdgeElems);
      y_local[lnode] = double(1.125)*double(curr_y)/double(meshEdgeElems);
      z_local[lnode] = double(1.125)*double(curr_z)/double(meshEdgeElems);
   }
   // volume calculations
   return  CalcElemVolume_Vec(x_local, y_local, z_local );
}


//initialise using the static data
static inline
void initialise(int myRank,
               int nx, int tp, int nr, int balance, int cost, Int8_t numRanks){

   g_numElem = nx*nx*nx;
   g_numNode = (nx+1)*(nx+1)*(nx+1);
   g_sizeX = nx ;
   g_sizeY = nx ;
   g_sizeZ = nx ;

   int m_numElem =  compute_local_size(g_numElem,(Int8_t)numRanks,myRank);
   int m_numNode =  compute_local_size(g_numNode,(Int8_t)numRanks,myRank);

   int starting_m_numElem =0;
   int starting_m_numNode =0;
   for (int i =0; i<myRank; i++){
      starting_m_numElem +=  compute_local_size(g_numElem,(Int8_t)numRanks,i);
      starting_m_numNode +=  compute_local_size(g_numNode,(Int8_t)numRanks,i);
   }

   int colLoc =0;
   int rowLoc =0;
   int planeLoc=0;


   int edgeElems = nx ;
   int edgeNodes = edgeElems+1 ;

   //! Setup Comm Buffer, Should not be necessary in final app
   rowMin = (rowLoc == 0)        ? 0 : 1;
   rowMax = (rowLoc == tp-1)     ? 0 : 1;
   colMin = (colLoc == 0)        ? 0 : 1;
   colMax = (colLoc == tp-1)     ? 0 : 1;
   planeMin = (planeLoc == 0)    ? 0 : 1;
   planeMax = (planeLoc == tp-1) ? 0 : 1;

   //All orgional
   for(int i=0; i<m_numElem;++i){
      p[i] = double(0.0);
      e[i] = double(0.0);
      q[i] = double(0.0);
      ss[i] = double(0.0);
   }

   for(int i=0; i<m_numElem;++i){
      v[i] = double(1.0);
   }

   for (int i = 0; i<m_numNode;++i){
      xd[i] = double(0.0);
      yd[i] = double(0.0);
      zd[i] = double(0.0);
   }

   for (int i = 0; i<m_numNode;++i){
   // for (int i = 0; i<m_numNode*3;++i){
      xdd[i] = double(0.0);
      ydd[i] = double(0.0);
      zdd[i] = double(0.0);
   }

   for (int i=0; i<m_numNode; ++i) {
      nodalMass[i] = double(0.0) ;
   }

   //!Build Mesh function !HERE

   // Happy with this
   int meshEdgeElems = tp * nx;   

   int curr_z,curr_y,curr_x;
   get_loc_from_index(edgeNodes,edgeNodes,edgeNodes,starting_m_numNode, &curr_x, &curr_y, &curr_z);


   int nidx = 0;

   while (nidx < m_numNode && curr_z<edgeNodes){
      double tz = double(1.125)*double(curr_z)/double(meshEdgeElems) ;
      while (nidx < m_numNode && curr_y<edgeNodes){
         double ty = double(1.125)*double(curr_y)/double(meshEdgeElems);
         while (nidx < m_numNode && curr_x<edgeNodes){
            double tx = double(1.125)*double(curr_x)/double(meshEdgeElems);
            x[nidx] = tx ;
            y[nidx] = ty ;
            z[nidx] = tz ;
            ++nidx;
            ++curr_x;
         }
         curr_x=0;
         ++curr_y;
      }
      curr_y=0;
      ++curr_z;
   }
   
   //Happy with this for now
   int zidx = 0 ;
   get_loc_from_index(edgeElems,edgeElems,edgeElems,starting_m_numElem, &curr_x, &curr_y, &curr_z);
   nidx = (curr_z*edgeElems*edgeElems) 
         + (curr_y*edgeElems) 
         + (curr_x) 

         + (curr_y)
         + (curr_z*edgeElems)

         + (curr_z*edgeNodes);

   while (zidx < m_numElem && curr_z<edgeElems){
      while (zidx < m_numElem && curr_y<edgeElems){
         while (zidx < m_numElem && curr_x<edgeElems){
            int *localNode = &nodelist[zidx*int(8)] ;

            localNode[0] = nidx                                       ;
            localNode[1] = nidx                                   + 1 ;
            localNode[2] = nidx                       + edgeNodes + 1 ;
            localNode[3] = nidx                       + edgeNodes     ;
            localNode[4] = nidx + edgeNodes*edgeNodes                 ;
            localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
            localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
            localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;

            ++zidx;
            ++nidx ;
            ++curr_x;
         }
         curr_x=0;
         ++curr_y;
         ++nidx ;
      }
      curr_y=0;
      ++curr_z;
      nidx += edgeNodes ;
   }

   //! End Build Mesh Function

   //! Start Create Region Sets
   srand(0);

   m_numReg = nr;
   m_regElemSize = (int*) malloc(m_numReg * sizeof(int));
   m_regElemlist_2 = (int*) malloc(m_numElem * sizeof(int));
   m_regNumList = (int*) malloc(m_numElem * sizeof(int));

   int nextIndex = 0;
   //if we only have one region just fill it
   // Fill out the regNumList with material numbers, which are always
   // the region index plus one 
   if(m_numReg == 1){
      while(nextIndex < m_numElem){
         m_regNumList[nextIndex] = 1;
         nextIndex++;
      }
      m_regElemSize[0] = 0;
   } else {//If we have more than one region distribute the elements.
      int regionNum;
      int regionVar;
      int lastReg = -1;
      int binSize;
      int elements;
      int runto = 0;
      int costDenominator = 0;
      int* regBinEnd = (int*) malloc(m_numReg * sizeof(int));

      //Determine the relative weights of all the regions.  This is based off the -b flag.  Balance is the value passed into b.  
      for(int i=0; i<m_numReg;++i){
         m_regElemSize[i] = 0;
         costDenominator += pow((i+1), balance);//Total sum of all regions weights
         regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
      }
      //Until all elements are assigned
      while (nextIndex < m_numElem) {
         //pick the region
         regionVar = rand() % costDenominator;
         int i = 0;
         while(regionVar >= regBinEnd[i]) i++;
         //rotate the regions based on MPI rank.  Rotation is Rank % m_numRegions this makes each domain have a different region with 
         //the highest representation
         regionNum = ((i+myRank)% m_numReg) +1;
         while(regionNum == lastReg){
            regionVar = rand() % costDenominator;
            i = 0;
            while(regionVar >= regBinEnd[i]) i++;
            regionNum = ((i + myRank) % m_numReg) + 1;
         }
         //Pick the bin size of the region and determine the number of elements.
         binSize = rand() % 1000;
         if(binSize < 773) {
	         elements = rand() % 15 + 1;
	      }
	      else if(binSize < 937) {
	         elements = rand() % 16 + 16;
	      }
	      else if(binSize < 970) {
	         elements = rand() % 32 + 32;
	      }
	      else if(binSize < 974) {
	         elements = rand() % 64 + 64;
	      } 
	      else if(binSize < 978) {
	         elements = rand() % 128 + 128;
	      }
	      else if(binSize < 981) {
	         elements = rand() % 256 + 256;
	      }
	      else
	         elements = rand() % 1537 + 512;
         runto = elements + nextIndex;
	      //Store the elements.  If we hit the end before we run out of elements then just stop.
         while (nextIndex < runto && nextIndex < m_numElem) {
	         m_regNumList[nextIndex] = regionNum;
	         nextIndex++;
	      }
         lastReg = regionNum;
      }
      free(regBinEnd); 
   }
   // Convert m_regNumList to region index sets
   // First, count size of each region 

   //re-writen for consecuative array reation
   int totalSize=0;
   for (int i=0 ; i<m_numElem ; ++i) {
      int r = m_regNumList[i]-1; // region index == regnum-1
      m_regElemSize[r]++;
   }
   
   int* reg_offset = (int*)malloc(m_numReg * sizeof(int));
   reg_offset[0]=0;
   for (int r=0; r<m_numReg-1; r++ ){
      reg_offset[r+1]=reg_offset[r]+m_regElemSize[r];
      m_regElemSize[r]=0;
   }
   m_regElemSize[m_numReg-1]=0;
   
   // Third, fill index sets
   for (int i=0 ; i<m_numElem ; ++i) {
      int r = m_regNumList[i]-1;       // region index == regnum-1
      int regndx = m_regElemSize[r]++; // Note increment
      m_regElemlist_2[reg_offset[r]+regndx] = i+starting_m_numElem;
   }

   //! End Create Region Sets

   //! Setup Symmetry Planes Function !HERE

   for (int i = 0; i<m_numNode;++i){
      t_symmX[i] = FREE_NODE; 
      t_symmY[i] = FREE_NODE;
      t_symmZ[i] = FREE_NODE;
   }

   //new Version   
   nidx = 0 ;
   for (int i=0; i<edgeNodes; ++i) {
      int planeInc = i*edgeNodes*edgeNodes ;
      int rowInc   = i*edgeNodes ;
      for (int j=0; j<edgeNodes; ++j) {
         if (planeLoc == 0) {
            if ( rowInc + j>=starting_m_numNode && rowInc + j<starting_m_numNode+m_numNode ){
               t_symmZ[rowInc + j-starting_m_numNode] |= BOUNDARY_NODE;
            }
         }
         if (rowLoc == 0) {
            if ( planeInc + j>=starting_m_numNode && planeInc + j<starting_m_numNode+m_numNode ){
               t_symmY[planeInc+j-starting_m_numNode] |= BOUNDARY_NODE;
            }
         }
         if (colLoc == 0) {
            if ( planeInc + j*edgeNodes>=starting_m_numNode && planeInc + j*edgeNodes<starting_m_numNode+m_numNode ){
               t_symmX[planeInc + j*edgeNodes-starting_m_numNode] |= BOUNDARY_NODE;
            }
         }
         ++nidx ;
      }
   }
   //! End Setup Symmetry Plnes Function

   //! Setup Elem Connectivity Function
   for (int i=0; i<m_numElem; ++i) {
      lxim[i] = std::max(0,i+starting_m_numElem-1);
      lxip[i] = std::min(g_numElem-1  ,i+starting_m_numElem+1) ;
   }

   for (int i=0; i<m_numElem; ++i ){
      if (i+starting_m_numElem < edgeElems){
         letam[i] = i+starting_m_numElem;
      }else{
         letam[i] = i+starting_m_numElem-edgeElems ;
      }

      if (i+starting_m_numElem >= g_numElem-edgeElems){
         letap[i] = i+starting_m_numElem;
      }else{
         letap[i] = i+starting_m_numElem+edgeElems ;
      }
   }

   for (int i=0; i<m_numElem; ++i ){
      if (i+starting_m_numElem < edgeElems*edgeElems){
         lzetam[i] = i+starting_m_numElem;
      }else{
         lzetam[i] = i+starting_m_numElem-edgeElems*edgeElems ;
      }

      if (i+starting_m_numElem >= g_numElem-edgeElems*edgeElems){
         lzetap[i] = i+starting_m_numElem;
      }else{
         lzetap[i] = i+starting_m_numElem+edgeElems*edgeElems ;
      }
   }

   //! End Selem Connectivity Function


   //! Setup Boundary Conditions Function !HERE
   // set up boundary condition information
   int ghostIdx[6] ;  // offsets to ghost locations

   //New version
   for (int i=0; i<m_numElem; ++i) {
      elemBC[i] = int(0) ;
   }
   for (int i=0; i<6; ++i) {
      ghostIdx[i] = INT_MIN ;
   }

   int pidx = g_numElem ;
   if (planeMin != 0) {
      ghostIdx[0] = pidx ;
      pidx += g_sizeX*g_sizeY ;
   }
   if (planeMax != 0) {
      ghostIdx[1] = pidx ;
      pidx += g_sizeX*g_sizeY ;
   }
   if (rowMin != 0) {
      ghostIdx[2] = pidx ;
      pidx += g_sizeX*g_sizeZ ;
   }
   if (rowMax != 0) {
      ghostIdx[3] = pidx ;
      pidx += g_sizeX*g_sizeZ ;
   }
   if (colMin != 0) {
      ghostIdx[4] = pidx ;
      pidx += g_sizeY*g_sizeZ ;
   } 
   if (colMax != 0) {
      ghostIdx[5] = pidx ;
   }

   for (int i=0; i<edgeElems; ++i) {

      int planeInc = i*edgeElems*edgeElems ;
      int rowInc   = i*edgeElems ;
      for (int j=0; j<edgeElems; ++j) {
         if (planeLoc == 0) {
            if ( (rowInc+j) >=starting_m_numElem && (rowInc+j)<starting_m_numElem+m_numElem ){
               elemBC[rowInc+j-starting_m_numElem] |= ZETA_M_SYMM ;
            }
         }
         else {
            if ( (rowInc+j) >=starting_m_numElem && (rowInc+j)<starting_m_numElem+m_numElem ){
               elemBC[rowInc+j-starting_m_numElem] |= ZETA_M_COMM ;
               lzetam[rowInc+j-starting_m_numElem] = ghostIdx[0] + rowInc + j ;
            }
         }
         if (planeLoc == tp-1) {
            if ( (rowInc+j+g_numElem-(edgeElems*edgeElems)) >=starting_m_numElem && (rowInc+j+g_numElem-(edgeElems*edgeElems))<starting_m_numElem+m_numElem ){
               elemBC[rowInc+j+g_numElem-(edgeElems*edgeElems)-starting_m_numElem] |= ZETA_P_FREE;
            }
         }
         else {
            if ( (rowInc+j+g_numElem-edgeElems*edgeElems) >=starting_m_numElem && (rowInc+j+g_numElem-edgeElems*edgeElems)<starting_m_numElem+m_numElem ){
               elemBC[rowInc+j+g_numElem-edgeElems*edgeElems-starting_m_numElem] |= ZETA_P_COMM ;
               lzetap[rowInc+j+g_numElem-edgeElems*edgeElems-starting_m_numElem] = ghostIdx[1] + rowInc + j ;
            }
         }
         if (rowLoc == 0) {
            if ( (planeInc+j) >=starting_m_numElem && (planeInc+j)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j-starting_m_numElem] |= ETA_M_SYMM ;
            }
         }
         else {
            if ( (planeInc+j) >=starting_m_numElem && (planeInc+j)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j-starting_m_numElem] |= ETA_M_COMM ;
               letam[planeInc+j-starting_m_numElem] = ghostIdx[2] + rowInc + j ;
            }
         }
         if (rowLoc == tp-1) {
            if ( (planeInc+j+edgeElems*edgeElems-edgeElems) >=starting_m_numElem && (planeInc+j+edgeElems*edgeElems-edgeElems)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j+edgeElems*edgeElems-edgeElems-starting_m_numElem] |= ETA_P_FREE ;
            }
         }
         else {
            if ( (planeInc+j+edgeElems*edgeElems-edgeElems) >=starting_m_numElem && (planeInc+j+edgeElems*edgeElems-edgeElems)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j+edgeElems*edgeElems-edgeElems-starting_m_numElem] |=  ETA_P_COMM ;
               letap[planeInc+j+edgeElems*edgeElems-edgeElems-starting_m_numElem] = ghostIdx[3] +  rowInc + j ;
            }
         }
         if (colLoc == 0) {
            if ( (planeInc+j*edgeElems) >=starting_m_numElem && (planeInc+j*edgeElems)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j*edgeElems-starting_m_numElem] |= XI_M_SYMM ;
            }
         }
         else {
            if ( (planeInc+j*edgeElems) >=starting_m_numElem && (planeInc+j*edgeElems)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j*edgeElems-starting_m_numElem] |= XI_M_COMM ;
               lxim[planeInc+j*edgeElems-starting_m_numElem] = ghostIdx[4] + rowInc + j ;
            }
         }
         if (colLoc == tp-1) {
            if ( (planeInc+j*edgeElems+edgeElems-1) >=starting_m_numElem && (planeInc+j*edgeElems+edgeElems-1)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j*edgeElems+edgeElems-1-starting_m_numElem] |= XI_P_FREE ;
            }
         }
         else {
            if ( (planeInc+j*edgeElems+edgeElems-1) >=starting_m_numElem && (planeInc+j*edgeElems+edgeElems-1)<starting_m_numElem+m_numElem ){
               elemBC[planeInc+j*edgeElems+edgeElems-1-starting_m_numElem] |= XI_P_COMM ;
               lxip[planeInc+j*edgeElems+edgeElems-1-starting_m_numElem] = ghostIdx[5] + rowInc + j ;
            }
         }
      }
   }

   m_region_i_to_lxim=(int*)malloc(sizeof(int)*m_numElem);
   m_region_i_to_lxip=(int*)malloc(sizeof(int)*m_numElem);
   m_region_i_to_letam=(int*)malloc(sizeof(int)*m_numElem);
   m_region_i_to_letap=(int*)malloc(sizeof(int)*m_numElem);
   m_region_i_to_lzetam=(int*)malloc(sizeof(int)*m_numElem);
   m_region_i_to_lzetap=(int*)malloc(sizeof(int)*m_numElem);
   for (int i=0; i<m_numElem;i++){
      m_region_i_to_lxim[i]=lxim[m_regElemlist_2[i]-starting_m_numElem];
      m_region_i_to_lxip[i]=lxip[m_regElemlist_2[i]-starting_m_numElem];
      m_region_i_to_letam[i]=letam[m_regElemlist_2[i]-starting_m_numElem];
      m_region_i_to_letap[i]=letap[m_regElemlist_2[i]-starting_m_numElem];
      m_region_i_to_lzetam[i]=lzetam[m_regElemlist_2[i]-starting_m_numElem];
      m_region_i_to_lzetap[i]=lzetap[m_regElemlist_2[i]-starting_m_numElem];
   }


   //! End SetupBC Function

   // Setup defaults

   // These can be changed (requires recompile) if you want to run
   // with a fixed timestep, or to a different end time, but it's
   // probably easier/better to just run a fixed number of timesteps
   // using the -i flag in 2.x

   m_dtfixed = double(-1.0e-6) ; // Negative means use courant condition
   m_stoptime  = double(1.0e-2); // *double(edgeElems*tp/45.0) ;

   // Initial conditions
   m_deltatimemultlb = double(1.1) ;
   m_deltatimemultub = double(1.2) ;
   m_dtcourant = double(1.0e+20) ;
   m_dthydro   = double(1.0e+20) ;
   m_dtmax     = double(1.0e-2) ;
   m_time    = double(0.) ;
   m_cycle   = int(0) ;

   //! Difficult
   // initialize field data 
   for (int i=0; i<m_numElem; ++i) {

      double x_local[8], y_local[8], z_local[8] ;
      int *m_elemToNode = &nodelist[i*int(8)] ;

      for( int lnode=0 ; lnode<8 ; ++lnode )
      {
        int gnode = m_elemToNode[lnode];
        get_loc_from_index(edgeNodes,edgeNodes,edgeNodes,gnode, &curr_x, &curr_y, &curr_z);

        x_local[lnode] = double(1.125)*double(curr_x)/double(meshEdgeElems);
        y_local[lnode] = double(1.125)*double(curr_y)/double(meshEdgeElems);
        z_local[lnode] = double(1.125)*double(curr_z)/double(meshEdgeElems);
      }

      // volume calculations
      double volume = CalcElemVolume_Vec(x_local, y_local, z_local);

      volo[i] = volume;
      elemMass[i] = volume ;
   }
   for (int i=0; i<m_numNode; ++i) {
      int *m_elemToNode = &nodelist[i*int(8)];
      get_loc_from_index(edgeNodes,edgeNodes,edgeNodes,i+starting_m_numNode, &curr_x, &curr_y, &curr_z);

      nodalMass[i] =0;
      if (curr_x<edgeElems && curr_y<edgeElems && curr_z<edgeElems)
      nodalMass[i] += volumeAt((curr_x  ) + (curr_y  )*edgeElems + (curr_z  )*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ;
      if (curr_x<edgeElems && curr_y<edgeElems && curr_z>0)
      nodalMass[i] += volumeAt((curr_x  ) + (curr_y  )*edgeElems + (curr_z-1)*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ; 
      if (curr_x<edgeElems && curr_y>0         && curr_z<edgeElems)
      nodalMass[i] += volumeAt((curr_x  ) + (curr_y-1)*edgeElems + (curr_z  )*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ;
      if (curr_x<edgeElems && curr_y>0         && curr_z>0)
      nodalMass[i] += volumeAt((curr_x  ) + (curr_y-1)*edgeElems + (curr_z-1)*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ; 
      if (curr_x>0         && curr_y<edgeElems && curr_z<edgeElems)
      nodalMass[i] += volumeAt((curr_x-1) + (curr_y  )*edgeElems + (curr_z  )*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ;
      if (curr_x>0         && curr_y<edgeElems && curr_z>0)
      nodalMass[i] += volumeAt((curr_x-1) + (curr_y  )*edgeElems + (curr_z-1)*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ;
      if (curr_x>0         && curr_y>0         && curr_z<edgeElems)
      nodalMass[i] += volumeAt((curr_x-1) + (curr_y-1)*edgeElems + (curr_z  )*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ;
      if (curr_x>0         && curr_y>0         && curr_z>0)
      nodalMass[i] += volumeAt((curr_x-1) + (curr_y-1)*edgeElems + (curr_z-1)*edgeElems*edgeElems   ,meshEdgeElems,edgeNodes,edgeElems)/double(8.0) ; 
      
      // nodalMass[i] =0;
      // x   y    z -1
      // x   y    z -1
      // x   y -1 z -1
      // x   y -1 z -1
      // x-1 y    z -1
      // x-1 y    z -1
      // x-1 y -1 z -1
      // x-1 y -1 z -1
   }

   op_printf("Voume at 0: %e, Nodal Mass: %e\n", volo[0], nodalMass[0]);

   // deposit initial energy
   // An energy of 3.948746e+7 is correct for a problem with
   // 45 zones along a side - we need to scale it
   if (myRank==0){
      const double ebase = double(3.948746e+7);
      double scale = (nx * tp)/double(45.0);
      double einit = ebase*scale*scale*scale;
      if (rowLoc + colLoc + planeLoc == 0) {
         // Dump into the first zone (which we know is in the corner)
         // of the domain that sits at the origin
         e[0] = einit;
      }
      m_deltatime = (double(.5)*cbrt(volo[0]))/sqrt(double(2.0)*einit);
   }

   for (int i = 0; i < m_numNode; i++)
   {
      loc[3*i]=x[i];
      loc[3*i +1]=y[i];
      loc[3*i +2]=z[i];
   }
}

//initialise using the static data
static inline
void initialiseSingular(int colLoc,
               int rowLoc, int planeLoc,
               int nx, int tp, int nr, int balance, int cost, Int8_t numRanks){
   printf("%d %d %d %d %d %d\n",colLoc,rowLoc,planeLoc,nx,tp,nr);

   int edgeElems = nx ;
   int edgeNodes = edgeElems+1 ;

   // sizeX = edgeElems ;
   // sizeY = edgeElems ;
   // sizeZ = edgeElems ;
   // m_numElem = edgeElems * edgeElems * edgeElems;
   // m_numNode = edgeNodes * edgeNodes * edgeNodes;

   g_numElem = nx*nx*nx;
   g_numNode = (nx+1)*(nx+1)*(nx+1);
   g_sizeX = nx ;
   g_sizeY = nx ;
   g_sizeZ = nx ;

   printf("%d %d %d %d %d\n",g_sizeX,g_sizeY,g_sizeZ,g_numElem,g_numNode);



   printf("HERE NOW 3.1\n");
   //! Setup Comm Buffer, Should not be necessary in final app
   rowMin = (rowLoc == 0)        ? 0 : 1;
   rowMax = (rowLoc == tp-1)     ? 0 : 1;
   colMin = (colLoc == 0)        ? 0 : 1;
   colMax = (colLoc == tp-1)     ? 0 : 1;
   planeMin = (planeLoc == 0)    ? 0 : 1;
   planeMax = (planeLoc == tp-1) ? 0 : 1;

   printf("HERE NOW 3.1.1\n");
   for(int i=0; i<g_numElem;++i){
      g_p[i] = double(0.0);
      g_e[i] = double(0.0);
      g_q[i] = double(0.0);
      g_ss[i] = double(0.0);
   }

   for(int i=0; i<g_numElem;++i){
      g_v[i] = double(1.0);
   }

   for (int i = 0; i<g_numNode;++i){
      g_xd[i] = double(0.0);
      g_yd[i] = double(0.0);
      g_zd[i] = double(0.0);
   }

   for (int i = 0; i<g_numNode;++i){
   // for (int i = 0; i<m_numNode*3;++i){
      g_xdd[i] = double(0.0);
      g_ydd[i] = double(0.0);
      g_zdd[i] = double(0.0);
   }

   for (int i=0; i<g_numNode; ++i) {
      g_nodalMass[i] = double(0.0) ;
   }

   printf("HERE NOW 3.2\n");
   //!Build Mesh function !HERE
   int meshEdgeElems = tp * nx;
   // op_printf("Building Mesh with planeloc: %d, col: %d, rowLoc: %d, meshEdge: %d\n", planeLoc, colLoc, rowLoc, meshEdgeElems);
   int nidx = 0;
   double tz = double(1.125)*double(planeLoc*nx)/double(meshEdgeElems) ;
   for (int plane=0; plane<edgeNodes; ++plane) {
      double ty = double(1.125)*double(rowLoc*nx)/double(meshEdgeElems) ;
      for (int row=0; row<edgeNodes; ++row) {
         double tx = double(1.125)*double(colLoc*nx)/double(meshEdgeElems) ;
         for (int col=0; col<edgeNodes; ++col) {
         g_x[nidx] = tx ;
         g_y[nidx] = ty ;
         g_z[nidx] = tz ;
         ++nidx ;
         // tx += ds ; // may accumulate roundoff... 
         tx = double(1.125)*double(colLoc*nx+col+1)/double(meshEdgeElems) ;
         }
         // ty += ds ;  // may accumulate roundoff... 
         ty = double(1.125)*double(rowLoc*nx+row+1)/double(meshEdgeElems) ;
      }
      // tz += ds ;  // may accumulate roundoff... 
      tz = double(1.125)*double(planeLoc*nx+plane+1)/double(meshEdgeElems) ;
   }


   

   // embed hexehedral elements in nodal point lattice 
   int zidx = 0 ;
   nidx = 0 ;
   printf("HERE NOW 3.3\n");

   for (int plane=0; plane<edgeElems; ++plane) {
      for (int row=0; row<edgeElems; ++row) {
         for (int col=0; col<edgeElems; ++col) {
         int *localNode = &g_nodelist[zidx*int(8)] ;

         localNode[0] = nidx                                       ;
         localNode[1] = nidx                                   + 1 ;
         localNode[2] = nidx                       + edgeNodes + 1 ;
         localNode[3] = nidx                       + edgeNodes     ;
         localNode[4] = nidx + edgeNodes*edgeNodes                 ;
         localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
         localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
         localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
         ++zidx ;
         ++nidx ;
         }
         ++nidx ;
      }
      nidx += edgeNodes ;
   }

   //! End Build Mesh Function
   printf("HERE NOW 3.4\n");

   //! Start Create Region Sets
   srand(0);
   int myRank = 0;

   m_numReg = nr;
   g_m_regElemSize = (int*) malloc(m_numReg * sizeof(int));
   g_m_regElemlist = (int**) malloc(m_numReg * sizeof(int*));
   g_m_regNumList = (int*) malloc(g_numElem * sizeof(int));

   int nextIndex = 0;
   //if we only have one region just fill it
   // Fill out the regNumList with material numbers, which are always
   // the region index plus one 
   if(m_numReg == 1){
      printf("HERE NOW 3.4.1\n");

      while(nextIndex < g_numElem){
         g_m_regNumList[nextIndex] = 1;
         nextIndex++;
      }
      g_m_regElemSize[0] = 0;
   } else {//If we have more than one region distribute the elements.
      printf("HERE NOW 3.4.2\n");

      int regionNum;
      int regionVar;
      int lastReg = -1;
      int binSize;
      int elements;
      int runto = 0;
      int costDenominator = 0;
      int* g_regBinEnd = (int*) malloc(m_numReg * sizeof(int));
      //Determine the relative weights of all the regions.  This is based off the -b flag.  Balance is the value passed into b.  
      for(int i=0; i<m_numReg;++i){
         g_m_regElemSize[i] = 0;
         costDenominator += pow((i+1), balance);//Total sum of all regions weights
         g_regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
      }
      //Until all elements are assigned
      while (nextIndex < g_numElem) {
         //pick the region
         regionVar = rand() % costDenominator;
         int i = 0;
         while(regionVar >= g_regBinEnd[i]) i++;
         //rotate the regions based on MPI rank.  Rotation is Rank % m_numRegions this makes each domain have a different region with 
         //the highest representation
         regionNum = ((i+myRank)% m_numReg) +1;
         while(regionNum == lastReg){
            regionVar = rand() % costDenominator;
            i = 0;
            while(regionVar >= g_regBinEnd[i]) i++;
            regionNum = ((i + myRank) % m_numReg) + 1;
         }
         //Pick the bin size of the region and determine the number of elements.
         binSize = rand() % 1000;
         if(binSize < 773) {
	         elements = rand() % 15 + 1;
	      }
	      else if(binSize < 937) {
	         elements = rand() % 16 + 16;
	      }
	      else if(binSize < 970) {
	         elements = rand() % 32 + 32;
	      }
	      else if(binSize < 974) {
	         elements = rand() % 64 + 64;
	      } 
	      else if(binSize < 978) {
	         elements = rand() % 128 + 128;
	      }
	      else if(binSize < 981) {
	         elements = rand() % 256 + 256;
	      }
	      else
	         elements = rand() % 1537 + 512;
         runto = elements + nextIndex;
	      //Store the elements.  If we hit the end before we run out of elements then just stop.
         while (nextIndex < runto && nextIndex < g_numElem) {
	         g_m_regNumList[nextIndex] = regionNum;
	         nextIndex++;
	      }
         lastReg = regionNum;
      }
      free(g_regBinEnd); 
   }
   // Convert m_regNumList to region index sets
   // First, count size of each region 
      printf("HERE NOW 3.4.3\n");


   for (int i=0 ; i<g_numElem ; ++i) {
      int r = g_m_regNumList[i]-1; // region index == regnum-1
      g_m_regElemSize[r]++;
   }

   // Second, allocate each region index set
   for (int i=0 ; i<m_numReg ; ++i) {
      g_m_regElemlist[i] = (int*) malloc(g_m_regElemSize[i]*sizeof(int));
      g_m_regElemSize[i] = 0;
   }

   // Third, fill index sets
   for (int i=0 ; i<g_numElem ; ++i) {
      int r = g_m_regNumList[i]-1;       // region index == regnum-1
      int regndx = g_m_regElemSize[r]++; // Note increment
      g_m_regElemlist[r][regndx] = i;
   }
   printf("HERE NOW 3.4.4.1 %d\n",g_numElem);





   //! End Create Region Sets
   printf("HERE NOW 3.5\n");

   //! Setup Symmetry Planes Function !HERE
   for (int i = 0; i<g_numNode;++i){ 
   // for (int i = 0; i<m_numNode*3;++i){
      g_t_symmX[i] |= FREE_NODE; 
      g_t_symmY[i] |= FREE_NODE;
      g_t_symmZ[i] |= FREE_NODE;
   }
   nidx = 0 ;
   for (int i=0; i<edgeNodes; ++i) {
      int planeInc = i*edgeNodes*edgeNodes ;
      int rowInc   = i*edgeNodes ;
      for (int j=0; j<edgeNodes; ++j) {
         if (planeLoc == 0) {
            // symmZ[nidx] = rowInc   + j ;
            g_t_symmZ[rowInc + j] |= BOUNDARY_NODE;
         }
         if (rowLoc == 0) {
            // symmY[nidx] = planeInc + j ;
            g_t_symmY[planeInc+j] |= BOUNDARY_NODE;
         }
         if (colLoc == 0) {
            // symmX[nidx] = planeInc + j*edgeNodes ;
            g_t_symmX[planeInc + j*edgeNodes] |= BOUNDARY_NODE;
         }
         ++nidx ;
      }
   }

   //! End Setup Symmetry Plnes Function
   printf("HERE NOW 3.6\n");

   //! Setup Elem Connectivity Function
   g_lxim[0] = 0 ;
   for (int i=1; i<g_numElem; ++i) {
      g_lxim[i]   = i-1 ;
      g_lxip[i-1] = i ;
   }
   g_lxip[g_numElem-1] = g_numElem-1 ;

   for (int i=0; i<edgeElems; ++i) {
      g_letam[i] = i ; 
      g_letap[g_numElem-edgeElems+i] = g_numElem-edgeElems+i ;
   }
   for (int i=edgeElems; i<g_numElem; ++i) {
      g_letam[i] = i-edgeElems ;
      g_letap[i-edgeElems] = i ;
   }

   for (int i=0; i<edgeElems*edgeElems; ++i) {
      g_lzetam[i] = i ;
      g_lzetap[g_numElem-edgeElems*edgeElems+i] = g_numElem-edgeElems*edgeElems+i ;
   }
   for (int i=edgeElems*edgeElems; i<g_numElem; ++i) {
      g_lzetam[i] = i - edgeElems*edgeElems ;
      g_lzetap[i-edgeElems*edgeElems] = i ;
   }
   //! End Selem Connectivity Function
   printf("HERE NOW 3.7\n");

   //! Setup Boundary Conditions Function !HERE
   // set up boundary condition information
   int ghostIdx[6] ;  // offsets to ghost locations
   for (int i=0; i<g_numElem; ++i) {
      g_elemBC[i] = int(0) ;
   }
   for (int i=0; i<6; ++i) {
      ghostIdx[i] = INT_MIN ;
   }

  int pidx = g_numElem ;
  if (planeMin != 0) {
    ghostIdx[0] = pidx ;
    pidx += g_sizeX*g_sizeY ;
  }

  if (planeMax != 0) {
    ghostIdx[1] = pidx ;
    pidx += g_sizeX*g_sizeY ;
  }

  if (rowMin != 0) {
    ghostIdx[2] = pidx ;
    pidx += g_sizeX*g_sizeZ ;
  }

  if (rowMax != 0) {
    ghostIdx[3] = pidx ;
    pidx += g_sizeX*g_sizeZ ;
  }

  if (colMin != 0) {
    ghostIdx[4] = pidx ;
    pidx += g_sizeY*g_sizeZ ;
  }

  if (colMax != 0) {
    ghostIdx[5] = pidx ;
  }

  // symmetry plane or free surface BCs 
  for (int i=0; i<edgeElems; ++i) {

   int planeInc = i*edgeElems*edgeElems ;
   int rowInc   = i*edgeElems ;
   for (int j=0; j<edgeElems; ++j) {
      if (planeLoc == 0) {
	      g_elemBC[rowInc+j] |= ZETA_M_SYMM ;
      }
      else {
	      g_elemBC[rowInc+j] |= ZETA_M_COMM ;
	      g_lzetam[rowInc+j] = ghostIdx[0] + rowInc + j ;
      }
      if (planeLoc == tp-1) {
	      g_elemBC[rowInc+j+g_numElem-edgeElems*edgeElems] |= ZETA_P_FREE;
      }
      else {
	      g_elemBC[rowInc+j+g_numElem-edgeElems*edgeElems] |= ZETA_P_COMM ;
	      g_lzetap[rowInc+j+g_numElem-edgeElems*edgeElems] = ghostIdx[1] + rowInc + j ;
      }
      if (rowLoc == 0) {
	      g_elemBC[planeInc+j] |= ETA_M_SYMM ;
      }
      else {
	      g_elemBC[planeInc+j] |= ETA_M_COMM ;
	      g_letam[planeInc+j] = ghostIdx[2] + rowInc + j ;
      }
      if (rowLoc == tp-1) {
	      g_elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |= ETA_P_FREE ;
      }
      else {
	      g_elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |=  ETA_P_COMM ;
	      g_letap[planeInc+j+edgeElems*edgeElems-edgeElems] = ghostIdx[3] +  rowInc + j ;
      }
      if (colLoc == 0) {
	      g_elemBC[planeInc+j*edgeElems] |= XI_M_SYMM ;
      }
      else {
	      g_elemBC[planeInc+j*edgeElems] |= XI_M_COMM ;
	      g_lxim[planeInc+j*edgeElems] = ghostIdx[4] + rowInc + j ;
      }

      if (colLoc == tp-1) {
	      g_elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_FREE ;
      }
      else {
	      g_elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_COMM ;
	      g_lxip[planeInc+j*edgeElems+edgeElems-1] = ghostIdx[5] + rowInc + j ;
      }
    }
   }
   //! End SetupBC Function
   printf("HERE NOW 3.8\n");

   // Setup defaults

   // These can be changed (requires recompile) if you want to run
   // with a fixed timestep, or to a different end time, but it's
   // probably easier/better to just run a fixed number of timesteps
   // using the -i flag in 2.x

   m_dtfixed = double(-1.0e-6) ; // Negative means use courant condition
   m_stoptime  = double(1.0e-2); // *double(edgeElems*tp/45.0) ;

   // Initial conditions
   m_deltatimemultlb = double(1.1) ;
   m_deltatimemultub = double(1.2) ;
   m_dtcourant = double(1.0e+20) ;
   m_dthydro   = double(1.0e+20) ;
   m_dtmax     = double(1.0e-2) ;
   m_time    = double(0.) ;
   m_cycle   = int(0) ;
   
   printf("HERE NOW 3.9\n");


   double mytotal =0;
   for (int i = 0; i < edgeNodes*edgeNodes*edgeNodes; i++)
   {
      mytotal+=g_x[i];
   }
   printf("curstep %e %d\n",mytotal,edgeNodes);



   // initialize field data 
   for (int i=0; i<g_numElem; ++i) {

      double x_local[8], y_local[8], z_local[8] ;
      int *g_elemToNode = &g_nodelist[i*int(8)] ;
      for( int lnode=0 ; lnode<8 ; ++lnode )
      {
        int gnode = g_elemToNode[lnode];
        x_local[lnode] = g_x[gnode];
        y_local[lnode] = g_y[gnode];
        z_local[lnode] = g_z[gnode];
      }

      // volume calculations
      double volume = CalcElemVolume_Vec(x_local, y_local, z_local );

      g_volo[i] = volume ;
      g_elemMass[i] = volume ;
      for (int j=0; j<8; ++j) {
         int idx = g_elemToNode[j] ;
         g_nodalMass[idx] += volume / double(8.0) ;
      }
   }
   printf("HERE NOW 3.10\n");

   op_printf("Voume at 0: %e, Nodal Mass: %e\n", g_volo[0], g_nodalMass[0]);
   printf("HERE NOW 3.11\n");

   // deposit initial energy
   // An energy of 3.948746e+7 is correct for a problem with
   // 45 zones along a side - we need to scale it
   const double ebase = double(3.948746e+7);
   double scale = (nx * tp)/double(45.0);
   double einit = ebase*scale*scale*scale;
   if (rowLoc + colLoc + planeLoc == 0) {
      // Dump into the first zone (which we know is in the corner)
      // of the domain that sits at the origin
      g_e[0] = einit;
   }
   printf("HERE NOW 3.12\n");

   m_deltatime = (double(.5)*cbrt(g_volo[0]))/sqrt(double(2.0)*einit);

   for (int i = 0; i < edgeNodes*edgeNodes*edgeNodes; i++)
   {
      g_loc[3*i]=g_x[i];
      g_loc[3*i +1]=g_y[i];
      g_loc[3*i +2]=g_z[i];
   }
   
}


static inline
Domain initialiseALL(struct cmdLineOpts opts,int myRank,Int8_t numRanks){
   //generate the global size
   int g_numElem = opts.nx*opts.nx*opts.nx;
   int g_numNode = (opts.nx+1)*(opts.nx+1)*(opts.nx+1);

   //Compute the local size
   int m_numElem =  compute_local_size(g_numElem,(Int8_t)numRanks,myRank);
   int m_numNode =  compute_local_size(g_numNode,(Int8_t)numRanks,myRank);
   m_numReg = opts.numReg;
   printf("HERE 0.0.1\n");
   allocateVars(m_numElem,m_numNode);
   if (opts.creation == Creation_Parallel){
      initialise( myRank,opts.nx, 1, opts.numReg,opts.balance, opts.cost, numRanks);
   }else if (opts.creation == Creation_Root){
      if (myRank == 0){
         allocateGlobalVars(g_numElem,g_numNode);
         initialiseSingular(0,0,0,opts.nx,1,opts.numReg,opts.balance, opts.cost,(Int8_t)numRanks);
      }
      distributeGlobalElems(myRank,numRanks,g_numElem,m_numElem,g_numNode,m_numNode);
   }
   MPI_Bcast(&m_deltatime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   Domain domain= initOp2Vars(myRank,m_numElem,m_numNode);
   printf("HERE 0.0.3\n");
   return domain;
}