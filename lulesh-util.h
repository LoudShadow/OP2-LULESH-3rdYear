#include <math.h>

/* -t partiotioning library <S,PK,PG,PKG,K> */
#define Partition_PK  4
#define Partition_S   1
#define Partition_PG  2
#define Partition_PGK 3
#define Partition_K   0

#define Creation_Parallel 0
#define Creation_Root 1

#define Hide_Time 0
#define Show_Time 1

#define Visualization_None 0
#define Visualization_All 1
#define Visualization_End 2

/* Helper function for converting strings to ints, with error checking */
template<typename IntT>
int StrToInt(const char *token, IntT *retVal)
{
   const char *c ;
   char *endptr ;
   const int decimal_base = 10 ;

   if (token == NULL)
      return 0 ;
   
   c = token ;
   *retVal = strtol(c, &endptr, decimal_base) ;
   if((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
      return 1 ;
   else
      return 0 ;
}


static int compute_local_size(int global_size, int mpi_comm_size,
                              int mpi_rank) {
  int local_size = global_size / mpi_comm_size;
  int remainder = (int)fmod(global_size, mpi_comm_size);

  if (mpi_rank < remainder) {
    local_size = local_size + 1;
  }
  return local_size;
}

//Given a cube size and an aray index caculate its x,y,z
static inline
void get_loc_from_index(int dx,int dy,int dz,int location,
      int *x, int *y, int *z){

   *x = location % dx ;
   *y = (location / dx) % dy ;
   *z = location / (dx*dy) ;
}

static inline
int get_rank_from_index(int index, int global_size, int number_of_ranks){
   int currentAccumulated =0;
   int local_size = global_size / number_of_ranks;
   int remainder = global_size % number_of_ranks;

   int rank=-1;
   while (currentAccumulated<=index){
      rank+=1;
      currentAccumulated+=local_size;
      if (rank < remainder) {
         currentAccumulated = currentAccumulated + 1;
      }      
   }
   return rank;
}

static inline
int calc_Single_Elem_Offset_to_node(int loc_x,int loc_y,int loc_z,int edgeNodes){
   return (loc_x  ) + (loc_y  )*edgeNodes + (loc_z  )*edgeNodes*edgeNodes -loc_y     -(loc_z  ) -2*(loc_z  * edgeNodes);
}



static void scatter_double_array(double *g_array, double *l_array,
                                 int comm_size, int g_size, int l_size,
                                 int elem_size) {
  int *sendcnts = (int *)malloc(comm_size * sizeof(int));
  int *displs = (int *)malloc(comm_size * sizeof(int));
  int disp = 0;

  for (int i = 0; i < comm_size; i++) {
    sendcnts[i] = elem_size * compute_local_size(g_size, comm_size, i);
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + sendcnts[i];
  }

  MPI_Scatterv(g_array, sendcnts, displs, MPI_DOUBLE, l_array,
               l_size * elem_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(sendcnts);
  free(displs);
}

static void scatter_int_array(int *g_array, int *l_array, int comm_size,
                              int g_size, int l_size, int elem_size) {
  int *sendcnts = (int *)malloc(comm_size * sizeof(int));
  int *displs = (int *)malloc(comm_size * sizeof(int));
  int disp = 0;

  for (int i = 0; i < comm_size; i++) {
    sendcnts[i] = elem_size * compute_local_size(g_size, comm_size, i);
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + sendcnts[i];
  }

  MPI_Scatterv(g_array, sendcnts, displs, MPI_INT, l_array, l_size * elem_size,
               MPI_INT, 0, MPI_COMM_WORLD);
  free(sendcnts);
  sendcnts=NULL;
  free(displs);
}

static void ParseError(const char *message, int myRank)
{
   if (myRank == 0) {
      printf("%s\n", message);     
      MPI_Abort(MPI_COMM_WORLD, -1);

   }
}
static void PrintCommandLineOptions(char *execname, int myRank)
{
   if (myRank == 0) {

      printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf("\n\n");
   }
}

void ParseCommandLineOptions(int argc, char *argv[],
                             int myRank, struct cmdLineOpts *opts)
{
   if(argc > 1) {
      int i = 1;

      while(i < argc) {
         int ok;
         /* -i <iterations> */
         if(strcmp(argv[i], "-i") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -i", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->its));
            if(!ok) {
               ParseError("Parse Error on option -i integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -s <size, sidelength> */
         else if(strcmp(argv[i], "-s") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -s\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->nx));
            if(!ok) {
               ParseError("Parse Error on option -s integer value required after argument\n", myRank);
            }
            i+=2;
         }
	 /* -r <numregions> */
         else if (strcmp(argv[i], "-r") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -r\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numReg));
            if (!ok) {
               ParseError("Parse Error on option -r integer value required after argument\n", myRank);
            }
            i+=2;
         }
	 /* -f <numfilepieces> */
         else if (strcmp(argv[i], "-f") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -f\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numFiles));
            if (!ok) {
               ParseError("Parse Error on option -f integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -p */
         else if (strcmp(argv[i], "-p") == 0) {
            opts->showProg = 1;
            i++;
         }
         /* -q */
         else if (strcmp(argv[i], "-q") == 0) {
            opts->quiet = 1;
            i++;
         }
         else if (strcmp(argv[i], "-b") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -b\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->balance));
            if (!ok) {
               ParseError("Parse Error on option -b integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-c") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -c\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->cost));
            if (!ok) {
               ParseError("Parse Error on option -c integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -v */
         else if (strcmp(argv[i], "-v") == 0) {
            #if VIZ_MESH
            if ( strcmp(argv[i+1], "all") == 0 ){
               opts->partition = Visualization_All;
            } 
            if ( strcmp(argv[i+1], "end") == 0 ){
               opts->partition = Visualization_End;
            }                 
            #else
            ParseError("Use of -v requires compiling with -DVIZ_MESH\n", myRank);
            #endif
            i++;
         }
         /* -h */
         else if (strcmp(argv[i], "-h") == 0) {
            PrintCommandLineOptions(argv[0], myRank);
            MPI_Abort(MPI_COMM_WORLD, 0);

         }
         else if(strcmp(argv[i], "OP_NO_REALLOC" ) == 0){
            i++;
         }
         else if(strcmp(argv[i], "-l") == 0){
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -l\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->itert));
            if(!ok) {
               ParseError("Parse Error on option -l integer value required after argument\n", myRank);
            }
            i+=2;
         }
            else if(strcmp(argv[i], "-z") == 0){
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -z\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->fnm));
            if(!ok) {
               ParseError("Parse Error on option -z integer value required after argument\n", myRank);
            }
            i+=2;
         }
         
         /* -t partiotioning library <S,PK,PG,PKG,K> */
         else if(strcmp(argv[i], "-t") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing letter argumnet to -t\n", myRank);
            }
            if ( strcmp(argv[i+1], "S") == 0 ){
               opts->partition = Partition_S;
            }
            if ( strcmp(argv[i+1], "PK") == 0 ){
               opts->partition = Partition_PK;
            }
            if ( strcmp(argv[i+1], "PG") == 0 ){
               opts->partition = Partition_PG;
            }
            if ( strcmp(argv[i+1], "PGK") == 0 ){
               opts->partition = Partition_PGK;
            }
            if ( strcmp(argv[i+1], "K") == 0 ){
               opts->partition = Partition_K;
            }
            i+=2;
         }

         /* -t initialiseation method <S,P> */
         else if(strcmp(argv[i], "-d") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing letter argumnet to -i\n", myRank);
            }
            if ( strcmp(argv[i+1], "S") == 0 ){
               opts->creation = Creation_Root;
            }
            if ( strcmp(argv[i+1], "P") == 0 ){
               opts->creation = Creation_Parallel;
            }
            i+=2;
         }

         /* -t initialiseation method <S,PK,PG,PKG,K> */
         else if(strcmp(argv[i], "-ti") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing letter argumnet to -i\n", myRank);
            }
            if ( strcmp(argv[i+1], "S") == 0 ){
               opts->time = Show_Time;
            }
            if ( strcmp(argv[i+1], "H") == 0 ){
               opts->time = Hide_Time;
            }
            i+=2;
         }


         /* -f <numfilepieces> */
         else {
            char msg[80];
            i++;
            // PrintCommandLineOptions(argv[0], myRank);
            // sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
            // ParseError(msg, myRank);
         }
      }
   }
}

