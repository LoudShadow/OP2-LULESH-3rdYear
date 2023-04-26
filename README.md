# OP2-LULESH

This is the README for the OP2 version of LULESH 2.0

The original source code can be found under https://github.com/LLNL/LULESH and more information on https://asc.llnl.gov/codes/proxy-apps/lulesh

### Before compiling

This repository requires a working version of the OP2 repository found under https://github.com/OP-DSL/OP2-Common  

Both the makefile requires the environment variable OP2_INSTALL_PATH to be set to the root of the OP2 directory.

### Arguments
There are a number of command line arguments that can be used to adjust the running of the file.

-i [NUM]
  Number of iterations to run, The program will run until either NUM iteration or 0.01 seconds of simulation time whichever
  occurs first.
-s [NUM]
  Size of the mesh to simulate. The mesh will have NUM^3 elements and (NUM+1)^3 nodes. NOTE in this implementation the
  size of the mesh is the same regardless of the number of processes. (Default:30)
-r [NUM]
  Specify the number of regions within the mesh. Regions operate as a pseudo material implementation. (Default:1)
-p
  Flag to show the progress. Display progress information each simulation step.
-q
  Suppress standard out displays. Takes priority over -p
-b [NUM]
  sets the amount of load imbalance between regions. For this option 0 is no imbalance and imbalance increases with higher inputs. (Default:1)
-c
  sets the cost of more expensive regions. Half the regions always have no extra
  cost. 45% have an extra cost for the EOS equal to the number entered and 5% have an extra
  cost of 10 times the extra cost entered.
-v <all|end>
  Create visit files for the simulation, 'all' creates a file every simulation step 'end' creates a single
  Silo file at the end of the simulation.  
-h
  Show the command line arguments (help)
-g <S,PK,PG,PKG,K>
  Select the partitioning library to use with the program.
  S: PTSCOTCH
  PK: PARMETIS KWAY
  PG: PARMETIS GEOM
  PGK: PARMETIS GEOMKWAY
  K: KAHIP KWAY

  default KAHIP
-d <S|P>
  Initialization method for the creation of the mesh
  S: Singular mesh generation on the root node
  P: Parallel mesh generation on nodes in parallel
-t
  Show the OP2 timing measurements for each kernel at the 
  end of the file.

### Structure
The core of the project is spread among a number of key files.

- lulesh.h       - The primary file
- lulesh_mpi.h   - adjustments for the use of MPI files 
- lulesh-init.cc - Used for mesh generation
- lulesh-util.cc - Additional utility functions 
- lulesh-viz.cc  - A file to handle writing to SILO files

### LIbrary linking
paasdsjf;lasjflasdjfsdjfasdf
dsjflka;sjf;lasjdfl;asjfasljkfd
sfl;asfkasldfjkaslfjaslfdkaslfj
asdfl;kasjdf;lasjkflasdjkfalsjf
sadjfasl;fjasl;dfjadslfjaslfjas