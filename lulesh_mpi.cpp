/*
  This is a Version 2.0 MPI + OpenMP implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
* Addition of regions to make work more representative of multi-material codes
* Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
  more representative of our actual working set sizes
* Single source distribution supports pure serial, pure OpenMP, MPI-only, 
  and MPI+OpenMP
* Addition of ability to visualize the mesh using VisIt 
  https://wci.llnl.gov/codes/visit/download.html
* Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

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

*Notable changes in LULESH 2.0

* Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions
*
* The concept of "regions" was added, although every region is the same ideal
*    gas material, and the same sedov blast wave problem is still the only
*    problem its hardcoded to solve.
* Regions allow two things important to making this proxy app more representative:
*   Four of the LULESH routines are now performed on a region-by-region basis,
*     making the memory access patterns non-unit stride
*   Artificial load imbalances can be easily introduced that could impact
*     parallelization strategies.  
* The load balance flag changes region assignment.  Region number is raised to
*   the power entered for assignment probability.  Most likely regions changes
*   with MPI process id.
* The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
*   entered multiple. The cost of 5% is 10x the entered multiple.
* MPI and OpenMP were added, and coalesced into a single version of the source
*   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
*
* Possible Differences in Final Release (other changes possible)
*
* High Level mesh structure to allow data structure transformations
* Different default parameters
* Minor code performance changes and cleanup

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
* CMake based build system

//////////////

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <climits>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include <op_seq.h>
// #include <hdf5.h>
#include <mpi.h>
#include "lulesh.h"

// #include "lulesh-util.h"
#include "lulesh-init.h"

#include "lulesh-viz.h"
// #include "lulesh-visit.cc"

#define USE_DIRTY_BIT_OPT 0

// #include "const.h"
int myRank;
struct cmdLineOpts opts;
double  m_hgcoef = double(3.0);            // hourglass control 
// Cutoffs (treat as constants)
double  m_e_cut = double(1.0e-7);             // energy tolerance 
double  m_p_cut = double(1.0e-7) ;             // pressure tolerance 
double  m_q_cut = double(1.0e-7) ;             // q tolerance 
double  m_v_cut = double(1.0e-10) ;             // relative volume tolerance 
double  m_u_cut = double(1.0e-7) ;             // velocity tolerance 

// Other constants (usually setable, but hardcoded in this proxy app)

double  m_ss4o3 = (double(4.0)/double(3.0));
double  m_qstop = double(1.0e+12);             // excessive q indicator 
double  m_monoq_max_slope = double(1.0);
double  m_monoq_limiter_mult = double(2.0);
double  m_qlc_monoq = double(0.5);         // linear term coef for q 
double  m_qqc_monoq = (double(2.0)/double(3.0));         // quadratic term coef for q 
double  m_qqc = double(2.0);
double  m_eosvmax = double(1.0e+9);
double  m_eosvmin = double(1.0e-9);
double  m_pmin = double(0.);              // pressure floor 
double  m_emin = double(-1.0e+15);              // energy floor 
double  m_dvovmax = double(0.1);           // maximum allowable volume change 
double  m_refdens = double(1.0);           // reference density 

//Consts not defined originally, but for CUDA version
double  m_qqc2 = double(64.0) * m_qqc * m_qqc ;
double  m_ptiny = double(1.e-36) ;
double  m_gamma_t[4*8];
double  m_twelfth = double(1.0)/double(12.0);
double  m_sixth = double(1.0) / double(6.0);
double  m_c1s = double(2.0)/double(3.0);
double  m_ssc_thresh = double(.1111111e-36);
double  m_ssc_low = double(.3333333e-18);


#define FILE_NAME_PATH "file_out.h5"

// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

#include "FBHourglassForceForElems.h"
#include "CalcVolumeDerivatives.h"
#include "CheckForNegativeElementVolume.h"
#include "IntegrateStressForElemsLoop.h"
#include "setForceToZero.h"
#include "CalcAccelForNodes.h"
#include "CalcVeloForNodes.h"
#include "CalcPosForNodes.h"
#include "CalcKinematicsForElem.h"
#include "CalcLagrangeElemRemaining.h"
#include "CalcMonotonicQGradientsForElem.h"
#include "CalcHalfStepBVC.h"
#include "CalcPHalfstep.h"
#include "CalcBVC.h"
#include "updateVolumesForElem.h"
#include "NoExcessiveArtificialViscosity.h"
#include "CalcMonotonicQRegionForElem.h"
#include "ALE3DRelevantCheck.h"
#include "CopyEOSValsIntoArray.h"
#include "CalcHalfSteps.h"
#include "CheckEOSLowerBound.h"
#include "CheckEOSUpperBound.h"
#include "CalcEOSWork.h"
#include "CopyTempEOSVarsBack.h"
#include "CopyVelocityToTempArray.h"
#include "ApplyLowerBoundToVelocity.h"
#include "ApplyUpperBoundToVelocity.h"
#include "CalcNewE.h"
#include "CalcNewEStep2.h"
#include "CalcNewEStep3.h"
#include "CalcNewEStep4.h"
#include "CalcQNew.h"
#include "CalcPNew.h"
#include "CalcSoundSpeedForElem.h"
#include "BoundaryX.h"
#include "BoundaryY.h"
#include "BoundaryZ.h"
#include "CalcHydroConstraint.h"
#include "initStressTerms.h"
#include "CalcCourantConstraint.h"
#include "CalcSpeed.h"
// Define arrays and constants
// Should maybe be moved to a header file to avoid clutter or to the main function to not be global
// Element-centered


static inline 
void logVec(char *funcName, op_dat value){
   std::cout << std::scientific << std::setprecision(6);

   printf("%s\n",funcName);
   printf("\n");
   for (int i =0 ; i<64;i++){
      std::cout << i << "     " << std::right << (double)((double *)value->data)[i] << "\n";
   }
   printf("\n");
}

int m_numElem ;
int m_numNode ;
struct Domain domain;

// This Function can be used when only reading from an HDF5 file
// It replaces the initialise function
static inline
void readOp2VarsFromHDF5(char* file){
   domain.nodes = op_decl_set_hdf5(file, "nodes");
   domain.elems = op_decl_set_hdf5(file, "elems");

   m_numElem = domain.elems->size;
   m_numNode = domain.nodes->size;

   domain.p_nodelist = op_decl_map_hdf5(domain.elems, domain.nodes, 8, file, "nodelist");
   
   domain.p_lxim = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lxim");
   domain.p_lxip = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lxip");
   domain.p_letam = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "letam");
   domain.p_letap = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "letap");
   domain.p_lzetam = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lzetam");
   domain.p_lzetap = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lzetap");

   //Node Centred
   domain.p_t_symmX = op_decl_dat_hdf5(domain.nodes, 1, "int", file, "t_symmX");
   domain.p_t_symmY = op_decl_dat_hdf5(domain.nodes, 1, "int", file, "t_symmY");
   domain.p_t_symmZ = op_decl_dat_hdf5(domain.nodes, 1, "int", file, "t_symmZ");

   domain.p_x = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_x");
   domain.p_y = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_y");
   domain.p_z = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_z");
   domain.p_xd = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_xd");
   domain.p_yd = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_yd");
   domain.p_zd = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_zd");
   domain.p_xdd = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_xdd");
   domain.p_ydd = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_ydd");
   domain.p_zdd = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_zdd");
   domain.p_fx = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_fx");
   domain.p_fy = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_fy");
   domain.p_fz = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_fz");

   domain.p_nodalMass = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_nodalMass");
   //Elem Centred
   domain.p_e = op_decl_dat_hdf5(domain.elems, 1, "double", file, "domain.p_e");
   domain.p_p = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_p");
   domain.p_q = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_q");
   domain.p_ql = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_ql");
   domain.p_qq = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_qq");
   domain.p_v = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_v");
   domain.p_volo = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_volo");
   domain.p_delv = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_delv");
   domain.p_vdov = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_vdov");
   domain.p_arealg = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_arealg");

   domain.p_dxx = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_dxx");
   domain.p_dyy = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_dyy");
   domain.p_dzz = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_dzz");
   
   domain.p_ss = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_ss");
   domain.p_elemMass = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_elemMass");
   domain.p_vnew = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_vnew");
   domain.p_vnewc = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_vnewc");

   //Temporary
   // p_sigxx = op_decl_dat_hdf5(domain.elems, 3, "double", file, "p_sigxx");
   domain.p_sigxx = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_sigxx");
   domain.p_sigyy = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_sigyy");
   domain.p_sigzz = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_sigzz");
   domain.p_determ = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_determ");

   domain.p_dvdx = op_decl_dat_hdf5(domain.elems, 8, "double", file, "dvdx");
   domain.p_dvdy = op_decl_dat_hdf5(domain.elems, 8, "double", file, "dvdy");
   domain.p_dvdz = op_decl_dat_hdf5(domain.elems, 8, "double", file, "dvdz");
   domain.p_x8n = op_decl_dat_hdf5(domain.elems, 8, "double", file, "x8n");
   domain.p_y8n = op_decl_dat_hdf5(domain.elems, 8, "double", file, "y8n");
   domain.p_z8n = op_decl_dat_hdf5(domain.elems, 8, "double", file, "z8n");

   domain.p_delv_xi = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_delv_xi"); 
   domain.p_delv_eta = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_delv_eta"); 
   domain.p_delv_zeta = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_delv_zeta"); 

   domain.p_delx_xi = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_delx_xi"); 
   domain.p_delx_eta = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_delx_eta"); 
   domain.p_delx_zeta = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_delx_zeta"); 

   domain.p_elemBC = op_decl_dat_hdf5(domain.elems, 1, "int", file, "p_elemBC");

   //EOS temp variables
   domain.p_e_old = op_decl_dat_hdf5(domain.elems, 1, "double", file, "e_old"); 
   domain.p_delvc = op_decl_dat_hdf5(domain.elems, 1, "double", file, "delvc");
   domain.p_p_old = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_old");
   domain.p_q_old = op_decl_dat_hdf5(domain.elems, 1, "double", file, "q_old");
   domain.p_compression = op_decl_dat_hdf5(domain.elems, 1, "double", file, "compression");
   domain.p_compHalfStep = op_decl_dat_hdf5(domain.elems, 1, "double", file, "compHalfStep");
   domain.p_qq_old = op_decl_dat_hdf5(domain.elems, 1, "double", file, "qq_old");
   domain.p_ql_old = op_decl_dat_hdf5(domain.elems, 1, "double", file, "ql_old");
   domain.p_work = op_decl_dat_hdf5(domain.elems, 1, "double", file, "work");
   domain.p_p_new = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_new");
   domain.p_e_new = op_decl_dat_hdf5(domain.elems, 1, "double", file, "e_new");
   domain.p_q_new = op_decl_dat_hdf5(domain.elems, 1, "double", file, "q_new");
   domain.p_bvc = op_decl_dat_hdf5(domain.elems, 1, "double", file, "bvc"); ;
   domain.p_pbvc = op_decl_dat_hdf5(domain.elems, 1, "double", file, "pbvc");
   domain.p_pHalfStep = op_decl_dat_hdf5(domain.elems, 1, "double", file, "pHalfStep");

   op_get_const_hdf5("deltatime", 1, "double", (char *)&m_deltatime,  file);
}

// This function was used as a hybrid for reading initialised variables and initalising the ones that are set to 0 itself
static inline
void readAndInitVars(char* file){
   domain.nodes = op_decl_set_hdf5(file, "nodes");
   domain.elems = op_decl_set_hdf5(file, "elems");

   m_numElem = domain.elems->size;
   m_numNode = domain.nodes->size;

   // Allocate Elems
   // e = (double*) malloc(m_numElem * sizeof(double));
   p = (double*) malloc(m_numElem * sizeof(double));
   q = (double*) malloc(m_numElem * sizeof(double));
   ss = (double*) malloc(m_numElem * sizeof(double));
   v = (double*) malloc(m_numElem * sizeof(double));
   delv = (double*) malloc(m_numElem * sizeof(double));
   m_vdov = (double*) malloc(m_numElem * sizeof(double));
   arealg = (double*) malloc(m_numElem * sizeof(double));
   vnew = (double*) malloc(m_numElem * sizeof(double));
   vnewc = (double*) malloc(m_numElem * sizeof(double));
   // Allocate Nodes
   xd = (double*) malloc(m_numNode * sizeof(double)); // Velocities
   yd = (double*) malloc(m_numNode * sizeof(double));
   zd = (double*) malloc(m_numNode * sizeof(double));
   xdd = (double*) malloc(m_numNode * sizeof(double)); //Accelerations
   ydd = (double*) malloc(m_numNode * sizeof(double));
   zdd = (double*) malloc(m_numNode * sizeof(double));
   m_fx = (double*) malloc(m_numNode * sizeof(double)); // Forces
   m_fy = (double*) malloc(m_numNode * sizeof(double));
   m_fz = (double*) malloc(m_numNode * sizeof(double));
   ql = (double*) malloc(m_numElem * sizeof(double));
   qq = (double*) malloc(m_numElem * sizeof(double));
   //Temporary Vars
   sigxx = (double*) malloc(m_numElem*  sizeof(double));
   sigyy = (double*) malloc(m_numElem*  sizeof(double));
   sigzz = (double*) malloc(m_numElem*  sizeof(double));
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


   for(int i=0; i<m_numElem;++i){
      // p[i] = double(0.0);
      e[i] = double(0.0);
      q[i] = double(0.0);
      ss[i] = double(0.0);
   }
   // p_e = op_decl_dat(domain.elems, 1, "double", e, "p_e");
   domain.p_p = op_decl_dat(domain.elems, 1, "double", p, "p_p");
   domain.p_q = op_decl_dat(domain.elems, 1, "double", q, "p_q");
   domain.p_ss = op_decl_dat(domain.elems, 1, "double", ss, "p_ss");

   for(int i=0; i<m_numElem;++i){
      v[i] = double(1.0);
   }
   domain.p_v = op_decl_dat(domain.elems, 1, "double", v, "p_v");

   for (int i = 0; i<m_numNode;++i){
      xd[i] = double(0.0);
      yd[i] = double(0.0);
      zd[i] = double(0.0);
   }
   domain.p_xd = op_decl_dat(domain.nodes, 1, "double", xd, "p_xd");
   domain.p_yd = op_decl_dat(domain.nodes, 1, "double", yd, "p_yd");
   domain.p_zd = op_decl_dat(domain.nodes, 1, "double", zd, "p_zd");

   for (int i = 0; i<m_numNode;++i){
   // for (int i = 0; i<m_numNode*3;++i){
      xdd[i] = double(0.0);
      ydd[i] = double(0.0);
      zdd[i] = double(0.0);
   }
   domain.p_xdd = op_decl_dat(domain.nodes, 1, "double", xdd, "p_xdd");
   domain.p_ydd = op_decl_dat(domain.nodes, 1, "double", ydd, "p_ydd");
   domain.p_zdd = op_decl_dat(domain.nodes, 1, "double", zdd, "p_zdd");

   domain.p_ql = op_decl_dat(domain.elems, 1, "double", ql, "domain.p_ql");
   domain.p_qq = op_decl_dat(domain.elems, 1, "double", qq, "p_qq");
   domain.p_delv = op_decl_dat(domain.elems, 1, "double", delv, "p_delv");
   domain.p_vdov = op_decl_dat(domain.elems, 1, "double", m_vdov, "p_vdov");
   domain.p_arealg = op_decl_dat(domain.elems, 1, "double", arealg, "p_arealg");
   domain.p_vnew = op_decl_dat(domain.elems, 1, "double", vnew, "p_vnew");
   domain.p_vnewc = op_decl_dat(domain.elems, 1, "double", vnewc, "p_vnewc");
   domain.p_fx = op_decl_dat(domain.nodes, 1, "double", m_fx, "p_fx");
   domain.p_fy = op_decl_dat(domain.nodes, 1, "double", m_fy, "p_fy");
   domain.p_fz = op_decl_dat(domain.nodes, 1, "double", m_fz, "p_fz");

   //Temporary Vars
   domain.p_sigxx = op_decl_dat(domain.elems, 1, "double", sigxx, "p_sigxx");
   domain.p_sigyy = op_decl_dat(domain.elems, 1, "double", sigyy, "p_sigyy");
   domain.p_sigzz = op_decl_dat(domain.elems, 1, "double", sigzz, "p_sigzz");
   domain.p_determ = op_decl_dat(domain.elems, 1, "double", determ, "p_determ");

   domain.p_dxx = op_decl_dat(domain.elems, 1, "double", dxx, "p_dxx");
   domain.p_dyy = op_decl_dat(domain.elems, 1, "double", dyy, "p_dyy");
   domain.p_dzz = op_decl_dat(domain.elems, 1, "double", dzz, "p_dzz");

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

   //HDF5 Read
   domain.p_nodelist = op_decl_map_hdf5(domain.elems, domain.nodes, 8, file, "nodelist");
   domain.p_lxim = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lxim");
   domain.p_lxip = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lxip");
   domain.p_letam = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "letam");
   domain.p_letap = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "letap");
   domain.p_lzetam = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lzetam");
   domain.p_lzetap = op_decl_map_hdf5(domain.elems, domain.elems, 1, file, "lzetap");

   domain.p_x = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_x");
   domain.p_y = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_y");
   domain.p_z = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_z");
   domain.p_nodalMass = op_decl_dat_hdf5(domain.nodes, 1, "double", file, "p_nodalMass");
   domain.p_volo = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_volo");
   domain.p_e = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_e");
   domain.p_elemMass = op_decl_dat_hdf5(domain.elems, 1, "double", file, "p_elemMass");
   domain.p_elemBC = op_decl_dat_hdf5(domain.elems, 1, "int", file, "p_elemBC");
   domain.p_t_symmX = op_decl_dat_hdf5(domain.nodes, 1, "int", file, "t_symmX");
   domain.p_t_symmY = op_decl_dat_hdf5(domain.nodes, 1, "int", file, "t_symmY");
   domain.p_t_symmZ = op_decl_dat_hdf5(domain.nodes, 1, "int", file, "t_symmZ");
   op_get_const_hdf5("deltatime", 1, "double", (char *)&m_deltatime,  file);

      //! Start Create Region Sets
   srand(0);
   int myRank = 0;

   m_numReg = 1;
   m_regElemSize = (int*) malloc(m_numReg * sizeof(int));
   m_regElemlist = (int**) malloc(m_numReg * sizeof(int));
   int nextIndex = 0;
   //if we only have one region just fill it
   // Fill out the regNumList with material numbers, which are always
   // the region index plus one 
   if(m_numReg == 1){
      m_regNumList = (int*) malloc(m_numElem * sizeof(int));
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
         costDenominator += pow((i+1), 1);//Total sum of all regions weights
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
      delete [] regBinEnd; 
   }
      // Convert m_regNumList to region index sets
   // First, count size of each region 
   for (int i=0 ; i<m_numElem ; ++i) {
      int r = m_regNumList[i]-1; // region index == regnum-1
      m_regElemSize[r]++;
   }
   // Second, allocate each region index set
   for (int i=0 ; i<m_numReg ; ++i) {
      m_regElemlist[i] = (int*) malloc(m_regElemSize[i]*sizeof(int));
      m_regElemSize[i] = 0;
   }
   // Third, fill index sets
   for (int i=0 ; i<m_numElem ; ++i) {
      int r = m_regNumList[i]-1;       // region index == regnum-1
      int regndx = m_regElemSize[r]++; // Note increment
      m_regElemlist[r][regndx] = i;
   }
   //! End Create Region Sets
   // free(e);
   free(p);
   free(q);
   free(ss);
   free(v);
   free(delv);
   free(m_vdov);
   free(arealg);
   free(vnew);
   free(vnewc);
   free(xd);
   free(yd);
   free(zd);
   free(xdd);
   free(ydd);
   free(zdd);
   free(m_fx);
   free(m_fy);
   free(m_fz);
   free(ql);
   free(qq);
   free(sigxx);
   free(sigyy);
   free(sigzz);
   free(determ);
   free(dvdx);
   free(dvdy);
   free(dvdz);
   free(x8n);
   free(y8n);
   free(z8n);
   free(dxx);
   free(dyy);
   free(dzz);
   free(delx_xi);
   free(delx_eta);
   free(delx_zeta);
   free(delv_xi);
   free(delv_eta);
   free(delv_zeta);
   free(e_old);
   free(delvc);
   free(p_old);
   free(q_old);
   free(compression);
   free(compHalfStep);
   free(qq_old);
   free(ql_old);
   free(work);
   free(p_new);
   free(e_new);
   free(q_new);
   free(bvc);
   free(pbvc);
   free(pHalfStep);


}

/* Work Routines */

static inline
void TimeIncrement(int myrank)
{
   double targetdt = m_stoptime - m_time ;
   if ((m_dtfixed <= double(0.0)) && (m_cycle != int(0))) {
      double ratio ;
      double olddt = m_deltatime ;

      /* This will require a reduction in parallel */
      /* Reduction is done in CalcHydroConstraint and CalcCourantConstraint for OP2*/
      double gnewdt = double(1.0e+20) ;
      double newdt ;
      if (m_dtcourant < gnewdt) {
         gnewdt = m_dtcourant / double(2.0) ;
      }
      if (m_dthydro < gnewdt) {
         gnewdt = m_dthydro * double(2.0) / double(3.0) ;
      }

      newdt = gnewdt;
      
      ratio = newdt / olddt ;
      if (ratio >= double(1.0)) {
         if (ratio < m_deltatimemultlb) {
            newdt = olddt ;
         }
         else if (ratio > m_deltatimemultub) {
            newdt = olddt*m_deltatimemultub ;
         }
      }
      if (newdt > m_dtmax) {
         newdt = m_dtmax ;
      }
      m_deltatime = newdt ;

   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > m_deltatime) &&
       (targetdt < (double(4.0) * m_deltatime / double(3.0))) ) {
      targetdt = double(2.0) * m_deltatime / double(3.0) ;
   }

   if (targetdt < m_deltatime) {
      m_deltatime = targetdt ;
   }

   m_time += m_deltatime ;

   ++m_cycle ;
}



static inline
void InitStressTermsForElems()
{
   // pull in the stresses appropriate to the hydro integration
   // std::cout<<"init stress at " << myRank << "\n";
   op_par_loop(initStressTerms, "initStressTerms", domain.elems,
               op_arg_dat(domain.p_sigxx, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_sigyy, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_sigzz, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_p, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_q, -1, OP_ID, 1, "double", OP_READ));

   // MPI_Barrier(MPI_COMM_WORLD);
}

/******************************************/

static inline
void IntegrateStressForElems()
{
//   if (numthreads > 1) {
//      fx_elem = Allocate<double>(numElem8) ;
//      fy_elem = Allocate<double>(numElem8) ;
//      fz_elem = Allocate<double>(numElem8) ;
//   }
  // loop over all elements
   op_par_loop(IntegrateStressForElemsLoop, "IntegrateStressForElemsLoop", domain.elems,
               op_arg_dat(domain.p_x, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_y, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_z, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_fx, 0, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fx, 1, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fx, 2, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 3, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 4, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 5, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 6, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 7, domain.p_nodelist, 1, "double", OP_INC),
               op_arg_dat(domain.p_fy, 0, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fy, 1, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fy, 2, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 3, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 4, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 5, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 6, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 7, domain.p_nodelist, 1, "double", OP_INC),
               op_arg_dat(domain.p_fz, 0, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fz, 1, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fz, 2, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 3, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 4, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 5, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 6, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 7, domain.p_nodelist, 1, "double", OP_INC),
               op_arg_dat(domain.p_determ, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_sigxx, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_sigyy, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_sigzz, -1, OP_ID, 1, "double", OP_READ)
               );


}

/******************************************/

static inline
void CalcFBHourglassForceForElems()
{
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/
  
/*************************************************/
/*    compute the hourglass modes */
   op_par_loop(FBHourglassForceForElems, "CalcFBHourglassForceForElems", domain.elems,
               op_arg_dat(domain.p_xd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_xd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_xd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_yd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_yd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_yd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_zd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_zd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_zd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_fx, 0, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fx, 1, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fx, 2, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 3, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 4, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 5, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 6, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fx, 7, domain.p_nodelist, 1, "double", OP_INC),
               op_arg_dat(domain.p_fy, 0, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fy, 1, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fy, 2, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 3, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 4, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 5, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 6, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fy, 7, domain.p_nodelist, 1, "double", OP_INC),
               op_arg_dat(domain.p_fz, 0, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fz, 1, domain.p_nodelist, 1, "double", OP_INC), op_arg_dat(domain.p_fz, 2, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 3, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 4, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 5, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 6, domain.p_nodelist, 1, "double", OP_INC),op_arg_dat(domain.p_fz, 7, domain.p_nodelist, 1, "double", OP_INC),
               op_arg_dat(domain.p_dvdx, -1, OP_ID, 8, "double", OP_READ), 
               op_arg_dat(domain.p_dvdy, -1, OP_ID, 8, "double", OP_READ), 
               op_arg_dat(domain.p_dvdz, -1, OP_ID, 8, "double", OP_READ), 
               op_arg_dat(domain.p_x8n, -1, OP_ID, 8, "double", OP_READ), 
               op_arg_dat(domain.p_y8n, -1, OP_ID, 8, "double", OP_READ), 
               op_arg_dat(domain.p_z8n, -1, OP_ID, 8, "double", OP_READ),
               op_arg_dat(domain.p_determ, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_ss, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_elemMass, -1, OP_ID, 1, "double", OP_READ)
   );


   // MPI_Barrier(MPI_COMM_WORLD);
}

/******************************************/

static inline
void CalcHourglassControlForElems()
{
   op_par_loop(CalcVolumeDerivatives, "CalcVolumeDerivatives", domain.elems,
               op_arg_dat(domain.p_x, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_y, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_z, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_dvdx, -1, OP_ID, 8, "double", OP_WRITE), 
               op_arg_dat(domain.p_dvdy, -1, OP_ID, 8, "double", OP_WRITE), 
               op_arg_dat(domain.p_dvdz, -1, OP_ID, 8, "double", OP_WRITE), 
               op_arg_dat(domain.p_x8n, -1, OP_ID, 8, "double", OP_WRITE), 
               op_arg_dat(domain.p_y8n, -1, OP_ID, 8, "double", OP_WRITE), 
               op_arg_dat(domain.p_z8n, -1, OP_ID, 8, "double", OP_WRITE), 
               op_arg_dat(domain.p_v, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_determ, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_volo, -1, OP_ID, 1, "double", OP_READ)
               );
   if ( m_hgcoef > double(0.) ) {
      CalcFBHourglassForceForElems() ;
   }

   return ;
}

/******************************************/

static inline
void CalcVolumeForceForElems()
{
   if (m_numElem != 0) {
      /* Sum contributions to total stress tensor */
      // op_print("Init stress");
      InitStressTermsForElems();

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      // std::cout<<"Integrate stress at " << myRank << "\n";
      IntegrateStressForElems() ;

      // check for negative element volume
      op_par_loop(CheckForNegativeElementVolume, "CheckForNegativeElementVolume", domain.elems,
                  op_arg_dat(domain.p_determ, -1, OP_ID, 1, "double", OP_READ));

      CalcHourglassControlForElems() ;
   }
}

/******************************************/

static inline void CalcForceForNodes()
{
   // op_print("Force to zero");
   op_par_loop(setForceToZero, "setForceToZero", domain.nodes,
               op_arg_dat(domain.p_fx, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_fy, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_fz, -1, OP_ID, 1, "double", OP_WRITE));


  /* Calcforce calls partial, force, hourq */

  CalcVolumeForceForElems() ;

}

/******************************************/

static inline
void CalcAccelerationForNodes()
{   
   op_par_loop(CalcAccelForNodes, "CalcAccelForNodes", domain.nodes,
               op_arg_dat(domain.p_xdd, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_ydd, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_zdd, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_fx, -1, OP_ID, 1, "double", OP_READ), op_arg_dat(domain.p_fy, -1, OP_ID, 1, "double", OP_READ), op_arg_dat(domain.p_fz, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_nodalMass, -1, OP_ID, 1, "double", OP_READ)
               );

}

/******************************************/

static inline
void ApplyAccelerationBoundaryConditionsForNodes()
{
   //Possible improvement would be to check if the current rank has a boundary node at pos 0
   op_par_loop(BoundaryX, "BoundaryX", domain.nodes,
               op_arg_dat(domain.p_xdd, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_t_symmX, -1, OP_ID, 1, "int", OP_READ)
               );


   op_par_loop(BoundaryY, "BoundaryY", domain.nodes,
               op_arg_dat(domain.p_ydd, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_t_symmY, -1, OP_ID, 1, "int", OP_READ)
               );


   op_par_loop(BoundaryZ, "BoundaryZ", domain.nodes,
               op_arg_dat(domain.p_zdd, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_t_symmZ, -1, OP_ID, 1, "int", OP_READ)
               );
}

/******************************************/

static inline
void CalcVelocityForNodes()
{
   op_par_loop(CalcVeloForNodes, "CalcVeloForNodes", domain.nodes,
               op_arg_dat(domain.p_xd, -1, OP_ID, 1, "double", OP_RW), op_arg_dat(domain.p_yd, -1, OP_ID, 1, "double", OP_RW), op_arg_dat(domain.p_zd, -1, OP_ID, 1, "double", OP_RW), 
               op_arg_dat(domain.p_xdd, -1, OP_ID, 1, "double", OP_READ), op_arg_dat(domain.p_ydd, -1, OP_ID, 1, "double", OP_READ), op_arg_dat(domain.p_zdd, -1, OP_ID, 1, "double", OP_READ),
               op_arg_gbl(&m_deltatime, 1, "double", OP_READ) 
               );
}

/******************************************/

static inline
void CalcPositionForNodes()
{
   op_par_loop(CalcPosForNodes, "CalcPosForNodes", domain.nodes,
               op_arg_dat(domain.p_x, -1, OP_ID, 1, "double", OP_INC), op_arg_dat(domain.p_y, -1, OP_ID, 1, "double", OP_INC), op_arg_dat(domain.p_z, -1, OP_ID, 1, "double", OP_INC),
               op_arg_dat(domain.p_xd, -1, OP_ID, 1, "double", OP_READ), op_arg_dat(domain.p_yd, -1, OP_ID, 1, "double", OP_READ), op_arg_dat(domain.p_zd, -1, OP_ID, 1, "double", OP_READ),
               op_arg_gbl(&m_deltatime, 1, "double", OP_READ)
               );  
}

/******************************************/

static inline
void LagrangeNodal()
{
  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
//   op_print("Force For Nodes");
   CalcForceForNodes();

   // std::cout << "Acceleration at " << myRank << "\n";
   CalcAccelerationForNodes();
   // std::cout << "BCs at " << myRank << "\n";
   ApplyAccelerationBoundaryConditionsForNodes();

   // std::cout << "Velocity at " << myRank << "\n";
   CalcVelocityForNodes() ;
   // std::cout << "Calc Pos at " << myRank << "\n";
   CalcPositionForNodes();
   return;
}

/******************************************/

// static inline
// Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
//                const Real_t x2, const Real_t x3,
//                const Real_t x4, const Real_t x5,
//                const Real_t x6, const Real_t x7,
//                const Real_t y0, const Real_t y1,
//                const Real_t y2, const Real_t y3,
//                const Real_t y4, const Real_t y5,
//                const Real_t y6, const Real_t y7,
//                const Real_t z0, const Real_t z1,
//                const Real_t z2, const Real_t z3,
//                const Real_t z4, const Real_t z5,
//                const Real_t z6, const Real_t z7 )
// {
//   Real_t twelveth = Real_t(1.0)/Real_t(12.0);

//   Real_t dx61 = x6 - x1;
//   Real_t dy61 = y6 - y1;
//   Real_t dz61 = z6 - z1;

//   Real_t dx70 = x7 - x0;
//   Real_t dy70 = y7 - y0;
//   Real_t dz70 = z7 - z0;

//   Real_t dx63 = x6 - x3;
//   Real_t dy63 = y6 - y3;
//   Real_t dz63 = z6 - z3;

//   Real_t dx20 = x2 - x0;
//   Real_t dy20 = y2 - y0;
//   Real_t dz20 = z2 - z0;

//   Real_t dx50 = x5 - x0;
//   Real_t dy50 = y5 - y0;
//   Real_t dz50 = z5 - z0;

//   Real_t dx64 = x6 - x4;
//   Real_t dy64 = y6 - y4;
//   Real_t dz64 = z6 - z4;

//   Real_t dx31 = x3 - x1;
//   Real_t dy31 = y3 - y1;
//   Real_t dz31 = z3 - z1;

//   Real_t dx72 = x7 - x2;
//   Real_t dy72 = y7 - y2;
//   Real_t dz72 = z7 - z2;

//   Real_t dx43 = x4 - x3;
//   Real_t dy43 = y4 - y3;
//   Real_t dz43 = z4 - z3;

//   Real_t dx57 = x5 - x7;
//   Real_t dy57 = y5 - y7;
//   Real_t dz57 = z5 - z7;

//   Real_t dx14 = x1 - x4;
//   Real_t dy14 = y1 - y4;
//   Real_t dz14 = z1 - z4;

//   Real_t dx25 = x2 - x5;
//   Real_t dy25 = y2 - y5;
//   Real_t dz25 = z2 - z5;

// #define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
//    ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

//   Real_t volume =
//     TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
//        dy31 + dy72, dy63, dy20,
//        dz31 + dz72, dz63, dz20) +
//     TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
//        dy43 + dy57, dy64, dy70,
//        dz43 + dz57, dz64, dz70) +
//     TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
//        dy14 + dy25, dy61, dy50,
//        dz14 + dz25, dz61, dz50);

// #undef TRIPLE_PRODUCT

//   volume *= twelveth;

//   return volume ;
// }

// /******************************************/

// //inline
// Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
// {
// return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
//                        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
//                        z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
// }

//static inline
void CalcKinematicsForElems()
{
   op_par_loop(CalcKinematicsForElem, "CalcKinematicsForElem", domain.elems,
               op_arg_dat(domain.p_x, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_y, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_z, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_xd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_xd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_xd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_yd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_yd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_yd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_zd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_zd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_zd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_dxx, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_dyy, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_dzz, -1, OP_ID, 1, "double", OP_WRITE), 
               op_arg_dat(domain.p_vnew, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_volo, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_delv, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_v, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_arealg, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_gbl(&m_deltatime, 1, "double", OP_READ)
   );
}

/******************************************/

static inline
void CalcLagrangeElements()
{
   if (m_numElem > 0) {
      CalcKinematicsForElems() ;
      //TODO review should be ok
      op_par_loop(CalcLagrangeElemRemaining, "CalcLagrangeElemRemaining", domain.elems,
                  op_arg_dat(domain.p_dxx, -1, OP_ID, 1, "double", OP_RW), op_arg_dat(domain.p_dyy, -1, OP_ID, 1, "double", OP_RW), op_arg_dat(domain.p_dzz, -1, OP_ID, 1, "double", OP_RW), 
                  op_arg_dat(domain.p_vdov, -1, OP_ID, 1, "double", OP_WRITE),
                  op_arg_dat(domain.p_vnew, -1, OP_ID, 1, "double", OP_READ)
                  );
      // element loop to do some stuff not included in the elemlib function.
   }
}

/******************************************/

static inline
void CalcMonotonicQGradientsForElems()
{
   op_par_loop(CalcMonotonicQGradientsForElem, "CalcMonotonicQGradientsForElem", domain.elems,
               op_arg_dat(domain.p_x, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_x, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_x, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_y, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_y, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_y, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_z, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_z, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_z, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_xd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_xd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_xd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_xd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_yd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_yd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_yd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_yd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_zd, 0, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_zd, 1, domain.p_nodelist, 1, "double", OP_READ), op_arg_dat(domain.p_zd, 2, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 3, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 4, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 5, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 6, domain.p_nodelist, 1, "double", OP_READ),op_arg_dat(domain.p_zd, 7, domain.p_nodelist, 1, "double", OP_READ),
               op_arg_dat(domain.p_volo, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnew, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_delx_zeta, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_delv_zeta, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_delv_xi, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_delx_xi, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_delx_eta, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_delv_eta, -1, OP_ID, 1, "double", OP_WRITE)
               );
}

/******************************************/

static inline
void CalcMonotonicQRegionForElems(int r)
{
   op_set region_elems=domain.region_i[r];

   op_map map_core=domain.region_i_to_elems[r];

   op_map map_lxim=domain.region_i_to_lxim[r];
   op_map map_lxip=domain.region_i_to_lxip[r];
   op_map map_letam=domain.region_i_to_letam[r];
   op_map map_letap=domain.region_i_to_letap[r];
   op_map map_lzetam=domain.region_i_to_lzetam[r];
   op_map map_lzetap=domain.region_i_to_lzetap[r];

   op_par_loop(CalcMonotonicQRegionForElem, "CalcMonotonicQRegionForElem",region_elems,
               op_arg_dat(domain.p_delv_xi, 0, map_core, 1, "double", OP_READ), op_arg_dat(domain.p_delv_xi, 0, map_lxim, 1, "double", OP_READ),op_arg_dat(domain.p_delv_xi, 0, map_lxip, 1, "double", OP_READ),
               op_arg_dat(domain.p_delv_eta, 0, map_core, 1, "double", OP_READ), op_arg_dat(domain.p_delv_eta, 0, map_letam, 1, "double", OP_READ),op_arg_dat(domain.p_delv_eta, 0, map_letap, 1, "double", OP_READ),
               op_arg_dat(domain.p_delv_zeta, 0, map_core, 1, "double", OP_READ), op_arg_dat(domain.p_delv_zeta, 0, map_lzetam, 1, "double", OP_READ),op_arg_dat(domain.p_delv_zeta, 0, map_lzetap, 1, "double", OP_READ),
               op_arg_dat(domain.p_delx_xi, 0, map_core, 1, "double", OP_READ), op_arg_dat(domain.p_delx_eta, 0, map_core, 1, "double", OP_READ), op_arg_dat(domain.p_delx_zeta, 0, map_core, 1, "double", OP_READ),
               op_arg_dat(domain.p_elemBC, 0, map_core, 1, "int", OP_READ), 
               op_arg_dat(domain.p_vdov, 0, map_core, 1, "double", OP_READ),
               op_arg_dat(domain.p_qq, 0, map_core, 1, "double", OP_WRITE), op_arg_dat(domain.p_ql, 0, map_core, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_elemMass, 0, map_core, 1, "double", OP_READ), 
               op_arg_dat(domain.p_volo, 0, map_core, 1, "double", OP_READ), 
               op_arg_dat(domain.p_vnew, 0, map_core, 1, "double", OP_READ)
               );

}

/******************************************/

static inline
void CalcMonotonicQForElems()
{  
   //
   // calculate the monotonic q for all regions
   //
   // // The OP2 version does not support multiple regions yet
   // // Code is left here for future reference
   // for (int r=0 ; r<domain.numReg() ; ++r) {
   for (int r=0 ; r<m_numReg ; ++r) {
      // if (domain.regElemSize(r) > 0) {
      // if (m_regElemSize[r] > 0) {
      if(m_numElem > 0){
         // CalcMonotonicQRegionForElems(domain, r, ptiny) ;
         CalcMonotonicQRegionForElems(r) ;
      }
   }
}


/******************************************/

static inline
void CalcQForElems()
{
   //
   // MONOTONIC Q option
   //
   if (m_numElem != 0) {

      /* Calculate velocity gradients */
      CalcMonotonicQGradientsForElems();
    
      CalcMonotonicQForElems();
      // Free up memory
      // domain.DeallocateGradients();
      // DeallocGradients();

      /* Don't allow excessive artificial viscosity */
      op_par_loop(NoExcessiveArtificialViscosity, "NoExcessiveArtificialViscosity", domain.elems,
                  op_arg_dat(domain.p_q, -1, OP_ID, 1, "double", OP_READ));
   }
}

/******************************************/

static inline
void CalcPressureForElemsHalfstep(int region)
{
   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];
   op_par_loop(CalcHalfStepBVC, "CalcHalfStepBVC", current_set,
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_compHalfStep, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_pbvc, 0, current_map, 1, "double", OP_WRITE));
   domain.p_bvc->dirtybit=0;
   domain.p_pbvc->dirtybit=0;


   //NOTE changed p_pHalfStep to write review
   op_par_loop(CalcPHalfstep, "CalcPHalfstep", current_set,
               op_arg_dat(domain.p_pHalfStep, 0, current_map, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ)
               );
   domain.p_pHalfStep->dirtybit=0;
}

static inline
void CalcPressureForElems(int region)
{
   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];
   op_par_loop(CalcBVC, "CalcBVC", current_set,
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_compression, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_pbvc, 0, current_map, 1, "double", OP_WRITE));
   #if USE_DIRTY_BIT_OPT 
   domain.p_bvc->dirtybit=0;
   domain.p_pbvc->dirtybit=0;
   #endif
   op_par_loop(CalcPNew, "CalcPNew", current_set,0
               op_arg_dat(domain.p_p_new, 0, current_map, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ)
               );
   #if USE_DIRTY_BIT_OPT 
   domain.p_p_new->dirtybit=0;
   #endif
}

/******************************************/

static inline
void CalcEnergyForElems(int region)
{
   // double *pHalfStep = Allocate<double>(length) ;
   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];

   op_par_loop(CalcNewE, "CalcNewE", current_set,
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_WRITE),
               op_arg_dat(domain.p_e_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_delvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_p_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_work, 0, current_map, 1, "double", OP_READ)
   );
   #if USE_DIRTY_BIT_OPT 
   domain.p_e_new->dirtybit=0;
   #endif
   CalcPressureForElemsHalfstep(region);

   //NOTE domain.p_e_new may be an INC
   op_par_loop(CalcNewEStep2, "CalcNewEStep2", current_set,
               op_arg_dat(domain.p_compHalfStep, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_delvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q_new, 0, current_map, 1, "double", OP_RW),
               op_arg_dat(domain.p_pbvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_RW),
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_pHalfStep, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_ql_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_qq_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_p_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q_old, 0, current_map, 1, "double", OP_READ)
   );
   #if USE_DIRTY_BIT_OPT 
   domain.p_q_new->dirtybit=0;
   domain.p_e_new->dirtybit=0;
   #endif

   //NOTE domain.p_e_new may be an INC
   op_par_loop(CalcNewEStep3, "CalcNewEStep3", current_set,
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_RW),
               op_arg_dat(domain.p_work, 0, current_map, 1, "double", OP_READ)
   );
   #if USE_DIRTY_BIT_OPT 
   domain.p_e_new->dirtybit=0;
   #endif
   CalcPressureForElems(region);

   //NOTE domain.p_e_new may be an INC
   op_par_loop(CalcNewEStep4, "CalcNewEStep4", current_set,
               op_arg_dat(domain.p_delvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_pbvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_RW),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_p_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_ql_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_qq_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_p_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_pHalfStep, 0, current_map, 1, "double", OP_READ)
   );
   #if USE_DIRTY_BIT_OPT 
   domain.p_e_new->dirtybit=0;
   #endif
   CalcPressureForElems(region);
   //NOTE p_q_new could probably be a write
   op_par_loop(CalcQNew, "CalcQNew", current_set,
               op_arg_dat(domain.p_delvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_pbvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_p_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q_new, 0, current_map, 1, "double", OP_RW),
               op_arg_dat(domain.p_ql_old, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_qq_old, 0, current_map, 1, "double", OP_READ)
               );
   #if USE_DIRTY_BIT_OPT
   domain.p_q_new->dirtybit=0;
   #endif

   return ;
}

/******************************************/

static inline
void CalcSoundSpeedForElems(int region)
{
   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];
   op_par_loop(CalcSoundSpeedForElem, "CalcSoundSpeedForElem", current_set,
               op_arg_dat(domain.p_pbvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_bvc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_p_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_ss, 0, current_map, 1, "double", OP_WRITE)
               );
   domain.p_ss->dirtybit=0;
}

/******************************************/

static inline
void EvalEOSForElems(int region, int rep)
{

   double  e_cut = m_e_cut ;
   double  p_cut = m_p_cut ;
   double  ss4o3 = m_ss4o3 ;
   double  q_cut = m_q_cut ;

   double eosvmax = m_eosvmax ;
   double eosvmin = m_eosvmin ;
   double pmin    = m_pmin ;
   double emin    = m_emin ;
   double rho0    = m_refdens ;

   // These temporaries will be of different size for OP_ID
   // each call (due to different sized region element
   // lists)

   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];

   //loop to add load imbalance based on region number 
   op_par_loop(CopyEOSValsIntoArray, "CopyEOSValsIntoArray", current_set,
               op_arg_dat(domain.p_e_old, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_e, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_delvc, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_delv, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_p_old, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_p, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q_old, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_q, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_qq_old, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_qq, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_ql_old, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_ql, 0, current_map, 1, "double", OP_READ));
   #if USE_DIRTY_BIT_OPT 
   domain.p_e_old->dirtybit=0;
   domain.p_delvc->dirtybit=0;
   domain.p_p_old->dirtybit=0;
   domain.p_q_old->dirtybit=0;
   domain.p_qq_old->dirtybit=0;
   domain.p_ql_old->dirtybit=0;
   #endif
   for(int j = 0; j < rep; j++) {
      /* compress data, minimal set */



         op_par_loop(CalcHalfSteps, "CalcHalfSteps", current_set,
                     op_arg_dat(domain.p_compression, 0, current_map, 1, "double", OP_WRITE),
                     op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
                     op_arg_dat(domain.p_delvc, 0, current_map, 1, "double", OP_READ),
                     op_arg_dat(domain.p_compHalfStep, 0, current_map, 1, "double", OP_WRITE)
         );
#if USE_DIRTY_BIT_OPT 
         domain.p_compression->dirtybit=0;
         domain.p_ql_old->dirtybit=0;
#endif

      /* Check for v > eosvmax or v < eosvmin */
         if ( eosvmin != double(0.) ) {
            op_par_loop(CheckEOSLowerBound, "CheckEOSLowerBound", current_set,
                        op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
                        op_arg_dat(domain.p_compHalfStep, 0, current_map, 1, "double", OP_WRITE),
                        op_arg_dat(domain.p_compression, 0, current_map, 1, "double", OP_READ)
            );
         }
         #if USE_DIRTY_BIT_OPT 
         domain.p_compHalfStep->dirtybit=0;
         #endif
         if ( eosvmax != double(0.) ) {
            op_par_loop(CheckEOSUpperBound, "CheckEOSUpperBound", current_set,
                        op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
                        op_arg_dat(domain.p_compHalfStep, 0, current_map, 1, "double", OP_WRITE),
                        op_arg_dat(domain.p_compression, 0, current_map, 1, "double", OP_WRITE),
                        op_arg_dat(domain.p_p_old, 0, current_map, 1, "double", OP_WRITE)
            );
         }
         #if USE_DIRTY_BIT_OPT 
         domain.p_compression->dirtybit=0;
         domain.p_ql_old->dirtybit=0;
         domain.p_compHalfStep->dirtybit=0;
         #endif
         op_par_loop(CalcEOSWork, "CalcEOSWork", current_set,
                     op_arg_dat(domain.p_work, 0, current_map, 1 , "double", OP_WRITE));
         #if USE_DIRTY_BIT_OPT
         domain.p_work->dirtybit=0;
         #endif

      CalcEnergyForElems(region);
   }

   op_par_loop(CopyTempEOSVarsBack, "CopyTempEOSVarsBack", current_set,
               op_arg_dat(domain.p_p, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_p_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_e, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_e_new, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(domain.p_q, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(domain.p_q_new, 0, current_map, 1, "double", OP_READ)
   );
   #if USE_DIRTY_BIT_OPT 
   domain.p_p->dirtybit=0;
   domain.p_e->dirtybit=0;
   domain.p_q->dirtybit=0;
   #endif

   CalcSoundSpeedForElems(region) ;


}

/******************************************/

static inline
void ApplyMaterialPropertiesForElems()
{
   if (m_numElem != 0) {

      // MPI_Barrier(MPI_COMM_WORLD);
      // op_print("CopyVelo");
      op_par_loop(CopyVelocityToTempArray, "CopyVelocityToTempArray", domain.elems,
                  op_arg_dat(domain.p_vnewc, -1, OP_ID, 1, "double", OP_WRITE),
                  op_arg_dat(domain.p_vnew, -1, OP_ID, 1, "double", OP_READ));

       // Bound the updated relative volumes with eosvmin/max
      // MPI_Barrier(MPI_COMM_WORLD);
      // op_print("Lower Bound");
       if (m_eosvmin != double(0.)) {
         op_par_loop(ApplyLowerBoundToVelocity, "ApplyLowerBoundToVelocity", domain.elems,
                  op_arg_dat(domain.p_vnewc, -1, OP_ID, 1, "double", OP_WRITE)
                  );

       }
      // MPI_Barrier(MPI_COMM_WORLD);
      // op_print("Upper bound");
       if (m_eosvmax != double(0.)) {
         op_par_loop(ApplyUpperBoundToVelocity, "ApplyUpperBoundToVelocity", domain.elems,
                  op_arg_dat(domain.p_vnewc, -1, OP_ID, 1, "double", OP_WRITE)
                  );
       }

       // This check may not make perfect sense in LULESH, but
       // it's representative of something in the full code -
       // just leave it in, please
      // MPI_Barrier(MPI_COMM_WORLD);
      // op_print("Ale3d");
      op_par_loop(ALE3DRelevantCheck, "ALE3DRelevantCheck", domain.elems,
                  op_arg_dat(domain.p_v, -1, OP_ID, 1, "double", OP_READ)
                  );

    

   for (int r=0 ; r<m_numReg ; r++) {
       int rep;
       //Determine load imbalance for this region
       //round down the number with lowest cost
       //  if(r < domain.numReg()/2)
      if(r < m_numReg/2)
         rep = 1;
         //you don't get an expensive region unless you at least have 5 regions
      else if(r < (m_numReg - (m_numReg+15)/20))
         rep = 1 + m_cost;
       //very expensive regions
      else
         rep = 10 * (1+ m_cost);
      //    MPI_Barrier(MPI_COMM_WORLD);
         // op_print("Eval EOS");
       EvalEOSForElems( r , rep);
   }
   #if USE_DIRTY_BIT_OPT
   domain.p_e_old->dirtybit=1; //only here
   domain.p_delvc->dirtybit=1; //only here
   domain.p_p_old->dirtybit=1; //only here
   domain.p_q_old->dirtybit=1; //only here
   domain.p_qq_old->dirtybit=1; //only here
   domain.p_ql_old->dirtybit=1; //only here
   domain.p_compHalfStep->dirtybit=1; //only here
   domain.p_work->dirtybit=1; //only here
   domain.p_p->dirtybit=1; //global can be copied out
   domain.p_e->dirtybit=1; //global can be copied out
   domain.p_q->dirtybit=1; //global can be copied out
   domain.p_e_new->dirtybit=1;      //only here
   domain.p_q_new->dirtybit=1;      //only here
   domain.p_p_new->dirtybit=1;      //only here
   domain.p_bvc->dirtybit=1;        //only here
   domain.p_pbvc->dirtybit=1;       //only here
   #endif

   //  Release(&vnewc) ;
  }
}

/******************************************/

static inline
void UpdateVolumesForElems()
{
   if (m_numElem != 0) {
      op_par_loop(updateVolumesForElem, "UpdateVolumesForElems", domain.elems,
               op_arg_dat(domain.p_vnew, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_v, -1, OP_ID, 1, "double", OP_WRITE)
               );

   }
   return ;
}

/******************************************/

static inline
void LagrangeElements()
{
   // op_print("Calc Lagrange Elements");
  CalcLagrangeElements() ;

  /* Calculate Q.  (Monotonic q option requires communication) */
   //   MPI_Barrier(MPI_COMM_WORLD);
   //   op_print("Q For Elems");
  CalcQForElems() ;

   // MPI_Barrier(MPI_COMM_WORLD);
   // op_print("Material props");
  ApplyMaterialPropertiesForElems() ;

// MPI_Barrier(MPI_COMM_WORLD);
   // op_print("Update Vols for Elems");
  UpdateVolumesForElems() ;
}

/******************************************/

static inline
void CalcCourantConstraintForElems(int region)
{
   op_set currRegion= domain.region_i[region];
   op_map currMap= domain.region_i_to_elems[region];
   op_par_loop(CalcCourantConstraint, "CalcCourantConstraint", currRegion,
               op_arg_dat(domain.p_ss, 0, currMap, 1, "double", OP_READ),
               op_arg_dat(domain.p_vdov, 0, currMap, 1, "double", OP_READ),
               op_arg_dat(domain.p_arealg, 0, currMap, 1, "double", OP_READ),
               op_arg_gbl(&m_dtcourant, 1, "double", OP_MIN)
   );

   return ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(int region)
{
   op_set currRegion= domain.region_i[region];
   op_map currMap= domain.region_i_to_elems[region];
   op_par_loop(CalcHydroConstraint, "CalcHydroConstraint", currRegion,
               op_arg_dat(domain.p_vdov, 0, currMap, 1, "double", OP_READ),
               op_arg_gbl(&m_dthydro, 1, "double", OP_MIN)            
   );
   return ;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems() {

   // Initialize conditions to a very large value
   m_dtcourant = 1.0e+20;
   m_dthydro = 1.0e+20;

   for (int r=0 ; r < m_numReg ; ++r) { 
      /* evaluate time constraint */
      CalcCourantConstraintForElems(r) ;

      /* check hydro constraint */
      CalcHydroConstraintForElems(r) ;


   }
}

/******************************************/

static inline
void LagrangeLeapFrog()
{
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   // op_print("Nodal");
   LagrangeNodal();

   // op_dump_dat()
   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   // MPI_Barrier(MPI_COMM_WORLD);
   // op_print("Elements");
   LagrangeElements();
  
   CalcTimeConstraintsForElems();
}


/******************************************/





void VerifyAndWriteFinalOutput(double elapsed_time,
                               int cycle,
                               int nx,
                               int numRanks, double* e)
{
   // GrindTime1 only takes a single domain into account, and is thus a good way to measure
   // processor speed indepdendent of MPI parallelism.
   // GrindTime2 takes into account speedups from MPI parallelism.
   // Cast to 64-bit integer to avoid overflow
   Int8_t nx8 = nx;
   double grindTime1 = (((elapsed_time*1e6)/cycle)*numRanks)/((nx8*nx8*nx8));
   double grindTime2 = ((elapsed_time*1e6)/cycle)/(nx8*nx8*nx8);

   int ElemId = 0;
   std::cout << "Run completed:\n";
   std::cout << "   Problem size        =  " << nx       << "\n";
   std::cout << "   MPI tasks           =  " << numRanks << "\n";
   std::cout << "   Iteration count     =  " << cycle << "\n";
   std::cout << "   Final Origin Energy =  ";
   std::cout << std::scientific << std::setprecision(6);
   // std::cout << std::setw(12) << locDom.e(ElemId) << "\n";
   std::cout << std::setw(12) << e[ElemId] << "\n";

   double   MaxAbsDiff = double(0.0);
   double TotalAbsDiff = double(0.0);
   double   MaxRelDiff = double(0.0);

   for (int j=0; j<nx; ++j) {
      for (int k=j+1; k<nx; ++k) {
         double AbsDiff = FABS(e[j*nx+k]-e[k*nx+j]);
         TotalAbsDiff  += AbsDiff;

         if (MaxAbsDiff <AbsDiff) MaxAbsDiff = AbsDiff;

         double RelDiff = AbsDiff / e[k*nx+j];

         if (MaxRelDiff <RelDiff)  MaxRelDiff = RelDiff;
      }
   }

   // Quick symmetry check
   std::cout << "   Testing Plane 0 of Energy Array on rank 0:\n";
   std::cout << "        MaxAbsDiff   = " << std::setw(12) << MaxAbsDiff   << "\n";
   std::cout << "        TotalAbsDiff = " << std::setw(12) << TotalAbsDiff << "\n";
   std::cout << "        MaxRelDiff   = " << std::setw(12) << MaxRelDiff   << "\n";

   // Timing information
   std::cout.unsetf(std::ios_base::floatfield);
   std::cout << std::setprecision(4);
   std::cout << "\nElapsed time         = " << std::setw(10) << elapsed_time << " (s)\n";
   std::cout << std::setprecision(8);
   std::cout << "Grind time (us/z/c)  = "  << std::setw(10) << grindTime1 << " (per dom)  ("
             << std::setw(10) << elapsed_time << " overall)\n";
   std::cout << "FOM                  = " << std::setw(10) << 1000.0/grindTime2 << " (z/s)\n\n";

   return ;
}

void InitMeshDecomp(int numRanks, int myRank,
                    int *col, int *row, int *plane, int *side)
{
   int testProcs;
   int dx, dy, dz;
   int myDom;
   
   // Assume cube processor layout for now 
   testProcs = int(cbrt(double(numRanks))+0.5) ;
   if (testProcs*testProcs*testProcs != numRanks) {
      printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;    
      MPI_Abort(MPI_COMM_WORLD, -1) ;
   }
   if (sizeof(double) != 4 && sizeof(double) != 8) {
      printf("MPI operations only support float and double right now...\n");
      MPI_Abort(MPI_COMM_WORLD, -1) ;
   }
   if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
      printf("corner element comm buffers too small.  Fix code.\n") ;
      MPI_Abort(MPI_COMM_WORLD, -1) ;
   }

   dx = testProcs ;
   dy = testProcs ;
   dz = testProcs ;

   // temporary test
   if (dx*dy*dz != numRanks) {
      printf("error -- must have as many domains as procs\n") ;
      MPI_Abort(MPI_COMM_WORLD, -1) ;
   }
   int remainder = dx*dy*dz % numRanks ;
   if (myRank < remainder) {
      myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
   }
   else {
      myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
         (myRank - remainder)*(dx*dy*dz/numRanks) ;
   }

   *col = myDom % dx ;
   *row = (myDom / dx) % dy ;
   *plane = myDom / (dx*dy) ;
   *side = testProcs;

   return;
}

//Neded to render regiosn in file

/******************************************/

int main(int argc, char *argv[])
{
   op_init(argc, argv, 1);

   int numRanks ;

// std::cout << "CHENCK THAT STUFF IS PRINTING\n";
//    std::cout << "USING MPI HERE JUST TO BE SURE\n";

   MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   /* Set defaults that can be overridden by command line opts */
   opts.its = 9999999; // Iterations
   opts.nx  = 30; //Size
   opts.numReg = 1;
   opts.numFiles = (int)(numRanks+10)/9;
   opts.showProg = 0;
   opts.quiet = 0;
   opts.viz = 0;
   opts.balance = 1;
   opts.cost = 1;
   opts.itert = 1;
   m_numReg = 1;


   ParseCommandLineOptions(argc, argv, myRank, &opts);

   if ((myRank == 0) && (opts.quiet == 0)) {
      std::cout << "Running problem size " << opts.nx << "^3 per domain until completion\n";
      std::cout << "Num processors: "      << numRanks << "\n";
      #if _OPENMP
      std::cout << "Num threads: " << omp_get_max_threads() << "\n";
      #endif
      std::cout << "Total number of elements: " << (opts.nx*opts.nx*opts.nx) << " \n\n";
      std::cout << "To run other sizes, use -s <integer>.\n";
      std::cout << "To run a fixed number of iterations, use -i <integer>.\n";
      std::cout << "To run a more or less balanced region set, use -b <integer>.\n";
      std::cout << "To change the relative costs of regions, use -c <integer>.\n";
      std::cout << "To print out progress, use -p\n";
      std::cout << "To write an output file for VisIt, use -v\n";
      std::cout << "See help (-h) for more options\n\n";
   }

   // Set up the mesh and decompose. Assumes regular cubes for now
   int col, row, plane, side;
   // InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);


   //locDom
   int edgeElems = opts.nx;
   int edgeNodes = edgeElems+1;

   sizeX = edgeElems ;
   sizeY = edgeElems ;
   sizeZ = edgeElems ;
   m_numElem = edgeElems * edgeElems * edgeElems;
   m_numNode = edgeNodes * edgeNodes * edgeNodes;


   //generate the global size
   g_numElem = opts.nx*opts.nx*opts.nx;
   g_numNode = (opts.nx+1)*(opts.nx+1)*(opts.nx+1);

   //Compute the local size
   m_numElem =  compute_local_size(g_numElem,(Int8_t)numRanks,myRank);
   m_numNode =  compute_local_size(g_numNode,(Int8_t)numRanks,myRank);

   MPI_Barrier(MPI_COMM_WORLD);

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

   m_gamma_t[0] = double( 1.);
   m_gamma_t[1] = double( 1.);
   m_gamma_t[2] = double(-1.);
   m_gamma_t[3] = double(-1.);
   m_gamma_t[4] = double(-1.);
   m_gamma_t[5] = double(-1.);
   m_gamma_t[6] = double( 1.);
   m_gamma_t[7] = double( 1.);
   m_gamma_t[8] = double( 1.);
   m_gamma_t[9] = double(-1.);
   m_gamma_t[10] = double(-1.);
   m_gamma_t[11] = double( 1.);
   m_gamma_t[12] = double(-1.);
   m_gamma_t[13] = double( 1.);
   m_gamma_t[14] = double( 1.);
   m_gamma_t[15] = double(-1.);
   m_gamma_t[16] = double( 1.);
   m_gamma_t[17] = double(-1.);
   m_gamma_t[18] = double( 1.);
   m_gamma_t[19] = double(-1.);
   m_gamma_t[20] = double( 1.);
   m_gamma_t[21] = double(-1.);
   m_gamma_t[22] = double( 1.);
   m_gamma_t[23] = double(-1.);
   m_gamma_t[24] = double(-1.);
   m_gamma_t[25] = double( 1.);
   m_gamma_t[26] = double(-1.);
   m_gamma_t[27] = double( 1.);
   m_gamma_t[28] = double( 1.);
   m_gamma_t[29] = double(-1.);
   m_gamma_t[30] = double( 1.);
   m_gamma_t[31] = double(-1.);

   //Declare Constants
   op_decl_const(1, "double", &m_e_cut);
   op_decl_const(1, "double", &m_p_cut);
   op_decl_const(1, "double", &m_q_cut);
   op_decl_const(1, "double", &m_v_cut);
   op_decl_const(1, "double", &m_u_cut);

   op_decl_const(1, "double", &m_hgcoef);
   op_decl_const(1, "double", &m_ss4o3);
   op_decl_const(1, "double", &m_qstop);
   op_decl_const(1, "double", &m_monoq_max_slope);
   op_decl_const(1, "double", &m_monoq_limiter_mult);
   op_decl_const(1, "double", &m_qlc_monoq);
   op_decl_const(1, "double", &m_qqc_monoq);
   op_decl_const(1, "double", &m_qqc);
   op_decl_const(1, "double", &m_eosvmax);
   op_decl_const(1, "double", &m_eosvmin);
   op_decl_const(1, "double", &m_pmin);
   op_decl_const(1, "double", &m_emin);
   op_decl_const(1, "double", &m_dvovmax);
   op_decl_const(1, "double", &m_refdens);
   op_decl_const(1, "double", &m_qqc2);

   op_decl_const(1, "double", &m_ptiny);
   op_decl_const(32, "double", m_gamma_t);
   op_decl_const(1, "double", &m_twelfth);
   op_decl_const(1, "double", &m_sixth);
   op_decl_const(1, "double", &m_c1s);
   op_decl_const(1, "double", &m_ssc_thresh);
   op_decl_const(1, "double", &m_ssc_low);

   char file[] = FILE_NAME_PATH;
   // readAndInitVars(file);->
   // readOp2VarsFromHDF5(file);   

   domain = initialiseALL(opts,myRank,(Int8_t)numRanks);

   writeFileADHFJ(myRank,
   domain.p_x->set->size+domain.p_x->set->exec_size+domain.p_x->set->nonexec_size,
   (double *)domain.p_x->data,
   (double *)domain.p_y->data,
   (double *)domain.p_z->data,"/home/joseph/3rdYear/testingFolder/partitioning/Before/Node");

   double * speed=(double *)malloc(m_numNode*sizeof(double));
   op_dat p_speed=op_decl_dat(domain.nodes, 1, "double", speed, "p_speed");

   int siz = op_get_size(domain.elems);
   std::cout <<  "Global Size: " << siz << ", Local Elems: " << domain.elems->size << ", Local Nodes: "<< domain.nodes->size << "\n";
   MPI_Barrier(MPI_COMM_WORLD);
   if (myRank==0){printf("\n\n");}
   op_diagnostic_output();

   op_dump_to_hdf5("/home/joseph/3rdYear/TestOut");
   switch (opts.partition)
   {
      case Partition_S:
         op_partition("PTSCOTCH", "KWAY", domain.nodes, domain.p_nodelist, p_loc);
         break;
      case Partition_PK:
         op_partition("PARMETIS", "KWAY", domain.nodes, domain.p_nodelist, p_loc);
         break;
      case Partition_PG:
         op_partition("PARMETIS", "GEOM", domain.nodes, domain.p_nodelist, p_loc);
         break;
      case Partition_PGK:
         op_partition("PARMETIS", "GEOMKWAY", domain.nodes, domain.p_nodelist, p_loc);
         break;
      case Partition_K:
         op_partition("KAHIP", "KWAY", domain.nodes, domain.p_nodelist, p_loc);
         break;
   }

   // op_dump_to_hdf5("file_out.h5");
   // op_write_const_hdf5("deltatime", 1, "double", (char*)&m_deltatime, FILE_NAME_PATH);

   MPI_Barrier(MPI_COMM_WORLD);
   if (myRank==0){printf("\n\n");}

   if (myRank==0){
      printf("size %d\n",domain.p_x->set->size);
      printf("exec_size x%d\n",domain.p_x->set->exec_size);
      printf("nonexec_size x%d\n",domain.p_x->set->nonexec_size);

      printf("exec_size y%d\n",domain.p_y->set->exec_size);
      printf("nonexec_size y %d\n",domain.p_y->set->nonexec_size);

      printf("exec_size z %d\n",domain.p_z->set->exec_size);
      printf("nonexec_size z%d\n",domain.p_z->set->nonexec_size);
      printf("m_nodes %d \n",m_numNode);
   }



   // SHOW THE CUBE AS DIVIDED
   writeFileADHFJ(myRank,
   domain.p_x->set->size+domain.p_x->set->exec_size+domain.p_x->set->nonexec_size,
   (double *)domain.p_x->data,
   (double *)domain.p_y->data,
   (double *)domain.p_z->data,"/home/joseph/3rdYear/testingFolder/partitioning/After/Node");

   printf("DONE\n");
   // writeFileADHFJ(myRank,p_x->set->nonexec_size+p_x->set->exec_size,(double *)p_x->data,(double *)p_y->data,(double *)p_z->data,"/home/joseph/3rdYear/testingFolder/partitioning/All/Node");
   // double *outputX = (double*) malloc(m_numNode*sizeof(double));
   // double *outputY = (double*) malloc(m_numNode*sizeof(double));
   // double *outputZ = (double*) malloc(m_numNode*sizeof(double));
   // op_fetch_data(domain.p_x, outputX);
   // op_fetch_data(domain.p_z,outputY);
   // op_fetch_data(domain.p_z,outputZ);
   // writeFileADHFJ(myRank,m_numNode,outputX,outputY,outputZ,"/home/joseph/3rdYear/testingFolder/partitioning/After/Node");
   // free(outputX);
   // free(outputY);
   // free(outputZ);

   MPI_Barrier(MPI_COMM_WORLD);

   
   // BEGIN timestep to solution */
   double cpu_t1, cpu_t2, wall_t1, wall_t2;
   op_timers(&cpu_t1, &wall_t1);

   // !Main Loop
   while((m_time < m_stoptime) && (m_cycle < opts.its)) {
      TimeIncrement(myRank) ;
      LagrangeLeapFrog() ;

      if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank==0)) {
         std::cout << "rank = " << myRank       << ", "
                   << "cycle = " << m_cycle       << ", "
                   << std::scientific
                   << "time = " << double(m_time) << ", "
                   << "dt="     << double(m_deltatime) << "\n";
         std::cout.unsetf(std::ios_base::floatfield);
      }
   }

   op_timers(&cpu_t2, &wall_t2);
   double walltime = wall_t2 - wall_t1;

   // Write out final viz file */
   // Currently no support for visualisation
   // if (opts.viz) {
      // DumpToVisit(*locDom, opts.numFiles, myRank, numRanks) ;
   // }

   // writeFileADHFJ(myRank,
   //    p_x->set->size+p_x->set->exec_size+p_x->set->nonexec_size,
   //    (double *)p_x->data,
   //    (double *)p_y->data,
   //    (double *)p_z->data,"/home/joseph/3rdYear/testingFolder/partitioning/All/Node");

   double *verify_e = (double*) malloc(m_numElem*sizeof(double));
   op_fetch_data(domain.p_e, verify_e);

   if ((myRank == 0) && (opts.quiet == 0)) {
      VerifyAndWriteFinalOutput(walltime, m_cycle, opts.nx, numRanks, verify_e);
   }

   if (opts.time){op_timing_output();}

   
   op_par_loop(CalcSpeed, "CalcSpeed", domain.nodes,
         op_arg_dat(p_speed, -1, OP_ID, 1, "double", OP_WRITE),
         op_arg_dat(domain.p_xd, -1, OP_ID, 1, "double", OP_READ),
         op_arg_dat(domain.p_yd, -1, OP_ID, 1, "double", OP_READ),
         op_arg_dat(domain.p_zd, -1, OP_ID, 1, "double", OP_READ));   
      
   if (opts.viz){
      writeSiloFile(myRank, m_cycle, g_numElem, g_numNode, m_numElem, m_numNode, nodelist, domain.p_x,
            domain.p_y, domain.p_z, domain.p_e, domain.p_p, domain.p_v, domain.p_q, domain.p_xd, domain.p_yd, 
            domain.p_zd, p_speed
         );
   }


   printf("finishing\n");

   // op_dump_to_hdf5("/home/joseph/3rdYear/TestOut");

   op_exit(); 

   return 0 ;
}
