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


#if !defined(USE_MPI)
#define USE_MPI 0
#endif

#if USE_MPI
#include <mpi.h>
#endif
#include "lulesh.h"


#include "lulesh-init.h"
#include "lulesh-viz.h"

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
   domain.p_sigxx->dirtybit=1;
   domain.p_sigxx->dirty_hd=1;
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
   domain.p_sigxx->dirtybit=1;
   domain.p_sigxx->dirty_hd=1;
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
   // op_print("Force For Nodes");
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
   for (int r=0 ; r<m_numReg ; ++r) {
      if(m_numElem > 0){
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

   op_dat local_region_i_p_e_new = domain.region_i_p_e_new[region];
   op_dat local_region_i_p_p_new = domain.region_i_p_p_new[region];

   op_dat local_region_i_p_compHalfStep = domain.region_i_p_compHalfStep[region];
   op_dat local_region_i_p_pHalfStep = domain.region_i_p_pHalfStep[region];
   op_dat local_region_i_p_bvc = domain.region_i_p_bvc[region];
   op_dat local_region_i_p_pbvc = domain.region_i_p_pbvc[region];

   op_par_loop(CalcHalfStepBVC, "CalcHalfStepBVC", current_set,
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(local_region_i_p_compHalfStep, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_pbvc, -1, OP_ID, 1, "double", OP_WRITE));

   //NOTE changed p_pHalfStep to write review
   op_par_loop(CalcPHalfstep, "CalcPHalfstep", current_set,
               op_arg_dat(local_region_i_p_pHalfStep, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ)
               );

}

static inline
void CalcPressureForElems(int region)
{
   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];

   op_dat local_region_i_p_e_new = domain.region_i_p_e_new[region];
   op_dat local_region_i_p_p_new = domain.region_i_p_p_new[region];
   op_dat local_region_i_p_compression = domain.region_i_p_compression[region];
   op_dat local_region_i_p_bvc = domain.region_i_p_bvc[region];
   op_dat local_region_i_p_pbvc = domain.region_i_p_pbvc[region];

   op_par_loop(CalcBVC, "CalcBVC", current_set,
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(local_region_i_p_compression, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_pbvc, -1, OP_ID, 1, "double", OP_WRITE));

   op_par_loop(CalcPNew, "CalcPNew", current_set,
               op_arg_dat(local_region_i_p_p_new, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ)
               );
}

/******************************************/

static inline
void CalcEnergyForElems(int region)
{
   // double *pHalfStep = Allocate<double>(length) ;
   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];

   op_dat local_region_i_p_e_old = domain.region_i_p_e_old[region];
   op_dat local_region_i_p_delvc = domain.region_i_p_delvc[region];
   op_dat local_region_i_p_p_old = domain.region_i_p_p_old[region];
   op_dat local_region_i_p_q_old = domain.region_i_p_q_old[region];
   op_dat local_region_i_p_qq_old = domain.region_i_p_qq_old[region];
   op_dat local_region_i_p_ql_old = domain.region_i_p_ql_old[region];

   op_dat local_region_i_p_compHalfStep = domain.region_i_p_compHalfStep[region];
   op_dat local_region_i_p_pHalfStep = domain.region_i_p_pHalfStep[region];
   op_dat local_region_i_p_work = domain.region_i_p_work[region];

   op_dat local_region_i_p_e_new = domain.region_i_p_e_new[region];
   op_dat local_region_i_p_q_new = domain.region_i_p_q_new[region];
   op_dat local_region_i_p_p_new = domain.region_i_p_p_new[region];
   op_dat local_region_i_p_bvc = domain.region_i_p_bvc[region];
   op_dat local_region_i_p_pbvc = domain.region_i_p_pbvc[region];

   op_par_loop(CalcNewE, "CalcNewE", current_set,
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_WRITE),
               op_arg_dat(local_region_i_p_e_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_delvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_p_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_q_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_work, -1, OP_ID, 1, "double", OP_READ)
   );
   CalcPressureForElemsHalfstep(region);

   //NOTE domain.p_e_new may be an INC
   op_par_loop(CalcNewEStep2, "CalcNewEStep2", current_set,
               op_arg_dat(local_region_i_p_compHalfStep, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_delvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_q_new, -1, OP_ID, 1, "double", OP_RW),
               op_arg_dat(local_region_i_p_pbvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_RW),
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_pHalfStep, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_ql_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_qq_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_p_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_q_old, -1, OP_ID, 1, "double", OP_READ)
   );


   //NOTE domain.p_e_new may be an INC
   op_par_loop(CalcNewEStep3, "CalcNewEStep3", current_set,
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_RW),
               op_arg_dat(local_region_i_p_work, -1, OP_ID, 1, "double", OP_READ)
   );

   CalcPressureForElems(region);

   //NOTE domain.p_e_new may be an INC
   op_par_loop(CalcNewEStep4, "CalcNewEStep4", current_set,
               op_arg_dat(local_region_i_p_delvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_pbvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_RW),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_p_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_ql_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_qq_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_p_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_q_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_q_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_pHalfStep, -1, OP_ID, 1, "double", OP_READ)
   );

   CalcPressureForElems(region);
   //NOTE p_q_new could probably be a write
   op_par_loop(CalcQNew, "CalcQNew", current_set,
               op_arg_dat(local_region_i_p_delvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_pbvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_p_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_q_new, -1, OP_ID, 1, "double", OP_RW),
               op_arg_dat(local_region_i_p_ql_old, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_qq_old, -1, OP_ID, 1, "double", OP_READ)
               );
   return ;
}

/******************************************/

static inline
void CalcSoundSpeedForElems(int region)
{
   op_set current_set= domain.region_i[region];
   op_map current_map = domain.region_i_to_elems[region];

   op_dat local_region_i_p_e_new = domain.region_i_p_e_new[region];
   op_dat local_region_i_p_p_new = domain.region_i_p_p_new[region];
   op_dat local_region_i_p_bvc = domain.region_i_p_bvc[region];
   op_dat local_region_i_p_pbvc = domain.region_i_p_pbvc[region];

   op_par_loop(CalcSoundSpeedForElem, "CalcSoundSpeedForElem", current_set,
               op_arg_dat(local_region_i_p_pbvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_bvc, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_p_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_ss, 0, current_map, 1, "double", OP_WRITE)
               );
   #if USE_DIRTY_BIT_OPT
   domain.p_ss->dirtybit=0;
   #endif
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

   op_dat local_region_i_p_e_old = domain.region_i_p_e_old[region];
   op_dat local_region_i_p_delvc = domain.region_i_p_delvc[region];
   op_dat local_region_i_p_p_old = domain.region_i_p_p_old[region];
   op_dat local_region_i_p_q_old = domain.region_i_p_q_old[region];
   op_dat local_region_i_p_qq_old = domain.region_i_p_qq_old[region];
   op_dat local_region_i_p_ql_old = domain.region_i_p_ql_old[region];

   op_dat local_region_i_p_compHalfStep = domain.region_i_p_compHalfStep[region];
   op_dat local_region_i_p_compression = domain.region_i_p_compression[region];
   op_dat local_region_i_p_work = domain.region_i_p_work[region];

   op_dat local_region_i_p_e_new = domain.region_i_p_e_new[region];
   op_dat local_region_i_p_q_new = domain.region_i_p_q_new[region];
   op_dat local_region_i_p_p_new = domain.region_i_p_p_new[region];

   op_dat local_region_i_p_bvc = domain.region_i_p_bvc[region];
   op_dat local_region_i_p_pbvc = domain.region_i_p_pbvc[region];

   op_par_loop(CopyEOSValsIntoArray, "CopyEOSValsIntoArray", current_set,
               op_arg_dat(local_region_i_p_e_old, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_e, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_delvc, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_delv, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_p_old, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_p, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_q_old, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_q, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_qq_old, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_qq, 0, current_map, 1, "double", OP_READ),
               op_arg_dat(local_region_i_p_ql_old, -1, OP_ID, 1, "double", OP_WRITE), op_arg_dat(domain.p_ql, 0, current_map, 1, "double", OP_READ));

   for(int j = 0; j < rep; j++) {
      /* compress data, minimal set */


         op_par_loop(CalcHalfSteps, "CalcHalfSteps", current_set,
                     op_arg_dat(local_region_i_p_compression, -1, OP_ID, 1, "double", OP_WRITE),
                     op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
                     op_arg_dat(local_region_i_p_delvc, -1, OP_ID, 1, "double", OP_READ),
                     op_arg_dat(local_region_i_p_compHalfStep, -1, OP_ID, 1, "double", OP_WRITE)
         );

      /* Check for v > eosvmax or v < eosvmin */
         if ( eosvmin != double(0.) ) {
            op_par_loop(CheckEOSLowerBound, "CheckEOSLowerBound", current_set,
                        op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
                        op_arg_dat(local_region_i_p_compHalfStep, -1, OP_ID, 1, "double", OP_WRITE),
                        op_arg_dat(local_region_i_p_compression, -1, OP_ID, 1, "double", OP_READ)
            );
         }

         if ( eosvmax != double(0.) ) {
            op_par_loop(CheckEOSUpperBound, "CheckEOSUpperBound", current_set,
                        op_arg_dat(domain.p_vnewc, 0, current_map, 1, "double", OP_READ),
                        op_arg_dat(local_region_i_p_compHalfStep, -1, OP_ID, 1, "double", OP_WRITE),
                        op_arg_dat(local_region_i_p_compression, -1, OP_ID, 1, "double", OP_WRITE),
                        op_arg_dat(local_region_i_p_p_old, -1, OP_ID, 1, "double", OP_WRITE)
            );
         }

         op_par_loop(CalcEOSWork, "CalcEOSWork", current_set,
                     op_arg_dat(local_region_i_p_work, -1, OP_ID, 1 , "double", OP_WRITE));
      CalcEnergyForElems(region);
   }
   #if USE_DIRTY_BIT_OPT 
   local_region_i_p_p_new->dirtybit=0;
   local_region_i_p_e_new->dirtybit=0;
   local_region_i_p_q_new->dirtybit=0;
   #endif
   op_par_loop(CopyTempEOSVarsBack, "CopyTempEOSVarsBack", current_set,
               op_arg_dat(domain.p_p, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(local_region_i_p_p_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_e, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(local_region_i_p_e_new, -1, OP_ID, 1, "double", OP_READ),
               op_arg_dat(domain.p_q, 0, current_map, 1, "double", OP_WRITE), op_arg_dat(local_region_i_p_q_new, -1, OP_ID, 1, "double", OP_READ)
   );
   // This imporvens things by about 1 second on a 5 proces run size 30 takes ~ 11 seconds
   //    or a 5-10% imporvment
   // without this the full data sets for p_p p_e and -p_q are exhnaged each time this combined with 
   // code later on only exchanges it once an interesting outcome
   // the exchange is not required as it is write only and every domain is disjoint
   // This is an area that OP2 is not desinged for as there is no knowlage of distinct subsets 
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

      // op_print("CopyVelo");
      op_par_loop(CopyVelocityToTempArray, "CopyVelocityToTempArray", domain.elems,
                  op_arg_dat(domain.p_vnewc, -1, OP_ID, 1, "double", OP_WRITE),
                  op_arg_dat(domain.p_vnew, -1, OP_ID, 1, "double", OP_READ));

       // Bound the updated relative volumes with eosvmin/max
      // op_print("Lower Bound");
       if (m_eosvmin != double(0.)) {
         op_par_loop(ApplyLowerBoundToVelocity, "ApplyLowerBoundToVelocity", domain.elems,
                  op_arg_dat(domain.p_vnewc, -1, OP_ID, 1, "double", OP_WRITE)
                  );

       }
      // op_print("Upper bound");
       if (m_eosvmax != double(0.)) {
         op_par_loop(ApplyUpperBoundToVelocity, "ApplyUpperBoundToVelocity", domain.elems,
                  op_arg_dat(domain.p_vnewc, -1, OP_ID, 1, "double", OP_WRITE)
                  );
       }

       // This check may not make perfect sense in LULESH, but
       // it's representative of something in the full code -
       // just leave it in, please
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
   domain.p_p->dirtybit=1; //global can be copied out
   domain.p_e->dirtybit=1; //global can be copied out
   domain.p_q->dirtybit=1; //global can be copied out
   domain.p_ss->dirtybit=1;
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
   //   op_print("Q For Elems");
  CalcQForElems() ;

   // op_print("Material props");
  ApplyMaterialPropertiesForElems() ;

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

int main(int argc, char *argv[])
{
   op_init(argc, argv, 1);
   //MPI node defaults
   int numRanks = 1;
   myRank =0;

   #if USE_MPI
   MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   #endif

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

   printf("COMPLETE HERE 0\n");
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

   #if USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
   #endif

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
 
   domain = initialiseALL(opts,myRank,(Int8_t)numRanks);

   double * speed=(double *)malloc(m_numNode*sizeof(double));
   op_dat p_speed=op_decl_dat(domain.nodes, 1, "double", speed, "p_speed");

   int siz = op_get_size(domain.elems);
   std::cout <<  "Global Size: " << siz << ", Local Elems: " << domain.elems->size << ", Local Nodes: "<< domain.nodes->size << "\n";
   op_print("\n\n");
   op_diagnostic_output();

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
   op_print("\n\n");
   
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
      
      //Create a silo file if selected at runtime
      if (opts.viz ==Visualization_All ||
         ((m_time > m_stoptime) || (m_cycle > opts.its) && Visualization_End) 
         ){
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
   }
   }

   op_timers(&cpu_t2, &wall_t2);
   double walltime = wall_t2 - wall_t1;

   double *verify_e = (double*) malloc(m_numElem*sizeof(double));
   op_fetch_data(domain.p_e, verify_e);

   if ((myRank == 0) && (opts.quiet == 0)) {
      VerifyAndWriteFinalOutput(walltime, m_cycle, opts.nx, numRanks, verify_e);
   }

   if (opts.time){op_timing_output();}
   
   op_exit(); 

   return 0 ;
}
