//
// auto-generated by op2.py
//

#define double_ALIGN 128
#define float_ALIGN 64
#define int_ALIGN 64
#ifdef VECTORIZE
#define SIMD_VEC 4
#define ALIGNED_double __attribute__((aligned(double_ALIGN)))
#define ALIGNED_float __attribute__((aligned(float_ALIGN)))
#define ALIGNED_int __attribute__((aligned(int_ALIGN)))
  #ifdef __ICC
    #define DECLARE_PTR_ALIGNED(X, Y) __assume_aligned(X, Y)
  #else
    #define DECLARE_PTR_ALIGNED(X, Y)
  #endif
#else
#define ALIGNED_double
#define ALIGNED_float
#define ALIGNED_int
#define DECLARE_PTR_ALIGNED(X, Y)
#endif

// global constants
extern double m_e_cut;
extern double m_p_cut;
extern double m_q_cut;
extern double m_v_cut;
extern double m_u_cut;
extern double m_hgcoef;
extern double m_ss4o3;
extern double m_qstop;
extern double m_monoq_max_slope;
extern double m_monoq_limiter_mult;
extern double m_qlc_monoq;
extern double m_qqc_monoq;
extern double m_qqc;
extern double m_eosvmax;
extern double m_eosvmin;
extern double m_pmin;
extern double m_emin;
extern double m_dvovmax;
extern double m_refdens;
extern double m_qqc2;
extern double m_ptiny;
extern double m_gamma_t[32];
extern double m_twelfth;
extern double m_sixth;
extern double m_c1s;
extern double m_ssc_thresh;
extern double m_ssc_low;

// header
#include "op_lib_cpp.h"

// user kernel files
#include "initStressTerms_veckernel.cpp"
#include "IntegrateStressForElemsLoop_veckernel.cpp"
#include "FBHourglassForceForElems_veckernel.cpp"
#include "CalcVolumeDerivatives_veckernel.cpp"
#include "CheckForNegativeElementVolume_veckernel.cpp"
#include "setForceToZero_veckernel.cpp"
#include "CalcAccelForNodes_veckernel.cpp"
#include "BoundaryX_veckernel.cpp"
#include "BoundaryY_veckernel.cpp"
#include "BoundaryZ_veckernel.cpp"
#include "CalcVeloForNodes_veckernel.cpp"
#include "CalcPosForNodes_veckernel.cpp"
#include "CalcKinematicsForElem_veckernel.cpp"
#include "CalcLagrangeElemRemaining_veckernel.cpp"
#include "CalcMonotonicQGradientsForElem_veckernel.cpp"
#include "CalcMonotonicQRegionForElem_veckernel.cpp"
#include "NoExcessiveArtificialViscosity_veckernel.cpp"
#include "CalcHalfStepBVC_veckernel.cpp"
#include "CalcPHalfstep_veckernel.cpp"
#include "CalcBVC_veckernel.cpp"
#include "CalcPNew_veckernel.cpp"
#include "CalcNewE_veckernel.cpp"
#include "CalcNewEStep2_veckernel.cpp"
#include "CalcNewEStep3_veckernel.cpp"
#include "CalcNewEStep4_veckernel.cpp"
#include "CalcQNew_veckernel.cpp"
#include "CalcSoundSpeedForElem_veckernel.cpp"
#include "CopyEOSValsIntoArray_veckernel.cpp"
#include "CalcHalfSteps_veckernel.cpp"
#include "CheckEOSLowerBound_veckernel.cpp"
#include "CheckEOSUpperBound_veckernel.cpp"
#include "CalcEOSWork_veckernel.cpp"
#include "CopyTempEOSVarsBack_veckernel.cpp"
#include "CopyVelocityToTempArray_veckernel.cpp"
#include "ApplyLowerBoundToVelocity_veckernel.cpp"
#include "ApplyUpperBoundToVelocity_veckernel.cpp"
#include "ALE3DRelevantCheck_veckernel.cpp"
#include "updateVolumesForElem_veckernel.cpp"
#include "CalcCourantConstraint_veckernel.cpp"
#include "CalcHydroConstraint_veckernel.cpp"
