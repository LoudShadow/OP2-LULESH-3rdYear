//
// auto-generated by op2.py
//

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
#include "initStressTerms_seqkernel.cpp"
#include "IntegrateStressForElemsLoop_seqkernel.cpp"
#include "FBHourglassForceForElems_seqkernel.cpp"
#include "CalcVolumeDerivatives_seqkernel.cpp"
#include "CheckForNegativeElementVolume_seqkernel.cpp"
#include "setForceToZero_seqkernel.cpp"
#include "CalcAccelForNodes_seqkernel.cpp"
#include "BoundaryX_seqkernel.cpp"
#include "BoundaryY_seqkernel.cpp"
#include "BoundaryZ_seqkernel.cpp"
#include "CalcVeloForNodes_seqkernel.cpp"
#include "CalcPosForNodes_seqkernel.cpp"
#include "CalcKinematicsForElem_seqkernel.cpp"
#include "CalcLagrangeElemRemaining_seqkernel.cpp"
#include "CalcMonotonicQGradientsForElem_seqkernel.cpp"
#include "CalcMonotonicQRegionForElem_seqkernel.cpp"
#include "NoExcessiveArtificialViscosity_seqkernel.cpp"
#include "CalcHalfStepBVC_seqkernel.cpp"
#include "CalcPHalfstep_seqkernel.cpp"
#include "CalcBVC_seqkernel.cpp"
#include "CalcPNew_seqkernel.cpp"
#include "CalcNewE_seqkernel.cpp"
#include "CalcNewEStep2_seqkernel.cpp"
#include "CalcNewEStep3_seqkernel.cpp"
#include "CalcNewEStep4_seqkernel.cpp"
#include "CalcQNew_seqkernel.cpp"
#include "CalcSoundSpeedForElem_seqkernel.cpp"
#include "CopyEOSValsIntoArray_seqkernel.cpp"
#include "CalcHalfSteps_seqkernel.cpp"
#include "CheckEOSLowerBound_seqkernel.cpp"
#include "CheckEOSUpperBound_seqkernel.cpp"
#include "CalcEOSWork_seqkernel.cpp"
#include "CopyTempEOSVarsBack_seqkernel.cpp"
#include "CopyVelocityToTempArray_seqkernel.cpp"
#include "ApplyLowerBoundToVelocity_seqkernel.cpp"
#include "ApplyUpperBoundToVelocity_seqkernel.cpp"
#include "ALE3DRelevantCheck_seqkernel.cpp"
#include "updateVolumesForElem_seqkernel.cpp"
#include "CalcCourantConstraint_seqkernel.cpp"
#include "CalcHydroConstraint_seqkernel.cpp"
