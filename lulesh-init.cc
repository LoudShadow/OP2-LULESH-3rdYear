
#include <op_seq.h>

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
struct Domain domain;