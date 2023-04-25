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