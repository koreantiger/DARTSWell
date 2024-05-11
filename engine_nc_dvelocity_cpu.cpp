#include <algorithm>
#include <cmath>
#include <cstring>
#include <time.h>
#include <functional>
#include <string>
#include <iomanip>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include "engine_nc_dvelocity_cpu.hpp"


template<uint8_t NC>
int
engine_nc_dvelocity_cpu<NC>::init(conn_mesh *mesh_, std::vector <ms_well*> &well_list_,
  std::vector <operator_set_gradient_evaluator_iface*> & acc_flux_op_set_list_,
  sim_params *params_, timer_node* timer_)
{

  time_t rawtime;
  struct tm *timeinfo;
  char buffer[1024];

  mesh = mesh_;
  wells = well_list_;
  acc_flux_op_set_list = acc_flux_op_set_list_;
  params = params_;
  timer = timer_;

  // Instantiate Jacobian in CSR format
  if (!Jacobian)
  {
    Jacobian = new csr_matrix<1>;
    Jacobian->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
  }

  // figure out if this is GPU engine from its name. 
  int is_gpu_engine = engine_name.find(" GPU ") != std::string::npos;


  // allocate Jacobian
  // if (!is_gpu_engine)
  {
    // for CPU engines we need full init
    index_t ctr;
    index_t nb = mesh_->n_blocks;
    index_t n_velocities = mesh_->n_conns / 2;
    index_t total_nnz = (mesh_->n_conns + mesh_->n_blocks) * NC * NC + // A
      n_velocities * NC * 2 + // B
      n_velocities * (2 * NC + 1);  // C and D
    if (acc_active)
    { // add additional nnz term since since we need to carry out the derivative wrt the perforated reservoir block as well. 
      ctr = additional_nnz_acc(Jacobian);
      total_nnz += NC * ctr;
    }
    (static_cast<csr_matrix<1> *>(Jacobian))->init(NC * nb + n_velocities, NC * nb + n_velocities, 1, total_nnz);
  }
  // else
  // {
  //   // for GPU engines we need only structure - rows_ptr and cols_ind
  //   // they are filled on CPU and later copied to GPU
  //   (static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_struct(mesh_->n_blocks, mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);
  // }
#ifdef WITH_GPU
  if (params->linear_type >= params->GPU_GMRES_CPR_AMG)
  {
    (static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_device(mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);
  }
#endif
  // Create linear solver
  if (!linear_solver)
  {

    linear_solver = new linsolv_superlu<1>;

  }

  n_vars = get_n_vars();
  n_ops = get_n_ops();
  nc = get_n_comps();
  z_var = get_z_var();
  X_init.resize(nc * mesh->n_blocks + mesh->n_conns / 2);    // add connection unknowns as well ( replace mesh->n... to n_velocities)
  PV.resize(mesh->n_blocks);
  RV.resize(mesh->n_blocks);
  old_z.resize(nc);
  new_z.resize(nc);
  FIPS.resize(nc);

  for (index_t i = 0; i < mesh->n_blocks; i++)
  {
    X_init[nc * i] = mesh->pressure[i];
    for (uint8_t c = 0; c < nc - 1; c++)
    {
      X_init[nc * i + c + 1] = mesh->composition[i * (nc - 1) + c];
    }
    PV[i] = mesh->volume[i] * mesh->poro[i];
    RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
  }
  // Fill velocity 
  for (index_t i = 0; i < mesh->n_conns / 2; i++)
  {
    X_init[nc  * mesh->n_blocks + i] = mesh->velocity[i];
  }

  op_vals_arr.resize(n_ops * mesh->n_blocks);
  op_ders_arr.resize(n_ops * nc * mesh->n_blocks);

  //
  t = 0;

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  stat = sim_stat();

  print_header();

  //acc_flux_op_set->init_timer_node(&timer->node["jacobian assembly"].node["interpolation"]);
    // initialize jacobian structure
  if (acc_active == 1)
  { // take into account acceleration pressure losses in the moment equation of MS-well
    init_jacobian_structure_acc(Jacobian);
  }
  else
  {// normal csr initialization.
    init_jacobian_structure(Jacobian);
  }


  //init_jacobian_structure(Jacobian);
  //init_jacobian_structure_acc(Jacobian);

#ifdef WITH_GPU
  if (params->linear_type >= sim_params::GPU_GMRES_CPR_AMG)
  {
    timer->node["jacobian assembly"].node["send_to_device"].start();
    Jacobian->copy_struct_to_device();
    timer->node["jacobian assembly"].node["send_to_device"].stop();
  }
#endif

  linear_solver->init_timer_nodes(&timer->node["linear solver setup"], &timer->node["linear solver solve"]);
  // initialize linear solver
  linear_solver->init(Jacobian, params->max_i_linear, params->tolerance_linear);

  //Xn.resize (n_vars * mesh->n_blocks);
  RHS.resize(nc * mesh->n_blocks + mesh->n_conns / 2);
  dX.resize(nc * mesh->n_blocks + mesh->n_conns / 2);

  sprintf(buffer, "\nSTART SIMULATION\n-------------------------------------------------------------------------------------------------------------\n");
  std::cout << buffer << std::flush;
  sprintf(buffer, "\nSTART SIMULATION\n-------------------------------------------------------------------------------------------------------------\n");
  std::cout << buffer << std::flush;

  // let wells initialize their state
  for (ms_well *w : wells)
  {
    w->initialize_control(X_init);
  }

  Xn = X = X_init;
  dt = params->first_ts;
  prev_usual_dt = dt;
  // initialize arrays for every operator set
  block_idxs.resize(acc_flux_op_set_list.size());
  op_axis_min.resize(acc_flux_op_set_list.size());
  op_axis_max.resize(acc_flux_op_set_list.size());
  // initialize arrays for every operator set
  for (int r = 0; r < acc_flux_op_set_list.size(); r++)
  {
    block_idxs[r].clear();
    op_axis_min[r].resize(nc);
    op_axis_max[r].resize(nc);
    for (int j = 0; j < nc; j++)
    {
      op_axis_min[r][j] = acc_flux_op_set_list[r]->get_axis_min(j);
      op_axis_max[r][j] = acc_flux_op_set_list[r]->get_axis_max(j);
    }
  }

  // create a block list for every operator set
  index_t idx = 0;
  for (auto op_region : mesh->op_num)
  {
    block_idxs[op_region].emplace_back(idx++);
  }

  // fill cell-center unknowns 
  X_cc.resize(nc * mesh->n_blocks);
  for (index_t i = 0; i < nc * mesh->n_blocks; i++)
  {
    X_cc[i] = X[i];
  }

  // Use only cell-center unknowns for evaluation_with_derivative since OBL doesn't depends on (interface unknown) velocity

  for (int r = 0; r < acc_flux_op_set_list.size(); r++)
    acc_flux_op_set_list[r]->evaluate_with_derivatives(X_cc, block_idxs[r], op_vals_arr, op_ders_arr);
  op_vals_arr_n = op_vals_arr;

  time_data.clear();
  time_data_report.clear();

  if (params->log_transform == 0)
  {
    min_zc = acc_flux_op_set_list[0]->get_axis_min(z_var) * params->obl_min_fac;
    max_zc = 1 - min_zc * params->obl_min_fac;
    //max_zc = acc_flux_op_set_list[0]->get_maxzc();
  }
  else if (params->log_transform == 1)
  {
    min_zc = exp(acc_flux_op_set_list[0]->get_axis_min(z_var)) * params->obl_min_fac; //log based composition
    max_zc = exp(acc_flux_op_set_list[0]->get_axis_max(z_var)); //log based composition
  }

  init_WellDynamicProperties(X_init);
  Xn = X = X_init;
  return 0;
}

// We fill Jacobian according to the following structure:
//        | p1 z1 | p2 z2 | ... | p_nb z_nb |  v_1 | .... v_n_conn|

//     General form : r_i = PV(alpha - alpha_n) + dt(beta_i*U_l - beta_i - 1 * U_{ l - 1 }), from left to right                | i-1 |l-1  i  |l i  |    --> 
//     General form : r_i = PV(alpha - alpha_n) + dt(beta_i*U_{l-1} - beta_i + 1 * U_{ l }), from right to left                | i-1 |l-1  i  |l i  |    <--

// mb_1_1      
// mb_2_1
// mb_1_2
// mb_2_2
// ....
// mom_1   rl  = u1 + tran[1] * lamba * dp  ;
// mom_2
// ....
// mom_n_conn
//
// Jacobian structure  J = [ A B ; C D ]
//  A B
//  C D



template<uint8_t NC>
int
engine_nc_dvelocity_cpu<NC>::init_jacobian_structure(csr_matrix_base *jacobian)
{
  const char n_vars = get_n_vars();

  // init Jacobian structure
  index_t *rows_ptr = jacobian->get_rows_ptr();
  index_t *diag_ind = jacobian->get_diag_ind();
  index_t *cols_ind = jacobian->get_cols_ind();
  index_t *row_thread_starts = jacobian->get_row_thread_starts();

  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  index_t n_velocities = n_conns / 2;
  index_t iw;
  index_t n_res_blocks = mesh->n_res_blocks;

  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  //std::vector <value_t> velocity_sorted;


#ifdef _OPENMP
#pragma omp parallel
  {
    int id, nt;
    index_t local, start, end;

    id = omp_get_thread_num();
    nt = omp_get_num_threads();
    start = row_thread_starts[id];
    end = row_thread_starts[id + 1];
    // 'first touch' rows_ptr, cols_ind, diag_ind

    numa_set(rows_ptr, 0, start, end);
    // since the length of rows_ptr is n_blocks+1, take care of the last entry (using last thread)
    if (id == nt - 1)
    {
      rows_ptr[n_blocks] = rows_ptr[n_blocks];
    }
    numa_set(diag_ind, -1, start, end);
  }
#else
  rows_ptr[0] = 0;
  memset(diag_ind, -1, n_blocks * NC * sizeof(index_t)); // t_long <-----> index_t
#endif
  // now we have to split the work into two loops

  // 1. Fill out rows_ptr
  index_t j = 0, k = 0;
  for (index_t i = 0; i < n_blocks; i++)
  {
    // take the last number from previous row
    rows_ptr[NC * i + 1] = rows_ptr[NC * i];
    for (; j < n_conns && block_m[j] == i; j++)
    {
      if (diag_ind[NC * i] < 0 && block_p[j] > i)
      {
        // block vars entries
        rows_ptr[NC * i + 1] += NC;
        diag_ind[NC * i] = k;
      }
      // block vars entries
      rows_ptr[NC * i + 1] += NC;
      // velocity entry
      rows_ptr[NC * i + 1]++;
      k += NC;
    }
    if (diag_ind[NC * i] < 0)
    {
      rows_ptr[NC * i + 1] += NC;
      diag_ind[NC * i] = k;
      k += NC;
    }

    int row_nnz = rows_ptr[NC * i + 1] - rows_ptr[NC * i];
    for (index_t eq = 1; eq < NC; eq++)
    {
      rows_ptr[NC * i + eq + 1] = rows_ptr[NC * i + eq] + row_nnz;
      diag_ind[NC * i + eq] = diag_ind[NC * i + eq - 1] + row_nnz;

      //k += row_nnz;
      k = rows_ptr[NC * i + eq + 1];
    }
  }


  for (index_t i = 0; i < n_velocities; i++)
  {
    // Each block connected by the interface gives NC entries + one diagonal velocity entry
    rows_ptr[NC * n_blocks + i + 1] = rows_ptr[NC * n_blocks + i] + 2 * NC + 1;
    diag_ind[NC * n_blocks + i] = diag_ind[NC * n_blocks + i - 1] + 2 * NC + 1;
  }


  // Now we know rows_ptr and can 'first touch' cols_ind
#ifdef _OPENMP
#pragma omp parallel
  {
    int id, nt;
    index_t local, start, end;

    id = omp_get_thread_num();
    start = row_thread_starts[id];
    end = row_thread_starts[id + 1];

    start = rows_ptr[start];
    end = rows_ptr[end];
    numa_set(cols_ind, 0, start, end);
  }
#endif

  // 2. Write cols_ind
  j = 0, k = 0;
  index_t j_start;
  ctrMeqI.resize(n_blocks);  // use it later(assembly) to idx in B part of assembly of jacobian
  for (index_t i = 0; i < n_blocks; i++)
  {
    // 1st loop through connections to fill the first row of block row i for A part
    j_start = j;
    ctrMeqI[i] = 0;
    for (; j < n_conns && block_m[j] == i; j++)
    {
      if (diag_ind[NC * i] == k)
      {
        for (index_t v = 0; v < NC; v++)
        {
          cols_ind[k++] = NC * i + v;
        }
      }
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * block_p[j] + v;
      }
      ctrMeqI[i] += 1;
    }

    if (diag_ind[NC * i] == k)
    {
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * i + v;
      }
    }

    // 2nd loop through connections to fill the first row of block row i for B part
    j = j_start;
    //previous implementation
    for (; j < n_conns && block_m[j] == i; j++)
    {
      cols_ind[k++] = NC * n_blocks + velocity_mapper[j];
    }

    // duplicate col_inds for all the rest equations in the block row i
    int row_nnz = rows_ptr[NC * i + 1] - rows_ptr[NC * i];

    for (index_t eq = 1; eq < NC; eq++)
    {
      for (; k < rows_ptr[NC * i + eq + 1]; k++)
      {
        cols_ind[k] = cols_ind[k - row_nnz];
      }
    }
  }

  // Now fill col_ind for C and D
  int ctr_d = 0;
  for (index_t j = 0; j < n_conns; j++)
  {
    if (block_m[j] < block_p[j])
    {
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * block_m[j] + v;
      }
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * block_p[j] + v;
      }
      cols_ind[k++] = NC * n_blocks + velocity_mapper[j];
      ctr_d += 1;

      // assign and save  the well_head_idx connection value
      if (block_m[j] >= n_res_blocks && block_p[j] >= n_res_blocks)
      {
        iw = return_well_number(block_m[j], block_p[j]);
        if (block_m[j] == wells[iw]->well_head_idx)
        {// assign the connection number to the well_head_idx_conn variable 
          wells[iw]->well_head_idx_conn = velocity_mapper[j];
        }
      }
    }
  }

  //Jacobian->write_matrix_to_file("jac1010.csr");
  //cpr_prec.init (&Jacobian, 0, 0);
  return 0;
};


template<uint8_t NC>
int
engine_nc_dvelocity_cpu<NC>::init_jacobian_structure_acc(csr_matrix_base *jacobian)
{
  const char n_vars = get_n_vars();

  // init Jacobian structure
  index_t *rows_ptr = jacobian->get_rows_ptr();
  index_t *diag_ind = jacobian->get_diag_ind();
  index_t *cols_ind = jacobian->get_cols_ind();
  index_t *row_thread_starts = jacobian->get_row_thread_starts();

  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  index_t n_velocities = n_conns / 2;

  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  //std::vector <value_t> velocity_sorted;
  index_t n_res_blocks = mesh->n_res_blocks;
  index_t i_w, i_r, iw;
  value_t wi;
#ifdef _OPENMP
#pragma omp parallel
  {
    int id, nt;
    index_t local, start, end;

    id = omp_get_thread_num();
    nt = omp_get_num_threads();
    start = row_thread_starts[id];
    end = row_thread_starts[id + 1];
    // 'first touch' rows_ptr, cols_ind, diag_ind

    numa_set(rows_ptr, 0, start, end);
    // since the length of rows_ptr is n_blocks+1, take care of the last entry (using last thread)
    if (id == nt - 1)
    {
      rows_ptr[n_blocks] = rows_ptr[n_blocks];
    }
    numa_set(diag_ind, -1, start, end);
  }
#else
  rows_ptr[0] = 0;
  memset(diag_ind, -1, n_blocks * NC * sizeof(index_t)); // t_long <-----> index_t
#endif
  // now we have to split the work into two loops

  // 1. Fill out rows_ptr
  index_t j = 0, k = 0;
  for (index_t i = 0; i < n_blocks; i++)
  {
    // take the last number from previous row
    rows_ptr[NC * i + 1] = rows_ptr[NC * i];
    for (; j < n_conns && block_m[j] == i; j++)
    {
      if (diag_ind[NC * i] < 0 && block_p[j] > i)
      {
        // block vars entries
        rows_ptr[NC * i + 1] += NC;
        diag_ind[NC * i] = k;
      }
      // block vars entries
      rows_ptr[NC * i + 1] += NC;
      // velocity entry
      rows_ptr[NC * i + 1]++;
      k += NC;
    }
    if (diag_ind[NC * i] < 0)
    {
      rows_ptr[NC * i + 1] += NC;
      diag_ind[NC * i] = k;
      k += NC;
    }

    int row_nnz = rows_ptr[NC * i + 1] - rows_ptr[NC * i];
    for (index_t eq = 1; eq < NC; eq++)
    {
      rows_ptr[NC * i + eq + 1] = rows_ptr[NC * i + eq] + row_nnz;
      diag_ind[NC * i + eq] = diag_ind[NC * i + eq - 1] + row_nnz;

      //k += row_nnz;
      k = rows_ptr[NC * i + eq + 1];
    }
  }
  /*
  // block c + d original one
  for (index_t i = 0; i < n_velocities; i++)
  {
    // Each block connected by the interface gives NC entries + one diagonal velocity entry
    rows_ptr[N_NODE * n_blocks + i + 1] = rows_ptr[N_NODE * n_blocks + i] + 2 * N_NODE + 1;
    diag_ind[N_NODE * n_blocks + i] = diag_ind[N_NODE * n_blocks + i - 1] + 2 * N_NODE + 1;
  }

  */
  index_t ctr_idx = 0;
  for (index_t j = 0; j < n_conns; j++)
  {
    if (block_m[j] < block_p[j])
    {
      // Each block connected by the interface gives NC entries + one diagonal velocity entry
      rows_ptr[NC * n_blocks + ctr_idx + 1] = rows_ptr[NC * n_blocks + ctr_idx] + 2 * NC + 1;
      diag_ind[NC * n_blocks + ctr_idx] = diag_ind[NC * n_blocks + ctr_idx - 1] + 2 * NC + 1;

      // In case one of the block between connection is perforated initialize it as well since we need a derivative for acceleration term.
      if (block_m[j] >= n_res_blocks && block_p[j] >= n_res_blocks)
      {
        for (index_t iw = 0; iw < wells.size(); iw++)
        {
          // connections between well segments and reservoir
          for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
          {
            std::tie(i_w, i_r, wi) = wells[iw]->perforations[p];
            value_t wb = i_w + wells[iw]->well_head_idx + 1;  // perforated well block number
            if (block_m[j] == wb || block_p[j] == wb)
            {
              rows_ptr[NC * n_blocks + ctr_idx + 1] += NC;
            }
          }
        }
      }
      ctr_idx++;
    }
  }

  // Now we know rows_ptr and can 'first touch' cols_ind
#ifdef _OPENMP
#pragma omp parallel
  {
    int id, nt;
    index_t local, start, end;

    id = omp_get_thread_num();
    start = row_thread_starts[id];
    end = row_thread_starts[id + 1];

    start = rows_ptr[start];
    end = rows_ptr[end];
    numa_set(cols_ind, 0, start, end);
  }
#endif

  // 2. Write cols_ind
  j = 0, k = 0;
  index_t j_start;
  ctrMeqI.resize(n_blocks);  // use it later(assembly) to idx in B part of assembly of jacobian
  for (index_t i = 0; i < n_blocks; i++)
  {
    // 1st loop through connections to fill the first row of block row i for A part
    j_start = j;
    ctrMeqI[i] = 0;
    for (; j < n_conns && block_m[j] == i; j++)
    {
      if (diag_ind[NC * i] == k)
      {
        for (index_t v = 0; v < NC; v++)
        {
          cols_ind[k++] = NC * i + v;
        }
      }
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * block_p[j] + v;
      }
      ctrMeqI[i] += 1;
    }

    if (diag_ind[NC * i] == k)
    {
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * i + v;
      }
    }

    // 2nd loop through connections to fill the first row of block row i for B part
    j = j_start;
    //previous implementation
    for (; j < n_conns && block_m[j] == i; j++)
    {
      cols_ind[k++] = NC * n_blocks + velocity_mapper[j];
    }

    // duplicate col_inds for all the rest equations in the block row i
    int row_nnz = rows_ptr[NC * i + 1] - rows_ptr[NC * i];

    for (index_t eq = 1; eq < NC; eq++)
    {
      for (; k < rows_ptr[NC * i + eq + 1]; k++)
      {
        cols_ind[k] = cols_ind[k - row_nnz];
      }
    }
  }

  // Now fill col_ind for C and D 
  // also add perforated block in case we take into account acceleration losses
  int ctr_d = 0;
  ctrConblock.resize(n_conns);  // use it later(assembly) to idx in B part of assembly of jacobian
  for (index_t j = 0; j < n_conns; j++)
  {
    ctrConblock[j] = 2;            // default number of blocks for a given connection   []---[]
    if (block_m[j] < block_p[j])
    {
      // In case one of the blocks between connection is perforated initialize it as well since we need a derivative for acceleration term.
      if (block_m[j] >= n_res_blocks && block_p[j] >= n_res_blocks)
      {
        for (index_t iw = 0; iw < wells.size(); iw++)
        {
          // connections between well segments and reservoir
          for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
          {
            std::tie(i_w, i_r, wi) = wells[iw]->perforations[p];  // well block number is wrong!
            value_t wb = i_w + wells[iw]->well_head_idx + 1;      // perforated well block number
            if (block_m[j] == wb || block_p[j] == wb)             // i_w is not correct!    i_w + wells[iw]->well_head_idx + 1
            {
              for (index_t v = 0; v < NC ; v++)
              {
                cols_ind[k++] = (NC * i_r) + v;
              }
              ctrConblock[j] += 1;
            }
          }
        }
      }

      //old one 
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * block_m[j] + v;
      }
      for (index_t v = 0; v < NC; v++)
      {
        cols_ind[k++] = NC * block_p[j] + v;
      }
      cols_ind[k++] = NC * n_blocks + velocity_mapper[j];
      ctr_d += 1;
    }

	// assign and save  the well_head_idx connection value
	if (block_m[j] >= n_res_blocks && block_p[j] >= n_res_blocks)
	{
		iw = return_well_number(block_m[j], block_p[j]);
		if (block_m[j] == wells[iw]->well_head_idx)
		{// assign the connection number to the well_head_idx_conn variable 
			wells[iw]->well_head_idx_conn = velocity_mapper[j];
		}
	}
  }

  //Jacobian->write_matrix_to_file("jac_struct1_d.csr");
  //exit(0);
  //cpr_prec.init (&Jacobian, 0, 0);
  return 0;
};


template<uint8_t NC>
int
engine_nc_dvelocity_cpu<NC>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assembly_kernel)
{
  // We need extended connection list for that with all connections for each block

  index_t n_blocks = mesh->n_blocks;
  index_t n_res_blocks = mesh->n_res_blocks;
  index_t n_conns = mesh->n_conns;
  std::vector <value_t> &tran = mesh->tran;
  value_t* Jac = jacobian->get_values();
  index_t* diag_ind = jacobian->get_diag_ind();
  index_t* rows = jacobian->get_rows_ptr();
  index_t* cols = jacobian->get_cols_ind();
  index_t *row_thread_starts = jacobian->get_row_thread_starts();
  index_t csr_idx_start = 0;
  index_t csr_idx_end;

  auto idx = rows[NC * n_blocks];  // idx for starting block c
  // Find the columnblock index array

  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  CFL_max = 0;

  // Additional parameters for well-momentum equation 
  std::vector <value_t> &grav_coef = mesh->grav_coef;          // taking into account  hydraulic losses  = g * dz
  auto newtoniter = n_newton_last_dt;
  value_t vel_res_nonlinear = 0;
  index_t conn_perf, iw;

#ifdef _OPENMP
  //#pragma omp parallel reduction (max: CFL_max)
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    index_t start = row_thread_starts[id];
    index_t end = row_thread_starts[id + 1];

#else
  index_t start = 0;
  index_t end = n_blocks;
#endif //_OPENMP

  index_t j = 0, diag_idx, idx_blockB, jac_idx = 0;
  value_t p_diff, gamma_p_diff, gamma, velocity;
  value_t CFL_in[NC], CFL_out[NC];
  value_t CFL_max_local = 0;

  int connected_with_well;
  index_t  k = 0;
  index_t j_start;
  jac_idx_array.resize(NC);

  // rhs for node equations
  // Block A 
  for (index_t i = start; i < end; ++i)
  {
    // 1st loop through connections to fill the first row of block row i for A part
    j_start = j;
    k = rows[NC*i];
    for (uint8_t c = 0; c < NC; c++)
    {
      jac_idx_array[c] = rows[NC * i + c];   // 2 or nc? 
    }

    // fill diagonal part Block A  acc operators only
    for (uint8_t c = 0; c < NC; c++)
    {
      RHS[i * NC + c] = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]);
      CFL_out[c] = 0;
      CFL_in[c] = 0;
      connected_with_well = 0;
      diag_idx = diag_ind[NC * i + c];

      for (uint8_t v = 0; v < NC; v++)
      {
        Jac[diag_idx + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * NC + v];
      }
    }

    // fill offdiagonal + contributation to diagonal for block A
    for (; j < n_conns && block_m[j] == i; j++)   // jac_idx += NC
    {
      p_diff = X[block_p[j] * NC + P_VAR] - X[i * NC + P_VAR];
      velocity = X[NC*n_blocks + velocity_mapper[j]];
      if (diag_ind[NC * i] == k)
      {
        k += NC;
        // update jac_idx
        transform(jac_idx_array.begin(), jac_idx_array.end(), jac_idx_array.begin(),
          bind2nd(std::plus<double>(), NC));
      }

      gamma = dt;
      const double sign = (block_p[j] - block_m[j] > 0) ? 1. : -1.;

      if (p_diff < 0)
      {
        //outflow  ; 
        for (uint8_t c = 0; c < NC; c++)
        {
          CFL_out[c] += -gamma * op_vals_arr[i * N_OPS + FLUX_OP + c];
          RHS[i * NC + c] += gamma * op_vals_arr[i * N_OPS + FLUX_OP + c] * velocity * sign;
          diag_idx = diag_ind[NC * i + c];
          for (uint8_t v = 0; v < NC; v++)
          {
            Jac[diag_idx + v] += gamma * op_ders_arr[(i * N_OPS + FLUX_OP + c) * nc + v] * velocity * sign;
            // put zero for off_diagonal
            jac_idx = jac_idx_array[c];
            Jac[jac_idx + v] = 0;
          }
        }
      }
      else
      {
        //inflow        
        for (uint8_t c = 0; c < NC; c++)
        {
          CFL_out[c] += -gamma * op_vals_arr[i * N_OPS + FLUX_OP + c];
          RHS[i * NC + c] += gamma * op_vals_arr[block_p[j] * N_OPS + FLUX_OP + c] * velocity * sign;
          jac_idx = jac_idx_array[c];

          for (uint8_t v = 0; v < NC; v++)
          {
            Jac[jac_idx + v] = +gamma * op_ders_arr[(block_p[j] * N_OPS + FLUX_OP + c) * nc + v] * velocity * sign;
          }
        }
      }

      k += NC;
      // update jac_idx
      transform(jac_idx_array.begin(), jac_idx_array.end(), jac_idx_array.begin(),
        bind2nd(std::plus<double>(), NC));
    }
    // Block B   ( Derivative of node equations wrt velocity ) 
    j = j_start;
    int ctr = 0;
    for (; j < n_conns && block_m[j] == i; j++)
    {
      p_diff = X[block_p[j] * NC + P_VAR] - X[i * NC + P_VAR];
      const double sign = (block_p[j] - block_m[j] > 0) ? 1. : -1.;
      if (p_diff < 0)
      {
        // bm upstream 
        ctr += 1;
        for (uint8_t c = 0; c < NC; c++)
        {
          idx_blockB = rows[NC * i + c] + NC * ctrMeqI[i] + (NC - 1) + ctr;
          Jac[idx_blockB] = gamma * op_vals_arr[i * N_OPS + FLUX_OP + c] * sign;
        }
      }
      else
      {
        // bp upstream
        ctr += 1;
        for (uint8_t c = 0; c < NC; c++)
        {
          idx_blockB = rows[NC * i + c] + NC * ctrMeqI[i] + (NC - 1) + ctr;                // here's the bug! it is correct if the velocity mapper be the same order
          Jac[idx_blockB] = gamma * op_vals_arr[block_p[j] * N_OPS + FLUX_OP + c] * sign;   // Check block_p[j]  
        }
      }
    }

    //// previous inplementation		
    // calc CFL for reservoir cells, not connected with wells
    if (i < mesh->n_res_blocks && !connected_with_well)
    {
      for (uint8_t c = 0; c < NC; c++)
      {
        if ((PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]) > 1e-4)
        {
          CFL_max_local = std::max(CFL_max_local, CFL_in[c] / (PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]));
          CFL_max_local = std::max(CFL_max_local, CFL_out[c] / (PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]));
        }
      }
    }
  }

  // rhs for connection equation
  // Block C + D  

  for (index_t j = 0; j < n_conns; j++)
  {
    if (block_m[j] >= n_res_blocks && block_p[j] >= n_res_blocks)
    {

      // Well connections 
      if (block_m[j] < block_p[j])
      {

        // replace it to the real MSWell momentum equations!
        idx = rows[NC * n_blocks] + (NC * 2 + 1) * velocity_mapper[j];
        p_diff = X[block_p[j] * NC + P_VAR] - X[block_m[j] * NC + P_VAR];
        velocity = X[NC * n_blocks + velocity_mapper[j]];
        //if (newtoniter <2 && fabs(t) < 1e-15)
        if (vel_linear)
        { // linear velocity (Darcy) for well- connections  as well. 
          darcy_velocity(j, velocity, p_diff, gamma, X, Jacobian, RHS, params->assembly_kernel);
        }
        else
        {
          // MSwell momentum equation for well connections
          iw = return_well_number(block_m[j], block_p[j]);                // return the well number that the j connection belongs to. 
          conn_perf = connection_perforated(iw, block_m[j], block_p[j]);  // check if any side of the j connection is perforated. 
          if (conn_perf && acc_active)
          {
            momentum_mswell_withacceleration(iw, j, velocity, p_diff, gamma, X, Jacobian, RHS, params->assembly_kernel);
          }
          else
          {
            momentum_mswell(iw, j, velocity, p_diff, gamma, X, Jacobian, RHS, params->assembly_kernel);
          }
        }
      }
    }
    else
    {
      //  reservoir connections -Linear velocity  (DARCY VELOCITY) ;
      if (block_m[j] < block_p[j])
      {
        idx = rows[NC * n_blocks] + (NC * 2 + 1)*velocity_mapper[j];
        p_diff = X[block_p[j] * NC + P_VAR] - X[block_m[j] * NC + P_VAR];
        velocity = X[NC * n_blocks + velocity_mapper[j]];
        darcy_velocity(j, velocity, p_diff, gamma, X, Jacobian, RHS, params->assembly_kernel);
      }
    }
  }


#ifdef _OPENMP
#pragma omp critical 
  {
    if (CFL_max < CFL_max_local)
      CFL_max = CFL_max_local;
  }
  }
#else
  CFL_max = CFL_max_local;
#endif


  // Add to CSR jacobian
  for (ms_well *w : wells)
  {
    value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[nc * w->well_head_idx]]);  // why n_vars is equal to nc now?
    w->add_to_csr_jacobian(dt, X, jac_well_head, RHS);
  }

  //Jacobian->write_matrix_to_file("jac_struct_withwell.csr");
  //write_vector_to_file("jac_struct_withwell.rhs", RHS);
  //exit(0);
  /*
    for (index_t iw = 0; iw < wells.size(); iw++)
  {
    cross_flow(iw, X);
  }
  */
  /*
    for (ms_well *w : wells)
  {
    w->cross_flow(X);
  }
  */

  return 0;

};


template<uint8_t NC>
double
engine_nc_dvelocity_cpu<NC>::calc_newton_residual_L2()
{
  double residual = 0;

  std::vector<value_t> res(nc, 0);
  std::vector<value_t> norm(nc, 0);

  for (int i = 0; i < mesh->n_res_blocks; i++)
  {
    for (int c = 0; c < nc; c++)
    {
      res[c] += RHS[i * nc + c] * RHS[i * nc + c];
      norm[c] += (PV[i] * op_vals_arr[i * nc + c]) * (PV[i] * op_vals_arr[i * nc + c]);
    }
  }
  for (int c = 0; c < nc; c++)
  {
    residual = std::max(residual, sqrt(res[c] / norm[c]));
  }

  return residual;
}



template<uint8_t NC>
double
engine_nc_dvelocity_cpu<NC>::calc_well_residual_L2()
{

  double residual = 0;
  std::vector<value_t> res(nc, 0);
  std::vector<value_t> norm(nc, 0);

  std::vector<value_t> av_op(nc, 0);
  average_operator(av_op);

  for (ms_well *w : wells)
  {

    // first sum up RHS for well segments which have perforations
    int nperf = w->perforations.size();
    for (int ip = 0; ip < nperf; ip++)
    {
      for (int v = 0; v < nc; v++)
      {
        index_t i_w, i_r;
        value_t wi;
        std::tie(i_w, i_r, wi) = w->perforations[ip];

        res[v] += RHS[(w->well_body_idx + i_w) * nc + v] * RHS[(w->well_body_idx + i_w) * nc + v];
        norm[v] += PV[w->well_body_idx + i_w] * av_op[v] * PV[w->well_body_idx + i_w] * av_op[v];
      }
    }
    // and then add RHS for well control equations
    for (int v = 0; v < nc; v++)
    {
      // well constraints should not be normalized, so pre-multiply by norm
      res[v] += RHS[w->well_head_idx * nc + v] * RHS[w->well_head_idx * nc + v] * PV[w->well_body_idx] * av_op[v] * PV[w->well_body_idx] * av_op[v];
    }
  }

  for (int v = 0; v < nc; v++)
  {
    residual = std::max(residual, sqrt(res[v] / norm[v]));
  }

  return residual;

}

template<uint8_t NC>
double
engine_nc_dvelocity_cpu<NC>::calc_velocity_residual()
{
  /*
  both well and reservoir velocities
  */

  double residual = 0;

  // for connection variables
  for (int i = 0; i < mesh->n_conns / 2; i++)
  {
    residual += RHS[mesh->n_blocks* nc + i] * RHS[mesh->n_blocks* nc + i];
  }

  residual = sqrt(residual);
  return residual;
}


template<uint8_t NC>
void
engine_nc_dvelocity_cpu<NC>::calc_well_res_velocity_residual(value_t & res_wel_vel, value_t & res_res_vel)
{
  /*
  calculate well and reservoir velocity residuals.
  */
  res_wel_vel = 0;
  res_res_vel = 0;
  index_t well_head_conn = wells[0]->well_head_idx_conn; // connection to the first well head.
  index_t  ctr = RHS.size() - (mesh->n_blocks * N_NODE + well_head_conn);
  for (int i = 0; i < ctr; i++)
  {
    res_wel_vel += RHS[(mesh->n_blocks * N_NODE + well_head_conn) + i] * RHS[(mesh->n_blocks * N_NODE + well_head_conn) + i];
  }

  for (int i = 0; i < ((mesh->n_blocks * N_NODE + well_head_conn) - mesh->n_blocks * N_NODE); i++)
  {
    res_res_vel += RHS[mesh->n_blocks * N_NODE + i] * RHS[mesh->n_blocks * N_NODE + i];
  }

  res_wel_vel = sqrt(res_wel_vel);
  res_res_vel = sqrt(res_res_vel);
}


template<uint8_t NC>
int
engine_nc_dvelocity_cpu<NC>::run_single_newton_iteration(value_t deltat)
{
  // switch constraints if needed
  timer->node["jacobian assembly"].start();
  for (ms_well *w : wells)
  {
    w->check_constraints(deltat, X);
  }

  // evaluate all operators and their derivatives
  // fill cell-center unknowns 
  X_cc.resize(nc * mesh->n_blocks);
  for (index_t i = 0; i < nc * mesh->n_blocks; i++)
  {
    X_cc[i] = X[i];
  }
  timer->node["jacobian assembly"].node["interpolation"].start();

  for (int r = 0; r < acc_flux_op_set_list.size(); r++)
  {
    int result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X_cc, block_idxs[r], op_vals_arr, op_ders_arr);
    if (result < 0)
      return 0;
  }

  timer->node["jacobian assembly"].node["interpolation"].stop();

  // assemble jacobian
  assemble_jacobian_array(deltat, X, Jacobian, RHS, params->assembly_kernel);

  velocity_residual_last_dt = calc_velocity_residual();

  calc_well_res_velocity_residual(res_wel_vel, res_res_vel);

#ifdef WITH_GPU
  if (params->linear_type >= sim_params::GPU_GMRES_CPR_AMG)
  {
    timer->node["jacobian assembly"].node["send_to_device"].start();
    Jacobian->copy_values_to_device();
    timer->node["jacobian assembly"].node["send_to_device"].stop();
  }
#endif

  timer->node["jacobian assembly"].stop();
  return 0;
}

template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::post_newtonloop(value_t deltat, value_t time)
{
  int converged = 0;
  char buffer[1024];
  double well_tolerance_coefficient = 1e2;

  if (linear_solver_error_last_dt == 1) // linear solver setup failed
  {
    sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (linear solver setup failed) \n", deltat);
  }
  else if (linear_solver_error_last_dt == 2) // linear solver solve failed
  {
    sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (linear solver solve failed) \n", deltat);
  }
  else if (newton_residual_last_dt >= params->tolerance_newton) // no reservoir convergence reached
  {
    sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (newton residual reservoir) \n", deltat);
  }
  else if (well_residual_last_dt > well_tolerance_coefficient * params->tolerance_newton) // no well convergence reached
  {
    sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (newton residual wells) \n", deltat);
  }
  else
  {
    converged = 1;
  }

  if (!converged)
  {
    stat.n_newton_wasted += n_newton_last_dt;
    stat.n_linear_wasted += n_linear_last_dt;
    stat.n_timesteps_wasted++;
    converged = 0;

    X = Xn;

    std::cout << buffer << std::flush;
  }
  else //convergence reached
  {
    stat.n_newton_total += n_newton_last_dt;
    stat.n_linear_total += n_linear_last_dt;
    stat.n_timesteps_total++;
    converged = 1;

    print_timestep(time + deltat, deltat);

    time_data["time"].push_back(time + deltat);

    for (ms_well *w : wells)
    {
      w->calc_rates_velocity(X, op_vals_arr, time_data, mesh->n_blocks);
    }

    // calculate FIPS
    FIPS.assign(nc, 0);
    for (index_t i = 0; i < mesh->n_res_blocks; i++)
    {
      for (uint8_t c = 0; c < nc; c++)
      {
        // assumung ACC_OP is 0
        FIPS[c] += PV[i] * op_vals_arr[i * n_ops + 0 + c];
      }
    }

    for (uint8_t c = 0; c < nc; c++)
    {
      time_data["FIPS c " + std::to_string(c) + " (kmol)"].push_back(FIPS[c]);
    }

    Xn = X;
    op_vals_arr_n = op_vals_arr;
    t += dt;
  }
  return converged;
}

template<uint8_t NC>
void
engine_nc_dvelocity_cpu<NC>::apply_composition_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
  double sum_z, new_z;
  index_t nb = mesh->n_blocks;
  bool z_corrected;
  index_t n_corrected = 0;

  for (index_t i = 0; i < nb; i++)
  {
    sum_z = 0;
    z_corrected = false;

    // check all but one composition in grid block
    for (char c = 0; c < nc - 1; c++)
    {
      new_z = X[i * nc + z_var + c] - dX[i * nc + z_var + c];
      if (new_z < min_zc)
      {
        new_z = min_zc;
        z_corrected = true;
      }
      else if (new_z > 1 - min_zc)
      {
        new_z = 1 - min_zc;
        z_corrected = true;
      }
      sum_z += new_z;
    }
    // check the last composition
    new_z = 1 - sum_z;
    if (new_z < min_zc)
    {
      new_z = min_zc;
      z_corrected = true;
    }
    sum_z += new_z;

    if (z_corrected)
    {
      // normalize compositions and set appropriate update
      for (char c = 0; c < nc - 1; c++)
      {
        new_z = X[i * nc + z_var + c] - dX[i * nc + z_var + c];

        new_z = std::max(min_zc, new_z);
        new_z = std::min(1 - min_zc, new_z);

        new_z = new_z / sum_z;
        dX[i * nc + z_var + c] = X[i * nc + z_var + c] - new_z;
      }
      n_corrected++;
    }
  }
  if (n_corrected)
    std::cout << "Composition correction applied in " << n_corrected << " block(s)" << std::endl;
}


template<uint8_t NC>
void
engine_nc_dvelocity_cpu<NC>::apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
  value_t max_dx = params->newton_params[0];
  value_t ratio, dx;
  index_t n_corrected = 0;

  for (int i = 0; i < mesh->n_blocks; i++)
  {
    ratio = 1.0;
    old_z[nc - 1] = 1.0;
    new_z[nc - 1] = 1.0;
    for (int j = 0; j < nc - 1; j++)
    {
      old_z[j] = X[i * nc + j + z_var];
      old_z[nc - 1] -= old_z[j];
      new_z[j] = old_z[j] - dX[i * nc + j + z_var];
      new_z[nc - 1] -= new_z[j];
    }

    for (int j = 0; j < nc; j++)
    {
      dx = fabs(new_z[j] - old_z[j]);
      if (dx > 0.0001)  // if update is not too small
      {
        ratio = std::min<value_t>(ratio, max_dx / dx);  // update the ratio
      }
    }

    if (ratio < 1.0)  // perform chopping if ratio is below 1.0
    {
      n_corrected++;
      for (int j = z_var; j < z_var + nc - 1; j++)
      {
        dX[i * nc + j] *= ratio;
      }
    }
  }
  if (n_corrected)
    std::cout << "Local chop applied in " << n_corrected << " block(s)" << std::endl;
}

template<uint8_t NC>
void
engine_nc_dvelocity_cpu<NC>::average_operator(std::vector<value_t> &av_op)
{
  for (int c = 0; c < nc; c++)
  {
    av_op[c] = 0;
  }
  for (int i = 0; i < mesh->n_res_blocks; i++)
  {
    for (int c = 0; c < nc; c++)
    {
      av_op[c] += op_vals_arr[i * n_ops + c];
    }
  }
  for (int c = 0; c < nc; c++)
  {
    av_op[c] /= mesh->n_res_blocks;
  }
}


template<uint8_t NC>
int
engine_nc_dvelocity_cpu<NC>::print_timestep(value_t time, value_t deltat)
{
  double estimate;
  int hour, min, sec;
  char buffer[1024];
  char buffer2[1024];
  char line[] = "-------------------------------------------------------------------------------------------------------------\n";

  estimate = (clock() + timer->timer) / CLOCKS_PER_SEC;
  hour = estimate / 3600;
  estimate -= hour * 3600;
  min = estimate / 60;
  estimate -= min * 60;
  sec = estimate;

 // sprintf(buffer, "T = %.3lf, DT = %.3lf, NI = %d, LI = %d, RES = %.1e (%.1e) (%.1e), CFL=%.3lf (ELAPSED %02d:%02d:%02d",
 //   time, deltat, n_newton_last_dt, n_linear_last_dt, newton_residual_last_dt, well_residual_last_dt, velocity_residual_last_dt, CFL_max, hour, min, sec);
  sprintf(buffer, "T = %.3lf, DT = %.3lf, NI = %d, LI = %d, RES = %.1e (%.1e) (%.1e) (%.1e), CFL=%.3lf (ELAPSED %02d:%02d:%02d",
    time, deltat, n_newton_last_dt, n_linear_last_dt, newton_residual_last_dt, well_residual_last_dt, res_wel_vel, res_res_vel,CFL_max, hour, min, sec);
  
  if ((dt * params->mult_ts > params->max_ts || full_step_timer.timer) && t < stop_time)
  {
    if (!full_step_timer.timer)
    {
      full_step_timer.start();
      t_full_step = t;
    }
    else
    {
      full_step_timer.stop();
      estimate = (full_step_timer.timer) / (t - t_full_step) * (stop_time - t) / CLOCKS_PER_SEC;
      full_step_timer.start();
      hour = estimate / 3600;
      estimate -= hour * 3600;
      min = estimate / 60;
      estimate -= min * 60;
      sec = estimate;
      sprintf(buffer2, "%s, REMAINING %02d:%02d:%02d", buffer, hour, min, sec);
      sprintf(buffer, "%s", buffer2);
    }
  }
  sprintf(buffer2, "%s %s )\n%s", line, buffer, line);
  std::cout << buffer2 << std::flush;

  return 0;
}

template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::solve_linear_equation()
{
  int r_code;
  char buffer[1024];
  timer->node["linear solver setup"].start();
  r_code = linear_solver->setup(Jacobian);
  timer->node["linear solver setup"].stop();
  //cpr_prec.setup (&Jacobian);

  if (r_code)
  {
    printf("ERROR: Linear solver setup returned %d, newton residual %lf \n", r_code, newton_residual_last_dt);
    // provoke timestep cut
    newton_residual_last_dt = 1000;
    //break;
    exit(1);
  }

  timer->node["linear solver solve"].start();
  r_code = linear_solver->solve(&RHS[0], &dX[0]);
  timer->node["linear solver solve"].stop();


  if (r_code)
  {
    printf("ERROR: Linear solver solve returned %d\n", r_code);
    exit(1);
  }
  else
  {
   // sprintf(buffer, "\t #%d (%.4e, %.4e, %.4e): lin %d (%.1e)\n", n_newton_last_dt + 1, newton_residual_last_dt,
    //  well_residual_last_dt, velocity_residual_last_dt, linear_solver->get_n_iters(), linear_solver->get_residual());
    sprintf(buffer, "\t #%d (%.4e, %.4e, %.4e, % .4e): lin %d (%.1e)\n", n_newton_last_dt + 1, newton_residual_last_dt,
      well_residual_last_dt, res_wel_vel, res_res_vel, linear_solver->get_n_iters(), linear_solver->get_residual());
    std::cout << buffer << std::flush;
    n_linear_last_dt += linear_solver->get_n_iters();
  }
  return 0;
}






template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::darcy_velocity(index_t j, value_t velocity, value_t p_diff, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assembly_kernel)
{
  // linear velocity (Darcy) for well part as well. 
  index_t n_blocks = mesh->n_blocks;
  std::vector <value_t> &tran = mesh->tran;
  value_t* Jac = jacobian->get_values();
  index_t* rows = jacobian->get_rows_ptr();
  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  auto idx = rows[NC * n_blocks] + (NC * 2 + 1)*velocity_mapper[j];  // idx for starting block c

  std::vector <value_t> &grav_coef = mesh->grav_coef;        // taking into account  hydraulic losses  = g * dz
  index_t  jac_idx = 0;
  if (p_diff < 0)
  { // Bm is upwind, V > 0 
    RHS[NC * n_blocks + velocity_mapper[j]] = velocity + tran[j] * p_diff * op_vals_arr[block_m[j] * N_OPS + LAMBDA_OP];
    for (index_t v = 0; v < NC; v++)
    {
      Jac[idx + v] = tran[j] * op_ders_arr[(block_m[j] * N_OPS + LAMBDA_OP) * nc + v] * p_diff;
      if (v == 0)
      {
        Jac[idx + v] -= tran[j] * op_vals_arr[block_m[j] * N_OPS + LAMBDA_OP];
        Jac[idx + v + NC] = tran[j] * op_vals_arr[block_m[j] * N_OPS + LAMBDA_OP];
        // Block D identity
        Jac[idx + NC * 2 + v] = 1;
      }
      else
      {  // derivative wrt composition is zero!
        Jac[idx + v + NC] = 0;
      }
    }
  }
  else
  {  // Bp is upwind,  V < 0 
    RHS[NC * n_blocks + velocity_mapper[j]] = velocity + tran[j] * op_vals_arr[block_p[j] * N_OPS + LAMBDA_OP] * p_diff;
    jac_idx = idx + NC;
    for (index_t v = 0; v < NC; v++)
    {
      Jac[jac_idx + v] = tran[j] * op_ders_arr[(block_p[j] * N_OPS + LAMBDA_OP) * nc + v] * p_diff;
      if (v == 0)
      {
        Jac[jac_idx + v] += tran[j] * op_vals_arr[block_p[j] * N_OPS + LAMBDA_OP];
        Jac[idx + v] = -tran[j] * op_vals_arr[block_p[j] * N_OPS + LAMBDA_OP];
        Jac[idx + NC * 2 + v] = 1;
      }
      else
      {  // derivative wrt composition is zero!
        Jac[idx + v] = 0;
      }
    }
  }
}

/*

template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::momentum_mswell(index_t j, value_t velocity, value_t p_diff, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assembly_kernel)
{
  // assembling rhs for well connection equation ( R = P_m - P_p - (\deltaP_h + \deltaP_f + deltaP_a) ) 
  // assemble the jacobian associated to the well connection equation. 

  index_t n_blocks = mesh->n_blocks;
  std::vector <value_t> &tran = mesh->tran;
  value_t* Jac = jacobian->get_values();
  index_t* rows = jacobian->get_rows_ptr();
  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  auto idx = rows[NC * n_blocks] + (NC * 2 + 1)*velocity_mapper[j];  // idx for starting block c
  index_t  jac_idx = 0;
  std::vector <value_t> &grav_coef = mesh->grav_coef;                  // taking into account  hydraulic losses  = g * dz
  value_t dp_h, dp_f, dp_a, Re;							                           // dp_h = hydrostatic losses,  dp_f = friction losses, dp_a = acceleration losses, re = reynold number. 
  value_t D = wells[0]->segment_diameter;										           // Hydraulic diameter of each segment 
  value_t f;	                                                         // Fanning Friction Factor 
  value_t l = l = wells[0]->segment_depth_increment;                   // length define as segment depth increment;                                                     // length define as segment depth increment
  p_diff = X[block_p[j] * NC + P_VAR] - X[block_m[j] * NC + P_VAR];
  const double sign = (velocity < 0) ? -1. : 1.;
  const double C = (sign * 2. / (86400. * 86400. * 1.e+5));
  value_t roughness = wells[0]->segment_roughness;
  value_t frictionCoefficient = (1. / pow(-3.6 * log10(6.9 / 4000. + pow(roughness / (3.7 * D), 10. / 9.)), 2) - 16. / 2000.) / (4000. - 2000.);

  if (p_diff < 0)
  {
    // block_m is the upwind 
    calculateFanningFactor(velocity, roughness, block_m[j], f);
    calculatePressureLosses(velocity, roughness, block_m[j], j, dp_f, dp_h, f);
    //calculateFrictionLosses(velocity, roughness, block_m[j], dp_f, f);
    //dp_f = C * f * op_vals_arr[block_m[j] * N_OPS + DENS_OP] * velocity * velocity * l / D;                                         // friction losses      unit: m^3/day -> m^3/s and Pa -> bar
    dp_h = 0.5 * (op_vals_arr[block_m[j] * N_OPS + DENS_OP] + op_vals_arr[block_p[j] * N_OPS + DENS_OP]) * grav_coef[j];                                  // Hydrostatic losses   /  grav_cpef = g * Dm - Dp   m < p D2 - D3 < 0 ; D4 - D
    RHS[NC * n_blocks + velocity_mapper[j]] = X[block_m[j] * NC + P_VAR] - X[block_p[j] * NC + P_VAR] - (dp_h + dp_f);   // P_m - P_p - ()
    for (index_t v = 0; v < NC; v++)
    {
      Jac[idx + v] = -(0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * nc + v] * grav_coef[j] + C * f * op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * nc + v] * velocity * velocity  * l / D);
      if (v == 0)
      {
        Jac[idx + v] += 1;
        Jac[idx + v + NC] = -1 - 0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * nc + v] * grav_coef[j];

        // Block D 
        Jac[idx + NC * 2 + v] = -C * 2. * f * op_vals_arr[block_m[j] * N_OPS + DENS_OP] * velocity  * l / D;
      }
      else
      {  // derivative wrt composition is zero!
        Jac[idx + v + NC] = -0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * nc + v] * grav_coef[j];
      }
    }
  }
  else
  {
    // block_p is the upwind
    Re = op_vals_arr[block_p[j] * N_OPS + RE_OP] * fabs(velocity) * D * 1000 / (24. * 3600.);   // 1000 cP = 1 Pa * sec; (24. * 3600.)s = 1day
    calculateFanningFactor(velocity, roughness, block_p[j], f);
    calculatePressureLosses(velocity, roughness, block_p[j], j, dp_f, dp_h, f);                                                                              // Hydrostatic losses
    RHS[NC * n_blocks + velocity_mapper[j]] = X[block_m[j] * NC + P_VAR] - X[block_p[j] * NC + P_VAR] - (dp_h + dp_f);
    jac_idx = idx + NC;
    for (index_t v = 0; v < NC; v++)
    {
      Jac[jac_idx + v] = -(0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * nc + v] * grav_coef[j] + C * f * op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * nc + v] * velocity * velocity * l / D);
      if (v == 0) 
      {
        Jac[jac_idx + v] -= 1; // check it 

        Jac[idx + v] = 1 - 0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * nc + v] * grav_coef[j];
        // Block D
        Jac[idx + NC * 2 + v] = -C * 2. * f * op_vals_arr[block_p[j] * N_OPS + DENS_OP] * velocity * l / D;;
      }
      else
      {  // derivative wrt composition is zero!
        Jac[idx + v] = -0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * nc + v] * grav_coef[j];
      }
    }
  }
}
*/

template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::momentum_mswell(index_t iw, index_t j, value_t velocity, value_t p_diff, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assembly_kernel)
{
  // assembling rhs for well connection equation ( R = P_m - P_p - (\deltaP_h + \deltaP_f + deltaP_a) ) 
  // assemble the jacobian associated to the well connection equation. 

  index_t n_blocks = mesh->n_blocks;
  std::vector <value_t> &tran = mesh->tran;
  value_t* Jac = jacobian->get_values();
  index_t* rows = jacobian->get_rows_ptr();
  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  auto idx = rows[N_NODE * n_blocks + velocity_mapper[j]];  // idx for starting block c
  index_t  jac_idx = 0;
  std::vector <value_t> &grav_coef = mesh->grav_coef;                 // taking into account  hydraulic losses  = g * dz
  value_t dp_h, dp_f, dp_a, Re;							                          // dp_h = hydrostatic losses,  dp_f = friction losses, dp_a = acceleration losses, re = reynold number. 

  value_t D = wells[iw]->segments[block_m[j] - wells[iw]->well_head_idx].diameter;
  value_t f;	                                                        // Fanning Friction Factor 
  value_t  l = 0.5 *(wells[iw]->segments[block_m[j] - wells[iw]->well_head_idx].length + wells[iw]->segments[block_p[j] - wells[iw]->well_head_idx].length);

  p_diff = X[block_p[j] * N_NODE + P_VAR] - X[block_m[j] * N_NODE + P_VAR];
  const double sign = (velocity < 0) ? -1. : 1.;
  const double C = (sign * 2. / (86400. * 86400. * 1.e+5));

  value_t roughness = 0.0000000000000000001;
  value_t frictionCoefficient = (1. / pow(-3.6 * log10(6.9 / 4000. + pow(roughness / (3.7 * D), 10. / 9.)), 2) - 16. / 2000.) / (4000. - 2000.);

  if (p_diff < 0)
  {
    // block_m is the upwind 
    calculateFanningFactor(velocity, roughness, block_m[j], f);
    calculatePressureLosses(velocity, roughness, block_m[j], j, dp_f, dp_h, f, l, D);
    dp_h = 0.5 * (op_vals_arr[block_m[j] * N_OPS + DENS_OP] + op_vals_arr[block_p[j] * N_OPS + DENS_OP]) * grav_coef[j];                                  // Hydrostatic losses   /  grav_cpef = g * Dm - Dp   m < p D2 - D3 < 0 ; D4 - D
    RHS[N_NODE * n_blocks + velocity_mapper[j]] = X[block_m[j] * N_NODE + P_VAR] - X[block_p[j] * N_NODE + P_VAR] - (dp_h + dp_f);   // P_m - P_p - ()
    for (index_t v = 0; v < N_NODE; v++)
    {
      Jac[idx + v] = -(0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j] + C * f * op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * velocity * velocity  * l / D);
      if (v == 0)
      {
        Jac[idx + v] += 1;
        Jac[idx + v + N_NODE] = -1 - 0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];

        // Block D 
        Jac[idx + N_NODE * 2 + v] = -C * 2. * f * op_vals_arr[block_m[j] * N_OPS + DENS_OP] * velocity  * l / D;
      }
      else
      {  // derivative wrt composition is zero!
        Jac[idx + v + N_NODE] = -0.5 * op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
      }
    }
  }
  else
  {
    // block_p is the upwind
    Re = op_vals_arr[block_p[j] * N_OPS + RE_OP] * fabs(velocity) * D * 1000 / (24. * 3600.);   // 1000 cP = 1 Pa * sec; (24. * 3600.)s = 1day
    calculateFanningFactor(velocity, roughness, block_p[j], f);
    calculatePressureLosses(velocity, roughness, block_p[j], j, dp_f, dp_h, f, l, D);                                                                              // Hydrostatic losses
    RHS[N_NODE * n_blocks + velocity_mapper[j]] = X[block_m[j] * N_NODE + P_VAR] - X[block_p[j] * N_NODE + P_VAR] - (dp_h + dp_f);
    jac_idx = idx + N_NODE;
    for (index_t v = 0; v < N_NODE; v++)
    {
      Jac[jac_idx + v] = -(0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j] + C * f * op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * velocity * velocity * l / D);
      if (v == 0)
      {
        Jac[jac_idx + v] -= 1; // check it 

        Jac[idx + v] = 1 - 0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
        // Block D
        Jac[idx + N_NODE * 2 + v] = -C * 2. * f * op_vals_arr[block_p[j] * N_OPS + DENS_OP] * velocity * l / D;;
      }
      else
      {  // derivative wrt composition is zero!
        Jac[idx + v] = -0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
      }
    }
  }
}



template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::momentum_mswell_withacceleration(index_t iw, index_t j, value_t velocity, value_t p_diff, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assembly_kernel)
{
  // assembling rhs for well connection equation ( R = P_m - P_p - (\deltaP_h + \deltaP_f + deltaP_a) ) 
  // assemble the jacobian associated to the well connection equation. 

  index_t n_blocks = mesh->n_blocks;
  std::vector <value_t> &tran = mesh->tran;
  value_t* Jac = jacobian->get_values();
  index_t* rows = jacobian->get_rows_ptr();
  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  index_t idx = rows[N_NODE * n_blocks + velocity_mapper[j]] + (N_NODE * (ctrConblock[j] - 2));             // idx for starting block c
  index_t idx_perf = rows[N_NODE * n_blocks + velocity_mapper[j]]; ;                                      // idx to the upwind perforated reservoir block
  index_t idxD = rows[N_NODE * n_blocks + velocity_mapper[j]] + (N_NODE * ctrConblock[j]);               // index to entry of Block D
  index_t  jac_idx = 0;
  index_t idx_downwind;
  std::vector <value_t> &grav_coef = mesh->grav_coef;                                                    // taking into account  hydraulic losses  = g * dz
  value_t dp_h, dp_f, dp_a, Re;							                                                             // dp_h = hydrostatic losses,  dp_f = friction losses, dp_a = acceleration losses, re = reynold number.
  value_t D = wells[iw]->segments[block_m[j] - wells[iw]->well_head_idx].diameter;										                                // Hydraulic diameter of each segment 
  value_t f;	                                                                                           // Fanning Friction Factor 
  value_t  l = 0.5 *(wells[iw]->segments[block_m[j] - wells[iw]->well_head_idx].length + wells[iw]->segments[block_p[j] - wells[iw]->well_head_idx].length);                                                    // length define as segment depth increment;                                                     
  p_diff = X[block_p[j] * N_NODE + P_VAR] - X[block_m[j] * N_NODE + P_VAR];  // two times writing it [delete it] 
  const double sign = (velocity < 0) ? -1. : 1.;
  const double C = (sign * 2. / (86400. * 86400. * 1.e+5));
  value_t roughness = 0.0000000000000000001;
  value_t frictionCoefficient = (1. / pow(-3.6 * log10(6.9 / 4000. + pow(roughness / (3.7 * D), 10. / 9.)), 2) - 16. / 2000.) / (4000. - 2000.);
  value_t Area = wells[iw]->segments[block_m[j] - wells[iw]->well_head_idx].area;
  dp_a = 0;
  value_t m_tot = 0;
  //idx += N_NODE; // first block is the perforated reservoir block. [correct only one peforation and not both sided of the connectino be perforated]
  // in case two side of the connection is perforated.
  if (ctrConblock[j] > 3)
  { // two side of the connection is perforated.
    // upwind reservoir block 
    // downwind reservoir block. [// make downwind derivatives equal to zero = 0 ]

    if (p_diff < 0)
    {
      // block_m is the upwindreservoirIdxperf
      index_t downwind_res = reservoirIdxperf(iw, block_p[j]);
      index_t upwind_res = reservoirIdxperf(iw, block_m[j]);


      calculateFanningFactor(velocity, roughness, block_m[j], f);
      calculatePressureLosses(velocity, roughness, block_m[j], j, dp_f, dp_h, f, l, D);
      dp_h = 0.5 * (op_vals_arr[block_m[j] * N_OPS + DENS_OP] + op_vals_arr[block_p[j] * N_OPS + DENS_OP]) * grav_coef[j];                                  // Hydrostatic losses   /  grav_cpef = g * Dm - Dp   m < p D2 - D3 < 0 ; D4 - D
      calculateAccelerationlosses(iw, velocity, block_m[j], dp_a, m_tot, Area);
      RHS[N_NODE * n_blocks + velocity_mapper[j]] = X[block_m[j] * N_NODE + P_VAR] - X[block_p[j] * N_NODE + P_VAR] - (dp_h + dp_f + dp_a);   // P_m - P_p - ()

      for (index_t v = 0; v < N_NODE; v++)
      {
        Jac[idx + v] = -(0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j] + C * f * op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * velocity * velocity  * l / D);
        if (v == 0)
        {
          Jac[idx + v] += 1;
          Jac[idx + v + N_NODE] = -1 - 0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];

          // Block D
          //Jac[idx + N_NODE * 2 + v] = -C * 2. * f * op_vals_arr[block_m[j] * N_OPS + DENS_OP] * velocity  * l / D + m_tot;
          Jac[idxD] = -C * 2. * f * op_vals_arr[block_m[j] * N_OPS + DENS_OP] * velocity  * l / D + m_tot;
        }
        else
        {  // derivative wrt composition is zero!
          Jac[idx + v + N_NODE] = -0.5 * op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
        }
      }
      // fill perforated reservoir block derivative as well
      // for now just fill it with zero
      for (index_t v = 0; v < N_NODE; v++)
      {
        //idx_perf = rows[N_NODE * n_blocks + velocity_mapper[j]] + (N_NODE * upwind_res);
        //index_t idx_downwind = rows[N_NODE * n_blocks + velocity_mapper[j]] + (N_NODE * downwind_res);           

        if (upwind_res - downwind_res > 0)
        {
          // upwind connected reservoir  block is bigger
          idx_downwind = idx_perf;
          idx_perf += N_NODE;
        }
        else
        { //downwind is bigger and idx_perf is the first entry.
          idx_downwind = idx_perf + N_NODE;;
        }

        Jac[idx + v] += 2 * velocity * Der_mass_upblock[v];
        Jac[idx_perf + v] = 2 * velocity * Der_mass_perf[v];
        // make zero the downwinf
        Jac[idx_downwind + v] = 0; // 
      }
    }
    else
    {
      // block_p is the upwind
      index_t downwind_res = reservoirIdxperf(iw, block_m[j]);
      index_t upwind_res = reservoirIdxperf(iw, block_p[j]);
      if (upwind_res - downwind_res > 0)
      {
        // upwind connected reservoir  block is bigger
        idx_downwind = idx_perf;
        idx_perf += N_NODE;
      }
      else
      { //downwind
        idx_downwind = idx_perf + N_NODE;;
      }
      Re = op_vals_arr[block_p[j] * N_OPS + RE_OP] * fabs(velocity) * D * 1000 / (24. * 3600.);   // 1000 cP = 1 Pa * sec; (24. * 3600.)s = 1day
      calculateFanningFactor(velocity, roughness, block_p[j], f);
      calculatePressureLosses(velocity, roughness, block_p[j], j, dp_f, dp_h, f, l, D);           // Hydrostatic losses
      calculateAccelerationlosses(iw, velocity, block_p[j], dp_a, m_tot, Area);                  // Hydrostatic losses
      RHS[N_NODE * n_blocks + velocity_mapper[j]] = X[block_m[j] * N_NODE + P_VAR] - X[block_p[j] * N_NODE + P_VAR] - (dp_h + dp_f + dp_a);
      jac_idx = idx + N_NODE;
      for (index_t v = 0; v < N_NODE; v++)
      {
        Jac[jac_idx + v] = -(0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j] + C * f * op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * velocity * velocity * l / D);
        if (v == 0)
        {
          Jac[jac_idx + v] -= 1; // check it

          Jac[idx + v] = 1 - 0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
          // Block D
          //Jac[idx + N_NODE * 2 + v] = -C * 2. * f * op_vals_arr[block_p[j] * N_OPS + DENS_OP] * velocity * l / D + m_tot;
          Jac[idxD] = -C * 2. * f * op_vals_arr[block_p[j] * N_OPS + DENS_OP] * velocity * l / D + m_tot;
        }
        else
        {  // derivative wrt composition is zero!
          Jac[idx + v] = -0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
        }
      }
      // fill perforated reservoir block derivative as well
      // for now just fill it with zero


      for (index_t v = 0; v < N_NODE; v++)
      {
        // idx_perf = rows[N_NODE * n_blocks + velocity_mapper[j]] + (N_NODE * upwind_res);
        // index_t idx_downwind = rows[N_NODE * n_blocks + velocity_mapper[j]] + (N_NODE * downwind_res);

        Jac[jac_idx + v] += 2 * velocity * Der_mass_upblock[v];
        Jac[idx_perf + v] = 2 * velocity * Der_mass_perf[v];
        // make zero the downwind
        Jac[idx_downwind + v] = 0; // 
      }
    }
  }
  else
  {
    if (p_diff < 0)
    {
      // block_m is the upwind
      calculateFanningFactor(velocity, roughness, block_m[j], f);
      calculatePressureLosses(velocity, roughness, block_m[j], j, dp_f, dp_h, f, l, D);
      dp_h = 0.5 * (op_vals_arr[block_m[j] * N_OPS + DENS_OP] + op_vals_arr[block_p[j] * N_OPS + DENS_OP]) * grav_coef[j];                                  // Hydrostatic losses   /  grav_cpef = g * Dm - Dp   m < p D2 - D3 < 0 ; D4 - D
      calculateAccelerationlosses(iw, velocity, block_m[j], dp_a, m_tot, Area);
      RHS[N_NODE * n_blocks + velocity_mapper[j]] = X[block_m[j] * N_NODE + P_VAR] - X[block_p[j] * N_NODE + P_VAR] - (dp_h + dp_f + dp_a);   // P_m - P_p - ()

      for (index_t v = 0; v < N_NODE; v++)
      {
        Jac[idx + v] = -(0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j] + C * f * op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * velocity * velocity  * l / D);
        if (v == 0)
        {
          Jac[idx + v] += 1;
          Jac[idx + v + N_NODE] = -1 - 0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];

          // Block D
          //Jac[idx + N_NODE * 2 + v] = -C * 2. * f * op_vals_arr[block_m[j] * N_OPS + DENS_OP] * velocity  * l / D + m_tot;
          Jac[idxD] = -C * 2. * f * op_vals_arr[block_m[j] * N_OPS + DENS_OP] * velocity  * l / D + m_tot;
        }
        else
        {  // derivative wrt composition is zero!
          Jac[idx + v + N_NODE] = -0.5 * op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
        }
      }
      // fill perforated reservoir block derivative as well
      // for now just fill it with zero
      for (index_t v = 0; v < N_NODE; v++)
      {
        Jac[idx + v] += 2 * velocity * Der_mass_upblock[v];
        Jac[idx_perf + v] = 2 * velocity * Der_mass_perf[v];
        // just to test what happens if we neglect the derivative of massflux wrt the omega
        //  Jac[idx_perf + v] = 0; // for now just fill it with zero to test
      }
    }
    else
    {
      // block_p is the upwind
      Re = op_vals_arr[block_p[j] * N_OPS + RE_OP] * fabs(velocity) * D * 1000 / (24. * 3600.);   // 1000 cP = 1 Pa * sec; (24. * 3600.)s = 1day
      calculateFanningFactor(velocity, roughness, block_p[j], f);
      calculatePressureLosses(velocity, roughness, block_p[j], j, dp_f, dp_h, f, l, D);           // Hydrostatic losses
      calculateAccelerationlosses(iw, velocity, block_p[j], dp_a, m_tot, Area);                  // Hydrostatic losses
      RHS[N_NODE * n_blocks + velocity_mapper[j]] = X[block_m[j] * N_NODE + P_VAR] - X[block_p[j] * N_NODE + P_VAR] - (dp_h + dp_f + dp_a);
      jac_idx = idx + N_NODE;
      for (index_t v = 0; v < N_NODE; v++)
      {
        Jac[jac_idx + v] = -(0.5* op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j] + C * f * op_ders_arr[(block_p[j] * N_OPS + DENS_OP) * N_NODE + v] * velocity * velocity * l / D);
        if (v == 0)
        {
          Jac[jac_idx + v] -= 1; // check it

          Jac[idx + v] = 1 - 0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
          // Block D
          //Jac[idx + N_NODE * 2 + v] = -C * 2. * f * op_vals_arr[block_p[j] * N_OPS + DENS_OP] * velocity * l / D + m_tot;
          Jac[idxD] = -C * 2. * f * op_vals_arr[block_p[j] * N_OPS + DENS_OP] * velocity * l / D + m_tot;
        }
        else
        {  // derivative wrt composition is zero!
          Jac[idx + v] = -0.5* op_ders_arr[(block_m[j] * N_OPS + DENS_OP) * N_NODE + v] * grav_coef[j];
        }
      }
      // fill perforated reservoir block derivative as well
      // for now just fill it with zero


      for (index_t v = 0; v < N_NODE; v++)
      {
        Jac[jac_idx + v] += 2 * velocity * Der_mass_upblock[v];
        Jac[idx_perf + v] = 2 * velocity * Der_mass_perf[v];
      }
    }
  }


}

/*

template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::calculatePressureLosses(value_t velocity, value_t roughness, value_t block_up, index_t j, value_t & dp_f, value_t & dp_h, value_t f)
{
  calculateFrictionLosses(velocity, roughness, block_up, dp_f, f);   // calculate friction losses
  calculateHydrauliclosses(j, dp_h);                                 // calculate hydraulic losses
}
*/

template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::calculatePressureLosses(value_t velocity, value_t roughness, value_t block_up, index_t j, value_t & dp_f, value_t & dp_h, value_t f, value_t l, value_t D)
{
  calculateFrictionLosses(velocity, roughness, block_up, dp_f, f, l, D);   // calculate friction losses
  calculateHydrauliclosses(j, dp_h);                                 // calculate hydraulic losses
}
template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::calculateHydrauliclosses(index_t j, value_t &dp_h)
{
  dp_h = 0.5 * (op_vals_arr[mesh->block_m[j] * N_OPS + DENS_OP] + op_vals_arr[mesh->block_p[j] * N_OPS + DENS_OP]) * mesh->grav_coef[j];
  if (dp_h != 0)
  {
    value_t kia = 1;
  }
}


template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::init_WellDynamicProperties(std::vector<value_t> &X_init)
{
  /*
  initialize all dynamic properties of well segments (changing during simulation)  [for now only pressure initialization] 
  //3 options:
  1. all segments pressure equal to the pressure of the perforated reservoir block
  2. All segments pressure equal to the wellhead pressure
  3. Segments pressure initiale based on hydrostatic pressure changes from the well head
  */
  value_t grav_const = 9.80665e-5;
  for (index_t iw = 0; iw < wells.size(); iw++)
  {
    int r_i = std::get<1>(wells[iw]->perforations[0]);  // go through this part!
   // depth of the well head block - well controls work at this depth
    for (index_t p = 0; p < wells[iw]->n_segments + 1; p++)
    {
      if (p > 0)
      {
        // well pressure is equal to the pressure of the reservoir block 
        //X_init[(wells[iw]->well_head_idx + p)*nc] = X_init[r_i * nc];

        // based on hydrostatic pressure drop
        //X_init[(wells[iw]->well_head_idx + p)*nc] = X_init[(wells[iw]->well_head_idx)*nc] + ((mesh->depth[wells[iw]->well_head_idx + p] - mesh->depth[wells[iw]->well_head_idx]) * grav_const * op_vals_arr[wells[iw]->well_head_idx * N_OPS + DENS_OP]);  // rho *g * Delta D // for now density equal to the wellhead ( incompressible )

        //  All segments pressure equal to the well head presuure  
        X_init[(wells[iw]->well_head_idx + p)*nc] = X_init[(wells[iw]->well_head_idx)*nc];  // rho *g * Delta D
      }
    }
  }
}

template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::calculateFanningFactor(value_t velocity, value_t roughness, value_t block_up, value_t &f)
{

  value_t D = wells[0]->segment_diameter;										                                                                                    // Fanning Friction Factor 
  value_t frictionCoefficient = (1. / pow(-3.6 * log10(6.9 / 4000. + pow(roughness / (3.7 * D), 10. / 9.)), 2) - 16. / 2000.) / (4000. - 2000.);
  value_t Re;
  Re = op_vals_arr[block_up * N_OPS + RE_OP] * fabs(velocity) * D * 1000 / (24. * 3600.);   // 1000 cP = 1 Pa * sec; (24. * 3600.)s = 1day
  if (Re < 1e-06)  // Re too small, no friction
  {
    f = 0.;
    return;
  }
  if (Re < 2000)  // laminar flow
  {
    f = 16. / Re;
  }
  else if (Re > 4000)  // Haaland's formula
  {
    f = 1. / pow(-3.6 * log10(6.9 / Re + pow(roughness
      / (3.7 * D), 10. / 9.)), 2);
  }
  else  // uncertain region, use linear interpolation of Re = 2000 and Re = 4000
  {
    f = 16. / 2000. + (Re - 2000.) * frictionCoefficient;
  }
}


template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::calculateFrictionLosses(value_t velocity, value_t roughness, value_t block_up, value_t &dp_f, value_t f, value_t l, value_t D)
{
  const double sign = (velocity < 0) ? -1. : 1.;
  const double C = (sign * 2. / (86400. * 86400. * 1.e+5));
  dp_f = C * f * op_vals_arr[block_up * N_OPS + DENS_OP] * velocity * velocity * l / D;     // friction losses      unit: m^3/day -> m^3/s and Pa -> bar
}



/*

template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::calculateFrictionLosses(value_t velocity, value_t roughness, value_t block_up, value_t &dp_f, value_t f)
{
  value_t D = wells[0]->segment_diameter;										                                // Hydraulic diameter of each segment 
  value_t l = wells[0]->segment_depth_increment;                                            // length define as segment depth increment;  
  const double sign = (velocity < 0) ? -1. : 1.;
  const double C = (sign * 2. / (86400. * 86400. * 1.e+5));
  dp_f = C * f * op_vals_arr[block_up * N_OPS + DENS_OP] * velocity * velocity * l / D;            // friction losses      unit: m^3/day -> m^3/s and Pa -> bar
  if (dp_f != 0)
  {
    value_t debug = 1;
  }
}
*/


#define NORMAL_ZC //If you want to use logtransform of zc, i.e. X = [P, log(z1), ..., log(znc-1)] instead of [P, z1, ..., znc-1], comment this line!
template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::apply_newton_update(value_t dt)
{
  // make sure that X after update stays within OBL limits
  //apply_obl_axis_global_chop_correction(X, dX);

  timer->node["newton update"].node["composition correction"].start();
  if (nc > 1)
  {
#ifdef NORMAL_ZC
    apply_composition_correction(X, dX);
#else
    apply_composition_correction_new(X, dX);
#endif
  }
  timer->node["newton update"].node["composition correction"].stop();

  if (params->newton_type == sim_params::NEWTON_GLOBAL_CHOP)
  {
#ifdef NORMAL_ZC
    apply_global_chop_correction(X, dX);
#else
    apply_global_chop_correction_new(X, dX);
#endif
  }
  // apply local chop only if number of components is 2 and more
  else if (params->newton_type == sim_params::NEWTON_LOCAL_CHOP && nc > 1)
  {
#ifdef NORMAL_ZC
    apply_local_chop_correction(X, dX);
#else
    apply_local_chop_correction_new(X, dX);
#endif
  }

  apply_obl_axis_local_correction(X, dX);
  //apply_velocity_update(dt);
  // make newton update
  std::transform(X.begin(), X.end(), dX.begin(), X.begin(), std::minus<double>());
  return 0;
}



template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::apply_velocity_update(value_t dt)
{// make velocity positive in case it becomes negative
  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  index_t n_conns = mesh->n_conns;
  index_t n_velocities = n_conns / 2;
  value_t velocity;
  index_t n_blocks = mesh->n_blocks;

  for (index_t j = 0; j < n_conns; j++)
  {
    // NONLINEAR VELOCITY (HOMOGENEOUS MODEL)- well connections; DARRCY velocity
    if (block_m[j] < block_p[j])
    {
      // replace it to the real MSWell momentum equations!
      velocity = X[NC * n_blocks + velocity_mapper[j]];

      if (velocity < 0)
      {
        X[NC * n_blocks + velocity_mapper[j]] *= -1;
      }
    }
  }
}



template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
  double max_ratio = 0;
  index_t n_vars_total = X.size();
  index_t n_obl_fixes = 0;

  for (index_t i = 0; i < mesh->n_blocks; i++)
  {
    value_t *axis_min = &op_axis_min[mesh->op_num[i]][0];
    value_t *axis_max = &op_axis_max[mesh->op_num[i]][0];
    for (index_t v = 0; v < 1; v++)
    {
      value_t new_x = X[i * nc + v] - dX[i * nc + v];
      if (new_x > axis_max[v])
      {
        dX[i * nc + v] = X[i * nc + v] - axis_max[v];
        n_obl_fixes++;
      }
      if (new_x < axis_min[v])
      {
        dX[i * nc + v] = X[i * nc + v] - axis_min[v];
        n_obl_fixes++;
      }
    }
  }

  if (n_obl_fixes > 0)
  {
    std::cout << "OBL axis correction applied " << n_obl_fixes << " time(s) \n";
  }
}

//
template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::return_well_number(value_t bm, value_t bp)
{
  // check thhe given connection correspond to what well and return that.
  index_t i_w, i_r;
  value_t wi;
  for (index_t iw = 0; iw < wells.size(); iw++)
  {
    for (index_t p = wells[iw]->well_head_idx; p < wells[iw]->well_head_idx + wells[iw]->n_segments; p++)
    {

      if (bm == p || bp == p)
      {
        return iw;
      }
    }
  }
}


template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::connection_perforated(index_t iw, value_t bm, value_t bp)
{
  index_t i_w, i_r;
  value_t wi;
  for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
  {
    std::tie(i_w, i_r, wi) = wells[iw]->perforations[p];
    value_t wb = i_w + wells[iw]->well_head_idx + 1;  // perforated well block number
    if (bm == wb || bp == wb)
    {
      return 1;
    }
  }
  return 0;
}


template<uint8_t NC>
double engine_nc_dvelocity_cpu<NC>::calculateAccelerationlosses(index_t iw, value_t velocity, value_t block_up, value_t & dp_a, value_t & m_tot, value_t Area)
{
  // remember to input dp_a, Der_mass_perf, Der_mass_upblock  as zero
  index_t n_blocks = mesh->n_blocks;
  index_t n_res_blocks = mesh->n_res_blocks;
  index_t i_w, i_r;
  value_t wi;
  value_t m_flux;
  value_t c = (2. / (86400.*86400.*1e+5* Area));  // constant multiplier for consistent units
  dp_a = 0;
  m_tot = 0;
  // save partial derivative wrt unknowns
  Der_mass_perf.resize(NC);
  Der_mass_upblock.resize(NC);
  for (uint8_t v = 0; v < NC; v++)
  {
    Der_mass_perf[v] = 0;
    Der_mass_upblock[v] = 0;
  }


  // connections between well segments and reservoir // 
  for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
  {
    std::tie(i_w, i_r, wi) = wells[iw]->perforations[p];
    //if (block_up == (n_res_blocks - 1) + i_w + (n_blocks - n_res_blocks) / 2) // this is now only injection. let's change it in the implementation add_Wells that i_w be correct. // that should be chngd
    if (block_up == i_w + wells[iw]->well_head_idx + 1) // this is now only injection. let's change it in the implementation add_Wells that i_w be correct. // that should be changd
    {
      // upwind segment is peforated
      // calculate mass flux!  \rho * v_perfoated? * A ;  how to find v_perforated?/
      calculate_mass_flux(block_up, i_r, wi, m_flux);
      dp_a += m_flux;
      //m_flux *= 1 / 86400; // convert to kg to second
    }
  }
  m_tot = dp_a;

  //dp_a *=  2 * velocity / (86400.*86400.*1e+5* Area);   // kg/(m*day^2) to kg/(m*s^2) [Pa], then to bar
  // der_vel = dp_a * c;
  dp_a *= velocity * c;   // kg/(m*day^2) to kg/(m*s^2) [Pa], then to bar
  for (uint8_t v = 0; v < N_NODE; v++)
  {
    Der_mass_perf[v] *= c;
    Der_mass_upblock[v] *= c;
  }
  m_tot *= c;

  //dp_a = 0;
  //m_tot = 0;


  return m_tot;
}

template<uint8_t NC>
void engine_nc_dvelocity_cpu<NC>::calculate_mass_flux(value_t block_up, value_t i_r, value_t wi, value_t & m_flux)
{
  // calculate mass flux entering/outgoing the wellsegment to the reservoir
  value_t p_diff = X[block_up * NC + P_VAR] - X[i_r * NC + P_VAR];

  if (p_diff < 0)
  {//   i_r is the upwind
    m_flux = wi * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * p_diff * op_vals_arr[i_r * N_OPS + DENS_OP];


    for (uint8_t v = 0; v < NC; v++)
    {

      Der_mass_perf[v] = wi * op_ders_arr[(i_r * N_OPS + LAMBDA_OP) * NC + v] * p_diff *  op_vals_arr[i_r * N_OPS + DENS_OP] + op_ders_arr[(i_r * N_OPS + DENS_OP) * NC + v] * wi * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * p_diff;
      if (v == 0)
      {

        Der_mass_perf[v] -= wi * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * op_vals_arr[i_r * N_OPS + DENS_OP];
        Der_mass_upblock[v] = wi * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * op_vals_arr[i_r * N_OPS + DENS_OP];
      }
      else
      {
        Der_mass_upblock[v] = 0;
      }
    }

  }
  else
  {//   block_up is the upwind
    m_flux = wi * op_vals_arr[block_up * N_OPS + LAMBDA_OP] * p_diff *   op_vals_arr[i_r * N_OPS + DENS_OP];
    // find partial derivative wrt unknowns
    for (uint8_t v = 0; v < NC; v++)
    {

      Der_mass_upblock[v] = wi * op_ders_arr[(block_up * N_OPS + LAMBDA_OP) * NC + v] * p_diff *  op_vals_arr[block_up * N_OPS + DENS_OP] + op_ders_arr[(block_up * N_OPS + DENS_OP) * NC + v] * wi * op_vals_arr[block_up * N_OPS + LAMBDA_OP] * p_diff;
      if (v == 0)
      {
        Der_mass_perf[v] = -wi * op_vals_arr[block_up * N_OPS + LAMBDA_OP] * op_vals_arr[block_up * N_OPS + LAMBDA_OP] * op_vals_arr[block_up * N_OPS + DENS_OP];
        Der_mass_upblock[v] += wi * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * op_vals_arr[i_r * N_OPS + LAMBDA_OP] * op_vals_arr[i_r * N_OPS + DENS_OP];
      }
      else
      {
        Der_mass_perf[v] = 0;
      }
    }
  }
}

template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::reservoirIdxperf(index_t iw, value_t b)
{
  index_t i_w, i_r;
  value_t wi;
  for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
  {
    std::tie(i_w, i_r, wi) = wells[iw]->perforations[p];
    value_t wb = i_w + wells[iw]->well_head_idx + 1;  // perforated well block number
    if (b == wb)
    {
      return i_r;
    }
  }
}

template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::additional_nnz_acc(csr_matrix_base *jacobian)
{

  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  index_t n_velocities = n_conns / 2;

  std::vector <index_t> &block_m = mesh->block_m;
  std::vector <index_t> &block_p = mesh->block_p;
  std::vector <index_t> &velocity_mapper = mesh->conn_index_to_one_way;
  //std::vector <value_t> velocity_sorted;

  index_t n_res_blocks = mesh->n_res_blocks;

  index_t i_w, i_r;
  value_t wi;
  int ctr = 0;
  for (index_t j = 0; j < n_conns; j++)
  {
    if (block_m[j] < block_p[j])
    {

      // In case one of the blocks between connection is perforated initialize it as well since we need a derivative for acceleration term.
      if (block_m[j] >= n_res_blocks && block_p[j] >= n_res_blocks)
      {
        for (index_t iw = 0; iw < wells.size(); iw++)
        {
          // connections between well segments and reservoir
          for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
          {
            std::tie(i_w, i_r, wi) = wells[iw]->perforations[p];  // well block number is wrong!
            value_t wb = i_w + wells[iw]->well_head_idx + 1;      // perforated well block number
            if (block_m[j] == wb || block_p[j] == wb)             // i_w is not correct!    i_w + wells[iw]->well_head_idx + 1
            {
              ctr += 1;
            }
          }
        }
      }
    }
  }
  return ctr;
}


template<uint8_t NC>
int engine_nc_dvelocity_cpu<NC>::cross_flow(index_t iw, std::vector<value_t> &X)
{
  index_t i_w, i_r;
  value_t wi;
  bool is_producer = wells[iw]->isProducer();
  for (index_t p = 0; p < wells[iw]->perforations.size(); p++)
  {
    /*
    1. check if the well is producer or injector [ based on the name of the well ]
    2. check whether cross-flow happens or not for the given peforation. if it happends  print it out .
    */

    std::tie(i_w, i_r, wi) = wells[iw]->perforations[p];

    value_t potential_diff = X[(i_w + wells[iw]->well_head_idx + 1) * N_NODE + P_VAR] - X[i_r * N_NODE + P_VAR];
    bool is_cross_flow = (is_producer && potential_diff > 0) || (!(is_producer) && potential_diff < 0);
    if (is_cross_flow)
    {
      std::cout << "Cross-flow happens for the well " << iw << " for this iteration \n";
    }
  }
  return 0;
}


template struct recursive_instantiator_nc<engine_nc_dvelocity_cpu, 2, MAX_NC>;
