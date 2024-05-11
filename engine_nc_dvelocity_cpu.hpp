
#ifndef CPU_SIMULATOR_NC_HPP
#define CPU_SIMULATOR_NC_HPP

#include <vector>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>


#include "globals.h"
#include "ms_well.h"
#include "engine_base.h"
#include "csr_matrix.h"
#include "linsolv_iface.h"
#include "evaluator_iface.h"


template<uint8_t NC>
class engine_nc_dvelocity_cpu : public engine_base
{

public:
  // number of components
  const static uint8_t NC_ = NC;
  // number of primary variables : [P, Z_1, ... Z_(NC-1)]
  const static uint8_t N_VARS = NC + 1;  // +1 velocity ( interface unknown)
  // order of primary variables:
  const static uint8_t P_VAR = 0;
  const static uint8_t Z_VAR = 1;
  // number of operators: NC accumulation operators, NC flux operators
  //const static uint8_t N_OPS = 2 * NC + 2;   //  + 2 ( total mobbility & total density ) 
  const static uint8_t N_OPS = 2 * NC + 3;   //  + 2 ( total mobbility & total density & reynolds operatr ) 
  const static uint8_t N_NODE = NC ;  // +1 velocity (interface unknown) + enthalpy
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t FLUX_OP = NC;
  const static uint8_t LAMBDA_OP = 2 * NC;
  const static uint8_t DENS_OP = 2 * NC + 1;
  const static uint8_t RE_OP = 2 * NC + 2;



  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // number of variables per jacobian matrix block
  //const static uint8_t N_VARS_SQ = N_VARS * N_VARS;
  const static uint8_t N_VARS_SQ = NC * NC;

  const uint8_t get_n_vars() override { return N_VARS; };
  const uint8_t get_n_ops() { return N_OPS; };
  const uint8_t get_n_comps() { return NC; };
  const uint8_t get_z_var() { return Z_VAR; };
  bool vel_linear = false;
  const int acc_active = (vel_linear) ? 0 : 1.;

  engine_nc_dvelocity_cpu() { engine_name = "Multiphase " + std::to_string(NC) + "- momentum in the wellbore is " + ("%s", vel_linear ? "Darcy" : "ms-well momemntum") + "-component isothermal flow decoupled velocity CPU engine"; };

  //inline index_t get_z_idx(char c, index_t block_idx) { return block_idx * N_VARS + c + 1; };

  int init(conn_mesh *mesh_, std::vector <ms_well*> &well_list_,
    std::vector <operator_set_gradient_evaluator_iface*> & acc_flux_op_set_list_,
    sim_params *params_, timer_node* timer_);

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assemble_kernel = 0);
  int init_jacobian_structure(csr_matrix_base *jacobian);
  int init_jacobian_structure_acc(csr_matrix_base *jacobian);
  int additional_nnz_acc(csr_matrix_base *jacobian);
  int run_single_newton_iteration(value_t deltat);
  int solve_linear_equation();
  int post_newtonloop(value_t deltat, value_t time);
  int print_timestep(value_t time, value_t deltat);
  double calc_newton_residual_L2();
  double calc_well_residual_L2();
  double calc_velocity_residual();
  void calc_well_res_velocity_residual(value_t & res_wel_vel, value_t & res_res_vel);
  double velocity_residual_last_dt;
  void apply_composition_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
  void apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
  void average_operator(std::vector<value_t> &av_op);
  void darcy_velocity(index_t j, value_t velocity, value_t p_diff, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assemble_kernel = 0);
  void momentum_mswell(index_t iw, index_t j, value_t velocity, value_t p_diff, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assemble_kernel = 0);
  void momentum_mswell_withacceleration(index_t iw, index_t j, value_t velocity, value_t p_diff, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS, int assemble_kernel = 0);
  void calculateFanningFactor(value_t velocity, value_t roughness, value_t block_up, value_t &f);
  //void calculateFrictionLosses(value_t velocity, value_t roughness, value_t block_up, value_t & dp_f, value_t f);
  void calculateFrictionLosses(value_t velocity, value_t roughness, value_t block_up, value_t & dp_f, value_t f, value_t l, value_t D);
  void calculatePressureLosses(value_t velocity, value_t roughness, value_t block_up, index_t j, value_t &dp_f, value_t &dp_h, value_t f, value_t l, value_t D);
  void calculateHydrauliclosses(index_t j, value_t &dp_h);
  double calculateAccelerationlosses(index_t iw, value_t velocity, value_t block_up, value_t & dp_a, value_t & m_tot, value_t Area);
  void init_WellDynamicProperties(std::vector<value_t> &X_init);
  void calculate_mass_flux(value_t block_up, value_t i_r, value_t wi, value_t & m_flux);
  int return_well_number(value_t bm, value_t bp);
  int connection_perforated(index_t iw, value_t bm, value_t bp);
  int reservoirIdxperf(index_t iw, value_t b);
  int cross_flow(index_t iw, std::vector<value_t> &X);

  void apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
  int apply_newton_update(value_t dt);
  void apply_velocity_update(value_t dt);
  std::vector<value_t> X_cc;            //array of cell-center unknowns  
  std::vector <value_t> ctrMeqI;        // array including the number of time blockm is equal to I 
  std::vector<value_t> jac_idx_array;   // jac diagonal array with the size on NC
  std::vector <value_t> ctrConblock;   // array including the number of block we need to fill for the connection [default = 2, one sided perf (only one perf)= 3, 2 sided perf(only one perf per each side) = 4]
  std::vector <index_t> segment_perforated;
  std::vector<value_t> Der_mass_perf;
  std::vector<value_t> Der_mass_upblock;
  value_t res_wel_vel;
  value_t res_res_vel;

public:
};
#endif



