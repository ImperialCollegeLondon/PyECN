# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import sys
from pyecn.read_LUT import read_LUT
from tomli import load as load_toml

root_dir = Path("pyecn/")
try:
    input_file = sys.argv[1]
except IndexError:
    print("Enter config file name:")
    input_file = input()
with open(root_dir / input_file, mode="rb") as f:
    inputs = load_toml(f)

model_params = inputs["model"]
runtime_opts = inputs["runtime_options"]
op_conds = inputs["operating_conditions"]
cell_params = inputs["cell"]
cell_geometry = cell_params["geometry"]
cell_electrical = cell_params["electrical"]
cell_physical = cell_params["physical"]
cell_thermal = cell_params["thermal"]
cell_LUT_opts = cell_params["LUTs"]
if cell_params["Form_factor"] != "Pouch":
    can_params = inputs["can"]
    can_geometry = can_params["geometry"]
    can_physical = can_params["physical"]
    can_thermal = can_params["thermal"]
tab_params = inputs["tab"]
tab_geometry = tab_params["geometry"]
tab_physical = tab_params["physical"]
tab_thermal = tab_params["thermal"]
membrane_params = inputs["membrane"]
membrane_geometry = membrane_params["geometry"]
membrane_thermal = membrane_params["thermal"]
module_params = inputs["module"]
if cell_params["Form_factor"] == "Pouch":
    tab_weld_params = inputs["tab_weld"]
elif cell_params["Form_factor"] == "Cylindrical":
    ribbon_params = inputs["ribbon"]
solver_params = inputs["solver"]
postprocessing = inputs["postprocessing"]

General_conditions = model_params["General_conditions"]
status_FormFactor = cell_params["Form_factor"]
status_Eparam = cell_params["Eparam"]
status_Electrical_type = model_params["Electrical_type"]
status_ECN_method = model_params["ECN_method"]
status_Model = model_params["Model"]
status_EandT_coupling = model_params["E-T_coupling"]
status_Heat_gen = model_params["Heat_gen"]
status_Entropy = model_params["Entropy"]
nRC = model_params["nRC"]
ny = model_params["ny"]
nstack = model_params["nstack"]
nstack_real = cell_geometry["nstack_real"]
if status_FormFactor == "Prismatic":
    nx_cylindrical = model_params["nx_cylindrical"]
    nx_pouch = model_params["nx_pouch"]
    nx = nx_cylindrical+2*nx_pouch
else:
    nx = model_params["nx"]

if cell_params["Form_factor"] == "Cylindrical":
    status_Cap_heatgen = op_conds["Cap_heatgen"]
status_CC = model_params["CC"]
status_R0minusRcc = model_params["R0_minus_Rcc"]
status_LUTinterp = cell_LUT_opts["interp"]
status_scipy_interpkind = cell_LUT_opts["interp_order"]
status_linsolver_E = solver_params["Linsolver_E"]
status_linsolver_T = solver_params["Linsolver_T"]
status_Thermal_solver = solver_params["Thermal_solver"]
status_PostProcessor = postprocessing["PostProcessor"]
status_PostProcessor_Module = postprocessing["PostProcessor_module"]
status_visualization_method = postprocessing["Visualisation_method"]
status_mayavi_show_cell_num = postprocessing["mayavi"]["Show_cell_num"]
if cell_params["Form_factor"] == 'Cylindrical':
    postprocessing["Plot_CoreSep_T"] = 'No'
    status_plot_CoreSep_T = postprocessing["Plot_CoreSep_T"]
    postprocessing["Plot_any_nx"] = 'Yes'
    status_plot_any_nx = postprocessing["Plot_any_nx"]
    postprocessing["nx_plot"] = 100
    nx_plot = postprocessing["nx_plot"]
General_conditions_module = module_params["General_conditions_module"]
status_Module = model_params["Module"]
status_Module_4T = model_params["Module_4T"]
status_linsolver_E_Module = solver_params["Linsolver_E_module"]
status_ageing = model_params["Ageing"]
Q_charact_separate = runtime_opts["Q_charact_separate"]
status_BoL_ageing = model_params["BoL_ageing"]
status_PlotLUT = postprocessing["PlotLUT"]
status_PlotNode = postprocessing["PlotNode"]
status_fig1to9 = postprocessing["Fig1to9"]
status_PopFig_or_SaveGIF_instant = postprocessing["PopFig_or_SaveGIF_instant"]
status_PopFig_or_SaveGIF_replay = postprocessing["PopFig_or_SaveGIF_replay"]
status_plot_type_preprocess = postprocessing["Plot_type_preprocess"]
status_plot_type_postprocess = postprocessing["Plot_type_postprocess"]
status_Echeck = runtime_opts["Echeck"]
status_TemBC_VectorCN_check = runtime_opts["TemBC_vectorCN_check"]
status_TemBC_smoothening = runtime_opts["TemBC_smoothing"]
status_CheckBC_4T = runtime_opts["CheckBC_4T"]
status_get_unecessary_info = runtime_opts["Get_unnecessary_info"]
status_Count = runtime_opts["Count"]


C_rate = op_conds["C_rate"]
dt = op_conds["dt"]
soc_initial = op_conds["SoC_initial"]
status_discharge = op_conds["Current_direction"]
status_IVmode = op_conds["IV_mode"]
V_highlimit_single = op_conds["V_highlimit_single"]
V_lowlimit_single = op_conds["V_lowlimit_single"]
T_cooling = op_conds["T_cooling"]
T_initial = op_conds["T_initial"]
T_fixed = op_conds["T_fixed"]
Rcc = op_conds["Rcc"]
Rcap = op_conds["Rcap"]
status_discharge_Module = module_params["Current_direction_module"]
status_IVmode_Module = module_params["IV_mode_module"]
soc_initial_Module = module_params["SoC_initial_module"]
C_rate_Module = module_params["C_rate_module"]
if op_conds["I_ext_fpath"] == "":
    nt = int(3600/op_conds["C_rate"]/op_conds["dt"])
else:  # use time-varying current
    Table_I_ext = np.loadtxt(op_conds["I_ext_fpath"])
    nt = np.size(Table_I_ext)-1
if status_FormFactor != "Pouch":
    status_ThermalBC_Core = op_conds["Thermal_BC_core"]
    n_Air = op_conds["n_air"]
    status_AllConv_h = op_conds["AllConv_h"]
if status_Eparam=='Cylindrical_Cell2':     #This is done just for generating the results for Soft_X paper for CylindricalCell2
    nt=3334 

smoothening_stiffness = runtime_opts["Smoothing_stiffness"]
status_levels = postprocessing["Temp_levels"]
min_temp_limit = postprocessing["Temp_min"]
max_temp_limit = postprocessing["Temp_max"]
if op_conds["Current_direction"] == 1:
    postprocessing["GIFdownsample_num"] = 1500/op_conds["C_rate"]*op_conds["SoC_initial"]
else:
    postprocessing["GIFdownsample_num"] = 1500/op_conds["C_rate"]*(1-op_conds["SoC_initial"])
status_GIFdownsample_num = nt

LUT_dir = Path("pyecn/Input_LUTs")
LUT_Address1 = cell_params["Eparam"]
FileAddress_SOC_read = LUT_dir / LUT_Address1 / 'SoC.csv'
FileAddress_T_read = LUT_dir / LUT_Address1 / 'T.csv'
FileAddress_OCV_read = LUT_dir / LUT_Address1 / 'OCV-SoC.csv'
FileAddress_dVdT_read= LUT_dir / LUT_Address1 / 'dVdT-SoC.csv'
#FileAddress_dVdT_SOC_read= LUT_dir / LUT_Address1 / 'entropy'/ 'SoC.csv'

T_LUT = read_LUT(FileAddress_T_read).reshape(-1,1)
Temperature_Value_LUTs = np.size(T_LUT)
Temperature_names = []
R0_names = []
Ri_names = []
Ci_names = []
OCV_names = []
for i0_temp in range(int(Temperature_Value_LUTs)):
    T = T_LUT[i0_temp]
    m=int(T)
    string_temp = f'T{m}.csv'
    Temperature_names.append(string_temp)
    R0_names.append(LUT_dir / LUT_Address1 / f'R0-SoC-{string_temp}')
    Ri_names.append(LUT_dir / LUT_Address1 / f'Ri-SoC-{string_temp}')
    Ci_names.append(LUT_dir / LUT_Address1 / f'Ci-SoC-{string_temp}')
    OCV_names.append(LUT_dir / LUT_Address1 / f'OCV-SoC-{string_temp}')

Capacity_rated0 = cell_electrical["Capacity_rated0"]
delta_An_real = cell_geometry["delta_an_real"]
delta_Ca_real = cell_geometry["delta_ca_real"]
delta_Sep_real = cell_geometry["delta_sep_real"]
delta_Al_real = cell_geometry["delta_Al_real"]
delta_Cu_real = cell_geometry["delta_Cu_real"]
delta_El_real = cell_geometry["delta_el_real"]
Lx_electrodes_real = cell_geometry["Lx_electrodes_real"]
Ly_electrodes_real = cell_geometry["Ly_electrodes_real"]
Lx_cell = cell_geometry["Lx_cell"]
Ly_cell = cell_geometry["Ly_cell"]
SpecSheet_Nominal_Voltage = cell_electrical["Nominal_voltage"]
SpecSheet_Energy = cell_electrical["Energy"]
SpecSheet_Casing_rou_Polyamide = cell_physical["Casing_density_polyamide"]
SpecSheet_Casing_rou_Polypropylene = cell_physical["Casing_density_polypropylene"]
SpecSheet_Casing_rou_Al = cell_physical["Casing_density_Al"]
SpecSheet_Casing_delta_Polyamide = cell_geometry["Casing_delta_polyamide"]
SpecSheet_Casing_delta_Polypropylene = cell_geometry["Casing_delta_polypropylene"]
SpecSheet_Casing_delta_Al = cell_geometry["Casing_delta_Al"]

Conductivity_Al = cell_electrical["Conductivity_Al"]
Conductivity_Cu = cell_electrical["Conductivity_Cu"]
Lamda_An = cell_thermal["Conductivity_an"]
Lamda_Ca = cell_thermal["Conductivity_ca"]
Lamda_Sep = cell_thermal["Conductivity_sep"]
Lamda_Al = cell_thermal["Conductivity_Al"]
Lamda_Cu = cell_thermal["Conductivity_Cu"]
rou_An = cell_physical["Density_an"]
rou_Ca = cell_physical["Density_ca"]
rou_Sep = cell_physical["Density_sep"]
rou_Al = cell_physical["Density_Al"]
rou_Cu = cell_physical["Density_Cu"]
c_An = cell_thermal["Specific_heat_capacity_an"]
c_Ca = cell_thermal["Specific_heat_capacity_ca"]
c_Sep = cell_thermal["Specific_heat_capacity_sep"]
c_Al = cell_thermal["Specific_heat_capacity_Al"]
c_Cu = cell_thermal["Specific_heat_capacity_Cu"]

if cell_params["Form_factor"] != "Pouch":
    delta_Can_real = can_geometry["delta_real"]
    delta_Can_surface = can_geometry["delta_surface"]
    delta_Can_Base = can_geometry["delta_base"]
    c_Can_real = can_thermal["Specific_heat_capacity_real"]
    c_Can_surface = can_thermal["Specific_heat_capacity_surface"]
    c_Can_Base = can_thermal["Specific_heat_capacity_base"]
    Lamda_Can = can_thermal["Conductivity"]
    rou_Can = can_physical["Density"]
    Lamda_Sep_CanBase = can_thermal["Conductivity_sep_base"]

A_tab = tab_geometry["A"]
L_tab = tab_geometry["L"]
Lamda_tab = tab_thermal["Conductivity"]
rou_tab = tab_physical["Density"]
c_tab = tab_thermal["Specific_heat_capacity"]

if cell_params["Form_factor"] != "Pouch":
    delta_Membrane = membrane_geometry["delta"]
    Lamda_Membrane = membrane_thermal["Conductivity"]
    delta_core_real = cell_geometry["delta_core_real"]
    LG_Can = inputs["form-factor-specific"]["LG_can"]
    LG_Jellyroll = inputs["form-factor-specific"]["LG_jellyroll"]
if cell_params["Form_factor"] == "Prismatic":
    Lx_pouch = inputs["form-factor-specific"]["Lx_pouch"]

status_TabSurface_Scheme = op_conds["Tab_surface_scheme"]
status_Tab_ThermalPath = op_conds["Tab_thermal_path"]
if cell_params["Form_factor"] != "Pouch":
    status_ThermalPatition_Can = model_params["Thermal_partition_can"]
    status_CanSepFill_Membrane = op_conds["CanSepFill_membrane"]
    status_Can_Scheme = op_conds["Can_scheme"]

if status_FormFactor == "Prismatic":
    LUT_Scale_Factor_Rs_area = module_params["LUT_Scale_Factor_Rs_area"]
    LUT_Scale_Factor_Cs_area = 1/LUT_Scale_Factor_Rs_area

if status_FormFactor == "Cylindrical":
    status_thermal_tab = model_params["Thermal_tab"]          #'Tabless_virtual/Default'; positive/negative tab between jellyroll and can; 'Tabless_virtual': At positive tab, Al and Can are thermally connected. At negative tab, Cu and Can are thermally connected; 'Default': at positive and negative tab, tab and can are thermally insulated
    status_electrical_tab = model_params["Electrical_tab"]     #'Tabless_virtual/Default'

if status_Eparam=='Cylindrical_Cell2':
    LUT_Scale_Factor_Rs_area= 0.116840/0.632685                                                               # This is done when parameters needs to be scaled by scaling the electrode area of some real cell to reflect the behaviour of unknown desired cell
    LUT_Scale_Factor_Cs_area=1/LUT_Scale_Factor_Rs_area

if cell_params["Form_factor"] == "Pouch":
    h_pouch = op_conds["h_cooling"]

model_params["Scalefactor_z"] = (2*cell_geometry["nstack_real"]-1)/(2*model_params["nstack"]-1) #value in [1,+∞), resolution reduction in z (radial or thickness) direction, for E and T model; scalefactor_z=1 means full-scale simulation, no resolution reduction
scalefactor_z = model_params["Scalefactor_z"]

status_climit_vector = np.linspace(min_temp_limit, max_temp_limit, status_levels)

if runtime_opts["TemBC_vectorCN_check"] == 'Yes':
    TemCheck_floor = (np.append(T_cooling,T_initial)).min()
    TemCheck_floor = TemCheck_floor-abs(TemCheck_floor)*0.01      #why -abs(...) and 0.01: Example. In the case of T_cooling and T_initial both e.g. 20°C. If there is no abs(...), only exact 20.00°C (VectorCN all=20.00°C) can satisfy implicit stability requirement, which is not right cos there are heat gen/entropy and system error that can cause fluctuate thus outside 20°C
    TemCheck_ceil = (np.append(T_cooling,T_initial)).max()
    TemCheck_ceil = TemCheck_ceil+abs(TemCheck_ceil)*0.01

status_Cells_name = ['cell_1']
if postprocessing["PostProcessor"] == 'Yes':
    postprocessing["PostProcess_cell_id"] = 'cell_1'                        #Postprocess cell_1 by default
    status_PostProcess_cell_id = postprocessing["PostProcess_cell_id"]

print('###############################################################################')
print('ECN solving method for Electrical model are of two type: General/Neo')
print('General: calculating elementary current and voltage potential of every nodes including El nodes')
print('Neo: calculating elementary current and voltage potential of current collector nodes only')
print('###############################################################################')
print('Default General conditions for Electrical & Thermal model are as below: ')
print('Cell considered for the model is (status_Eparam): ', cell_params["Eparam"])

if cell_params["Form_factor"] != 'Pouch':
    print('No. of angular nodes are (nx): ', nx)
    print('No. of axial nodes are (ny): ', ny)
    print('No. of laps are (nstack): ', nstack)
else:
    print('No. of nodes along x-direction (Longest side-Length) (nx): ', nx)
    print('No. of nodes along y-direction (width) (ny): ', ny)
    print('No. of stacks/layers (nstack): ', nstack)

if cell_params["Form_factor"] == 'Prismatic' or cell_params["Form_factor"] == 'Cylindrical':
    print('Thermal model core BC (status_ThermalBC_Core): ', status_ThermalBC_Core)

status_BC_Module = module_params["BC_module"]
status_BC_Module_4T = module_params["BC_module_4T"]
status_Cells_num_Module = module_params["Cells_num_module"]
status_Parts_num_Module = module_params["Parts_num_module"]
status_Parts_name = []
if cell_params["Form_factor"] != "Prismatic":
    status_BC_ribbon = module_params["BC_ribbon"]
V_ext_Module = module_params["V_ext_module"]
status_Modules_name = module_params["Modules_name"]
status_Modules_4T_name = module_params["Modules_4T_name"]
status_PostProcess_Module_id = postprocessing["PostProcess_module_id"]
if model_params["Module"] == "Yes" or model_params["Module_4T"] == "Yes":
    if General_conditions_module == 'Default':
        if cell_params["Form_factor"] == 'Pouch':
            status_BC_Module = 'Pouch_weld_tab'
            status_Cells_num_Module = 1

        elif cell_params["Form_factor"] == 'Prismatic':
            status_BC_Module = 'Prismatic_Cell1'
            status_Cells_num_Module = 2

        elif cell_params["Form_factor"] == 'Cylindrical':
            status_BC_Module = 'Ribbon_cooling'
            status_Cells_num_Module = 2

        if model_params["Module_4T"] == 'Yes':
            status_BC_Module_4T = status_BC_Module

            if module_params["BC_module_4T"] == 'Pouch_weld_tab':
                status_Parts_num_Module = 5

            elif module_params["BC_module_4T"] == 'Prismatic_Cell1':
                status_Parts_num_Module = 3

            elif module_params["BC_module_4T"] == 'Ribbon_cooling':
                status_Parts_num_Module = status_Cells_num_Module+1
                status_BC_ribbon = 'Single_Tab_Cooling'

            else:
                status_Parts_num_Module = status_Cells_num_Module

        else:
            status_Parts_num_Module = 1


    if model_params["Module"] == 'Yes':
        if module_params["IV_mode_module"] == 0:

            Capacity_rated0_Ah_SingleRoll = cell_electrical["Capacity_rated0"]/3600  # This capacity value (As) is coming from input table(Excel)
            Capacity_rated0_Ah = Capacity_rated0_Ah_SingleRoll*status_Cells_num_Module  # Ah capacity of entire Cell
            Capacity_rated0_Module = Capacity_rated0_Ah_SingleRoll*3600*status_Cells_num_Module

            I_ext_Module = int(status_discharge_Module * C_rate_Module * Capacity_rated0_Module/3600)

            Capacity0_Module = Capacity_rated0_Module * soc_initial_Module

        elif module_params["IV_mode_module"] == 1:
            V_ext_Module = 4.2

        LUT_Scale_Factor_Rs = status_Cells_num_Module
        LUT_Scale_Factor_Cs = 1/status_Cells_num_Module


    status_Modules_name = ['module_1']
    status_Modules_4T_name = ['module_1_4T']

    if postprocessing["PostProcessor_module"] == 'Yes':
        status_PostProcess_Module_id = 'module_1'   # Postprocess module_1 by default

    if model_params["Module"] == 'Yes':
        status_Cells_name = []                      # simple processing------Defining cells/jellyrolls names
        for i0_temp in range(status_Cells_num_Module):
            string_temp = f'cell_{i0_temp+1}'
            status_Cells_name.append(string_temp)
                                                    # simple processing------cells/jellyrolls in parallel in Module/cell

        Cells_Ppos_link_Module = np.zeros([ status_Cells_num_Module,status_Cells_num_Module ])
        Cells_Ppos_link_Module[0,1:] = 1
        Cells_Pneg_link_Module = Cells_Ppos_link_Module
        Cells_S_link_Module = np.zeros([ status_Cells_num_Module,status_Cells_num_Module ])

    if model_params["Module_4T"] == 'Yes':
        status_Parts_name = []
        for i0_temp in range(status_Parts_num_Module):
            string_temp = f'part_{i0_temp+1}'
            status_Parts_name.append(string_temp)


    if cell_params["Form_factor"] != 'Prismatic':
        if status_visualization_method == 'mayavi':
            status_Cells_Xinterval_Module = 0.05    # spatial locations in a module for cells
            status_Cells_Yinterval_Module = 0
            status_Cells_Zinterval_Module = 0
            status_Cells_XYZ_Module = np.nan * np.ones([status_Cells_num_Module,3])
            status_Cells_XYZ_Module[:,0] = status_Cells_Xinterval_Module * np.arange(status_Cells_num_Module)
            status_Cells_XYZ_Module[:,1] = status_Cells_Yinterval_Module * np.arange(status_Cells_num_Module)
            status_Cells_XYZ_Module[:,2] = status_Cells_Zinterval_Module * np.arange(status_Cells_num_Module)


    print('###############################################################################')
    if General_conditions_module=='Default':
        print('Default General conditions for Electrical & Thermal model are as below:')

    if model_params["Module"] == 'Yes' and model_params["Module_4T"] == 'Yes':
        print('BCs for Electrical model is corresponding to (status_BC_Module): ', status_BC_Module)
        print('BCs for Thermal model is corresponding to (status_BC_Module_4T): ', status_BC_Module_4T)
        print('Type of linear solver for electrical model is (status_linsolver_E_Module): ', status_linsolver_E_Module)
        print('No. of Jellyroll/s in parallel inside the cell are (status_Cells_num_Module): ', status_Cells_num_Module)
        print('No. of thermal parts considered in the model are (status_Parts_num_Module): ', status_Parts_num_Module)
    print('###############################################################################')


    if model_params["Module_4T"] == "Yes":
        status_Cells_Xinterval_Module = postprocessing["mayavi"]["Cells_Xinterval_module"]
        status_Cells_Yinterval_Module = postprocessing["mayavi"]["Cells_Yinterval_module"]
        status_Cells_Zinterval_Module = postprocessing["mayavi"]["Cells_Zinterval_module"]
        if cell_params["Form_factor"] == "Prismatic":
            if status_BC_Module_4T == 'Prismatic_Cell1':
                nz_Can = inputs["can_prismatic"]["nz"]
                Lamda_Can = Lamda_Al
                rou_Can = rou_Al
                c_Can = c_Al
                status_TabSurface_Scheme_Module_4T = {
                    'cell_1':'Placeholder', 'cell_2':'Placeholder', 'part_3':'BaseCool_Prismatic_Cell1'     #part_3: 'AllConv/BaseCond/SingleSideCond'
                }
                h_Can = inputs["can_prismatic"]["h_can"]
                h_inner_jellyroll = inputs["can_prismatic"]["h_inner_jellyroll"]

                delta_Mylar = inputs["can_prismatic"]["delta_Mylar"]
                Lamda_Mylar = inputs["can_prismatic"]["lambda_Mylar"]

                status_Cells_XYZ_Module = np.nan * np.ones([status_Parts_num_Module,3])
                status_Cells_XYZ_Module[:,0] = status_Cells_Xinterval_Module * np.arange(status_Parts_num_Module)
                status_Cells_XYZ_Module[:,1] = status_Cells_Yinterval_Module * np.arange(status_Parts_num_Module)
                status_Cells_XYZ_Module[:,2] = status_Cells_Zinterval_Module * np.arange(status_Parts_num_Module)


                status_Cells_XYZ_Module[-1,0] = -33e-3*0.5   #same as above
                status_Cells_XYZ_Module[-1,1] = 45e-3*0.5    #same as above
                status_Cells_XYZ_Module[-1,2] = 0.11

            elif status_BC_Module_4T == 'Prismatic_touching':
                delta_Mylar = 152e-6                             #152e-6. If no Mylar, put 0 here, no need to change Lamda_Mylar.
                Lamda_Mylar = 0.14                               #0.14. If insulation, put 1e-100 here, no need to change delta_Mylar.

        elif cell_params["Form_factor"] == "Pouch":
            if status_BC_Module_4T == 'Pouch_weld_tab':
                weld_Lamda_pos = tab_weld_params["weld"]["thermal"]["lambda_pos"]
                weld_Lamda_neg = tab_weld_params["weld"]["thermal"]["lambda_neg"]
                weld_rouXc_pos = cell_physical["Density_Al"] * cell_thermal["Specific_heat_capacity_Al"]
                weld_rouXc_neg = cell_physical["Density_Cu"] * cell_thermal["Specific_heat_capacity_Cu"]
                nonweld_tab_Lamda = tab_weld_params["weld"]["nonweld_tab_lambda"]
                weld_n = tab_weld_params["weld"]["weld_n"]
                nonweld_n = tab_weld_params["weld"]["weld_n"] + 1

                #weld geometry - please see the user manual - Fig.6 and Fig.7 - to see which is which
                weld_L1 = tab_weld_params["weld"]["geometry"]["L1"]
                weld_W1 = tab_weld_params["weld"]["geometry"]["W1"]
                weld_W2 = tab_weld_params["weld"]["geometry"]["W2"]
                weld_W3 = tab_weld_params["weld"]["geometry"]["W3"]
                weld_T1_pos = tab_weld_params["weld"]["geometry"]["T1_pos"]
                weld_T1_neg = tab_weld_params["weld"]["geometry"]["T1_neg"]

                tab_Lamda_pos = tab_weld_params["tab"]["thermal"]["lambda_pos"]
                tab_Lamda_neg = tab_weld_params["tab"]["thermal"]["lambda_neg"]
                tab_rouXc_pos = cell_physical["Density_Al"] * cell_thermal["Specific_heat_capacity_Al"]
                tab_rouXc_neg = cell_physical["Density_Al"] * cell_thermal["Specific_heat_capacity_Al"] # not Cu?

                # tab geometry - please see the user manual - Fig.6 and Fig.7 - to see which is which
                tab_L1 = tab_weld_params["tab"]["geometry"]["L1"]
                tab_T1 = tab_weld_params["tab"]["geometry"]["T1"]
                cell_tab_L1 = tab_weld_params["tab"]["geometry"]["cell_tab_L1"]
                tab_nx = tab_weld_params["tab"]["geometry"]["nx"]
                tab_ny = tab_weld_params["weld"]["weld_n"] + tab_weld_params["weld"]["nonweld_n"]
                tab_nz = tab_weld_params["tab"]["geometry"]["nz"]
                #tab_W1 = 55e-3

                status_BC_tab = tab_weld_params["tab"]["thermal"]["BC"]
                T_cooling_tab = tab_weld_params["tab"]["thermal"]["T_cooling"]
                h_tab = tab_weld_params["tab"]["thermal"]["h"]

                # Settings for one-point welding
                welding_tech = tab_weld_params["weld"]["welding_tech"]
                if welding_tech == '1-point welding':
                    weld_W1 = 5e-3  + 7.5e-3 *2           # change the geometry to merge 3 welds into 1
                    weld_W2 = 0                           # change the geometry to merge 3 welds into 1
                    nonweld_tab_Lamda = cell_thermal["Conductivity_Al"]          # the thermal conductivity between two adjecent welds

                overlap_weld_tab = tab_weld_params["weld"]["overlap_weld_tab"]
                offset_y = tab_weld_params["weld"]["geometry"]["offset_y"]
                h_weldline = tab_weld_params["tab"]["thermal"]["h_weldline"]

                status_Cells_XYZ_Module = np.nan * np.ones([status_Parts_num_Module,3])
                status_Cells_XYZ_Module[:,0] = status_Cells_Xinterval_Module * np.arange(status_Parts_num_Module)
                status_Cells_XYZ_Module[:,1] = status_Cells_Yinterval_Module * np.arange(status_Parts_num_Module)
                status_Cells_XYZ_Module[:,2] = status_Cells_Zinterval_Module * np.arange(status_Parts_num_Module)

                status_Cells_XYZ_Module[1,0] = tab_T1  # Position of weld on cell edge in X-direction: distance in z-direction.
                status_Cells_XYZ_Module[1,1] = -overlap_weld_tab  # Position of weld on cell edge in Y-direction: distance in x-direction.
                status_Cells_XYZ_Module[1,2] = offset_y  # Position of weld on cell edge in Z-direction: distance in y-direction.
                status_Cells_XYZ_Module[2,0] = tab_T1  # Position of weld on cell center in X-direction: distance in z-direction.
                status_Cells_XYZ_Module[2,1] = Lx_cell - (weld_L1-overlap_weld_tab)  # Position of weld on cell center in Y-direction: distance in x-direction.
                status_Cells_XYZ_Module[2,2] = offset_y  # Position of weld on cell center in Z-direction: distance in y-direction.
                status_Cells_XYZ_Module[3,0] = tab_T1  # Position of tab on cell edge in X-direction: distance in z-direction.
                status_Cells_XYZ_Module[3,1] = -0.5*tab_L1  # Position of tab on cell edge in Y-direction: distance in x-direction.
                status_Cells_XYZ_Module[3,2] = offset_y  # Position of tab on cell edge in Z-direction: distance in y-direction.
                status_Cells_XYZ_Module[4,0] = tab_T1  # Position of tab on cell center in X-direction: distance in z-direction.
                status_Cells_XYZ_Module[4,1] = Lx_cell  # Position of tab on cell center in Y-direction: distance in x-direction.
                status_Cells_XYZ_Module[4,2] = offset_y  # Position of tab on cell center in Z-direction: distance in y-direction.

        elif cell_params["Form_factor"] == "Cylindrical":
            if status_BC_Module_4T == 'Ribbon_cooling':
                status_BC_ribbon = module_params["BC_ribbon"]
                T_cooling_ribbon = ribbon_params["T_cooling"]
                h_ribbon = ribbon_params["h"]

                S_cooling_size_x = ribbon_params["geometry"]["S_cooling_size_x"]
                S_cooling_size_y = ribbon_params["geometry"]["S_cooling_size_y"]
                S_cooling_h = ribbon_params["geometry"]["S_cooling_h"]
                S_cooling_cell_spacing = ribbon_params["geometry"]["S_cooling_cell_spacing"]
                S_cooling_ribbon_addlength = ribbon_params["geometry"]["S_cooling_ribbon_addlength"]
                Lz_ribbon = ribbon_params["geometry"]["Lz"]
                nz_ribbon = ribbon_params["geometry"]["nz"]
                Lamda_ribbon = cell_thermal["Conductivity_Al"]
                rou_ribbon = cell_physical["Density_Al"]
                c_ribbon = cell_thermal["Specific_heat_capacity_Al"]

                status_Cells_XYZ_Module = np.nan * np.ones([status_Parts_num_Module,3])
                status_Cells_XYZ_Module[:,0] = status_Cells_Xinterval_Module * np.arange(status_Parts_num_Module)
                status_Cells_XYZ_Module[:,1] = status_Cells_Yinterval_Module * np.arange(status_Parts_num_Module)
                status_Cells_XYZ_Module[:,2] = status_Cells_Zinterval_Module * np.arange(status_Parts_num_Module)

                status_Cells_XYZ_Module[-1,0] = 0
                status_Cells_XYZ_Module[-1,1] = 0.02   #0.02 is interval between cell and ribbon, only for visulisation. They are actually thermally connected
                status_Cells_XYZ_Module[-1,2] = 10e-3
else:
    status_Cells_num_Module = 1
    status_Parts_num_Module = 1
status_PID = "No"
