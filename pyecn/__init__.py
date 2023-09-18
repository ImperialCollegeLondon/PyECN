# -*- coding: utf-8 -*-
"""The main module for the PyECN program."""

def run() -> None:
    """Run PyECN."""
    import numpy as np
    from mayavi import mlab
    import scipy.sparse.linalg
    import scipy.sparse
    from scipy.optimize import fsolve
    import time
    import scipy.io as sio

    import pyecn.parse_inputs as ip

    inf=1e10

    #check inputs contradiction
    if ip.status_Model=='T' and (ip.status_Heat_gen=='Yes' or ip.status_Entropy=='Yes'):
        print('PyECN error: no E model then no heat gen or entropy \nhint: Check status_Model, status_Heat_gen and status_Entropy')
        raise Exception('exit')
    if ip.status_Model=='E' and ip.status_Heat_gen=='Yes':
        print('PyECN error: no T model then no heat gen \nhint: Check status_Model, status_Heat_gen and status_Entropy')
        raise Exception('exit')
    if (ip.status_Model != 'EandT') and (ip.status_EandT_coupling != 'NA'):
        print('PyECN error: only E or T model itself should not have any coupling \nhint: Check status_Model and status_EandT_coupling')
        raise Exception('exit')
    if ip.status_FormFactor == 'Cylindrical':
        if (ip.status_ThermalPatition_Can=='No') and (ip.status_Tab_ThermalPath == 'Yes'):
            print('PyECN error: if no Can, there should be no thermal tab either  \nhint: Check status_ThermalPatition_Can and status_Tab_ThermalPath')
            raise Exception('exit')

    if ip.ny==1:
        if ip.nx==1:     #lumped model
            LG=0
        else:         #1-layer lumped model, ny=1, nx>1
            print('PyECN error: nx>1,ny=1 should not happen')
            raise Exception('exit')
    if ip.nx==1 and not(ip.ny==1 and ip.nstack==1):
            print('PyECN error: nx=1 but not lumped is not ready. The code need further test')
            raise Exception('exit')
    if ip.nx==1 and not(ip.ny==1 and ip.nstack==1):
            print('PyECN error: nx=1 but not lumped is not ready. The code need further test')
            raise Exception('exit')
    if ip.status_PopFig_or_SaveGIF_replay=='GIF':
        if ip.nt < ip.status_GIFdownsample_num:
            print('PyECN error: if saving GIF, check nt >= status_GIFdownsample_num is satisfied')
            input('please press any key +Enter to break:')
            raise Exception('exit')
    if 'Table_I_ext' in globals():
        print('Current input is loaded from external file. Now nt is %d'%ip.nt)
        temp1=input('press y to continue, n to break:')
        if temp1 != 'y':
            raise Exception('exit')
    #↓↓↓MMMMMMMMMMMMMM  Module level  MMMMMMMMMMMMMMMMM↓↓↓

    ###   Generate Cells objects   ###
    from pyecn.Battery_Classes.Combined_potential.Core_class.core import Core

    for cell_index, cell_name in enumerate(ip.status_Cells_name):
        if ip.status_Electrical_type == 'ECN_Plain':
            globals()[cell_name] = Core({}, cell_index)

    #ONLY for 'Prismatic_Cell1' in module level
    if ip.status_Module_4T == 'Yes' and ip.status_BC_Module_4T == 'Prismatic_Cell1':  #only for Prismatic_Cell1, generate class instance 'part_i'
        from pyecn.Battery_Classes.Thermal_entitities.prismatic.can_prismatic import Can_Prismatic
        item = 'part_3'
        params_update = {
                         'R_cylinder':cell_1.delta_cell,
                         'status_Heat_gen': 'No',
                         'status_ECN_method': 'NA',
                         'status_Entropy': 'No',
                         'status_TabSurface_Scheme': ip.status_TabSurface_Scheme_Module_4T['part_3']
                        }
        globals()[item] = Can_Prismatic(params_update)
    #ONLY for 'ribbon cooling' in module level
    if ip.status_Module_4T == 'Yes' and ip.status_BC_Module_4T == 'Ribbon_cooling':  #only for ribbon cooling, generate class instance 'part_i'
        from pyecn.Battery_Classes.Thermal_entitities.cylindrical.ribbon import Ribbon
        item = f'part_{ip.status_Parts_num_Module}'
        params_update = {
                         'status_TabSurface_Scheme':ip.status_BC_ribbon,'T_cooling_ribbon':ip.T_cooling_ribbon, 'h_ribbon':ip.h_ribbon,
                         'S_cooling_nx':cell_1.S_cooling_nx, 'S_cooling_ny':cell_1.S_cooling_ny,
                         'S_cooling_cell_spacing':cell_1.S_cooling_cell_spacing,'S_cooling_ribbon_addlength':cell_1.S_cooling_ribbon_addlength,
                         'dx_ribbon':cell_1.delta_x1_Can_4T[cell_1.ind0_Geo_Can_node34to37_4T[0]],
                         'dy_ribbon':cell_1.delta_y1_Can_4T[cell_1.ind0_Geo_Can_node34to37_4T[0]],
                         'Lz_ribbon':ip.Lz_ribbon,
                         'status_Heat_gen': 'No',
                         'status_ECN_method': 'NA',
                         'status_Entropy': 'No'
                        }
        globals()[item] = Ribbon(params_update)

        print('S-shaped cooling thermal BC is applied.\n')
        print('Input rectangle in x,y dimension is: %.2fm x %.2fm.\nIn the current model, rectangle x,y dimension is: %.2fm x %.2fm;\nnumber of included nodes in x,y dimension is: %d x %d\n '%(cell_1.S_cooling_size_x,cell_1.S_cooling_size_y,cell_1.S_cooling_size_x_model,cell_1.S_cooling_size_y_model,cell_1.S_cooling_nx,cell_1.S_cooling_ny))
        temp1=input('press y to continue, n to break:')
        if temp1 != 'y':
            raise Exception('exit')
    #ONLY for pouch_weld_tab in module level
    if ip.status_Module_4T == 'Yes' and ip.status_BC_Module_4T == 'Pouch_weld_tab':  #only for ribbon cooling, generate class instance 'part_i'
        from pyecn.Battery_Classes.Thermal_entitities.pouch.weld import Weld

        item = 'part_2'
        params_update = {
                         'pos_neg':'pos',
                         'weld_T1':ip.weld_T1_pos,
                         'status_Heat_gen': 'No',
                         'status_ECN_method': 'NA',
                         'status_Entropy': 'No'
                        }
        globals()[item] = Weld(params_update)

        item = 'part_3'
        params_update = {
                         'pos_neg':'neg',
                         'weld_T1':ip.weld_T1_neg,
                         'status_Heat_gen': 'No',
                         'status_ECN_method': 'NA',
                         'status_Entropy': 'No'
                        }
        globals()[item] = Weld(params_update)

        from pyecn.Battery_Classes.Thermal_entitities.pouch.tab import Tab

        item = 'part_4'
        params_update = {
                         'status_BC_tab':ip.status_BC_tab,'T_cooling_tab':ip.T_cooling_tab,
                         'pos_neg':'pos',
                         'status_Heat_gen': 'No',
                         'status_ECN_method': 'NA',
                         'status_Entropy': 'No'
                        }
        globals()[item] = Tab(params_update)

        item = 'part_5'
        params_update = {
                         'status_BC_tab':ip.status_BC_tab,'T_cooling_tab':ip.T_cooling_tab,
                         'pos_neg':'neg',
                         'status_Heat_gen': 'No',
                         'status_ECN_method': 'NA',
                         'status_Entropy': 'No'
                        }
        globals()[item] = Tab(params_update)

    ###   Generate Module(Electrical) objects   ###
    if ip.status_Module == 'Yes':

        params_C2M = {}                                                 #params to be passed from cell objects (cell_1, cell_2,...) into module class. e.g. params_C2M: { 'cell_1': {'ntotal':12,'nCC':8}
        for item in ip.status_Cells_name:                               #                                                                                                 'cell_2': {'ntotal':12,'nCC':8} }  when nx,ny,nstack=2,2,1 in 2 cells: cell_1 and cell_2
            params_C2M[item] = {}
            cell_i = globals()[item]
            params_C2M[item]['ntotal'] = cell_i.ntotal         #nested dict
            params_C2M[item]['nCC'] = cell_i.nCC
            params_C2M[item]['nECN'] = cell_i.nECN
            params_C2M[item]['ntab'] = cell_i.ntab
            params_C2M[item]['status_ECN_method'] = cell_i.status_ECN_method
            params_C2M[item]['status_CC'] = cell_i.status_CC
            params_C2M[item]['node_positive_0ind'] = cell_i.node_positive_0ind
            params_C2M[item]['node_negative_0ind'] = cell_i.node_negative_0ind
            params_C2M[item]['nPos'] = cell_i.nPos
            params_C2M[item]['nNeg'] = cell_i.nNeg
            params_C2M[item]['Uini_pos'] = cell_i.Uini[cell_i.node_positive_0ind[0]]  #for Module level use: initialization of U_pndiff_plot_Module
            params_C2M[item]['Uini_neg'] = cell_i.Uini[cell_i.node_negative_0ind[0]]
            params_C2M[item]['module_name'] = ip.status_Modules_name

        from pyecn.Battery_Classes.Module_level.module import Module
        for item in ip.status_Modules_name:                    #Tailor the 'status' for specific module if not default settings (in inputs.py)
            params_update = {
                            'Cells_name':ip.status_Cells_name,
                            'Cells_attr':params_C2M
                            }
            globals()[item] = Module(params_update)

        for i0 in np.arange(np.size(ip.status_Cells_name)):  #Add attr 'module_ind0' into the cells objects
            globals()[ip.status_Cells_name[i0]].module_ind0 = 0
    ###   Generate Module(Thermal) objects   ###
    if ip.status_Module_4T == 'Yes':

        params_C2M = {}
        for i0 in np.arange(ip.status_Parts_num_Module):
            item_temp1 = ip.status_Parts_name[i0]
            item_temp2 = ip.status_Cells_name[i0] if i0 <= ip.status_Cells_num_Module-1 else ip.status_Parts_name[i0]
            params_C2M[item_temp1] = {}
            part_i = globals()[item_temp2]
            params_C2M[item_temp1]['n_4T_ALL'] = part_i.n_4T_ALL         #nested dict
            params_C2M[item_temp1]['rou_c_V_weights'] = part_i.rou_c_V_weights         #nested dict

        from pyecn.Battery_Classes.Module_level.module_4T import Module_4T
        for item in ip.status_Modules_4T_name:                                       #Tailor the 'status' for specific module if not default settings (in inputs.py)
            params_update = {
                            'Parts_name':ip.status_Parts_name,
                            'Parts_attr':params_C2M
                            }
            globals()[item] = Module_4T(params_update)

    #    params_C2M = {}
    #    for i0 in np.arange(ip.status_Parts_num_Module):
    #        item_temp1 = ip.status_Parts_name[i0]
    #        item_temp2 = ip.status_Cells_name[i0] if i0 <= ip.status_Cells_num_Module-1 else ip.status_Parts_name[i0]
    #        params_C2M[item_temp1] = {}
    #        part_i = globals()[item_temp2]
    #        for j0 in np.arange(len(module_1_4T.interface_dict[item_temp1])):
    #            string_temp = module_1_4T.interface_dict[item_temp1][j0]
    #            params_C2M[item_temp1][string_temp] = getattr(part_i,string_temp)
        params_C2M = {}
        for i0 in np.arange(module_1_4T.n_interface):    #nested dict in 3 layers
            item_temp1 = module_1_4T.status_Interfaces_name[i0]   #item_temp1: loop over 'interface_1', 'interface_2'...
            params_C2M[item_temp1] = {}
            for j0 in np.arange(2):
                if j0==0:
                    item_temp2_1 = module_1_4T.interface_string[item_temp1]['SideA_part_id']  #loop over 'part_1','part_2'
                if j0==1:
                    item_temp2_1 = module_1_4T.interface_string[item_temp1]['SideB_part_id']  #loop over 'part_1','part_2'
                item_temp2_2 = module_1_4T.Parts2Cells_name[item_temp2_1]
                part_i = globals()[item_temp2_2]
                params_C2M[item_temp1][item_temp2_1] = {}
                for k0 in np.arange(len(module_1_4T.interface_dict[item_temp1][item_temp2_1])):
                    string_temp = module_1_4T.interface_dict[item_temp1][item_temp2_1][k0]
                    params_C2M[item_temp1][item_temp2_1][string_temp] = getattr(part_i,string_temp)

        module_1_4T.fun_pre_Thermal_Module(params_C2M)
        module_1_4T.Tini_4T_Module = module_1_4T.fun_IC_4T_Module(params_C2M)

        params_C2M = {}
        for i0 in np.arange(ip.status_Parts_num_Module):
            item_temp1 = ip.status_Parts_name[i0]
            item_temp2 = module_1_4T.Parts2Cells_name[item_temp1]
            part_i = globals()[item_temp2]
            params_C2M[item_temp1] = part_i.MatrixCN

        module_1_4T.MatrixM_4T = module_1_4T.fun_matrixM_4T(params_C2M)             #pass the list containing instances into fun_matrixM in module.py
        if ip.status_linsolver_T=='BandSparse':
            [module_1_4T.length_MatrixM_4T, module_1_4T.ind0_l, module_1_4T.ind0_u, module_1_4T.ind0_r_expand, module_1_4T.ind0_c_expand]=module_1_4T.fun_band_matrix_precompute(module_1_4T.MatrixM_4T)
        module_1_4T.VectorM_preTp = module_1_4T.fun_vectorM_preTp()
        params_C2M = {}
        for i0 in np.arange(module_1_4T.n_interface):    #nested dict in 3 layers
            item_temp1 = module_1_4T.status_Interfaces_name[i0]   #item_temp1: loop over 'interface_1', 'interface_2'...
            params_C2M[item_temp1] = {}
            for j0 in np.arange(2):
                if j0==0:
                    item_temp2_1 = module_1_4T.interface_string[item_temp1]['SideA_part_id']  #loop over 'part_1','part_2'
                if j0==1:
                    item_temp2_1 = module_1_4T.interface_string[item_temp1]['SideB_part_id']  #loop over 'part_1','part_2'
                item_temp2_2 = module_1_4T.Parts2Cells_name[item_temp2_1]
                part_i = globals()[item_temp2_2]
                params_C2M[item_temp1][item_temp2_1] = {}
                for k0 in np.arange(len(module_1_4T.interface_dict_forBC[item_temp1][item_temp2_1])):
                    string_temp = module_1_4T.interface_dict_forBC[item_temp1][item_temp2_1][k0]
                    params_C2M[item_temp1][item_temp2_1][string_temp] = getattr(part_i,string_temp)
        module_1_4T.interface_attr_forBC = params_C2M                  #module_1_4T.interface_attr_forBC is stored in module_1_4T for fun_BC_4T_Module in the loop later
    #↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑

    t_begin=time.time()
    print('preprocessor done, running solving...\n')
    print('----------------------------------------')
    #=============================================================================================loop start for cycles
    step=1
    cycle=1
    while 1:   #cycle: 1,2...ncycle   loop for cycle       If status_ageing='No' i.e. BoL model, cycle=1 by default

        #-----------------------------------------------------------------------------------------loop start for single cycle (solver)
        while 1:
            t_1=time.time() if ip.status_Count=='Yes' else []
            print('running step=',step,'cycle=',cycle,'...');          ts_step_all=time.time() if ip.status_Count=='Yes' else []
            ############################################################################solve Electrical model
            if ip.status_Model=='EandT'or ip.status_Model=='E':
                #-------------------------------ECN model
                if ip.status_Electrical_type in ['ECN_Plain']:
                    for item in ip.status_Cells_name:  #i: cell_1,cell_2...   loop for all cells
                        cell_i = globals()[item]
                        if step==1:       #first time, get initial voltage potential for step=0
                            step=0                                                     #temporately change step to step=0 and will be changed back to step=1 later
                            cell_i.T1_4T_ALL=cell_i.Tini_4T_ALL
                            cell_i.T_ele=cell_i.T1_4T_ALL[ cell_i.List_ele2node_4T,0 ]                               #T of each element; in the form of 1,2...nECN
                            cell_i.SoC_ele=cell_i.coulomb_ele/cell_i.coulomb_ele_rated;                     #SoC of each element; in the form of 1,2...nECN
                            cell_i.SoC_ele_record[:,step]=cell_i.SoC_ele.reshape(-1)
                            cell_i.SoC_Cell_record[step]=np.sum(cell_i.coulomb_ele)/np.sum(cell_i.coulomb_ele_rated) #record SoC of each element and whole cell
                            cell_i.fun_BC(step)
                            if ip.status_ECN_method=='Neo':
                                cell_i.U3_neo=cell_i.fun_Uini_neo(step)
                                cell_i.fun_update_neo()
                            elif ip.status_ECN_method=='General':
                                cell_i.U3=cell_i.fun_Uini(step)
                                cell_i.U1=cell_i.U3
                            step=1
                        cell_i.T_ele = cell_i.T1_4T_ALL[ cell_i.List_ele2node_4T,0 ]                               #T of each element; in the form of 1,2...nECN
                        cell_i.coulomb_ele = cell_i.coulomb_ele-cell_i.dt*cell_i.U1_neo[cell_i.nCC:(cell_i.nCC+cell_i.nECN)] if ip.status_ECN_method=='Neo' else cell_i.coulomb_ele-cell_i.dt*cell_i.U1[cell_i.ntotal:(cell_i.ntotal+cell_i.nECN)]               #coulomb counting   Q=Σdt*I
                        cell_i.SoC_ele=cell_i.coulomb_ele/cell_i.coulomb_ele_rated                          #SoC of each element; in the form of 1,2...nECN
                        cell_i.SoC_ele_record[:,step]=cell_i.SoC_ele.reshape(-1)
                        cell_i.SoC_Cell_record[step]=np.sum(cell_i.coulomb_ele)/np.sum(cell_i.coulomb_ele_rated) #record SoC of each element and whole cell
                        cell_i.fun_BC(step);                                                                          ts_step_MatrixC=time.time() if ip.status_Count=='Yes' else []
                        if ip.status_ECN_method=='Neo':
                            cell_i.MatrixC_neo=cell_i.fun_matrixC_neo();                                              te_step_MatrixC=time.time() if ip.status_Count=='Yes' else []
                            cell_i.fun_IRi_neo(step)
                            cell_i.VectorI_neo=cell_i.fun_I_neo()
                        elif ip.status_ECN_method=='General':
                            cell_i.MatrixC=cell_i.fun_matrixC();                                                      te_step_MatrixC=time.time() if ip.status_Count=='Yes' else []
                            cell_i.VectorI=cell_i.fun_I()

                        if ip.status_CC=='No':   #if CC resistance is not considered, modify the MatrixC and VectorI for the CC node 0index
                            cell_i.fun_modifyCandI4NoCC_neo()                                                                if ip.status_ECN_method=='Neo' else cell_i.fun_modifyCandI4NoCC()
                        t_2=time.time() if ip.status_Count=='Yes' else []

                        if ip.status_Module == 'No':
                            if ip.status_ECN_method=='Neo':
                                if ip.status_linsolver_E=='Sparse':
                                    cell_i.U3_neo=scipy.sparse.linalg.spsolve(cell_i.MatrixC_neo,cell_i.VectorI_neo)
                                else:
                                    cell_i.U3_neo=np.linalg.solve(cell_i.MatrixC_neo,cell_i.VectorI_neo)
                                cell_i.U3_neo=np.reshape(cell_i.U3_neo,(-1,1))                                                   #U3 from np.linalg is 1d array, needs to be converted to column vector
                                cell_i.Ii_3=cell_i.U3_neo[cell_i.nCC:(cell_i.nCC+cell_i.nECN)]
                                cell_i.fun_update_neo()
                            elif ip.status_ECN_method=='General':
                                if ip.status_linsolver_E=='Sparse':
                                    cell_i.U3=scipy.sparse.linalg.spsolve(cell_i.MatrixC,cell_i.VectorI)
                                else:
                                    cell_i.U3=np.linalg.solve(cell_i.MatrixC,cell_i.VectorI)
                                cell_i.U3=np.reshape(cell_i.U3,(-1,1))          #U3 from np.linalg is 1d array, needs to be converted to column vector
                                cell_i.U1=cell_i.U3

                            if ip.status_ECN_method=='Neo':
                                #--------------change U3_neo form (nCC+nECN+ntab,1) back to U3 form (ntotal+nECN+ntab,1)
                                if ip.status_CC=='Yes':
                                    cell_i.U3=np.nan*np.zeros([cell_i.ntotal+cell_i.nECN+cell_i.ntab+1,1])
                                else:
                                    cell_i.U3=np.nan*np.zeros([cell_i.ntotal+cell_i.nECN+1,1])
                                cell_i.U3[ cell_i.List_Neo2General[cell_i.ind0_CC_neo,0] ]= cell_i.U3_neo[ cell_i.ind0_CC_neo ]
                                cell_i.U3[ cell_i.ntotal: ]= cell_i.U3_neo[ cell_i.nCC: ]
                #-------------------------------SPMe model
                elif ip.status_Electrical_type == 'SPMe':
                    for item in ip.status_Cells_name:  #i: cell_1,cell_2...   loop for all cells
                        cell_i = globals()[item]
                        if step==1:       #first time, get initial voltage potential for step=0
                            step=0                                                     #temporately change step to step=0 and will be changed back to step=1 later
                            cell_i.T1_4T_ALL=cell_i.Tini_4T_ALL
                            cell_i.T_ele=cell_i.T1_4T_ALL[ cell_i.List_ele2node_4T,0 ]                               #T of each element; in the form of 1,2...nECN
                            cell_i.SoC_ele=cell_i.coulomb_ele/cell_i.coulomb_ele_rated;                     #SoC of each element; in the form of 1,2...nECN
                            cell_i.SoC_ele_record[:,step]=cell_i.SoC_ele.reshape(-1); cell_i.SoC_Cell_record[step]=np.sum(cell_i.coulomb_ele)/np.sum(cell_i.coulomb_ele_rated) #record SoC of each element and whole cell
                            cell_i.SoC_Cell_record[step]=np.sum(cell_i.coulomb_ele)/np.sum(cell_i.coulomb_ele_rated) #record SoC of each element and whole cell
                            cell_i.fun_BC(step)
                            cell_i.U3_neo=cell_i.fun_Uini_neo(step)
                            cell_i.fun_update_neo()
                            if cell_i.status_WithElectrolyte == 'Yes':
                                cell_i.c1_ele_Ca, cell_i.c1_ele_An, cell_i.c1_ele_e = cell_i.cini_ele_Ca, cell_i.cini_ele_An, cell_i.cini_ele_e
                            elif cell_i.status_WithElectrolyte == 'No':
                                cell_i.c1_ele_Ca, cell_i.c1_ele_An = cell_i.cini_ele_Ca, cell_i.cini_ele_An
                            step=1
                        cell_i.T_ele=cell_i.T1_4T_ALL[ cell_i.List_ele2node_4T,0 ]                               #T of each element; in the form of 1,2...nECN
                        cell_i.coulomb_ele=cell_i.coulomb_ele-cell_i.dt*cell_i.U1_neo[cell_i.nCC:(cell_i.nCC+cell_i.nECN)]               #coulomb counting   Q=Σdt*I
    #                    if ip.status_ageing=='Yes':
    #                        cell_i.fun_EoL()
                        cell_i.SoC_ele=cell_i.coulomb_ele/cell_i.coulomb_ele_rated;                          #SoC of each element; in the form of 1,2...nECN
                        cell_i.SoC_ele_record[:,step]=cell_i.SoC_ele.reshape(-1); cell_i.SoC_Cell_record[step]=np.sum(cell_i.coulomb_ele)/np.sum(cell_i.coulomb_ele_rated) #record SoC of each element and whole cell
                        cell_i.SoC_Cell_record[step]=np.sum(cell_i.coulomb_ele)/np.sum(cell_i.coulomb_ele_rated) #record SoC of each element and whole cell
                        cell_i.fun_BC(step)
                        ts_step_MatrixC=time.time() if ip.status_Count=='Yes' else []
                        cell_i.MatrixC_neo=cell_i.fun_matrixC_neo();                                                  te_step_MatrixC=time.time() if ip.status_Count=='Yes' else []
                        cell_i.VectorI_neo=cell_i.fun_I_neo(step)
                        if ip.status_CC=='No':   #if CC resistance is not considered, modify the MatrixC and VectorI for the CC node 0index
                            cell_i.fun_modifyCandI4NoCC_neo()
                        t_2=time.time() if ip.status_Count=='Yes' else []

    #                    cell_i.U3_neo = fsolve(cell_i.fun_DAE2NLE, cell_i.U1_neo)                      #default xtol
    #                    cell_i.U3_neo = fsolve(cell_i.fun_DAE2NLE, cell_i.U1_neo, xtol=1e-10)          #xtol control. more accurate
                        cell_i.U3_neo = fsolve(cell_i.fun_DAE2NLE, cell_i.Uini_neo_trial, xtol=1e-10)   #xtol control + fixed initial trial. the most accurate but throtically slower as well

                        cell_i.U3_neo=np.reshape(cell_i.U3_neo,(-1,1)); cell_i.Ii_3=cell_i.U3_neo[cell_i.nCC:(cell_i.nCC+cell_i.nECN)]
                        cell_i.fun_update_neo()
                        cell_i.U_ele_SPMe_3 = cell_i.fun_U_ele_SPMe( cell_i.U3_neo[cell_i.nCC:(cell_i.nCC+cell_i.nECN),0] )
                        cell_i.U_ele_SPMe_record[:,step] = cell_i.U_ele_SPMe_3

                        #--------------change U3_neo form (nCC+nECN+ntab,1) back to U3 form (ntotal+nECN+ntab,1)
                        if ip.status_CC=='Yes':
                            cell_i.U3=np.nan*np.zeros([cell_i.ntotal+cell_i.nECN+cell_i.ntab+1,1])
                        else:
                            cell_i.U3=np.nan*np.zeros([cell_i.ntotal+cell_i.nECN+1,1])
                        cell_i.U3[ cell_i.List_Neo2General[cell_i.ind0_CC_neo,0] ]= cell_i.U3_neo[ cell_i.ind0_CC_neo ]
                        cell_i.U3[ cell_i.ntotal: ]= cell_i.U3_neo[ cell_i.nCC: ]
                        #--------------solve Diffusion model
                        cell_i.fun_update_cof_4D()
                        for i0 in np.arange(cell_i.nECN*3) if cell_i.status_WithElectrolyte == 'Yes' else np.arange(cell_i.nECN*2):      #solve diffusion model for cathode/anode/electrolyte
                            ind0_matID = i0%3 if cell_i.status_WithElectrolyte == 'Yes' else i0%2
                            ind0_ele = i0//3 if cell_i.status_WithElectrolyte == 'Yes' else i0//2
                            if ind0_matID == 0:                        #ind0_matID=0: Cathode
                                if cell_i.status_Diffusion_solver == 'CN':
                                    cell_i.MatrixDs_Ca=cell_i.fun_matrixDs_Ca(ind0_ele,step)
                                    cell_i.VectorDs_Ca=cell_i.fun_vectorDs_Ca(ind0_ele,step,cell_i.MatrixDs_Ca)
                                cell_i.c3_ele_Ca[ind0_ele]=cell_i.fun_Diffusion_Ca(ind0_ele, step, cell_i.c1_ele_Ca[ind0_ele], cell_i.c3_ele_Ca[ind0_ele])
                                cell_i.c1_ele_Ca[ind0_ele]=cell_i.c3_ele_Ca[ind0_ele]
                            elif ind0_matID == 1:                      #ind0_matID=1: Anode
                                if cell_i.status_Diffusion_solver == 'CN':
                                    cell_i.MatrixDs_An=cell_i.fun_matrixDs_An(ind0_ele,step)
                                    cell_i.VectorDs_An=cell_i.fun_vectorDs_An(ind0_ele,step,cell_i.MatrixDs_An)
                                cell_i.c3_ele_An[ind0_ele]=cell_i.fun_Diffusion_An(ind0_ele, step, cell_i.c1_ele_An[ind0_ele], cell_i.c3_ele_An[ind0_ele])
                                cell_i.c1_ele_An[ind0_ele]=cell_i.c3_ele_An[ind0_ele]
                            elif ind0_matID == 2:                      #ind0_matID=2: Electrolyte
                                if cell_i.status_Diffusion_solver == 'CN':
                                    cell_i.MatrixDe=cell_i.fun_matrixDe(ind0_ele,step)
                                    cell_i.VectorDe=cell_i.fun_vectorDe(ind0_ele,step,cell_i.MatrixDe)
                                cell_i.c3_ele_e[ind0_ele]=cell_i.fun_Diffusion_e(ind0_ele, step, cell_i.c1_ele_e[ind0_ele], cell_i.c3_ele_e[ind0_ele])
                                cell_i.c1_ele_e[ind0_ele]=cell_i.c3_ele_e[ind0_ele]
                        cell_i.c3_ele1_Ca_record[:,step] = cell_i.c3_ele_Ca[0]
                        cell_i.c3_ele1_An_record[:,step] = cell_i.c3_ele_An[0]
                        if cell_i.status_WithElectrolyte == 'Yes':
                            cell_i.c3_ele1_e_record[:,step] = cell_i.c3_ele_e[0]
                #↓↓↓MMMMMMMMMMMMMM  Module level  MMMMMMMMMMMMMMMMM↓↓↓
                if ip.status_Module == 'Yes':
                    module_1.fun_BC(step)
                    params_C2M = {}
                    for item in ip.status_Cells_name:
                        if ip.status_ECN_method == 'Neo':
                            params_C2M[item] = globals()[item].MatrixC_neo                 #dict: ['cell_1':cell_1, 'cell_2':cell_2, ...]. Make class instances cell_1, cell_2 etc. into a dict: params
                        elif ip.status_ECN_method == 'General':
                            params_C2M[item] = globals()[item].MatrixC                 #dict: ['cell_1':cell_1, 'cell_2':cell_2, ...]. Make class instances cell_1, cell_2 etc. into a dict: params
                    module_1.MatrixM = module_1.fun_matrixM(params_C2M)            #pass the list containing instances into fun_matrixM in module.py

                    params_C2M = {}
                    for item in ip.status_Cells_name:
                        if ip.status_ECN_method == 'Neo':
                            params_C2M[item] = globals()[item].VectorI_neo                 #dict: ['cell_1':cell_1, 'cell_2':cell_2, ...]. Make class instances cell_1, cell_2 etc. into a dict: params
                        elif ip.status_ECN_method == 'General':
                            params_C2M[item] = globals()[item].VectorI                 #dict: ['cell_1':cell_1, 'cell_2':cell_2, ...]. Make class instances cell_1, cell_2 etc. into a dict: params
                    module_1.VectorM = module_1.fun_vectorM(params_C2M)            #pass the list containing instances into fun_matrixM in module.py

                    if ip.status_linsolver_E_Module=='Sparse':
                        module_1.U3_Module=scipy.sparse.linalg.spsolve(module_1.MatrixM,module_1.VectorM)
                    else:
                        module_1.U3_Module=np.linalg.solve(module_1.MatrixM,module_1.VectorM)
                    for item in ip.status_Cells_name:  #i: cell_1,cell_2...   loop for all cells
                        cell_i = globals()[item]
                        if ip.status_ECN_method == 'Neo':
                            i0_row_temp = module_1.List_Cmatind2Mmatind[ cell_i.cell_ind0,:(module_1.Cells_size_MatrixC_or_neo[cell_i.cell_ind0]) ]
                            cell_i.U3_neo = module_1.U3_Module[i0_row_temp].reshape(-1,1)
                            cell_i.Ii_3=cell_i.U3_neo[cell_i.nCC:(cell_i.nCC+cell_i.nECN)]
                            cell_i.fun_update_neo()
                        elif ip.status_ECN_method == 'General':
                            i0_row_temp = module_1.List_Cmatind2Mmatind[ cell_i.cell_ind0,:(module_1.Cells_size_MatrixC_or_neo[cell_i.cell_ind0]) ]
                            cell_i.U3 = module_1.U3_Module[i0_row_temp].reshape(-1,1)
                            cell_i.U1 = cell_i.U3

                        if ip.status_ECN_method=='Neo':
                            #--------------change U3_neo form (nCC+nECN+ntab,1) back to U3 form (ntotal+nECN+ntab,1)
                            if cell_i.status_CC=='Yes':
                                cell_i.U3=np.nan*np.zeros([cell_i.ntotal+cell_i.nECN+cell_i.ntab+1,1])
                            else:
                                cell_i.U3=np.nan*np.zeros([cell_i.ntotal+cell_i.nECN+1,1])
                            cell_i.U3[ cell_i.List_Neo2General[cell_i.ind0_CC_neo,0] ]= cell_i.U3_neo[ cell_i.ind0_CC_neo ]
                            cell_i.U3[ cell_i.ntotal: ]= cell_i.U3_neo[ cell_i.nCC: ]
                #↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑

                for item in ip.status_Cells_name:  #i: cell_1,cell_2...   loop for all cells
                    cell_i = globals()[item]
                    cell_i.U_pndiff_plot[step]=cell_i.U3[cell_i.node_positive_0ind[0]]-cell_i.U3[cell_i.node_negative_0ind[0]]    #for plotting voltage difference between positive and negative
                    cell_i.I0_record[step]=cell_i.U3[-1]
                    cell_i.V_record[:,step]=cell_i.U3[:cell_i.ntotal].reshape(-1)
                    cell_i.I_ele_record[:,step]=cell_i.U3[cell_i.ntotal:(cell_i.ntotal+cell_i.nECN)].reshape(-1)
                    cell_i.Coulomb_Counting_As[step] = cell_i.Coulomb_Counting_As[step-1]+(-cell_i.U3[-1])*cell_i.dt if cell_i.status_ECN_method=='General' else cell_i.Coulomb_Counting_As[step-1]+(-cell_i.U3_neo[-1])*cell_i.dt
                    cell_i.Charge_Throughput_As[step]= cell_i.Charge_Throughput_As[step-1]+abs(cell_i.U3[-1]*cell_i.dt) if cell_i.status_ECN_method=='General' else cell_i.Charge_Throughput_As[step-1]+abs(cell_i.U3_neo[-1]*cell_i.dt)
                    cell_i.SoC[step]=(cell_i.Capacity0+cell_i.Coulomb_Counting_As[step])/cell_i.Capacity_rated   #SoC of entire cell, calculated by coulomb counting
                    t_3=time.time() if ip.status_Count=='Yes' else []
                #↓↓↓MMMMMMMMMMMMMM  Module level  MMMMMMMMMMMMMMMMM↓↓↓
                if ip.status_Module == 'Yes':
                    module_1.U_pndiff_plot_Module[step] = globals()[ip.status_Cells_name[0]].U3[globals()[ip.status_Cells_name[0]].node_positive_0ind[0]] - globals()[ip.status_Cells_name[-1]].U3[globals()[ip.status_Cells_name[-1]].node_negative_0ind[0]]
                    module_1.I0_record_Module[step] = module_1.U3_Module[-1]
                    module_1.Coulomb_Counting_As_Module[step] = module_1.Coulomb_Counting_As_Module[step-1]+(-module_1.I_ext_Module)*module_1.dt
                    module_1.Charge_Throughput_As_Module[step] = module_1.Charge_Throughput_As_Module[step-1]+abs(module_1.U3_Module[-1]*module_1.dt)
                    module_1.SoC_Module[step]=(module_1.Capacity0_Module+module_1.Coulomb_Counting_As_Module[step])/module_1.Capacity_rated_Module   #SoC of entire cell, calculated by coulomb counting
                #↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑

            ############################################################################solve Thermal model
            for i0 in np.arange(ip.status_Parts_num_Module):
                item = ip.status_Cells_name[i0] if i0 <= np.size(ip.status_Cells_name)-1 else ip.status_Parts_name[i0]
                part_i = globals()[item]

                if part_i.status_Model=='EandT' or part_i.status_Model=='T':
                    if step==1:       #first time, initial voltage potential
                        part_i.T1_4T_ALL=part_i.Tini_4T_ALL
                    if part_i.status_Heat_gen=='Yes':
                        if part_i.status_ECN_method=='Neo' or part_i.status_ECN_method=='SPMe':
                            part_i.q_4T_ALL=part_i.fun_HeatGen_neo_4T()     #get heat gen term of heat source; note thermal model is calculated on Thermal framework node set, all variables are ended with _4T
                        elif part_i.status_ECN_method=='General':
                            part_i.q_4T_ALL=part_i.fun_HeatGen_4T()     #get heat gen term of heat source; note thermal model is calculated on Thermal framework node set, all variables are ended with _4T
                        part_i.q_4T_record[:,step]=part_i.q_4T_ALL[:part_i.ntotal_4T,0]/part_i.V_stencil_4T_ALL[:part_i.ntotal_4T]      #record heat generation of each node
                    else:
                        part_i.q_4T_ALL=np.zeros([part_i.n_4T_ALL,1])
                    if part_i.status_Entropy=='Yes':
                        part_i.q_4T_ALL=part_i.fun_Entropy_4T()     #get entropy term of heat source; note thermal model is calculated on Thermal framework node set, all variables are ended with _4T
                    part_i.q_4T_ALL[:,0]=part_i.q_4T_ALL[:,0]/part_i.V_stencil_4T_ALL
                    if part_i.status_TemBC_smoothening=='Yes':
                        part_i.T_cooling_smoothened=part_i.T_cooling + (part_i.T_initial-part_i.T_cooling)/np.exp(part_i.smoothening_stiffness * part_i.t_record[step])
                    part_i.fun_BC_4T_ALL();                                                                                             ts_step_T=time.time() if ip.status_Count=='Yes' else []                #apply BC-temperature; For T3_4T at this line, only elements of constrained boundary node are filled, others are zero
                    if part_i.status_CheckBC_4T=='Yes' and step==1:
                        part_i.fun_BC_4T_check()
                    if part_i.status_Thermal_solver == 'CN':
                        part_i.VectorCN=part_i.fun_VectorCN()
                        if part_i.status_TemBC_VectorCN_check == 'Yes':
                            part_i.fun_implicit_TemBC_VectorCN_check()  #for Temperature constrained BC (Dirichlet BC), i.g. node 36 has initial 30°C, node 40 is suddenly assigned with 20°C, λΔT/Δz could numerically cause a large number that the heat is extracted unreasonably large.  In order to avoid this, Vector_CN is used as an indicator

                    if ip.status_Module_4T == 'No':
                        part_i.T3_4T_ALL=part_i.fun_Thermal(part_i.T1_4T_ALL, part_i.T3_4T_ALL, part_i.ind0_BCtem_ALL, part_i.ind0_BCtem_others_ALL, part_i.h_4T_ALL, part_i.Tconv_4T_ALL, part_i.ind0_BCconv_ALL, part_i.ind0_BCconv_others_ALL);                  te_step_T=time.time() if ip.status_Count=='Yes' else []        #calculate node temperature
                        #print('Tavg', part_i.T3_4T_ALL)
                        part_i.T_record[:,step]=part_i.T3_4T_ALL.reshape(-1)                                                                                                                                           #record T
                        part_i.T_avg_record[step], part_i.T_SD_record[step], part_i.T_Delta_record[step] =part_i.fun_weighted_avg_and_std(part_i.T3_4T_ALL,part_i.rou_c_V_weights)                                                                                                            #record T average and SD, all weighted by ρcV
                        part_i.T1_4T_ALL=part_i.T3_4T_ALL.copy()
                        if part_i.status_Echeck=='Yes':
                            part_i.fun_Echeck(step)

            #↓↓↓MMMMMMMMMMMMMM  Module level  MMMMMMMMMMMMMMMMM↓↓↓
            if ip.status_Module_4T == 'Yes':

                params_C2M = module_1_4T.interface_attr_forBC
                module_1_4T.fun_BC_4T_Module(params_C2M)

                params_C2M = {}
                for i0 in np.arange(ip.status_Parts_num_Module):
                    item_temp1 = ip.status_Parts_name[i0]
                    item_temp2 = ip.status_Cells_name[i0] if i0 <= ip.status_Cells_num_Module-1 else ip.status_Parts_name[i0]
                    params_C2M[item_temp1] = {}
                    params_C2M[item_temp1]['VectorCN'] = globals()[item_temp2].VectorCN
                    params_C2M[item_temp1]['q_4T_ALL'] = globals()[item_temp2].q_4T_ALL
                    params_C2M[item_temp1]['T1_4T_ALL'] = globals()[item_temp2].T1_4T_ALL
                    params_C2M[item_temp1]['T3_4T_ALL'] = globals()[item_temp2].T3_4T_ALL
                module_1_4T.VectorM_4T = module_1_4T.fun_vectorM_4T(params_C2M)             #pass the list containing instances into fun_vectorM in module.py

                if ip.status_linsolver_T=='BandSparse':
                    top=np.zeros([module_1_4T.length_MatrixM_4T,module_1_4T.length_MatrixM_4T]); bottom=np.zeros([module_1_4T.length_MatrixM_4T,module_1_4T.length_MatrixM_4T]); MatrixM_4T_expand=np.concatenate((top,module_1_4T.MatrixM_4T,bottom))
                    ab=MatrixM_4T_expand[module_1_4T.ind0_r_expand,module_1_4T.ind0_c_expand]
                    module_1_4T.T3_4T_Module=scipy.linalg.solve_banded((module_1_4T.ind0_l,module_1_4T.ind0_u),ab,module_1_4T.VectorCN)
                elif ip.status_linsolver_T=='Sparse':
                    module_1_4T.T3_4T_Module=scipy.sparse.linalg.spsolve(module_1_4T.MatrixM_4T,module_1_4T.VectorM_4T).reshape(-1,1)
                else:  #i.e. status_linsolver_T=='General'
                    module_1_4T.T3_4T_Module=np.linalg.solve(module_1_4T.MatrixM_4T,module_1_4T.VectorM_4T)
                for i0 in np.arange(ip.status_Parts_num_Module):              #i: part_1,part_2...   loop for all parts
                    item_temp1 = ip.status_Parts_name[i0]
                    item_temp2 = module_1_4T.Parts2Cells_name[item_temp1]
                    part_i = globals()[item_temp2]
                    i0_row_temp = module_1_4T.List_Cmatind2Mmatind_4T[ i0,:(part_i.n_4T_ALL) ]
                    part_i.T3_4T_ALL = module_1_4T.T3_4T_Module[i0_row_temp].reshape(-1,1)
                    part_i.T1_4T_ALL = part_i.T3_4T_ALL
                    part_i.T_record[:,step]=part_i.T3_4T_ALL.reshape(-1)                                                                                                                    #record T
                    part_i.T_avg_record[step], part_i.T_SD_record[step], part_i.T_Delta_record[step] =part_i.fun_weighted_avg_and_std(part_i.T3_4T_ALL,part_i.rou_c_V_weights)                                                                                                            #record T average and SD, all weighted by ρcV

                #--calculate volume-average T and SD
                if ip.status_BC_Module_4T == 'Prismatic_Cell1':
                    temp_T = np.concatenate((cell_1.T3_4T_ALL, cell_2.T3_4T_ALL))
                    module_1_4T.T_avg_record[step], module_1_4T.T_SD_record[step], module_1_4T.T_Delta_record[step] = module_1_4T.fun_weighted_avg_and_std(temp_T, module_1_4T.rou_c_V_weights)
                if ip.status_FormFactor == 'Prismatic':
                    part_3.fun_Echeck(step)                                                                                                            #record T average and SD, all weighted by ρcV
            #↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑
            t_4=time.time() if ip.status_Count=='Yes' else []
            ############################################################################plot temperature, voltage potential etc.
            if ip.status_PopFig_or_SaveGIF_instant == 'Fig' or ip.status_PopFig_or_SaveGIF_instant == 'GIF' and ip.status_fig1to9 == 'Yes':
                cell_1.fun_plot(step, status_climit_vector)
            t_5=time.time() if ip.status_Count=='Yes' else []
            ############################################################################print running time
            if ip.status_Count=='Yes':
                if (t_5-t_1)<=1e-10:
                    print('total time cost is %f (s), which is too small to calculate percentage of each part\n'%(t_5-t_1) )
                else:
                    print('%s for the entire cell is %f (mm)'%('Thickness' if ip.status_FormFactor=='Pouch' else 'Largest radius', cell_1.delta_cell*1e3))
                    if ip.status_Model=='EandT'or ip.status_Model=='E':
                        print('R0 for the entire cell is %f (Ω)'%(1/np.sum(1/cell_1.R0_pair[:,2])))
                        print('SoC for the entire cell is %.2f%%'%(cell_1.SoC[step]*100))
                        print('\ntime cost of the step is %f (s)' %(t_5-t_1) )
                        print('on 1-2 (MatrixC)   : %f (s), %.2f%%' %(t_2-t_1, (t_2-t_1)/(t_5-t_1)*100) )
                        if ip.status_Electrical_type != 'SPMe':
                            print('wherein interp: %f (s)' %(cell_1.duration_R0_interp + cell_1.duration_Ri_interp + cell_1.duration_Ci_interp + cell_1.duration_OCV_interp) )
                            print('wherein OCV interp: %f (s)' %cell_1.duration_OCV_interp )
                            print('wherein R0 interp: %f (s)' %cell_1.duration_R0_interp )
                            print('wherein Ri interp: %f (s)' %cell_1.duration_Ri_interp )
                            print('wherein Ci interp: %f (s)' %cell_1.duration_Ci_interp )
                        print('on 2-3 (Equation E): %f (s), %.2f%%' %(t_3-t_2, (t_3-t_2)/(t_5-t_1)*100) )
                    if ip.status_Model=='EandT'or ip.status_Model=='T':
                        print('on 3-4 (Equation T): %f (s), %.2f%%' %(t_4-t_3, (t_4-t_3)/(t_5-t_1)*100) )
                    print('on 4-5 (Plotting)  : %f (s), %.2f%%' %(t_5-t_4, (t_5-t_4)/(t_5-t_1)*100) )
                    print('expected remaining time: %.2f (s) or %.2f (min) or %.2f (h)' %((t_5-t_1)*(cell_1.nt-step),(t_5-t_1)*(cell_1.nt-step)/60,(t_5-t_1)*(cell_1.nt-step)/3600) )
                    print('----------------------------------------')
            ############################################################################judge discharge/charge limit
            if ip.status_Module == 'No':
                if step==cell_1.nt:
                    break                                           #if up to the given nt, end: break the cycle loop and cycles loop;   this is also in the break in cycles loop
            elif ip.status_Module == 'Yes':
                if step==module_1.nt:
                    break                                           #if up to the given nt, end: break the cycle loop and cycles loop;   this is also in the break in cycles loop
            step=step+1
        #-----------------------------------------------------------------------------------------loop end for single cycle
        if step==cell_1.nt or cycle==cell_1.ncycle:
            break                                               #if up to the given ncycle, break the cycles loop
        cycle=cycle+1
    #=============================================================================================loop end for cycles
    t_end=time.time()
    print('\nsolving done, running postprocessor...\n')
    status_climit_vector=np.linspace((cell_i.T3_4T_ALL).min()-273.15,(cell_i.T3_4T_ALL).max()-273.15,ip.status_levels)        #colorbar limit
    if ip.status_PostProcessor == 'Yes':
        if ip.status_PostProcess_cell_id in globals():
            globals()[ip.status_PostProcess_cell_id].fun_Postprocessor(step,t_begin,t_end, status_climit_vector)
        else:
            print('Current postprocessing is for %s which does NOT exist\nCheck status_PostProcess_cell_id in inputs.py' %ip.status_PostProcess_cell_id)
            raise Exception('exit')
        #3D visualization
        if ip.status_visualization_method == 'mayavi':
            #Temperature
            globals()[ip.status_PostProcess_cell_id].fig1 = mlab.figure(bgcolor=(1,1,1))
            plot_steps_available=np.where(~np.isnan(cell_i.T_avg_record))[0]      #in cycling mode, there are NaN values in the last cycles. So here plot_steps_available is all the steps with non-NaN values
            plot_step=plot_steps_available[-1]                             #plot the last step from non-NaN steps
            vmin, vmax = (cell_i.T_record[:,plot_step]-273.15).min(), (cell_i.T_record[:,plot_step]-273.15).max()
            cell_i = globals()[ip.status_PostProcess_cell_id]
            cell_i.fun_mayavi_by_node([0,0,0], cell_i.T_record[:,plot_step]-273.15, vmin, vmax, title_string = '°C', colormap_string = 'coolwarm')
        elif ip.status_visualization_method == 'plotly':
            #Temperature
            plot_steps_available=np.where(~np.isnan(cell_i.T_avg_record))[0]      #in cycling mode, there are NaN values in the last cycles. So here plot_step is the last non-NaN step
            plot_step=plot_steps_available[-1]                             #plot the last step of non-NaN steps
            vmin, vmax = (cell_i.T_record[:,plot_step]-273.15).min(), (cell_i.T_record[:,plot_step]-273.15).max()
            cell_i = globals()[ip.status_PostProcess_cell_id]
            cell_i.fun_plotly_T([0,0,0], plot_step, vmin, vmax)
    #↓↓↓MMMMMMMMMMMMMM  Module level  MMMMMMMMMMMMMMMMM↓↓↓
    if ip.status_Module == 'Yes' and ip.status_PostProcessor_Module == 'Yes':
        if ip.status_PostProcess_Module_id in globals():
            globals()[ip.status_PostProcess_Module_id].fun_Postprocessor(step,t_begin,t_end)
            if ip.status_Module_4T == 'Yes' and ip.status_FormFactor == 'Prismatic':
                params_C2M = {}
                globals()[ip.status_PostProcess_Module_id+'_4T'].fun_Postprocessor(step,t_begin,t_end,params_C2M)
            else:
                globals()[ip.status_PostProcess_Module_id+'_4T'].fun_Postprocessor(step,t_begin,t_end)

        else:
            print('Current Module input for postprocessing is %s which does NOT exist\nCheck status_PostProcess_Module_id in inputs.py' %ip.status_PostProcessor_Module)
            raise Exception('exit')
        #3D visualization
        if ip.status_visualization_method == 'mayavi':
            #Temperature
            module_1.fig1 = mlab.figure(bgcolor=(1,1,1))
            plot_steps_available=np.where(~np.isnan(cell_1.T_avg_record))[0]      #in cycling mode, there are NaN values in the last cycles. So here plot_steps_available is all the steps with non-NaN values
            plot_step=plot_steps_available[-1]                             #plot the last step from non-NaN steps
            if ip.status_FormFactor == 'Prismatic':
                v_min_max_temp = [ [(globals()[o].T3_4T_ALL-273.15).min(), (globals()[o].T3_4T_ALL-273.15).max()] for o in list(module_1_4T.Parts2Cells_name.values())]
            else:
                v_min_max_temp = [ [(globals()[o].T_record[:,plot_step]-273.15).min(), (globals()[o].T_record[:,plot_step]-273.15).max()] for o in list(module_1_4T.Parts2Cells_name.values())]
            vmin, vmax = np.min(v_min_max_temp),np.max(v_min_max_temp)
            for i0 in np.arange(ip.status_Parts_num_Module):              #i: part_1,part_2...   loop for all parts
                item_temp1 = ip.status_Parts_name[i0]
                item_temp2 = module_1_4T.Parts2Cells_name[item_temp1]
                part_i = globals()[item_temp2]
                part_i.fun_mayavi_by_node(module_1.Cells_XYZ_Module[i0,:], part_i.T_record[:,plot_step]-273.15, vmin, vmax, title_string = '°C', colormap_string = 'coolwarm')
    print('Postprocessing is done')
    mlab.show()
    if ip.status_Module == 'Yes' or ip.status_Module_4T == 'Yes':
        print('Switch for advance structure is ON')
    else:
        print('Switch for advance structure is OFF')
    #↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑MMMMMMMMMMMMMMMMMMMMMM↑↑↑
