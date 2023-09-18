# -*- coding: utf-8 -*-

import numpy as np
import pyecn.parse_inputs as ip
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
inf=1e10

class Module:
    def __init__(self,params_update):
        #---------copy attr in inputs.py into this class---------
        my_shelf = {}                       
        for key in dir(ip):
            if not key.startswith("__"):          #filter out internal attributes like __builtins__ etc
                my_shelf[key] = ip.__dict__[key]
        self.__dict__ = my_shelf.copy()
        self.__dict__.update(params_update)       #update the update(unique) inputs for this 'self'
        #--------------------------------------------------------        
        ####################################
        ###CREATE class - self ATTRIBUTES###
        ####################################
        self.Cells_num = len(self.Cells_name)      #Example in this section is based on 2 cells: 2,2,1; 2,2,1        Cells_num:2
        for i0 in np.arange(self.Cells_num):       #generate dict key 'size_MatrixC' in 'Cells_attr' for each cell
            string_temp = self.Cells_attr[self.Cells_name[i0]]['status_ECN_method']
            if string_temp == 'General':
                self.Cells_attr[self.Cells_name[i0]]['size_MatrixC_or_neo'] = [ o['ntotal']+o['nECN']+o['ntab']+1 if o['status_CC'] == 'Yes' else o['ntotal']+o['nECN']+1 for o in [ self.Cells_attr[self.Cells_name[i0]] ] ] 
            elif string_temp == 'Neo':
                self.Cells_attr[self.Cells_name[i0]]['size_MatrixC_or_neo'] = [ o['nCC']+o['nECN']+o['ntab']+1 if o['status_CC'] == 'Yes' else o['nCC']+o['nECN']+1 for o in [ self.Cells_attr[self.Cells_name[i0]] ] ] 
        #[ o['ntotal']+o['nECN']+o['ntab']+1 for o in self.Cells_attr.values() ] #outputs MatrixC size [21,21] for 2 cells

        self.Cells_size_MatrixC_or_neo = np.array( [ o['size_MatrixC_or_neo'] for o in self.Cells_attr.values() ] ).flatten()
        self.List_Cmatind2Mmatind = -9999*np.ones( [self.Cells_num, np.max( [ o['size_MatrixC_or_neo'] for o in self.Cells_attr.values()] )],dtype=int )      #List_Cmatind2Mmatind: Cell objects: (cell_1,cell_2) to Module matrixM ind0: (1,2,3...21, 22,23,24...42)-1. e.g. List_Cmatind2Mmatind: [[0,1,2,...20], 
                                                                                                                                                       #                                                                                                                                       [21,22,23,...41]]
        nbefore_temp = 0                                                                                                                    
        for i0 in np.arange(self.Cells_num):
            n_temp = self.Cells_size_MatrixC_or_neo[i0]
            self.List_Cmatind2Mmatind[i0,:n_temp] = np.arange(n_temp) + nbefore_temp
            nbefore_temp += n_temp
        self.Cells_nPos = [ o['nPos'] for o in self.Cells_attr.values()]
        self.Cells_nNeg = [ o['nNeg'] for o in self.Cells_attr.values()]
        self.Cells_node_positive_0ind = -9999*np.ones( [self.Cells_num, np.max( [ o['nPos'] for o in self.Cells_attr.values()] )],dtype=int )
        self.Cells_node_negative_0ind = -9999*np.ones( [self.Cells_num, np.max( [ o['nNeg'] for o in self.Cells_attr.values()] )],dtype=int )
        for i0 in np.arange(self.Cells_num):
            n_Pos_temp = self.Cells_attr[self.Cells_name[i0]]['nPos']
            n_Neg_temp = self.Cells_attr[self.Cells_name[i0]]['nNeg']
            self.Cells_node_positive_0ind[i0,:n_Pos_temp] = self.Cells_attr[self.Cells_name[i0]]['node_positive_0ind']   
            self.Cells_node_negative_0ind[i0,:n_Neg_temp] = self.Cells_attr[self.Cells_name[i0]]['node_negative_0ind']  
        ### getting Cells linking info in a Module ###
        self.Ppos_pair_Module = np.transpose(np.nonzero(self.Cells_Ppos_link_Module))   
        self.Pneg_pair_Module = np.transpose(np.nonzero(self.Cells_Pneg_link_Module))   
        self.S_pair_Module = np.transpose(np.nonzero(self.Cells_S_link_Module))   

#        self.Cells_Pind0_Module = -9999*np.ones([self.status_Cells_num_Module,self.status_Cells_num_Module],dtype=int)
#        self.Cells_Pind0_Module[self.Ppos_pair_Module[:,0],0] = self.Ppos_pair_Module[:,0]
#        self.Cells_Pind0_Module[ self.Ppos_pair_Module[:,0], self.Ppos_pair_Module[:,1] ] = self.Ppos_pair_Module[:,1]
#        self.Cells_Pnum_Module = -9999*np.ones([self.status_Cells_num_Module,1],dtype=int)
#        for i0 in np.unique(self.Ppos_pair_Module[:,0]):                    #for each row, sort non-9999 elements to the left side
#            row_temp = self.Cells_Pind0_Module[i0].copy()
#            self.Cells_Pind0_Module[i0] = np.concatenate((row_temp[row_temp!=-9999], row_temp[row_temp==-9999]))    
#            self.Cells_Pnum_Module[i0] = np.size(row_temp[row_temp!=-9999])

        self.Cells_link_Module = np.zeros([self.status_Cells_num_Module,self.status_Cells_num_Module])
        Ppos_pair_num = self.Ppos_pair_Module.shape[0]
        S_pair_num = self.S_pair_Module.shape[0]
        o1 = self.Ppos_pair_Module
        o2 = self.S_pair_Module
        #---------fill in P_pair rows in 'Cells_link_Module' 
        self.Cells_link_Module[ np.arange(Ppos_pair_num), self.Ppos_pair_Module[:,0] ] = -1
        self.Cells_link_Module[ np.arange(Ppos_pair_num), self.Ppos_pair_Module[:,1] ] = 1        
        #---------fill in S_pair rows in 'Cells_link_Module' 
        for i0 in np.arange(S_pair_num):
            #---------fill in '1' terms in S_pair rows in 'Cells_link_Module' 
            self.Cells_link_Module[ Ppos_pair_num+i0, self.S_pair_Module[i0,0] ] = 1
            o3 = o1[o1[:,0]==o2[i0,0]]
            self.Cells_link_Module[ Ppos_pair_num+i0, o3[:,1] ] = 1
            #---------fill in '-1' terms in S_pair rows in 'Cells_link_Module' 
            self.Cells_link_Module[ Ppos_pair_num+i0, self.S_pair_Module[i0,1] ] = -1
            o3 = o1[o1[:,0]==o2[i0,1]]
            self.Cells_link_Module[ Ppos_pair_num+i0, o3[:,1] ] = -1
        #---------fill in the last row in 'Cells_link_Module'
        o4 = self.status_Cells_num_Module-1       #e.g. 3 cells(ind0=0,1,2) in parallel. o4 is last cell(ind0=2), o4=2
        o5 = o1[o1[:,1]==o4]                      #find the cell(ind0=2) 's master cell row, o5=[[0,2]]
        if o5.size==0:                                #if the last cell is not connected in parallel with any cell 
            self.Cells_link_Module[ -1,o4 ] = 1       #fill in the last row in 'Cells_link_Module'            
        else:                                         #if the last cell is connected in parallel with other cell (in the case of 3 cells, cell(ind0=2) is connected with cells(ind0=0,1))
            master_ind0 = o5[0,0]                     #find the cell(ind0=2) 's master cell(ind0=0). cell(ind0=0) is the master cell as it connects cells(ind0=1,2). o5[0,0]=0
            o6 = np.unique(o1[o1[:,0]==master_ind0])  #collect the cells(ind0=0,1,2) that are connected in parallel for the last cell (or last row in 'Cells_link_Module'). o6=1,2       
            self.Cells_link_Module[ -1,o6 ] = 1       #fill in the last row in 'Cells_link_Module'
        ### getting Cells spatial locations in a Module ###
        self.Cells_XYZ_Module = self.status_Cells_XYZ_Module
        ### getting initial voltage potential and I ###
#        self.fun_IC()
        self.Charge_Throughput_As_Module = np.zeros([self.nt+1,1])      #overall coulomb counting, in the form of 1,2...nt
        self.U_pndiff_plot_Module = np.nan*np.zeros([self.nt+1])        #for plotting positive negative voltage difference in a Module
        self.U_pndiff_plot_Module[0] = 0                                #initial voltage of the module does not exist, unlike cell level. In cell level, initial voltage is OCV which is stable and under no current. 
                                                                        #in module level, even under no current, cells may inter charge and initial voltage unknown (only the first time step voltage can be calculated)
        self.I0_record_Module=np.nan*np.zeros([self.nt+1])              #record I0
        self.I0_record_Module[0] = 0                                    #respective to initial voltage (U_pndiff_plot_Module[0]), initial voltage and current are both meaningless

        self.Capacity_rated_Module=np.copy(self.Capacity_rated0_Module)
        self.Coulomb_Counting_As_Module=np.zeros([self.nt+1,1])       #overall coulomb counting, in the form of 1,2...nt
        self.Charge_Throughput_As_Module=np.zeros([self.nt+1,1])      #overall coulomb counting, in the form of 1,2...nt
        self.SoC_Module=np.nan*np.zeros([self.nt+1])                  #SoC for entire cell

    #########################################################   
    ####### function for calculating Qloss and Rinc #########
    #########################################################
    def fun_EoL_Eta_Miu(self,params_C2M,step):
        if self.status_BC_Module_4T == 'Prismatic_Cell1':
            if self.status_Electrical_type == 'ECN_Plain':
                self.Eta_plot_Module[step] = (self.Capacity_rated0_Module-self.Capacity_rated_Module)/self.Capacity_rated0_Module
                
                R0_BoL_module = 1/np.sum( 1/np.array([ params_C2M[o]['R0_BoL_cell'] for o in ip.status_Cells_name ]) )
                R0_MoL_module = 1/np.sum( 1/np.array([ params_C2M[o]['R0_MoL_cell'] for o in ip.status_Cells_name ]) )
                
                self.Miu_plot_Module[step] = R0_MoL_module/R0_BoL_module - 1
        else:
            print('PyECN error: only Prismatic_Cell1 is ready for cycling at module level now')
            raise Exception('exit')
    #########################################################   
    #####   function for Electrical initial condition   #####
    #########################################################
#    def fun_IC(self):
#    #------------------------------- input window --------------------------------
#        status_discharge=1                    #for discharge  
#    #    status_discharge=-1                  #for charge
#        self.status_discharge = status_discharge
    #########################################################   
    ############### function for Electrical BC ##############
    #########################################################
    def fun_BC(self,step):    
        #---------------constant current mode---------------
        self.status_IVmode=0
        #---------------------------------------------------
        self.I_ext_Module = self.status_discharge_Module * self.Capacity_rated0_Module/3600*self.C_rate_Module                  #for discharge, status_discharge=1; for charge, status_discharge=-1
            #----------------------user defined discharge----------------------
        if hasattr(self,'Table_I_ext'):   #if current is already loaded from external file
            self.I_ext_Module=self.Table_I_ext[step]                                                                      
    #########################################################   
    ################## function for MatrixM #################
    #########################################################
    def fun_matrixM(self,params_C2M):            #Example in this section is based on 3 cells: 2,2,1; 2,2,1; 4,3,2
        #--------------------------- diagonally append MatrixC(s) to form MatrixM ---------------------------
        item = ip.status_Cells_name[0]
        MatrixM = params_C2M[item]
        for item in ip.status_Cells_name[1:]:
            MatrixM = block_diag( MatrixM, params_C2M[item] )
        MatrixM = block_diag(MatrixM,0)
        #--------------------------- clear the I0_C1, I0_C2... rows ---------------------------
        ind0_Cell_temp = np.arange(self.Cells_num)                    #cell_1,cell_2,cell_3 i.g. all cells
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo[ind0_Cell_temp]-1]
        MatrixM[i0_row_temp] = 0
        #--------------------------- Assemble Cells_link_Module into MatrixM: P_pair rows ---------------------------        
        Ppos_pair_num = self.Ppos_pair_Module.shape[0]
        S_pair_num = self.S_pair_Module.shape[0]
        ind0_Cell_temp = np.repeat( np.arange(Ppos_pair_num).reshape(-1,1), self.Cells_num, axis=1 )                   #e.g. 5 cells in ppt: ind0_Cell_temp = np.array([[0,0,0,0,0],[1,1,1,1,1]]); 3 cells: ind0_Cell_temp = np.array([[0,0,0]])  
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo[ind0_Cell_temp]-1]
        ind0_Cell_temp = np.repeat( np.arange(self.Cells_num).reshape(1,-1), Ppos_pair_num, axis=0 )                   #e.g. 5 cells in ppt: ind0_Cell_temp = np.array([[0,1,2,3,4],[0,1,2,3,4]]); 3 cells: ind0_Cell_temp = np.array([[0,1,2]])
        i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_positive_0ind[ind0_Cell_temp,0]]       
        MatrixM[ i0_row_temp,i0_col_temp ] = self.Cells_link_Module[:Ppos_pair_num,:]
        #--------------------------- Assemble Cells_link_Module into MatrixM: S_pair rows ---------------------------        
        ind0_Cell_temp = np.repeat( np.arange(S_pair_num).reshape(-1,1)+Ppos_pair_num, self.Cells_num, axis=1 )        #e.g. 5 cells in ppt: ind0_Cell_temp = np.array([[2,2,2,2,2],[3,3,3,3,3]]); 3 cells: ind0_Cell_temp = np.array([[1,1,1]])         
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo[ind0_Cell_temp]-1]
        ind0_Cell_temp = np.repeat( np.arange(self.Cells_num).reshape(1,-1), S_pair_num, axis=0 )                      #e.g. 5 cells in ppt: ind0_Cell_temp = np.array([[0,1,2,3,4],[0,1,2,3,4]]); 3 cells: ind0_Cell_temp = np.array([[0,1,2]])
        i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo[ind0_Cell_temp]-1] 
        MatrixM[ i0_row_temp,i0_col_temp ] = self.Cells_link_Module[Ppos_pair_num:Ppos_pair_num+S_pair_num,:]
        #--------------------------- Assemble Cells_link_Module into MatrixM: the last row ---------------------------        
        ind0_Cell_temp = -1 * np.ones([1,self.Cells_num],dtype=int)                                                    #e.g. 5 cells in ppt: ind0_Cell_temp = np.array([[-1,-1,-1,-1,-1]]); 3 cells: ind0_Cell_temp = np.array([[-1,-1,-1]])
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo[ind0_Cell_temp]-1]
        ind0_Cell_temp = np.arange(self.Cells_num).reshape(1,-1)                                                       #e.g. 5 cells in ppt: ind0_Cell_temp = np.array([[0,1,2,3,4]]); 3 cells: ind0_Cell_temp = np.array([[0,1,2]])
        i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo[ind0_Cell_temp]-1] 
        MatrixM[ i0_row_temp,i0_col_temp ] = self.Cells_link_Module[-1,:]
        #--------------------------- MatrixM: -I0_M1 in the last low ---------------------------
        ind0_Cell_temp = -1                                           
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo[ind0_Cell_temp]-1]
        i0_col_temp = -1
        MatrixM[ i0_row_temp,i0_col_temp ] = -1

        #--------------------------- clear the Vneg_C1, Vneg_C2... rows ---------------------------        
        ind0_Cell_temp = np.arange(self.Cells_num-1)                    
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_negative_0ind[ind0_Cell_temp,0]]
        MatrixM[i0_row_temp] = 0
        #--------------------------- Modify V_neg in each cell: Pneg_pair rows ---------------------------
        Pneg_pair_num = self.Pneg_pair_Module.shape[0]
        S_pair_num = self.S_pair_Module.shape[0]
        ind0_Cell_temp = np.arange(Pneg_pair_num)                    
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_negative_0ind[ind0_Cell_temp,0]]
        ind0_Cell_temp = self.Pneg_pair_Module[:,0]          
        i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_negative_0ind[ind0_Cell_temp,0]]       
        MatrixM[ i0_row_temp,i0_col_temp ] = -1
        ind0_Cell_temp = self.Pneg_pair_Module[:,1]          
        i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_negative_0ind[ind0_Cell_temp,0]]       
        MatrixM[ i0_row_temp,i0_col_temp ] = 1
        #--------------------------- Modify V_neg in each cell: S_pair rows ---------------------------
        ind0_Cell_temp = np.arange(S_pair_num) + Pneg_pair_num                    
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_negative_0ind[ind0_Cell_temp,0]]
        ind0_Cell_temp = self.S_pair_Module[:,0]          
        i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_negative_0ind[ind0_Cell_temp,0]]       
        MatrixM[ i0_row_temp,i0_col_temp ] = 1
        ind0_Cell_temp = self.S_pair_Module[:,1]          
        i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_positive_0ind[ind0_Cell_temp,0]]       
        MatrixM[ i0_row_temp,i0_col_temp ] = -1

        #--------------------------- last equation (current/voltage control) ---------------------------
        if self.status_IVmode_Module == 0:      #current control mode
            MatrixM[ -1,-1 ] = inf
        elif self.status_IVmode_Module == 1:    #voltage control mode
            ind0_Cell_temp = 0                                        #cell_1 i.g. the first cell
            i0_col_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_positive_0ind[ind0_Cell_temp,0]]       
            MatrixM[ -1,i0_col_temp ] = inf

        return MatrixM
    #########################################################   
    ###############    function for VectorM    ##############
    #########################################################
    def fun_vectorM(self,params_C2M):
        #--------------------------- concatenate VectorI(s) to form VectorM ---------------------------
        VectorM = np.concatenate(( [o for o in params_C2M.values()] ))
        VectorM = np.concatenate(( VectorM,[0] ))
        #--------------------------- clear the I0_C1, I0_C2... rows ---------------------------
        ind0_Cell_temp = np.arange(self.Cells_num)                    #cell_1,cell_2,cell_3 i.g. all cells
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_size_MatrixC_or_neo-1]
        VectorM[i0_row_temp] = 0
        #--------------------------- clear the Vneg_C1, Vneg_C2... rows ---------------------------        
        ind0_Cell_temp = np.arange(self.Cells_num-1)                    
        i0_row_temp = self.List_Cmatind2Mmatind[ind0_Cell_temp,self.Cells_node_negative_0ind[ind0_Cell_temp,0]]
        VectorM[i0_row_temp] = 0

        #--------------------------- last equation (current/voltage control) ---------------------------
        if self.status_IVmode_Module == 0:      #current control mode
            VectorM[ -1 ] = inf * self.I_ext_Module
        elif self.status_IVmode_Module == 1:    #voltage control mode
            VectorM[ -1 ] = inf * self.V_ext_Module
        
        return VectorM       
    #########################################################   
    #########     function for Postprocessor     ############
    #########################################################
    def fun_Postprocessor(self,step,t_begin,t_end):        
        if ip.status_Model=='EandT'or ip.status_Model=='E':    
            # plt.figure(11)
            # plt.subplot(2,1,1)
            # plt.plot(self.Charge_Throughput_As_Module/3600,self.U_pndiff_plot_Module,'ro')    
            # plt.ylabel('Terminal voltage [V]') 
            # plt.subplot(2,1,2)
            # plt.plot(self.Charge_Throughput_As_Module/3600,self.I0_record_Module,'bo')
            # plt.xlabel('Charge throughput [Ah]'); plt.ylabel('Terminal current [A]')
            if ip.status_ageing=='Yes':                     #plot V-Ah for assigned cycles
                if ip.status_Electrical_type == 'ECN_Plain':
                    #---------------plot voltage with Ah in assigned cycles
                    import itertools
                    marker=itertools.cycle((',')) 
                    plt.figure()
                    for i0 in np.arange(np.size(ip.status_cycle_focus)):
                        step_plot=self.List_cycle2step[ip.status_cycle_focus[i0]-1];  step_plot=step_plot[~np.isnan(step_plot)].astype(int)
                        plt.plot((self.Charge_Throughput_As_Module[ step_plot-1 ]-self.Charge_Throughput_As_Module[ step_plot-1 ][0])/3600,self.U_pndiff_plot_Module[ step_plot-1 ],marker=next(marker),linewidth=2, label='Cycle %d'%ip.status_cycle_focus[i0])
                    plt.xlabel('Ah in a cycle'); plt.ylabel('Terminal voltage [V]') 
                    plt.legend()
                    #---------------plot voltage with Ah in assigned cycles
                    plt.figure()
                    for i0 in np.arange(np.size(ip.status_cycle_focus)):
                        step_plot=self.List_cycle2step[ip.status_cycle_focus[i0]-1];  step_plot=step_plot[~np.isnan(step_plot)].astype(int)
                        plt.plot((self.Charge_Throughput_As_Module[ step_plot-1 ]-self.Charge_Throughput_As_Module[ step_plot-1 ][0])/3600,self.SoC_Module[ step_plot-1 ],marker=next(marker),linewidth=2, label='Cycle %d'%ip.status_cycle_focus[i0])
                    plt.xlabel('Ah in a cycle'); plt.ylabel('Entire cell SoC') 
                    plt.legend()        
                    #---------------plot η,μ with Ah
                    plt.figure()
                    plt.subplot(1,2,1)
                    plt.plot(self.Charge_Throughput_As_Module/3600,self.Eta_plot_Module*100,'ro',markersize=1)
                    plt.xlabel('Charge throughput [Ah]');     plt.ylabel('η [%]')
                    plt.subplot(1,2,2)
                    plt.plot(self.Charge_Throughput_As_Module/3600,self.Miu_plot_Module*100,'ro',markersize=1)
                    plt.xlabel('Charge throughput [Ah]');     plt.ylabel('μ [%]')
                    plt.subplots_adjust(wspace = 0.4)  #spacing of the two subplots
                    #---------------plot SCDC(single cycle discharge Coulomb) with cycle number
                    plt.figure()
                    plt.plot(np.arange(self.ncycle+1),self.SCDC/3600,'ro',markersize=1)
                    plt.xlabel('Cycle number');     plt.ylabel('Discharge useable capacity [Ah]')
                    plt.title(self.status_TabSurface_Scheme)
                elif ip.status_Electrical_type == 'ECN_Physical':
                    #---------------plot voltage with Ah in assigned cycles
                    import itertools
                    marker=itertools.cycle((',')) 
                    plt.figure()
                    for i0 in np.arange(np.size(ip.status_cycle_focus)):
                        step_plot=self.List_cycle2step[ip.status_cycle_focus[i0]-1];  step_plot=step_plot[~np.isnan(step_plot)].astype(int)
                        plt.plot((self.Charge_Throughput_As_Module[ step_plot-1 ]-self.Charge_Throughput_As_Module[ step_plot-1 ][0])/3600,self.U_pndiff_plot_Module[ step_plot-1 ],marker=next(marker),linewidth=2, label='Cycle %d'%ip.status_cycle_focus[i0])
                    plt.xlabel('Ah in a cycle'); plt.ylabel('Terminal voltage [V]') 
                    plt.legend()
                    #---------------plot voltage with Ah in assigned cycles
                    plt.figure()
                    for i0 in np.arange(np.size(ip.status_cycle_focus)):
                        step_plot=self.List_cycle2step[ip.status_cycle_focus[i0]-1];  step_plot=step_plot[~np.isnan(step_plot)].astype(int)
                        plt.plot((self.Charge_Throughput_As_Module[ step_plot-1 ]-self.Charge_Throughput_As_Module[ step_plot-1 ][0])/3600,self.SoC_Module[ step_plot-1 ],marker=next(marker),linewidth=2, label='Cycle %d'%ip.status_cycle_focus[i0])
                    plt.xlabel('Ah in a cycle'); plt.ylabel('Entire cell SoC') 
                    plt.legend()        
                    #---------------plot η,μ with Ah
                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.plot(self.Charge_Throughput_As_Module/3600,self.Eta_plot_Module*100,'ro',markersize=1)
                    plt.xlabel('Charge throughput [Ah]');     plt.ylabel('η [%]')
                    plt.subplot(1,3,2)
                    plt.plot(self.Charge_Throughput_As_Module/3600,self.Miu_Ca_plot_Module*100,'ro',markersize=1)
                    plt.xlabel('Charge throughput [Ah]');     plt.ylabel('μ_Cathode [%]')
                    plt.subplot(1,3,3)
                    plt.plot(self.Charge_Throughput_As_Module/3600,self.Miu_An_plot_Module*100,'ro',markersize=1)
                    plt.xlabel('Charge throughput [Ah]');     plt.ylabel('μ_Anode [%]')
                    plt.subplots_adjust(wspace = 0.4)  #spacing of the two subplots
                    #---------------plot SCDC(single cycle discharge Coulomb) with cycle number
                    plt.figure()
                    plt.plot(np.arange(self.ncycle+1),self.SCDC/3600,'ro',markersize=1)
                    plt.xlabel('Cycle number');     plt.ylabel('Discharge useable capacity [Ah]')
                    plt.title(self.status_TabSurface_Scheme)
    
        
    
    
    
    
    
    
    
    
    
    
    
