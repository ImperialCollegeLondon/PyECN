# -*- coding: utf-8 -*-

import numpy as np
import pyecn.parse_inputs as ip
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
inf=1e10

class Module_4T:
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
        #get 'self.interface_string', which is used as string only in this python module module_4T.py
        if self.status_BC_Module_4T == 'Pouch_touching':
            self.n_interface = 1
            self.status_Interfaces_name = []
            for i0_temp in range(self.n_interface):
                string_temp = f'interface_{i0_temp+1}'
                self.status_Interfaces_name.append(string_temp)        

            interface_string = {}
            for i0 in np.arange(self.n_interface):
                item = f'interface_{i0+1}'
                interface_string[item] = {}
                interface_string[item]['SideA_part_id'] = f'part_{i0+1}'
                interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_back_4T'
                interface_string[item]['SideB_part_id'] = f'part_{self.status_Parts_num_Module}'
                interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_front_4T'

        if self.status_BC_Module_4T == 'Prismatic_touching':
            self.n_interface = 1
            self.status_Interfaces_name = []
            for i0_temp in range(self.n_interface):
                string_temp = f'interface_{i0_temp+1}'
                self.status_Interfaces_name.append(string_temp)        

            interface_string = {}
            for i0 in np.arange(self.n_interface):
                item = f'interface_{i0+1}'
                interface_string[item] = {}
                interface_string[item]['SideA_part_id'] = f'part_{i0+1}'
                interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_right_4T'
                interface_string[item]['SideB_part_id'] = f'part_{self.status_Parts_num_Module}'
                interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_left_4T'

        if self.status_BC_Module_4T == 'Prismatic_Cell1':
            self.n_interface = 3
            self.status_Interfaces_name = []
            for i0_temp in range(self.n_interface):
                string_temp = f'interface_{i0_temp+1}'
                self.status_Interfaces_name.append(string_temp)        
            
            interface_string = {}
            for i0 in np.arange(self.n_interface):
                item = f'interface_{i0+1}'
                interface_string[item] = {}
                if i0 == 0:
                    interface_string[item]['SideA_part_id'] = 'part_1'
                    interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_right_4T_4SepFill'
                    interface_string[item]['SideB_part_id'] = 'part_2'
                    interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_left_4T_4SepFill'
                if i0 == 1:
                    interface_string[item]['SideA_part_id'] = 'part_1'
                    interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_left_4T_4SepFill'
                    interface_string[item]['SideB_part_id'] = 'part_3'
                    interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_interface_2_can'
                if i0 == 2:
                    interface_string[item]['SideA_part_id'] = 'part_2'
                    interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_right_4T_4SepFill'
                    interface_string[item]['SideB_part_id'] = 'part_3'
                    interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_interface_3_can'

        if self.status_BC_Module_4T == 'Ribbon_cooling':
            self.n_interface = self.status_Cells_num_Module
            self.status_Interfaces_name = []
            for i0_temp in range(self.n_interface):
                string_temp = f'interface_{i0_temp+1}'
                self.status_Interfaces_name.append(string_temp)        
            
            interface_string = {}
            for i0 in np.arange(self.n_interface):
                item = f'interface_{i0+1}'
                interface_string[item] = {}
                interface_string[item]['SideA_part_id'] = f'part_{i0+1}'
                interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_S_cooling_node38_4T'
                interface_string[item]['SideB_part_id'] = f'part_{self.status_Parts_num_Module}'
                interface_string[item]['SideB_ind0_Geo'] = f'ind0_Geo_interface_{i0+1}_ribbon'

        if self.status_BC_Module_4T == 'Pouch_weld_tab':
            self.n_interface = 4
            self.status_Interfaces_name = []
            for i0_temp in range(self.n_interface):
                string_temp = f'interface_{i0_temp+1}'
                self.status_Interfaces_name.append(string_temp)        

            interface_string = {}
            for i0 in np.arange(self.n_interface):
                item = f'interface_{i0+1}'
                interface_string[item] = {}
                if i0 == 0:
                    interface_string[item]['SideA_part_id'] = 'part_2'
                    interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_front_4T'
                    interface_string[item]['SideB_part_id'] = 'part_4'
                    interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_link_4T'
                if i0 == 1:
                    interface_string[item]['SideA_part_id'] = 'part_1'
                    interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_left_Al_4T'
                    interface_string[item]['SideB_part_id'] = 'part_2'
                    interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_left_right_4T'
                if i0 == 2:
                    interface_string[item]['SideA_part_id'] = 'part_1'
                    interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_right_Cu_4T'
                    interface_string[item]['SideB_part_id'] = 'part_3'
                    interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_left_right_4T'
                if i0 == 3:
                    interface_string[item]['SideA_part_id'] = 'part_3'
                    interface_string[item]['SideA_ind0_Geo'] = 'ind0_Geo_front_4T'
                    interface_string[item]['SideB_part_id'] = 'part_5'
                    interface_string[item]['SideB_ind0_Geo'] = 'ind0_Geo_link_4T'


        self.interface_string = interface_string
        
        self.Parts_num = len(self.Parts_name)      
        self.Parts2Cells_name = {}          #e.g. self.Parts2Cells_name: {'part_1':'cell_1', 'part_2':'cell_2', 'part_3':'part_3'} when part_1 part_2 are cells and part_3 is only thermal block
        for i0 in np.arange(self.Parts_num):       
            self.Parts2Cells_name[self.status_Parts_name[i0]] = self.status_Cells_name[i0] if i0 <= ip.status_Cells_num_Module-1 else ip.status_Parts_name[i0]
        self.Parts_str2int = {}          #e.g. self.Parts_str2int: {'part_1':1, 'part_2':2, 'part_3':3'}
        for i0 in np.arange(self.Parts_num):       
            self.Parts_str2int[self.status_Parts_name[i0]] = i0+1
        Parts_size_MatrixCN = np.array( [ o['n_4T_ALL'] for o in self.Parts_attr.values() ] ).flatten()
        self.n_4T_Module = Parts_size_MatrixCN.sum()
        self.List_Cmatind2Mmatind_4T = -9999*np.ones( [self.Parts_num, np.max( [ o['n_4T_ALL'] for o in self.Parts_attr.values()] )],dtype=int )      #List_Cmatind2Mmatind: Cell objects: (cell_1,cell_2) to Module matrixM ind0: (1,2,3...21, 22,23,24...42)-1. e.g. List_Cmatind2Mmatind: [[0,1,2,...20], 
                                                                                                                                                       #                                                                                                                                       [21,22,23,...41]]
        nbefore_temp = 0                                                                                                                    
        for i0 in np.arange(self.Parts_num):
            n_temp = self.Parts_attr[self.Parts_name[i0]]['n_4T_ALL']
            self.List_Cmatind2Mmatind_4T[i0,:n_temp] = np.arange(n_temp) + nbefore_temp
            nbefore_temp += n_temp
        
        self.interface_dict = self.fun_interface_dict()
        self.interface_dict_forBC = self.fun_interface_dict_forBC()
        ### getting Cells linking info in a Module ###
#        self.Cells_link_Module = np.zeros([self.status_Cells_num_Module,self.status_Cells_num_Module])

        ### getting Cells spatial locations in a Module ###
#        self.Cells_XYZ_Module = self.status_Cells_XYZ_Module
        ### getting initial voltage potential and I ###
#        self.Charge_Throughput_As_Module = np.zeros([self.nt+1,1])      #overall coulomb counting, in the form of 1,2...nt
#        self.U_pndiff_plot_Module = np.nan*np.zeros([self.nt+1])        #for plotting positive negative voltage difference in a Module
#        self.U_pndiff_plot_Module[0] = 0                                #initial voltage of the module does not exist, unlike cell level. In cell level, initial voltage is OCV which is stable and under no current. 
#                                                                        #in module level, even under no current, cells may inter charge and initial voltage unknown (only the first time step voltage can be calculated)
#        self.I0_record_Module=np.nan*np.zeros([self.nt+1])              #record I0
#        self.I0_record_Module[0] = 0                                    #respective to initial voltage (U_pndiff_plot_Module[0]), initial voltage and current are both meaningless

        if self.status_BC_Module_4T == 'Prismatic_Cell1':
            self.rou_c_V_weights = np.concatenate(( self.Parts_attr['part_1']['rou_c_V_weights'], self.Parts_attr['part_2']['rou_c_V_weights'] ))                                               #prep for fun_weighted_avg_and_std 

        self.t_record=self.dt*np.arange(self.nt+1)
        self.T_avg_record = np.nan*np.zeros([self.nt+1]); self.T_SD_record = np.nan*np.zeros([self.nt+1]); self.T_Delta_record = np.nan*np.zeros([self.nt+1])
    #########################################################   
    #  functions for weighted avg & std of nodes temperature#
    #########################################################
    def fun_weighted_avg_and_std(self, T, rou_c_V_weights):
        weight_avg=np.average(T, weights=self.rou_c_V_weights)
        weight_std=np.sqrt( np.average( (T - weight_avg)**2, weights=self.rou_c_V_weights ) )
        T_Delta=np.max(T)-np.min(T)
        return (weight_avg,weight_std,T_Delta)
    #########################################################   
    # function for band matrix dimension and diagonal form  #
    #########################################################
    def fun_band_matrix_precompute(self, Matrix):                                                                                           #i.g. A=np.array([[1,2,0,0,0],[2,3,4,0,0],[3,1,5,0,0],[0,7,5,8,0],[0,0,0,0,1]]), length=5
        #---------------get band matrix dimension: ind0_l and ind0_u---------------
        length=Matrix.shape[0]
        counter_l=-length+1; counter_u=length-1
        for i0 in np.linspace(-length+1,-1,length-1,dtype=int):                                                                         #i.g. for A, i0: -4,-3,-2,-1,0
            if (np.diagonal(Matrix,i0)==np.zeros([length+i0])).all():
                counter_l=counter_l+1
            else:
                break
        for i0 in np.linspace(length-1,1,length-1,dtype=int):                                                                           #i.g. for A, i0: 4,3,2,1,0
            if (np.diagonal(Matrix,i0)==np.zeros([length-i0])).all():
                counter_u=counter_u-1
            else:
                break        
        ind0_l=counter_l; ind0_u=counter_u                                                                                              #i.g. for A, ind0_u=1, ind0_l=-2
        #---------------get ind0_r and ind0_c for the diagonal ordered form---------------
        #top=np.zeros([length,length]); bottom=np.zeros([length,length]); MatrixCN_expand=np.concatenate((top,MatrixCN,bottom))         #i.g. for A, A_expand=[np.zeros([5,5]),A,np.zeros([5,5])]
        temp1=np.arange(length); length_v=abs(ind0_l)+ind0_u+1; temp2=np.arange(length_v)
        ind0_r_expand=(length-ind0_u)*np.ones([length_v,length],dtype=int)+np.tile(temp1,(length_v,1))+np.tile(temp2,(length,1)).T      #i.g. for A, ind0_r_expand=[[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10],[7,8,9,10,11]]
        ind0_c_expand=np.tile(np.arange(length),(length_v,1))                                                                           #i.g. for A, ind0_r_expand=[[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]]
        #ab=MatrixCN_expand[ind0_r_expand,ind0_c_expand]                                                                                #i.g. for A, diagonal ordered form is ab=[[0,2,4,0,0],[1,3,5,8,1],[2,1,5,0,0],[3,7,0,0,0]]
        return length, abs(ind0_l), ind0_u, ind0_r_expand, ind0_c_expand            
    #########################################################   
    ######      function for interface attr dict      #######
    #########################################################
    def fun_interface_dict(self):              #to get 'self.interface_dict', which is passed to main function loop.py to generate 'interface_attr' for each cell object
        if self.status_BC_Module_4T == 'Pouch_touching':
            interface_dict = {}                           #thermal parts linking info in dict form
            interface_dict['interface_1'] = {}
            interface_dict['interface_1']['part_1'] = ['ind0_Geo_back_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_1']['part_2'] = ['ind0_Geo_front_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
        if self.status_BC_Module_4T == 'Prismatic_touching':
            interface_dict = {}                           #thermal parts linking info in dict form
            interface_dict['interface_1'] = {}
            interface_dict['interface_1']['part_1'] = ['ind0_Geo_right_4T_4SepFill',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
            interface_dict['interface_1']['part_2'] = ['ind0_Geo_left_4T_4SepFill',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
        if self.status_BC_Module_4T == 'Prismatic_Cell1':
            interface_dict = {}                           #thermal parts linking info in dict form
            interface_dict['interface_1'] = {}
            interface_dict['interface_1']['part_1'] = ['ind0_Geo_right_4T_4SepFill',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
            interface_dict['interface_1']['part_2'] = ['ind0_Geo_left_4T_4SepFill',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
            interface_dict['interface_2'] = {}
            interface_dict['interface_2']['part_1'] = ['ind0_Geo_left_4T_4SepFill',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
            interface_dict['interface_2']['part_3'] = ['ind0_Geo_interface_2_can',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Can_real','Lamda_Can',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
            interface_dict['interface_3'] = {}
            interface_dict['interface_3']['part_2'] = ['ind0_Geo_right_4T_4SepFill',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Cu','delta_Al','Lamda_Cu','Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
            interface_dict['interface_3']['part_3'] = ['ind0_Geo_interface_3_can',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'delta_Can_real','Lamda_Can',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL',
                                                       'delta_Mylar','Lamda_Mylar']
        if self.status_BC_Module_4T == 'Ribbon_cooling':
            interface_dict = {}                           #thermal parts linking info in dict form
            for i0 in np.arange(self.n_interface):
                item_temp1 = list(self.interface_string.keys())[i0]                 #'interface_1'
                interface_dict[item_temp1] = {}
                for j0 in np.arange(self.n_interface):
                    if j0 == 0:
                        item_temp2 = self.interface_string[item_temp1]['SideA_part_id']     #'part_1'
                        item_temp3 = self.interface_string[item_temp1]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    if j0 == 1:
                        item_temp2 = self.interface_string[item_temp1]['SideB_part_id']     #'part_3'
                        item_temp3 = self.interface_string[item_temp1]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'
                    if self.Parts_str2int[item_temp2]-1 <= self.status_Parts_num_Module-2:
                        interface_dict[item_temp1][item_temp2] = [item_temp3,
                                                                  'Lamda_4T_ALL','RouXc_4T_ALL',
                                                                  'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                                  'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                                  'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                                  'delta_Can_real','Lamda_Can',
                                                                  'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
                    if self.Parts_str2int[item_temp2]-1 == self.status_Parts_num_Module-1:
                         interface_dict[item_temp1][item_temp2] = [item_temp3,
                                                                  'Lamda_4T_ALL','RouXc_4T_ALL',
                                                                  'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                                  'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                                  'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                                  'delta_ribbon','Lamda_ribbon',
                                                                  'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
       
        if self.status_BC_Module_4T == 'Pouch_weld_tab':
            interface_dict = {}                           #thermal parts linking info in dict form
            interface_dict['interface_1'] = {}
            interface_dict['interface_1']['part_2'] = ['ind0_Geo_front_4T','ind0_Geo_node2_4_6_9_11_13_4T','ind0_Geo_node1_7_4T','ind0_Geo_node3_5_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'weld_Lamda_pos','nonweld_tab_Lamda',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_1']['part_4'] = ['ind0_Geo_link_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'tab_Lamda_pos',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_2'] = {}
            interface_dict['interface_2']['part_1'] = ['ind0_Geo_left_Al_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'Lamda_Al',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_2']['part_2'] = ['ind0_Geo_left_right_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'weld_Lamda_pos',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_3'] = {}
            interface_dict['interface_3']['part_1'] = ['ind0_Geo_right_Cu_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'Lamda_Cu',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_3']['part_3'] = ['ind0_Geo_left_right_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'weld_Lamda_neg',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_4'] = {}
            interface_dict['interface_4']['part_3'] = ['ind0_Geo_front_4T','ind0_Geo_node2_4_6_9_11_13_4T','ind0_Geo_node1_7_4T','ind0_Geo_node3_5_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'weld_Lamda_neg','nonweld_tab_Lamda',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict['interface_4']['part_5'] = ['ind0_Geo_link_4T',
                                                       'Lamda_4T_ALL','RouXc_4T_ALL',
                                                       'jx1_4T_ALL','jx2_4T_ALL','jy1_4T_ALL','jy2_4T_ALL','jz1_4T_ALL','jz2_4T_ALL',
                                                       'delta_x1_4T_ALL','delta_x2_4T_ALL','delta_y1_4T_ALL','delta_y2_4T_ALL','delta_z1_4T_ALL','delta_z2_4T_ALL',
                                                       'Delta_x1_4T_ALL','Delta_x2_4T_ALL','Delta_y1_4T_ALL','Delta_y2_4T_ALL','Delta_z1_4T_ALL','Delta_z2_4T_ALL',
                                                       'tab_Lamda_neg',
                                                       'T_initial','Tini_4T_ALL','h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
       
        return interface_dict
    def fun_interface_dict_forBC(self):             #same as 'self.interface_dict', but a subsection, to be used in each time step
        if self.status_BC_Module_4T == 'Pouch_touching':
            interface_dict_forBC = {}                           #thermal parts linking info in dict form
            interface_dict_forBC['interface_1'] = {}
            interface_dict_forBC['interface_1']['part_1'] = ['ind0_Geo_back_4T',
                                                             'h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict_forBC['interface_1']['part_2'] = ['ind0_Geo_front_4T',
                                                             'h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
        if self.status_BC_Module_4T == 'Prismatic_touching':
            interface_dict_forBC = {}                           #thermal parts linking info in dict form
            interface_dict_forBC['interface_1'] = {}
            interface_dict_forBC['interface_1']['part_1'] = ['ind0_Geo_right_4T_4SepFill',
                                                             'h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            interface_dict_forBC['interface_1']['part_2'] = ['ind0_Geo_left_4T_4SepFill',
                                                             'h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
        if self.status_BC_Module_4T == 'Prismatic_Cell1':
            interface_dict_forBC = {}                           #thermal parts linking info in dict form
            for i0 in np.arange(self.n_interface):
                item_temp1 = list(self.interface_string.keys())[i0]                 #'interface_1'
                interface_dict_forBC[item_temp1] = {}
                for j0 in np.arange(2):
                    if j0 == 0:
                        item_temp2 = self.interface_string[item_temp1]['SideA_part_id']     #'part_1'
                        item_temp3 = self.interface_string[item_temp1]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    if j0 == 1:
                        item_temp2 = self.interface_string[item_temp1]['SideB_part_id']     #'part_3'
                        item_temp3 = self.interface_string[item_temp1]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'
                    interface_dict_forBC[item_temp1][item_temp2] = [item_temp3,
                                                                    'h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']

        if self.status_BC_Module_4T == 'Ribbon_cooling':
            interface_dict_forBC = {}                           #thermal parts linking info in dict form
            for i0 in np.arange(self.n_interface):
                item_temp1 = list(self.interface_string.keys())[i0]                 #'interface_1'
                interface_dict_forBC[item_temp1] = {}
                for j0 in np.arange(2):
                    if j0 == 0:
                        item_temp2 = self.interface_string[item_temp1]['SideA_part_id']     #'part_1'
                        item_temp3 = self.interface_string[item_temp1]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    if j0 == 1:
                        item_temp2 = self.interface_string[item_temp1]['SideB_part_id']     #'part_3'
                        item_temp3 = self.interface_string[item_temp1]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'
                    interface_dict_forBC[item_temp1][item_temp2] = [item_temp3,
                                                                    'h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']
            
        if self.status_BC_Module_4T == 'Pouch_weld_tab':
            interface_dict_forBC = {}                           #thermal parts linking info in dict form
            for i0 in np.arange(self.n_interface):
                item_temp1 = list(self.interface_string.keys())[i0]                 #'interface_1'
                interface_dict_forBC[item_temp1] = {}
                for j0 in np.arange(2):
                    if j0 == 0:
                        item_temp2 = self.interface_string[item_temp1]['SideA_part_id']     #'part_1'
                        item_temp3 = self.interface_string[item_temp1]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    if j0 == 1:
                        item_temp2 = self.interface_string[item_temp1]['SideB_part_id']     #'part_3'
                        item_temp3 = self.interface_string[item_temp1]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'
                    interface_dict_forBC[item_temp1][item_temp2] = [item_temp3,
                                                                    'h_4T_ALL','Tconv_4T_ALL','ind0_BCtem_ALL']

        return interface_dict_forBC
    #########################################################   
    ######   function for Thermal initial condition   #######
    #########################################################
    def fun_IC_4T_Module(self,params_C2M):
        #---------------------------  append Tini_4T_ALL(s) to form Tini_4T_Module ---------------------------
        item = self.status_Parts_name[0]
        for j0 in self.status_Interfaces_name:
            if item in params_C2M[j0]:
                Tini_4T_Module = params_C2M[j0][item]['Tini_4T_ALL']                       #loop to find part_1
        for i0 in self.status_Parts_name[1:]:
            for j0 in self.status_Interfaces_name:
                if i0 in params_C2M[j0]:
                    Tini_4T_Module_temp = params_C2M[j0][i0]['Tini_4T_ALL']
            Tini_4T_Module = np.append( Tini_4T_Module, Tini_4T_Module_temp, axis=0 )      #loop to append part_2, part_3...

        self.fun_BC_4T_Module(params_C2M)

        return Tini_4T_Module
    #########################################################   
    #########    function for Thermal module BC    ##########
    #########################################################
    def fun_BC_4T_Module(self,params_C2M):    
        if self.status_BC_Module_4T == 'Pouch_touching':     
            
            #h_4T_Module
            h_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side,1+6])
            h_temp = params_C2M['interface_1']['part_1']['h_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            h_4T_Module[:self.n_nodes_1side,0] = self.ind0_interfaces_Module_4T[:self.n_nodes_1side]
            h_4T_Module[:self.n_nodes_1side,1:] = h_temp[ind0_nodes_temp,:]
            h_4T_Module[:self.n_nodes_1side,1+4] = np.nan
    
            h_temp = params_C2M['interface_1']['part_2']['h_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            h_4T_Module[self.n_nodes_1side:,0] = self.ind0_interfaces_Module_4T[self.n_nodes_1side:]
            h_4T_Module[self.n_nodes_1side:,1:] = h_temp[ind0_nodes_temp,:]
            h_4T_Module[self.n_nodes_1side:,1+5] = np.nan
            
            #Tconv_4T_Module
            Tconv_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side,1+6])
            Tconv_temp = params_C2M['interface_1']['part_1']['Tconv_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Tconv_4T_Module[:self.n_nodes_1side,0] = self.ind0_interfaces_Module_4T[:self.n_nodes_1side]
            Tconv_4T_Module[:self.n_nodes_1side,1:] = Tconv_temp[ind0_nodes_temp,:]
            Tconv_4T_Module[:self.n_nodes_1side,1+4] = np.nan
    
            Tconv_temp = params_C2M['interface_1']['part_2']['Tconv_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Tconv_4T_Module[self.n_nodes_1side:,0] = self.ind0_interfaces_Module_4T[self.n_nodes_1side:]
            Tconv_4T_Module[self.n_nodes_1side:,1:] = Tconv_temp[ind0_nodes_temp,:]
            Tconv_4T_Module[self.n_nodes_1side:,1+5] = np.nan

            #-------------ind0_BCtem_4T_Module
            ind0_BCtem_4T_Module = np.array([],dtype=int) 

        if self.status_BC_Module_4T == 'Prismatic_touching':     
            
            #h_4T_Module
            h_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side,1+6])
            h_temp = params_C2M['interface_1']['part_1']['h_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            h_4T_Module[:self.n_nodes_1side,0] = self.ind0_interfaces_Module_4T[:self.n_nodes_1side]
            h_4T_Module[:self.n_nodes_1side,1:] = h_temp[ind0_nodes_temp,:]
            h_4T_Module[:self.n_nodes_1side,1+5] = np.nan
    
            h_temp = params_C2M['interface_1']['part_2']['h_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            h_4T_Module[self.n_nodes_1side:,0] = self.ind0_interfaces_Module_4T[self.n_nodes_1side:]
            h_4T_Module[self.n_nodes_1side:,1:] = h_temp[ind0_nodes_temp,:]
            h_4T_Module[self.n_nodes_1side:,1+5] = np.nan
            
            #Tconv_4T_Module
            Tconv_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side,1+6])
            Tconv_temp = params_C2M['interface_1']['part_1']['Tconv_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Tconv_4T_Module[:self.n_nodes_1side,0] = self.ind0_interfaces_Module_4T[:self.n_nodes_1side]
            Tconv_4T_Module[:self.n_nodes_1side,1:] = Tconv_temp[ind0_nodes_temp,:]
            Tconv_4T_Module[:self.n_nodes_1side,1+5] = np.nan
    
            Tconv_temp = params_C2M['interface_1']['part_2']['Tconv_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Tconv_4T_Module[self.n_nodes_1side:,0] = self.ind0_interfaces_Module_4T[self.n_nodes_1side:]
            Tconv_4T_Module[self.n_nodes_1side:,1:] = Tconv_temp[ind0_nodes_temp,:]
            Tconv_4T_Module[self.n_nodes_1side:,1+5] = np.nan

            #-------------ind0_BCtem_4T_Module
            ind0_BCtem_4T_Module = np.array([],dtype=int) 

        if self.status_BC_Module_4T == 'Prismatic_Cell1':     
            
            h_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side_1d.sum(),1+6])
            Tconv_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side_1d.sum(),1+6])
            ind0_BCtem_4T_Module = np.array([],dtype=int)
            for i0 in np.arange(self.n_interface):
                #-------------h_4T_Module
                #SideA
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                h_temp = params_C2M[item_temp0][item_temp1]['h_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
    
                local_temp1 = np.append( 0, np.cumsum(self.n_nodes_1side_1d))
                local_temp2 = self.n_nodes_1side_1d[i0]
                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,0] = self.ind0_interfacesSideA_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1:] = h_temp[ind0_nodes_temp,:]
#                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1+5] = np.nan

                #SideB
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                h_temp = params_C2M[item_temp0][item_temp1]['h_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]

                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,0] = self.ind0_interfacesSideB_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1:] = h_temp[ind0_nodes_temp,:]
#                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1+5] = np.nan

                #-------------Tconv_4T_Module
                #SideA
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Tconv_temp = params_C2M[item_temp0][item_temp1]['Tconv_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
    
                local_temp1 = np.append( 0, np.cumsum(self.n_nodes_1side_1d))
                local_temp2 = self.n_nodes_1side_1d[i0]
                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,0] = self.ind0_interfacesSideA_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1:] = Tconv_temp[ind0_nodes_temp,:]
#                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1+5] = np.nan

                #SideB
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                Tconv_temp = params_C2M[item_temp0][item_temp1]['Tconv_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]

                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,0] = self.ind0_interfacesSideB_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1:] = Tconv_temp[ind0_nodes_temp,:]
#                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1+5] = np.nan

                #-------------ind0_BCtem_4T_Module
                if i0 == 1:
                    item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    item_temp3 = params_C2M[item_temp0][item_temp1]['ind0_BCtem_ALL']

                    ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                    ind0_BCtem_4T_Module = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, item_temp3 ] 

        if self.status_BC_Module_4T == 'Ribbon_cooling':

            h_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side_1d.sum(),1+6])
            Tconv_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side_1d.sum(),1+6])
            for i0 in np.arange(self.n_interface):
                #-------------h_4T_Module
                #SideA
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                h_temp = params_C2M[item_temp0][item_temp1]['h_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
    
                local_temp1 = np.append( 0, np.cumsum(self.n_nodes_1side_1d))
                local_temp2 = self.n_nodes_1side_1d[i0]
                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,0] = self.ind0_interfacesSideA_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1:] = h_temp[ind0_nodes_temp,:]
                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1+5] = np.nan

                #SideB
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                h_temp = params_C2M[item_temp0][item_temp1]['h_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]

                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,0] = self.ind0_interfacesSideB_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1:] = h_temp[ind0_nodes_temp,:]
                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1+5] = np.nan

                #-------------Tconv_4T_Module
                #SideA
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Tconv_temp = params_C2M[item_temp0][item_temp1]['Tconv_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
    
                local_temp1 = np.append( 0, np.cumsum(self.n_nodes_1side_1d))
                local_temp2 = self.n_nodes_1side_1d[i0]
                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,0] = self.ind0_interfacesSideA_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1:] = Tconv_temp[ind0_nodes_temp,:]
                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1+5] = np.nan

                #SideB
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                Tconv_temp = params_C2M[item_temp0][item_temp1]['Tconv_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]

                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,0] = self.ind0_interfacesSideB_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1:] = Tconv_temp[ind0_nodes_temp,:]
                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1+5] = np.nan

            #-------------ind0_BCtem_4T_Module
            ind0_BCtem_4T_Module = np.array([],dtype=int) 

        if self.status_BC_Module_4T == 'Pouch_weld_tab':     
            
            h_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side_1d.sum(),1+6])
            Tconv_4T_Module = np.nan * np.zeros([2*self.n_nodes_1side_1d.sum(),1+6])
            ind0_BCtem_4T_Module = np.array([],dtype=int)
            for i0 in np.arange(self.n_interface):
                #-------------h_4T_Module
                #SideA
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                h_temp = params_C2M[item_temp0][item_temp1]['h_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
    
                local_temp1 = np.append( 0, np.cumsum(self.n_nodes_1side_1d))
                local_temp2 = self.n_nodes_1side_1d[i0]
                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,0] = self.ind0_interfacesSideA_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1:] = h_temp[ind0_nodes_temp,:]
#                h_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1+5] = np.nan

                #SideB
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                h_temp = params_C2M[item_temp0][item_temp1]['h_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]

                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,0] = self.ind0_interfacesSideB_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1:] = h_temp[ind0_nodes_temp,:]
#                h_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1+5] = np.nan

                #-------------Tconv_4T_Module
                #SideA
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Tconv_temp = params_C2M[item_temp0][item_temp1]['Tconv_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
    
                local_temp1 = np.append( 0, np.cumsum(self.n_nodes_1side_1d))
                local_temp2 = self.n_nodes_1side_1d[i0]
                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,0] = self.ind0_interfacesSideA_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1:] = Tconv_temp[ind0_nodes_temp,:]
#                Tconv_4T_Module[local_temp1[i0]*2:local_temp2+local_temp1[i0]*2,1+5] = np.nan

                #SideB
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                Tconv_temp = params_C2M[item_temp0][item_temp1]['Tconv_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]

                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,0] = self.ind0_interfacesSideB_Module_4T[local_temp1[i0]:local_temp2+local_temp1[i0]]
                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1:] = Tconv_temp[ind0_nodes_temp,:]
#                Tconv_4T_Module[self.n_nodes_1side_1d[i0]+local_temp1[i0]*2:local_temp2+self.n_nodes_1side_1d[i0]+local_temp1[i0]*2,1+5] = np.nan

                #-------------ind0_BCtem_4T_Module
                if i0 == 1:
                    item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    item_temp3 = params_C2M[item_temp0][item_temp1]['ind0_BCtem_ALL']

                    ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                    ind0_BCtem_4T_Module = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, item_temp3 ] 


        self.h_4T_Module = h_4T_Module
        self.Tconv_4T_Module = Tconv_4T_Module            
        self.ind0_BCtem_4T_Module = ind0_BCtem_4T_Module            
    #########################################################   
    #########    function for pre Thermal module    #########
    #########################################################
    def fun_pre_Thermal_Module(self,params_C2M):    
        if self.status_BC_Module_4T == 'Pouch_touching':     
            #interface_1: part_1-ind0_Geo_back_4T & part_2-ind0_Geo_front_4T
            ind0_Part_temp = 0
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            prep_temp1 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 5,6,11,12     

            ind0_Part_temp = 1
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            prep_temp2 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 13,14,19,20      

            ind0_interfaces_Module_4T = np.concatenate(( prep_temp1, 
                                                         prep_temp2 ))                        #all nodes to be modified in MatrixM_4T. e.g. nodes 5,6,11,12, 13,14,19,20
            ind0_interfacesSideA_Module_4T = prep_temp1
            ind0_interfacesSideB_Module_4T = prep_temp2
            
            n_nodes_1side = np.size(prep_temp1)                                               #number of interface nodes (single side)
            self.n_nodes_1side = n_nodes_1side
            ########################################################################################################
            ###############################          prep Thermal interfaces          ##############################
            ########################################################################################################
            delta_Cu = params_C2M['interface_1']['part_1']['delta_Cu']
            delta_Al = params_C2M['interface_1']['part_1']['delta_Al']
            Lamda_Cu = params_C2M['interface_1']['part_1']['Lamda_Cu']
            Lamda_Al = params_C2M['interface_1']['part_1']['Lamda_Al']

            #jx1_4T_Module
            Parts_link_jx1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jx1_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_back_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jx1_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jx1_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_front_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp

            #jx2_4T_Module
            Parts_link_jx2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jx2_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_back_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jx2_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jx2_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_front_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp

            #jy1_4T_Module
            Parts_link_jy1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jy1_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_back_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jy1_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jy1_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_front_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jy1_4T_Module[n_nodes_1side:,1] = prep_temp

            #jy2_4T_Module
            Parts_link_jy2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jy2_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_back_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jy2_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jy2_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_front_4T'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jy2_4T_Module[n_nodes_1side:,1] = prep_temp

            #jz1_4T_Module
            Parts_link_jz1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 1                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1            
            Parts_link_jz1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jz1_4T_Module[:n_nodes_1side,1] = prep_temp
            
            ind0_Cell_temp = 1                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['jz1_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_front_4T'] ] -1    
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 14,15,20,21
            Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jz1_4T_Module[n_nodes_1side:,1] = prep_temp
            
            #jz2_4T_Module
            Parts_link_jz2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['jz2_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_back_4T'] ] -1
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 3,4,9,10
            Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp
            
            ind0_Cell_temp = 0                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 4,5,10,11            
            Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp
            
            #Lamda_4T_Module
            Parts_link_Lamda_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])
            Lamda_temp = params_C2M['interface_1']['part_1']['Lamda_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]
            Parts_link_Lamda_4T_Module[:n_nodes_1side,1+4] = (0.5*delta_Cu+0.5*delta_Al) / (0.5*delta_Cu/Lamda_Cu + 0.5*delta_Al/Lamda_Al)
            
            Lamda_temp = params_C2M['interface_1']['part_2']['Lamda_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
            Parts_link_Lamda_4T_Module[n_nodes_1side:,1+5] = (0.5*delta_Cu+0.5*delta_Al) / (0.5*delta_Cu/Lamda_Cu + 0.5*delta_Al/Lamda_Al)

            #Mis_4T_Module
            Parts_link_Mis_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])
            Parts_link_Mis_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Mis_4T_Module[:n_nodes_1side,1:] = 1                                        

            Parts_link_Mis_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Mis_4T_Module[n_nodes_1side:,1:] = 1

            #RouXc_4T_Module
            Parts_link_RouXc_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            RouXc_temp = params_C2M['interface_1']['part_1']['RouXc_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_RouXc_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_RouXc_4T_Module[:n_nodes_1side,1] = RouXc_temp[ind0_nodes_temp]

            RouXc_temp = params_C2M['interface_1']['part_2']['RouXc_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_RouXc_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_RouXc_4T_Module[n_nodes_1side:,1] = RouXc_temp[ind0_nodes_temp]

            #delta_x1_4T_Module
            Parts_link_delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_x1_temp = params_C2M['interface_1']['part_1']['delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_x1_4T_Module[:n_nodes_1side,1] = delta_x1_temp[ind0_nodes_temp]

            delta_x1_temp = params_C2M['interface_1']['part_2']['delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_x1_4T_Module[n_nodes_1side:,1] = delta_x1_temp[ind0_nodes_temp]

            #delta_x2_4T_Module
            Parts_link_delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_x2_temp = params_C2M['interface_1']['part_1']['delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_x2_4T_Module[:n_nodes_1side,1] = delta_x2_temp[ind0_nodes_temp]

            delta_x2_temp = params_C2M['interface_1']['part_2']['delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_x2_4T_Module[n_nodes_1side:,1] = delta_x2_temp[ind0_nodes_temp]

            #delta_y1_4T_Module
            Parts_link_delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_y1_temp = params_C2M['interface_1']['part_1']['delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_y1_4T_Module[:n_nodes_1side,1] = delta_y1_temp[ind0_nodes_temp]

            delta_y1_temp = params_C2M['interface_1']['part_2']['delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_y1_4T_Module[n_nodes_1side:,1] = delta_y1_temp[ind0_nodes_temp]

            #delta_y2_4T_Module
            Parts_link_delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_y2_temp = params_C2M['interface_1']['part_1']['delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_y2_4T_Module[:n_nodes_1side,1] = delta_y2_temp[ind0_nodes_temp]

            delta_y2_temp = params_C2M['interface_1']['part_2']['delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_y2_4T_Module[n_nodes_1side:,1] = delta_y2_temp[ind0_nodes_temp]

            #delta_z1_4T_Module
            Parts_link_delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_z1_temp = params_C2M['interface_1']['part_1']['delta_z1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_z1_4T_Module[:n_nodes_1side,1] = delta_z1_temp[ind0_nodes_temp]

            delta_z1_temp = params_C2M['interface_1']['part_2']['delta_z1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_z1_4T_Module[n_nodes_1side:,1] = delta_z1_temp[ind0_nodes_temp]

            #delta_z2_4T_Module
            Parts_link_delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_z2_temp = params_C2M['interface_1']['part_1']['delta_z2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_z2_4T_Module[:n_nodes_1side,1] = delta_z2_temp[ind0_nodes_temp]

            delta_z2_temp = params_C2M['interface_1']['part_2']['delta_z2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_z2_4T_Module[n_nodes_1side:,1] = delta_z2_temp[ind0_nodes_temp]
            
            #Delta_x1_4T_Module
            Parts_link_Delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_x1_temp = params_C2M['interface_1']['part_1']['Delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_Delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_x1_4T_Module[:n_nodes_1side,1] = Delta_x1_temp[ind0_nodes_temp]

            Delta_x1_temp = params_C2M['interface_1']['part_2']['Delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_Delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_x1_4T_Module[n_nodes_1side:,1] = Delta_x1_temp[ind0_nodes_temp]

            #Delta_x2_4T_Module
            Parts_link_Delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_x2_temp = params_C2M['interface_1']['part_1']['Delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_Delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_x2_4T_Module[:n_nodes_1side,1] = Delta_x2_temp[ind0_nodes_temp]

            Delta_x2_temp = params_C2M['interface_1']['part_2']['Delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_Delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_x2_4T_Module[n_nodes_1side:,1] = Delta_x2_temp[ind0_nodes_temp]

            #Delta_y1_4T_Module
            Parts_link_Delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_y1_temp = params_C2M['interface_1']['part_1']['Delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_Delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_y1_4T_Module[:n_nodes_1side,1] = Delta_y1_temp[ind0_nodes_temp]

            Delta_y1_temp = params_C2M['interface_1']['part_2']['Delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_Delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_y1_4T_Module[n_nodes_1side:,1] = Delta_y1_temp[ind0_nodes_temp]

            #Delta_y2_4T_Module
            Parts_link_Delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_y2_temp = params_C2M['interface_1']['part_1']['Delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_Delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_y2_4T_Module[:n_nodes_1side,1] = Delta_y2_temp[ind0_nodes_temp]

            Delta_y2_temp = params_C2M['interface_1']['part_2']['Delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_Delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_y2_4T_Module[n_nodes_1side:,1] = Delta_y2_temp[ind0_nodes_temp]

            #Delta_z1_4T_Module
            Parts_link_Delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = 0.5*delta_Cu+0.5*delta_Al

            Delta_z1_temp = params_C2M['interface_1']['part_2']['Delta_z1_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_front_4T']
            Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = Delta_z1_temp[ind0_nodes_temp]

            #Delta_z2_4T_Module
            Parts_link_Delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_z2_temp = params_C2M['interface_1']['part_1']['Delta_z2_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_back_4T']
            Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = Delta_z2_temp[ind0_nodes_temp]

            Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = 0.5*delta_Cu+0.5*delta_Al
            
        if self.status_BC_Module_4T == 'Prismatic_touching':     
            #interface_1: part_1-ind0_Geo_back_4T & part_2-ind0_Geo_front_4T
            ind0_Part_temp = 0
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            prep_temp1 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 5,6,11,12     

            ind0_Part_temp = 1
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            prep_temp2 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 13,14,19,20      

            ind0_interfaces_Module_4T = np.concatenate(( prep_temp1, 
                                                         prep_temp2 ))                        #all nodes to be modified in MatrixM_4T. e.g. nodes 5,6,11,12, 13,14,19,20
            ind0_interfacesSideA_Module_4T = prep_temp1
            ind0_interfacesSideB_Module_4T = prep_temp2
            
            n_nodes_1side = np.size(prep_temp1)                                               #number of interface nodes (single side)
            self.n_nodes_1side = n_nodes_1side
            ########################################################################################################
            ###############################          prep Thermal interfaces          ##############################
            ########################################################################################################
            delta_Cu = params_C2M['interface_1']['part_1']['delta_Cu']
            delta_Al = params_C2M['interface_1']['part_1']['delta_Al']
            Lamda_Cu = params_C2M['interface_1']['part_1']['Lamda_Cu']
            Lamda_Al = params_C2M['interface_1']['part_1']['Lamda_Al']

            #jx1_4T_Module
            Parts_link_jx1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jx1_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jx1_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jx1_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp

            #jx2_4T_Module
            Parts_link_jx2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jx2_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jx2_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jx2_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp

            #jy1_4T_Module
            Parts_link_jy1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jy1_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jy1_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jy1_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jy1_4T_Module[n_nodes_1side:,1] = prep_temp

            #jy2_4T_Module
            Parts_link_jy2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind_nodes_temp = params_C2M['interface_1']['part_1']['jy2_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jy2_4T_Module[:n_nodes_1side,1] = prep_temp

            ind0_Cell_temp = 1                                                
            ind_nodes_temp = params_C2M['interface_1']['part_2']['jy2_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill'] ]
            TF_temp = ind_nodes_temp != -9999
            prep_temp = np.zeros([n_nodes_1side],dtype=int)
            prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
            prep_temp[~TF_temp] = -9999
            Parts_link_jy2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jy2_4T_Module[n_nodes_1side:,1] = prep_temp

            #jz1_4T_Module
            Parts_link_jz1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 0                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['jz1_4T_ALL'][ params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill'] ] -1    
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1            
            Parts_link_jz1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jz1_4T_Module[:n_nodes_1side,1] = prep_temp
            
            ind0_Cell_temp = 1                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['jz1_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill'] ] -1    
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 14,15,20,21
            Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jz1_4T_Module[n_nodes_1side:,1] = prep_temp
            
            #jz2_4T_Module
            Parts_link_jz2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
            ind0_Cell_temp = 1                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            ind0_nodes_temp = np.flip(ind0_nodes_temp.reshape(self.ny,-1),axis=1).reshape(-1)    #flip order
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 3,4,9,10
            Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp
            
            ind0_Cell_temp = 0                                                
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            ind0_nodes_temp = np.flip(ind0_nodes_temp.reshape(self.ny,-1),axis=1).reshape(-1)    #flip order
            prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 4,5,10,11            
            Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp
            
            #Lamda_4T_Module
            Parts_link_Lamda_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])
            #Mylar layer (if there is)
            delta_Mylar = params_C2M['interface_1']['part_1']['delta_Mylar']
            Lamda_Mylar = params_C2M['interface_1']['part_1']['Lamda_Mylar']

            Lamda_temp = params_C2M['interface_1']['part_1']['Lamda_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]
            Parts_link_Lamda_4T_Module[:n_nodes_1side,1+5] = (0.5*delta_Cu+0.5*delta_Cu+delta_Mylar) / (0.5*delta_Cu/Lamda_Cu + 0.5*delta_Cu/Lamda_Cu + delta_Mylar/Lamda_Mylar)
            
            Lamda_temp = params_C2M['interface_1']['part_2']['Lamda_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
            Parts_link_Lamda_4T_Module[n_nodes_1side:,1+5] = (0.5*delta_Cu+0.5*delta_Cu+delta_Mylar) / (0.5*delta_Cu/Lamda_Cu + 0.5*delta_Cu/Lamda_Cu + delta_Mylar/Lamda_Mylar)

            #Mis_4T_Module
            Parts_link_Mis_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])
            Parts_link_Mis_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Mis_4T_Module[:n_nodes_1side,1:] = 1                                        

            Parts_link_Mis_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Mis_4T_Module[n_nodes_1side:,1:] = 1

            #RouXc_4T_Module
            Parts_link_RouXc_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            RouXc_temp = params_C2M['interface_1']['part_1']['RouXc_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_RouXc_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_RouXc_4T_Module[:n_nodes_1side,1] = RouXc_temp[ind0_nodes_temp]

            RouXc_temp = params_C2M['interface_1']['part_2']['RouXc_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_RouXc_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_RouXc_4T_Module[n_nodes_1side:,1] = RouXc_temp[ind0_nodes_temp]

            #delta_x1_4T_Module
            Parts_link_delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_x1_temp = params_C2M['interface_1']['part_1']['delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_x1_4T_Module[:n_nodes_1side,1] = delta_x1_temp[ind0_nodes_temp]

            delta_x1_temp = params_C2M['interface_1']['part_2']['delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_x1_4T_Module[n_nodes_1side:,1] = delta_x1_temp[ind0_nodes_temp]

            #delta_x2_4T_Module
            Parts_link_delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_x2_temp = params_C2M['interface_1']['part_1']['delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_x2_4T_Module[:n_nodes_1side,1] = delta_x2_temp[ind0_nodes_temp]

            delta_x2_temp = params_C2M['interface_1']['part_2']['delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_x2_4T_Module[n_nodes_1side:,1] = delta_x2_temp[ind0_nodes_temp]

            #delta_y1_4T_Module
            Parts_link_delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_y1_temp = params_C2M['interface_1']['part_1']['delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_y1_4T_Module[:n_nodes_1side,1] = delta_y1_temp[ind0_nodes_temp]

            delta_y1_temp = params_C2M['interface_1']['part_2']['delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_y1_4T_Module[n_nodes_1side:,1] = delta_y1_temp[ind0_nodes_temp]

            #delta_y2_4T_Module
            Parts_link_delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_y2_temp = params_C2M['interface_1']['part_1']['delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_y2_4T_Module[:n_nodes_1side,1] = delta_y2_temp[ind0_nodes_temp]

            delta_y2_temp = params_C2M['interface_1']['part_2']['delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_y2_4T_Module[n_nodes_1side:,1] = delta_y2_temp[ind0_nodes_temp]

            #delta_z1_4T_Module
            Parts_link_delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_z1_temp = params_C2M['interface_1']['part_1']['delta_z1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_z1_4T_Module[:n_nodes_1side,1] = delta_z1_temp[ind0_nodes_temp]

            delta_z1_temp = params_C2M['interface_1']['part_2']['delta_z1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_z1_4T_Module[n_nodes_1side:,1] = delta_z1_temp[ind0_nodes_temp]

            #delta_z2_4T_Module
            Parts_link_delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            delta_z2_temp = params_C2M['interface_1']['part_1']['delta_z2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_delta_z2_4T_Module[:n_nodes_1side,1] = delta_z2_temp[ind0_nodes_temp]

            delta_z2_temp = params_C2M['interface_1']['part_2']['delta_z2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_delta_z2_4T_Module[n_nodes_1side:,1] = delta_z2_temp[ind0_nodes_temp]
            
            #Delta_x1_4T_Module
            Parts_link_Delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_x1_temp = params_C2M['interface_1']['part_1']['Delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_Delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_x1_4T_Module[:n_nodes_1side,1] = Delta_x1_temp[ind0_nodes_temp]

            Delta_x1_temp = params_C2M['interface_1']['part_2']['Delta_x1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_Delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_x1_4T_Module[n_nodes_1side:,1] = Delta_x1_temp[ind0_nodes_temp]

            #Delta_x2_4T_Module
            Parts_link_Delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_x2_temp = params_C2M['interface_1']['part_1']['Delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_Delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_x2_4T_Module[:n_nodes_1side,1] = Delta_x2_temp[ind0_nodes_temp]

            Delta_x2_temp = params_C2M['interface_1']['part_2']['Delta_x2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_Delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_x2_4T_Module[n_nodes_1side:,1] = Delta_x2_temp[ind0_nodes_temp]

            #Delta_y1_4T_Module
            Parts_link_Delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_y1_temp = params_C2M['interface_1']['part_1']['Delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_Delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_y1_4T_Module[:n_nodes_1side,1] = Delta_y1_temp[ind0_nodes_temp]

            Delta_y1_temp = params_C2M['interface_1']['part_2']['Delta_y1_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_Delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_y1_4T_Module[n_nodes_1side:,1] = Delta_y1_temp[ind0_nodes_temp]

            #Delta_y2_4T_Module
            Parts_link_Delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_y2_temp = params_C2M['interface_1']['part_1']['Delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_Delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_y2_4T_Module[:n_nodes_1side,1] = Delta_y2_temp[ind0_nodes_temp]

            Delta_y2_temp = params_C2M['interface_1']['part_2']['Delta_y2_4T_ALL'].reshape(-1)
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_Delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_y2_4T_Module[n_nodes_1side:,1] = Delta_y2_temp[ind0_nodes_temp]

            #Delta_z1_4T_Module
            Parts_link_Delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Delta_z1_temp = params_C2M['interface_1']['part_1']['Delta_z1_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_1']['ind0_Geo_right_4T_4SepFill']
            Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = Delta_z1_temp[ind0_nodes_temp]

            Delta_z1_temp = params_C2M['interface_1']['part_2']['Delta_z1_4T_ALL']
            ind0_nodes_temp = params_C2M['interface_1']['part_2']['ind0_Geo_left_4T_4SepFill']
            Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = Delta_z1_temp[ind0_nodes_temp]

            #Delta_z2_4T_Module
            Parts_link_Delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
            Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
            Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = 0.5*delta_Cu+0.5*delta_Cu

            Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
            Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = 0.5*delta_Cu+0.5*delta_Cu


        if self.status_BC_Module_4T == 'Prismatic_Cell1':     
            self.n_nodes_1side_1d = np.zeros([self.n_interface],dtype=int)  #unique for 'Ribbon_cooling'
            for i0 in np.arange(self.n_interface):
                #interface_1: part_1-ind0_Geo_back_4T & part_2-ind0_Geo_front_4T
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'

                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'

                ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp1 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 67-70,137-140,207-210     

                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill

                ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp2 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 275-278,345-348,415-418      
    
                ind0_interfaces_Module_4T = np.concatenate(( prep_temp1, 
                                                             prep_temp2 ))                        #all nodes to be modified in MatrixM_4T. e.g. nodes 67-70,137-140,207-210, 275-278,345-348,415-418
                
                ind0_interfacesSideA_Module_4T = prep_temp1
                ind0_interfacesSideB_Module_4T = prep_temp2
                
                n_nodes_1side = np.size(prep_temp1)                                               #number of interface nodes (single side)
                self.n_nodes_1side = n_nodes_1side
                
                self.n_nodes_1side_1d[i0] = n_nodes_1side
                ########################################################################################################
                ###############################          prep Thermal interfaces          ##############################
                ########################################################################################################
    
                #-------------jx1_4T_Module
                Parts_link_jx1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jx1_4T_Module[:n_nodes_1side,1] = prep_temp
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jx2_4T_Module
                Parts_link_jx2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jx2_4T_Module[:n_nodes_1side,1] = prep_temp    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jy1_4T_Module
                Parts_link_jy1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jy1_4T_Module[:n_nodes_1side,1] = prep_temp    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jy1_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jy2_4T_Module
                Parts_link_jy2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jy2_4T_Module[:n_nodes_1side,1] = prep_temp    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jy2_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jz1_4T_Module
                Parts_link_jz1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)

                if item_temp0 == 'interface_1':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ] -1    
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1            
                    Parts_link_jz1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz1_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ] -1    
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 14,15,20,21
                    Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz1_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ] -1    
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1            
                    Parts_link_jz1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz1_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz1_4T_Module[n_nodes_1side:,1] = -9999
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ] -1    
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1            
                    Parts_link_jz1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz1_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_2'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = np.flip(ind0_nodes_temp.reshape(self.ny,-1),axis=1).reshape(-1)    #flip order
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 3,4,9,10
                    Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz1_4T_Module[n_nodes_1side:,1] = prep_temp                
                
                #-------------jz2_4T_Module
                Parts_link_jz2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)

                if item_temp0 == 'interface_1':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = np.flip(ind0_nodes_temp.reshape(self.ny,-1),axis=1).reshape(-1)    #flip order
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 3,4,9,10
                    Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_interface_1'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = np.flip(ind0_nodes_temp.reshape(self.ny,-1),axis=1).reshape(-1)    #flip order
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 4,5,10,11            
                    Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 3,4,9,10
                    Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_interface_1'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 4,5,10,11            
                    Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = np.flip(ind0_nodes_temp.reshape(self.ny,-1),axis=1).reshape(-1)    #flip order
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 3,4,9,10
                    Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz2_4T_Module[n_nodes_1side:,1] = -9999
                
                #-------------Lamda_4T_Module
                Parts_link_Lamda_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])

                if item_temp0 == 'interface_1':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    Lamda_Cu_SideA = params_C2M[item_temp0][item_temp1]['Lamda_Cu']
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    delta_Cu_SideB = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    Lamda_Cu_SideB = params_C2M[item_temp0][item_temp1]['Lamda_Cu']
                    #Mylar layer (if there is)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']      
                    delta_Mylar = params_C2M[item_temp0][item_temp1]['delta_Mylar']
                    Lamda_Mylar = params_C2M[item_temp0][item_temp1]['Lamda_Mylar']
    
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1+5] = (0.5*delta_Cu_SideA+0.5*delta_Cu_SideB+delta_Mylar) / (0.5*delta_Cu_SideA/Lamda_Cu_SideA + 0.5*delta_Cu_SideB/Lamda_Cu_SideB + delta_Mylar/Lamda_Mylar)                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+5] = (0.5*delta_Cu_SideA+0.5*delta_Cu_SideB+delta_Mylar) / (0.5*delta_Cu_SideA/Lamda_Cu_SideA + 0.5*delta_Cu_SideB/Lamda_Cu_SideB + delta_Mylar/Lamda_Mylar)
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    Lamda_Cu_SideA = params_C2M[item_temp0][item_temp1]['Lamda_Cu']
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    delta_Can_SideB = params_C2M[item_temp0][item_temp1]['delta_Can_real']
                    Lamda_Can_SideB = params_C2M[item_temp0][item_temp1]['Lamda_Can']
    
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1+5] = (0.5*delta_Cu_SideA+0.5*delta_Can_SideB+delta_Mylar) / (0.5*delta_Cu_SideA/Lamda_Cu_SideA + 0.5*delta_Can_SideB/Lamda_Can_SideB + delta_Mylar/Lamda_Mylar)                
#                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1+5] = 0                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+5] = (0.5*delta_Cu_SideA+0.5*delta_Can_SideB+delta_Mylar) / (0.5*delta_Cu_SideA/Lamda_Cu_SideA + 0.5*delta_Can_SideB/Lamda_Can_SideB + delta_Mylar/Lamda_Mylar)
#                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+5] = 0
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    Lamda_Cu_SideA = params_C2M[item_temp0][item_temp1]['Lamda_Cu']
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    delta_Can_SideB = params_C2M[item_temp0][item_temp1]['delta_Can_real']
                    Lamda_Can_SideB = params_C2M[item_temp0][item_temp1]['Lamda_Can']
    
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1+5] = (0.5*delta_Cu_SideA+0.5*delta_Can_SideB+delta_Mylar) / (0.5*delta_Cu_SideA/Lamda_Cu_SideA + 0.5*delta_Can_SideB/Lamda_Can_SideB + delta_Mylar/Lamda_Mylar)                
#                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1+5] = 0                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+4] = (0.5*delta_Cu_SideA+0.5*delta_Can_SideB+delta_Mylar) / (0.5*delta_Cu_SideA/Lamda_Cu_SideA + 0.5*delta_Can_SideB/Lamda_Can_SideB + delta_Mylar/Lamda_Mylar)
#                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+4] = 0
    
                #-------------Mis_4T_Module
                Parts_link_Mis_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])

                #SideA
                Parts_link_Mis_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Mis_4T_Module[:n_nodes_1side,1:] = 1                                        
                #SideB
                Parts_link_Mis_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Mis_4T_Module[n_nodes_1side:,1:] = 1

                #-------------RouXc_4T_Module
                Parts_link_RouXc_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                RouXc_temp = params_C2M[item_temp0][item_temp1]['RouXc_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_RouXc_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_RouXc_4T_Module[:n_nodes_1side,1] = RouXc_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                RouXc_temp = params_C2M[item_temp0][item_temp1]['RouXc_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_RouXc_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_RouXc_4T_Module[n_nodes_1side:,1] = RouXc_temp[ind0_nodes_temp]
    
                #-------------delta_x1_4T_Module
                Parts_link_delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_x1_temp = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_x1_4T_Module[:n_nodes_1side,1] = delta_x1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_x1_temp = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_x1_4T_Module[n_nodes_1side:,1] = delta_x1_temp[ind0_nodes_temp]
    
                #-------------delta_x2_4T_Module
                Parts_link_delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_x2_temp = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_x2_4T_Module[:n_nodes_1side,1] = delta_x2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_x2_temp = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_x2_4T_Module[n_nodes_1side:,1] = delta_x2_temp[ind0_nodes_temp]
    
                #-------------delta_y1_4T_Module
                Parts_link_delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_y1_temp = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_y1_4T_Module[:n_nodes_1side,1] = delta_y1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_y1_temp = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_y1_4T_Module[n_nodes_1side:,1] = delta_y1_temp[ind0_nodes_temp]
    
                #-------------delta_y2_4T_Module
                Parts_link_delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_y2_temp = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_y2_4T_Module[:n_nodes_1side,1] = delta_y2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_y2_temp = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_y2_4T_Module[n_nodes_1side:,1] = delta_y2_temp[ind0_nodes_temp]
    
                #-------------delta_z1_4T_Module
                Parts_link_delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_z1_temp = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_z1_4T_Module[:n_nodes_1side,1] = delta_z1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_z1_temp = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_z1_4T_Module[n_nodes_1side:,1] = delta_z1_temp[ind0_nodes_temp]
    
                #-------------delta_z2_4T_Module
                Parts_link_delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_z2_temp = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_z2_4T_Module[:n_nodes_1side,1] = delta_z2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_z2_temp = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_z2_4T_Module[n_nodes_1side:,1] = delta_z2_temp[ind0_nodes_temp]
                
                #-------------Delta_x1_4T_Module
                Parts_link_Delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_x1_temp = params_C2M[item_temp0][item_temp1]['Delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_x1_4T_Module[:n_nodes_1side,1] = Delta_x1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_x1_temp = params_C2M[item_temp0][item_temp1]['Delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_x1_4T_Module[n_nodes_1side:,1] = Delta_x1_temp[ind0_nodes_temp]
    
                #-------------Delta_x2_4T_Module
                Parts_link_Delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_x2_temp = params_C2M[item_temp0][item_temp1]['Delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_x2_4T_Module[:n_nodes_1side,1] = Delta_x2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_x2_temp = params_C2M[item_temp0][item_temp1]['Delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_x2_4T_Module[n_nodes_1side:,1] = Delta_x2_temp[ind0_nodes_temp]
    
                #-------------Delta_y1_4T_Module
                Parts_link_Delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_y1_temp = params_C2M[item_temp0][item_temp1]['Delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_y1_4T_Module[:n_nodes_1side,1] = Delta_y1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_y1_temp = params_C2M[item_temp0][item_temp1]['Delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_y1_4T_Module[n_nodes_1side:,1] = Delta_y1_temp[ind0_nodes_temp]
    
                #-------------Delta_y2_4T_Module
                Parts_link_Delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_y2_temp = params_C2M[item_temp0][item_temp1]['Delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_y2_4T_Module[:n_nodes_1side,1] = Delta_y2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_y2_temp = params_C2M[item_temp0][item_temp1]['Delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_y2_4T_Module[n_nodes_1side:,1] = Delta_y2_temp[ind0_nodes_temp]
    
                #-------------Delta_z1_4T_Module
                Parts_link_Delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    Delta_z1_temp = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = Delta_z1_temp[ind0_nodes_temp]    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']         
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']      
                    Delta_z1_temp = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = Delta_z1_temp[ind0_nodes_temp]
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    Delta_z1_temp = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = Delta_z1_temp[ind0_nodes_temp]    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    delta_Can_SideB = params_C2M[item_temp0][item_temp1]['delta_Can_real']

                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = -9999
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    Delta_z1_temp = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = Delta_z1_temp[ind0_nodes_temp]    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    delta_Can_SideB = params_C2M[item_temp0][item_temp1]['delta_Can_real']

                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = 0.5*delta_Cu_SideA+0.5*delta_Can_SideB
    
                #-------------Delta_z2_4T_Module
                Parts_link_Delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    delta_Cu_SideB = params_C2M[item_temp0][item_temp1]['delta_Cu']

                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = 0.5*delta_Cu_SideA + 0.5*delta_Cu_SideB    
                    #SideB
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = 0.5*delta_Cu_SideA + 0.5*delta_Cu_SideB
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    delta_Can_SideB = params_C2M[item_temp0][item_temp1]['delta_Can_real']

                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = 0.5*delta_Cu_SideA + 0.5*delta_Can_SideB    
                    #SideB
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = 0.5*delta_Cu_SideA + 0.5*delta_Can_SideB
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    delta_Cu_SideA = params_C2M[item_temp0][item_temp1]['delta_Cu']
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    delta_Can_SideB = params_C2M[item_temp0][item_temp1]['delta_Can_real']

                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = 0.5*delta_Cu_SideA + 0.5*delta_Can_SideB    
                    #SideB
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = -9999

                if i0 == 0:
                    ind0_interfaces_Module_4T_temp = ind0_interfaces_Module_4T.copy()
                    ind0_interfacesSideA_Module_4T_temp = ind0_interfacesSideA_Module_4T.copy()
                    ind0_interfacesSideB_Module_4T_temp = ind0_interfacesSideB_Module_4T.copy()
                    Parts_link_jx1_4T_Module_temp = Parts_link_jx1_4T_Module.copy()
                    Parts_link_jx2_4T_Module_temp = Parts_link_jx2_4T_Module.copy()
                    Parts_link_jy1_4T_Module_temp = Parts_link_jy1_4T_Module.copy()
                    Parts_link_jy2_4T_Module_temp = Parts_link_jy2_4T_Module.copy()
                    Parts_link_jz1_4T_Module_temp = Parts_link_jz1_4T_Module.copy()
                    Parts_link_jz2_4T_Module_temp = Parts_link_jz2_4T_Module.copy()
                    Parts_link_Lamda_4T_Module_temp = Parts_link_Lamda_4T_Module.copy()
                    Parts_link_Mis_4T_Module_temp = Parts_link_Mis_4T_Module.copy()
                    Parts_link_RouXc_4T_Module_temp = Parts_link_RouXc_4T_Module.copy()
                    Parts_link_delta_x1_4T_Module_temp = Parts_link_delta_x1_4T_Module.copy()
                    Parts_link_delta_x2_4T_Module_temp = Parts_link_delta_x2_4T_Module.copy()
                    Parts_link_delta_y1_4T_Module_temp = Parts_link_delta_y1_4T_Module.copy()
                    Parts_link_delta_y2_4T_Module_temp = Parts_link_delta_y2_4T_Module.copy()
                    Parts_link_delta_z1_4T_Module_temp = Parts_link_delta_z1_4T_Module.copy()
                    Parts_link_delta_z2_4T_Module_temp = Parts_link_delta_z2_4T_Module.copy()
                    Parts_link_Delta_x1_4T_Module_temp = Parts_link_Delta_x1_4T_Module.copy()
                    Parts_link_Delta_x2_4T_Module_temp = Parts_link_Delta_x2_4T_Module.copy()
                    Parts_link_Delta_y1_4T_Module_temp = Parts_link_Delta_y1_4T_Module.copy()
                    Parts_link_Delta_y2_4T_Module_temp = Parts_link_Delta_y2_4T_Module.copy()
                    Parts_link_Delta_z1_4T_Module_temp = Parts_link_Delta_z1_4T_Module.copy()
                    Parts_link_Delta_z2_4T_Module_temp = Parts_link_Delta_z2_4T_Module.copy()
                if i0 >= 1:
                    ind0_interfaces_Module_4T = np.append(ind0_interfaces_Module_4T_temp,ind0_interfaces_Module_4T,axis=0); ind0_interfaces_Module_4T_temp = ind0_interfaces_Module_4T
                    ind0_interfacesSideA_Module_4T = np.append(ind0_interfacesSideA_Module_4T_temp,ind0_interfacesSideA_Module_4T,axis=0); ind0_interfacesSideA_Module_4T_temp = ind0_interfacesSideA_Module_4T
                    ind0_interfacesSideB_Module_4T = np.append(ind0_interfacesSideB_Module_4T_temp,ind0_interfacesSideB_Module_4T,axis=0); ind0_interfacesSideB_Module_4T_temp = ind0_interfacesSideB_Module_4T
                    Parts_link_jx1_4T_Module = np.append(Parts_link_jx1_4T_Module_temp,Parts_link_jx1_4T_Module,axis=0); Parts_link_jx1_4T_Module_temp = Parts_link_jx1_4T_Module
                    Parts_link_jx2_4T_Module = np.append(Parts_link_jx2_4T_Module_temp,Parts_link_jx2_4T_Module,axis=0); Parts_link_jx2_4T_Module_temp = Parts_link_jx2_4T_Module
                    Parts_link_jy1_4T_Module = np.append(Parts_link_jy1_4T_Module_temp,Parts_link_jy1_4T_Module,axis=0); Parts_link_jy1_4T_Module_temp = Parts_link_jy1_4T_Module
                    Parts_link_jy2_4T_Module = np.append(Parts_link_jy2_4T_Module_temp,Parts_link_jy2_4T_Module,axis=0); Parts_link_jy2_4T_Module_temp = Parts_link_jy2_4T_Module
                    Parts_link_jz1_4T_Module = np.append(Parts_link_jz1_4T_Module_temp,Parts_link_jz1_4T_Module,axis=0); Parts_link_jz1_4T_Module_temp = Parts_link_jz1_4T_Module
                    Parts_link_jz2_4T_Module = np.append(Parts_link_jz2_4T_Module_temp,Parts_link_jz2_4T_Module,axis=0); Parts_link_jz2_4T_Module_temp = Parts_link_jz2_4T_Module
                    Parts_link_Lamda_4T_Module = np.append(Parts_link_Lamda_4T_Module_temp,Parts_link_Lamda_4T_Module,axis=0); Parts_link_Lamda_4T_Module_temp = Parts_link_Lamda_4T_Module
                    Parts_link_Mis_4T_Module = np.append(Parts_link_Mis_4T_Module_temp,Parts_link_Mis_4T_Module,axis=0); Parts_link_Mis_4T_Module_temp = Parts_link_Mis_4T_Module
                    Parts_link_RouXc_4T_Module = np.append(Parts_link_RouXc_4T_Module_temp,Parts_link_RouXc_4T_Module,axis=0); Parts_link_RouXc_4T_Module_temp = Parts_link_RouXc_4T_Module
                    Parts_link_delta_x1_4T_Module = np.append(Parts_link_delta_x1_4T_Module_temp,Parts_link_delta_x1_4T_Module,axis=0); Parts_link_delta_x1_4T_Module_temp = Parts_link_delta_x1_4T_Module
                    Parts_link_delta_x2_4T_Module = np.append(Parts_link_delta_x2_4T_Module_temp,Parts_link_delta_x2_4T_Module,axis=0); Parts_link_delta_x2_4T_Module_temp = Parts_link_delta_x2_4T_Module
                    Parts_link_delta_y1_4T_Module = np.append(Parts_link_delta_y1_4T_Module_temp,Parts_link_delta_y1_4T_Module,axis=0); Parts_link_delta_y1_4T_Module_temp = Parts_link_delta_y1_4T_Module
                    Parts_link_delta_y2_4T_Module = np.append(Parts_link_delta_y2_4T_Module_temp,Parts_link_delta_y2_4T_Module,axis=0); Parts_link_delta_y2_4T_Module_temp = Parts_link_delta_y2_4T_Module
                    Parts_link_delta_z1_4T_Module = np.append(Parts_link_delta_z1_4T_Module_temp,Parts_link_delta_z1_4T_Module,axis=0); Parts_link_delta_z1_4T_Module_temp = Parts_link_delta_z1_4T_Module
                    Parts_link_delta_z2_4T_Module = np.append(Parts_link_delta_z2_4T_Module_temp,Parts_link_delta_z2_4T_Module,axis=0); Parts_link_delta_z2_4T_Module_temp = Parts_link_delta_z2_4T_Module
                    Parts_link_Delta_x1_4T_Module = np.append(Parts_link_Delta_x1_4T_Module_temp,Parts_link_Delta_x1_4T_Module,axis=0); Parts_link_Delta_x1_4T_Module_temp = Parts_link_Delta_x1_4T_Module
                    Parts_link_Delta_x2_4T_Module = np.append(Parts_link_Delta_x2_4T_Module_temp,Parts_link_Delta_x2_4T_Module,axis=0); Parts_link_Delta_x2_4T_Module_temp = Parts_link_Delta_x2_4T_Module
                    Parts_link_Delta_y1_4T_Module = np.append(Parts_link_Delta_y1_4T_Module_temp,Parts_link_Delta_y1_4T_Module,axis=0); Parts_link_Delta_y1_4T_Module_temp = Parts_link_Delta_y1_4T_Module
                    Parts_link_Delta_y2_4T_Module = np.append(Parts_link_Delta_y2_4T_Module_temp,Parts_link_Delta_y2_4T_Module,axis=0); Parts_link_Delta_y2_4T_Module_temp = Parts_link_Delta_y2_4T_Module
                    Parts_link_Delta_z1_4T_Module = np.append(Parts_link_Delta_z1_4T_Module_temp,Parts_link_Delta_z1_4T_Module,axis=0); Parts_link_Delta_z1_4T_Module_temp = Parts_link_Delta_z1_4T_Module
                    Parts_link_Delta_z2_4T_Module = np.append(Parts_link_Delta_z2_4T_Module_temp,Parts_link_Delta_z2_4T_Module,axis=0); Parts_link_Delta_z2_4T_Module_temp = Parts_link_Delta_z2_4T_Module


        if self.status_BC_Module_4T == 'Ribbon_cooling':     
            self.n_nodes_1side_1d = np.zeros([self.n_interface],dtype=int)  #unique for 'Ribbon_cooling'
            for i0 in np.arange(self.n_interface):
                #interface_1: part_1-ind0_Geo_back_4T & part_2-ind0_Geo_front_4T
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'

                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'

                ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp1 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 5,6,11,12     

                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'

                ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp2 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 13,14,19,20      
    
                ind0_interfaces_Module_4T = np.concatenate(( prep_temp1, 
                                                             prep_temp2 ))                        #all nodes to be modified in MatrixM_4T. e.g. nodes 5,6,11,12, 13,14,19,20
                
                ind0_interfacesSideA_Module_4T = prep_temp1
                ind0_interfacesSideB_Module_4T = prep_temp2
                
                n_nodes_1side = np.size(prep_temp1)                                               #number of interface nodes (single side)
                self.n_nodes_1side = n_nodes_1side
                
                self.n_nodes_1side_1d[i0] = n_nodes_1side
                ########################################################################################################
                ###############################          prep Thermal interfaces          ##############################
                ########################################################################################################
    
                #-------------jx1_4T_Module
                Parts_link_jx1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jx1_4T_Module[:n_nodes_1side,1] = prep_temp
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jx2_4T_Module
                Parts_link_jx2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jx2_4T_Module[:n_nodes_1side,1] = prep_temp    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jy1_4T_Module
                Parts_link_jy1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jy1_4T_Module[:n_nodes_1side,1] = prep_temp    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jy1_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jy2_4T_Module
                Parts_link_jy2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jy2_4T_Module[:n_nodes_1side,1] = prep_temp    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                TF_temp = ind_nodes_temp != -9999
                prep_temp = np.zeros([n_nodes_1side],dtype=int)
                prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                prep_temp[~TF_temp] = -9999
                Parts_link_jy2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jy2_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jz1_4T_Module
                Parts_link_jz1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ] -1    
                prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1            
                Parts_link_jz1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jz1_4T_Module[:n_nodes_1side,1] = prep_temp                
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ] -1    
                prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 14,15,20,21
                Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jz1_4T_Module[n_nodes_1side:,1] = prep_temp
                
                #-------------jz2_4T_Module
                Parts_link_jz2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 3,4,9,10
                Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp                
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_interface_1'    
                ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              #e.g. nodes 4,5,10,11            
                Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp
                
                #-------------Lamda_4T_Module
                Parts_link_Lamda_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])

                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                delta_Can = params_C2M[item_temp0][item_temp1]['delta_Can_real']
                Lamda_Can = params_C2M[item_temp0][item_temp1]['Lamda_Can']
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                delta_ribbon = params_C2M[item_temp0][item_temp1]['delta_ribbon']
                Lamda_ribbon = params_C2M[item_temp0][item_temp1]['Lamda_ribbon']

                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                Parts_link_Lamda_4T_Module[:n_nodes_1side,1+5] = (0.5*delta_ribbon+0.5*delta_Can) / (0.5*delta_ribbon/Lamda_ribbon + 0.5*delta_Can/Lamda_Can)                
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                Parts_link_Lamda_4T_Module[n_nodes_1side:,1+5] = (0.5*delta_ribbon+0.5*delta_Can) / (0.5*delta_ribbon/Lamda_ribbon + 0.5*delta_Can/Lamda_Can)
    
                #-------------Mis_4T_Module
                Parts_link_Mis_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])

                #SideA
                Parts_link_Mis_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Mis_4T_Module[:n_nodes_1side,1:] = 1                                        
                #SideB
                Parts_link_Mis_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Mis_4T_Module[n_nodes_1side:,1:] = 1

                #-------------RouXc_4T_Module
                Parts_link_RouXc_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                RouXc_temp = params_C2M[item_temp0][item_temp1]['RouXc_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_RouXc_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_RouXc_4T_Module[:n_nodes_1side,1] = RouXc_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                RouXc_temp = params_C2M[item_temp0][item_temp1]['RouXc_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_RouXc_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_RouXc_4T_Module[n_nodes_1side:,1] = RouXc_temp[ind0_nodes_temp]
    
                #-------------delta_x1_4T_Module
                Parts_link_delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_x1_temp = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_x1_4T_Module[:n_nodes_1side,1] = delta_x1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_x1_temp = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_x1_4T_Module[n_nodes_1side:,1] = delta_x1_temp[ind0_nodes_temp]
    
                #-------------delta_x2_4T_Module
                Parts_link_delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_x2_temp = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_x2_4T_Module[:n_nodes_1side,1] = delta_x2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_x2_temp = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_x2_4T_Module[n_nodes_1side:,1] = delta_x2_temp[ind0_nodes_temp]
    
                #-------------delta_y1_4T_Module
                Parts_link_delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_y1_temp = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_y1_4T_Module[:n_nodes_1side,1] = delta_y1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_y1_temp = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_y1_4T_Module[n_nodes_1side:,1] = delta_y1_temp[ind0_nodes_temp]
    
                #-------------delta_y2_4T_Module
                Parts_link_delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_y2_temp = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_y2_4T_Module[:n_nodes_1side,1] = delta_y2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_y2_temp = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_y2_4T_Module[n_nodes_1side:,1] = delta_y2_temp[ind0_nodes_temp]
    
                #-------------delta_z1_4T_Module
                Parts_link_delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_z1_temp = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_z1_4T_Module[:n_nodes_1side,1] = delta_z1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_z1_temp = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_z1_4T_Module[n_nodes_1side:,1] = delta_z1_temp[ind0_nodes_temp]
    
                #-------------delta_z2_4T_Module
                Parts_link_delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_z2_temp = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_z2_4T_Module[:n_nodes_1side,1] = delta_z2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_z2_temp = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_z2_4T_Module[n_nodes_1side:,1] = delta_z2_temp[ind0_nodes_temp]
                
                #-------------Delta_x1_4T_Module
                Parts_link_Delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_x1_temp = params_C2M[item_temp0][item_temp1]['Delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_x1_4T_Module[:n_nodes_1side,1] = Delta_x1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_x1_temp = params_C2M[item_temp0][item_temp1]['Delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_x1_4T_Module[n_nodes_1side:,1] = Delta_x1_temp[ind0_nodes_temp]
    
                #-------------Delta_x2_4T_Module
                Parts_link_Delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_x2_temp = params_C2M[item_temp0][item_temp1]['Delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_x2_4T_Module[:n_nodes_1side,1] = Delta_x2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_x2_temp = params_C2M[item_temp0][item_temp1]['Delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_x2_4T_Module[n_nodes_1side:,1] = Delta_x2_temp[ind0_nodes_temp]
    
                #-------------Delta_y1_4T_Module
                Parts_link_Delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_y1_temp = params_C2M[item_temp0][item_temp1]['Delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_y1_4T_Module[:n_nodes_1side,1] = Delta_y1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_y1_temp = params_C2M[item_temp0][item_temp1]['Delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_y1_4T_Module[n_nodes_1side:,1] = Delta_y1_temp[ind0_nodes_temp]
    
                #-------------Delta_y2_4T_Module
                Parts_link_Delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_y2_temp = params_C2M[item_temp0][item_temp1]['Delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_y2_4T_Module[:n_nodes_1side,1] = Delta_y2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_y2_temp = params_C2M[item_temp0][item_temp1]['Delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_y2_4T_Module[n_nodes_1side:,1] = Delta_y2_temp[ind0_nodes_temp]
    
                #-------------Delta_z1_4T_Module
                Parts_link_Delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                Delta_z1_temp = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = Delta_z1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                Delta_z1_temp = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL']
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = Delta_z1_temp[ind0_nodes_temp]
    
                #-------------Delta_z2_4T_Module
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                delta_Can = params_C2M[item_temp0][item_temp1]['delta_Can_real']
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                delta_ribbon = params_C2M[item_temp0][item_temp1]['delta_ribbon']

                Parts_link_Delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = 0.5*delta_Can+0.5*delta_ribbon    
                #SideB
                Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = 0.5*delta_Can+0.5*delta_ribbon

                if i0 == 0:
                    ind0_interfaces_Module_4T_temp = ind0_interfaces_Module_4T.copy()
                    ind0_interfacesSideA_Module_4T_temp = ind0_interfacesSideA_Module_4T.copy()
                    ind0_interfacesSideB_Module_4T_temp = ind0_interfacesSideB_Module_4T.copy()
                    Parts_link_jx1_4T_Module_temp = Parts_link_jx1_4T_Module.copy()
                    Parts_link_jx2_4T_Module_temp = Parts_link_jx2_4T_Module.copy()
                    Parts_link_jy1_4T_Module_temp = Parts_link_jy1_4T_Module.copy()
                    Parts_link_jy2_4T_Module_temp = Parts_link_jy2_4T_Module.copy()
                    Parts_link_jz1_4T_Module_temp = Parts_link_jz1_4T_Module.copy()
                    Parts_link_jz2_4T_Module_temp = Parts_link_jz2_4T_Module.copy()
                    Parts_link_Lamda_4T_Module_temp = Parts_link_Lamda_4T_Module.copy()
                    Parts_link_Mis_4T_Module_temp = Parts_link_Mis_4T_Module.copy()
                    Parts_link_RouXc_4T_Module_temp = Parts_link_RouXc_4T_Module.copy()
                    Parts_link_delta_x1_4T_Module_temp = Parts_link_delta_x1_4T_Module.copy()
                    Parts_link_delta_x2_4T_Module_temp = Parts_link_delta_x2_4T_Module.copy()
                    Parts_link_delta_y1_4T_Module_temp = Parts_link_delta_y1_4T_Module.copy()
                    Parts_link_delta_y2_4T_Module_temp = Parts_link_delta_y2_4T_Module.copy()
                    Parts_link_delta_z1_4T_Module_temp = Parts_link_delta_z1_4T_Module.copy()
                    Parts_link_delta_z2_4T_Module_temp = Parts_link_delta_z2_4T_Module.copy()
                    Parts_link_Delta_x1_4T_Module_temp = Parts_link_Delta_x1_4T_Module.copy()
                    Parts_link_Delta_x2_4T_Module_temp = Parts_link_Delta_x2_4T_Module.copy()
                    Parts_link_Delta_y1_4T_Module_temp = Parts_link_Delta_y1_4T_Module.copy()
                    Parts_link_Delta_y2_4T_Module_temp = Parts_link_Delta_y2_4T_Module.copy()
                    Parts_link_Delta_z1_4T_Module_temp = Parts_link_Delta_z1_4T_Module.copy()
                    Parts_link_Delta_z2_4T_Module_temp = Parts_link_Delta_z2_4T_Module.copy()
                if i0 >= 1:
                    ind0_interfaces_Module_4T = np.append(ind0_interfaces_Module_4T_temp,ind0_interfaces_Module_4T,axis=0); ind0_interfaces_Module_4T_temp = ind0_interfaces_Module_4T
                    ind0_interfacesSideA_Module_4T = np.append(ind0_interfacesSideA_Module_4T_temp,ind0_interfacesSideA_Module_4T,axis=0); ind0_interfacesSideA_Module_4T_temp = ind0_interfacesSideA_Module_4T
                    ind0_interfacesSideB_Module_4T = np.append(ind0_interfacesSideB_Module_4T_temp,ind0_interfacesSideB_Module_4T,axis=0); ind0_interfacesSideB_Module_4T_temp = ind0_interfacesSideB_Module_4T
                    Parts_link_jx1_4T_Module = np.append(Parts_link_jx1_4T_Module_temp,Parts_link_jx1_4T_Module,axis=0); Parts_link_jx1_4T_Module_temp = Parts_link_jx1_4T_Module
                    Parts_link_jx2_4T_Module = np.append(Parts_link_jx2_4T_Module_temp,Parts_link_jx2_4T_Module,axis=0); Parts_link_jx2_4T_Module_temp = Parts_link_jx2_4T_Module
                    Parts_link_jy1_4T_Module = np.append(Parts_link_jy1_4T_Module_temp,Parts_link_jy1_4T_Module,axis=0); Parts_link_jy1_4T_Module_temp = Parts_link_jy1_4T_Module
                    Parts_link_jy2_4T_Module = np.append(Parts_link_jy2_4T_Module_temp,Parts_link_jy2_4T_Module,axis=0); Parts_link_jy2_4T_Module_temp = Parts_link_jy2_4T_Module
                    Parts_link_jz1_4T_Module = np.append(Parts_link_jz1_4T_Module_temp,Parts_link_jz1_4T_Module,axis=0); Parts_link_jz1_4T_Module_temp = Parts_link_jz1_4T_Module
                    Parts_link_jz2_4T_Module = np.append(Parts_link_jz2_4T_Module_temp,Parts_link_jz2_4T_Module,axis=0); Parts_link_jz2_4T_Module_temp = Parts_link_jz2_4T_Module
                    Parts_link_Lamda_4T_Module = np.append(Parts_link_Lamda_4T_Module_temp,Parts_link_Lamda_4T_Module,axis=0); Parts_link_Lamda_4T_Module_temp = Parts_link_Lamda_4T_Module
                    Parts_link_Mis_4T_Module = np.append(Parts_link_Mis_4T_Module_temp,Parts_link_Mis_4T_Module,axis=0); Parts_link_Mis_4T_Module_temp = Parts_link_Mis_4T_Module
                    Parts_link_RouXc_4T_Module = np.append(Parts_link_RouXc_4T_Module_temp,Parts_link_RouXc_4T_Module,axis=0); Parts_link_RouXc_4T_Module_temp = Parts_link_RouXc_4T_Module
                    Parts_link_delta_x1_4T_Module = np.append(Parts_link_delta_x1_4T_Module_temp,Parts_link_delta_x1_4T_Module,axis=0); Parts_link_delta_x1_4T_Module_temp = Parts_link_delta_x1_4T_Module
                    Parts_link_delta_x2_4T_Module = np.append(Parts_link_delta_x2_4T_Module_temp,Parts_link_delta_x2_4T_Module,axis=0); Parts_link_delta_x2_4T_Module_temp = Parts_link_delta_x2_4T_Module
                    Parts_link_delta_y1_4T_Module = np.append(Parts_link_delta_y1_4T_Module_temp,Parts_link_delta_y1_4T_Module,axis=0); Parts_link_delta_y1_4T_Module_temp = Parts_link_delta_y1_4T_Module
                    Parts_link_delta_y2_4T_Module = np.append(Parts_link_delta_y2_4T_Module_temp,Parts_link_delta_y2_4T_Module,axis=0); Parts_link_delta_y2_4T_Module_temp = Parts_link_delta_y2_4T_Module
                    Parts_link_delta_z1_4T_Module = np.append(Parts_link_delta_z1_4T_Module_temp,Parts_link_delta_z1_4T_Module,axis=0); Parts_link_delta_z1_4T_Module_temp = Parts_link_delta_z1_4T_Module
                    Parts_link_delta_z2_4T_Module = np.append(Parts_link_delta_z2_4T_Module_temp,Parts_link_delta_z2_4T_Module,axis=0); Parts_link_delta_z2_4T_Module_temp = Parts_link_delta_z2_4T_Module
                    Parts_link_Delta_x1_4T_Module = np.append(Parts_link_Delta_x1_4T_Module_temp,Parts_link_Delta_x1_4T_Module,axis=0); Parts_link_Delta_x1_4T_Module_temp = Parts_link_Delta_x1_4T_Module
                    Parts_link_Delta_x2_4T_Module = np.append(Parts_link_Delta_x2_4T_Module_temp,Parts_link_Delta_x2_4T_Module,axis=0); Parts_link_Delta_x2_4T_Module_temp = Parts_link_Delta_x2_4T_Module
                    Parts_link_Delta_y1_4T_Module = np.append(Parts_link_Delta_y1_4T_Module_temp,Parts_link_Delta_y1_4T_Module,axis=0); Parts_link_Delta_y1_4T_Module_temp = Parts_link_Delta_y1_4T_Module
                    Parts_link_Delta_y2_4T_Module = np.append(Parts_link_Delta_y2_4T_Module_temp,Parts_link_Delta_y2_4T_Module,axis=0); Parts_link_Delta_y2_4T_Module_temp = Parts_link_Delta_y2_4T_Module
                    Parts_link_Delta_z1_4T_Module = np.append(Parts_link_Delta_z1_4T_Module_temp,Parts_link_Delta_z1_4T_Module,axis=0); Parts_link_Delta_z1_4T_Module_temp = Parts_link_Delta_z1_4T_Module
                    Parts_link_Delta_z2_4T_Module = np.append(Parts_link_Delta_z2_4T_Module_temp,Parts_link_Delta_z2_4T_Module,axis=0); Parts_link_Delta_z2_4T_Module_temp = Parts_link_Delta_z2_4T_Module
                                
        if self.status_BC_Module_4T == 'Pouch_weld_tab':     
            self.n_nodes_1side_1d = np.zeros([self.n_interface],dtype=int)  #unique for 'Ribbon_cooling'
            for i0 in np.arange(self.n_interface):
                #interface_1: part_1-ind0_Geo_back_4T & part_2-ind0_Geo_front_4T
                item_temp0 = list(self.interface_string.keys())[i0]                 #'interface_1'

                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'

                ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp1 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 67-70,137-140,207-210     

                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill

                ind0_Part_temp = self.Parts_str2int[item_temp1]-1
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                prep_temp2 = self.List_Cmatind2Mmatind_4T[ ind0_Part_temp, ind0_nodes_temp ]      #e.g. nodes 275-278,345-348,415-418      
    
                ind0_interfaces_Module_4T = np.concatenate(( prep_temp1, 
                                                             prep_temp2 ))                        #all nodes to be modified in MatrixM_4T. e.g. nodes 67-70,137-140,207-210, 275-278,345-348,415-418
                
                ind0_interfacesSideA_Module_4T = prep_temp1
                ind0_interfacesSideB_Module_4T = prep_temp2
                
                n_nodes_1side = np.size(prep_temp1)                                               #number of interface nodes (single side)
                self.n_nodes_1side = n_nodes_1side
                
                self.n_nodes_1side_1d[i0] = n_nodes_1side
                ########################################################################################################
                ###############################          prep Thermal interfaces          ##############################
                ########################################################################################################
    
                #-------------jx1_4T_Module
                Parts_link_jx1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                if item_temp0 == 'interface_1':
                    #SideA
                    Parts_link_jx1_4T_Module[:n_nodes_1side,0] = -8888  #No need for nodes in Interface_1 SideA, because this is done in Interface_2
                    Parts_link_jx1_4T_Module[:n_nodes_1side,1] = -8888
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = ind0_nodes_temp.reshape(self.nstack,-1).T.reshape(-1)
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1
                    Parts_link_jx1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jx1_4T_Module[:n_nodes_1side,1] = prep_temp
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    prep_temp = -9999
                    Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp-1 ] +1
                    Parts_link_jx1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jx1_4T_Module[:n_nodes_1side,1] = prep_temp
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = ind0_nodes_temp.reshape(-1,self.nstack).T.reshape(-1)
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1
                    Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_jx1_4T_Module[:n_nodes_1side,0] = -8888  #No need for nodes in Interface_4 SideA, because this is done in Interface_3
                    Parts_link_jx1_4T_Module[:n_nodes_1side,1] = -8888
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']          
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']        
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jx1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx1_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jx2_4T_Module
                Parts_link_jx2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                if item_temp0 == 'interface_1':
                    #SideA
                    Parts_link_jx2_4T_Module[:n_nodes_1side,0] = -8888  #No need for nodes in Interface_1 SideA, because this is done in Interface_2
                    Parts_link_jx2_4T_Module[:n_nodes_1side,1] = -8888    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp-1 ] +1
                    Parts_link_jx2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jx2_4T_Module[:n_nodes_1side,1] = prep_temp    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = ind0_nodes_temp.reshape(-1,self.nstack).T.reshape(-1)
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1
                    Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    ind0_nodes_temp = ind0_nodes_temp.reshape(self.nstack,-1).T.reshape(-1)
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1
                    Parts_link_jx2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jx2_4T_Module[:n_nodes_1side,1] = prep_temp    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_jx2_4T_Module[:n_nodes_1side,0] = -8888  #No need for nodes in Interface_4 SideA, because this is done in Interface_3
                    Parts_link_jx2_4T_Module[:n_nodes_1side,1] = -8888    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jx2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jx2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jx2_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jy1_4T_Module
                Parts_link_jy1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_jy1_4T_Module[:n_nodes_1side,0] = -8888
                    Parts_link_jy1_4T_Module[:n_nodes_1side,1] = -8888    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jy1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jy1_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_2' or item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jy1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jy1_4T_Module[:n_nodes_1side,1] = prep_temp    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jy1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jy1_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jy2_4T_Module
                Parts_link_jy2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)
                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_jy2_4T_Module[:n_nodes_1side,0] = -8888
                    Parts_link_jy2_4T_Module[:n_nodes_1side,1] = -8888    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jy2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jy2_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_2' or item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jy2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jy2_4T_Module[:n_nodes_1side,1] = prep_temp    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_2'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_left_4T_4SepFill'    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jy2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jy2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jy2_4T_Module[n_nodes_1side:,1] = prep_temp
    
                #-------------jz1_4T_Module
                Parts_link_jz1_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_jz1_4T_Module[:n_nodes_1side,0] = -8888
                    Parts_link_jz1_4T_Module[:n_nodes_1side,1] = -8888                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']      
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']       
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1                              
                    Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz1_4T_Module[n_nodes_1side:,1] = prep_temp
                if item_temp0 == 'interface_2' or item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]    
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jz1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz1_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jz1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]    
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jz1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz1_4T_Module[n_nodes_1side:,1] = prep_temp                
                
                #-------------jz2_4T_Module
                Parts_link_jz2_4T_Module = np.zeros([2*n_nodes_1side,2],dtype=int)

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_jz2_4T_Module[:n_nodes_1side,0] = -8888
                    Parts_link_jz2_4T_Module[:n_nodes_1side,1] = -8888                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right_4T_4SepFill'
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1]['jz2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ] -1    
                    prep_temp = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind0_nodes_temp ] +1
                    Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp                
                if item_temp0 == 'interface_2':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jz2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]    
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jz2_4T_ALL']
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999

                    item_temp1 = self.interface_string['interface_1']['SideB_part_id']     
                    item_temp2 = self.interface_string['interface_1']['SideB_ind0_Geo']    
                    ind0_nodes_tab = params_C2M['interface_1'][item_temp1][item_temp2]
                    ind0_nodes_tab = self.List_Cmatind2Mmatind_4T[ 3, ind0_nodes_tab ]
                    prep_temp[:self.tab_ny] = ind0_nodes_tab + 1
                    
                    Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp                
                if item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jz2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]    
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999
                    Parts_link_jz2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_jz2_4T_Module[:n_nodes_1side,1] = prep_temp                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    
                    ind0_Cell_temp = self.Parts_str2int[item_temp1]-1                                                
                    ind_nodes_temp = params_C2M[item_temp0][item_temp1]['jz2_4T_ALL']
                    TF_temp = ind_nodes_temp != -9999
                    prep_temp = np.zeros([n_nodes_1side],dtype=int)
                    prep_temp[TF_temp] = self.List_Cmatind2Mmatind_4T[ ind0_Cell_temp, ind_nodes_temp[TF_temp]-1 ] +1
                    prep_temp[~TF_temp] = -9999

                    item_temp1 = self.interface_string['interface_4']['SideB_part_id']     
                    item_temp2 = self.interface_string['interface_4']['SideB_ind0_Geo']    
                    ind0_nodes_tab = params_C2M['interface_4'][item_temp1][item_temp2]
                    ind0_nodes_tab = self.List_Cmatind2Mmatind_4T[ 4, ind0_nodes_tab ]
                    prep_temp[:self.tab_ny] = ind0_nodes_tab + 1
                    
                    Parts_link_jz2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_jz2_4T_Module[n_nodes_1side:,1] = prep_temp                
                
                #-------------Lamda_4T_Module
                Parts_link_Lamda_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    delta_z2_weld = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    Lamda_weld = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2],5 ]

                    nonweld_tab_Lamda = params_C2M[item_temp0][item_temp1]['nonweld_tab_Lamda']
                    ind0_Geo_node1_7_4T = params_C2M[item_temp0][item_temp1]['ind0_Geo_node1_7_4T']
                    ind0_Geo_node3_5_4T = params_C2M[item_temp0][item_temp1]['ind0_Geo_node3_5_4T']

                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_z1_tab = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    Lamda_tab = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2],4 ]
    
                    #SideA
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = np.nan                                        
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+4] = (delta_z2_weld+delta_z1_tab) / (delta_z2_weld/Lamda_weld + delta_z1_tab/Lamda_tab) 
                    Parts_link_Lamda_4T_Module[ind0_Geo_node1_7_4T+n_nodes_1side,1+4] = nonweld_tab_Lamda  #modify for nonweld                
                    Parts_link_Lamda_4T_Module[ind0_Geo_node3_5_4T+n_nodes_1side,1+4] = nonweld_tab_Lamda  #modify for nonweld                
                if item_temp0 == 'interface_2':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    delta_z1_cell = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_z1_cell_flip = delta_z1_cell.reshape(-1,self.nstack).T.reshape(-1)
                    Lamda_cell = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2],0 ]
                    Lamda_cell_flip = Lamda_cell.reshape(-1,self.nstack).T.reshape(-1)

                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_x2_weld = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x2_weld_flip = delta_x2_weld.reshape(self.nstack,-1).T.reshape(-1)
                    Lamda_weld = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2],1 ]
                    Lamda_weld_flip = Lamda_weld.reshape(self.nstack,-1).T.reshape(-1)
    
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]                                        
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1+0] = (delta_x2_weld_flip+self.cell_tab_L1+delta_z1_cell) / (delta_x2_weld_flip/Lamda_weld_flip + self.cell_tab_L1/self.Lamda_Al + delta_z1_cell/Lamda_cell)                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+1] = (delta_x2_weld+self.cell_tab_L1+delta_z1_cell_flip) / (delta_x2_weld/Lamda_weld + self.cell_tab_L1/self.Lamda_Al + delta_z1_cell_flip/Lamda_cell_flip) 
                if item_temp0 == 'interface_3':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    delta_z2_cell = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_z2_cell_flip = delta_z2_cell.reshape(-1,self.nstack).T.reshape(-1)
                    Lamda_cell = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2],1 ]
                    Lamda_cell_flip = Lamda_cell.reshape(-1,self.nstack).T.reshape(-1)

                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_x1_weld = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x1_weld_flip = delta_x1_weld.reshape(self.nstack,-1).T.reshape(-1)
                    Lamda_weld = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2],0 ]
                    Lamda_weld_flip = Lamda_weld.reshape(self.nstack,-1).T.reshape(-1)
    
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1:] = Lamda_temp[ ind0_nodes_temp,: ]                                        
                    Parts_link_Lamda_4T_Module[:n_nodes_1side,1+1] = (delta_x1_weld_flip+self.cell_tab_L1+delta_z2_cell) / (delta_x1_weld_flip/Lamda_weld_flip + self.cell_tab_L1/self.Lamda_Cu + delta_z2_cell/Lamda_cell)                
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Lamda_temp = params_C2M[item_temp0][item_temp1]['Lamda_4T_ALL']
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1:] = Lamda_temp[ ind0_nodes_temp,: ]
                    Parts_link_Lamda_4T_Module[n_nodes_1side:,1+0] = (delta_x1_weld+self.cell_tab_L1+delta_z2_cell_flip) / (delta_x1_weld/Lamda_weld + self.cell_tab_L1/self.Lamda_Cu + delta_z2_cell_flip/Lamda_cell_flip) 
    
                #-------------Mis_4T_Module
                Parts_link_Mis_4T_Module = np.nan * np.zeros([2*n_nodes_1side,7])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    temp1 = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp2 = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp3 = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp4 = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    S5_weld = 0.5*(temp1+temp2) * 0.5*(temp3+temp4)

                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    temp1 = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp2 = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp3 = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp4 = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    S4_tab = 0.5*(temp1+temp2) * 0.5*(temp3+temp4)
                    
                    Smax = np.max( np.concatenate((S5_weld.reshape(-1,1),S4_tab.reshape(-1,1)),axis=1), axis=1 )
                        
                    #SideA
                    Parts_link_Mis_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Mis_4T_Module[:n_nodes_1side,1:] = np.nan                                        
                    #SideB
                    Parts_link_Mis_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Mis_4T_Module[n_nodes_1side:,1:] = 1
                    Parts_link_Mis_4T_Module[n_nodes_1side:,1+4] = S5_weld/Smax 
                if item_temp0 == 'interface_2':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    temp1 = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp2 = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp3 = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp4 = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    S0_cell = 0.5*(temp1+temp2) * 0.5*(temp3+temp4)
                    S0_cell_flip = S0_cell.reshape(-1,self.nstack).T.reshape(-1)

                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    temp1 = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp2 = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp3 = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp4 = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    S1_weld = 0.5*(temp1+temp2) * 0.5*(temp3+temp4)
                    S1_weld_flip = S1_weld.reshape(self.nstack,-1).T.reshape(-1)
                    
                    Smax_cell = np.max( np.concatenate((S0_cell.reshape(-1,1),S1_weld_flip.reshape(-1,1)),axis=1), axis=1 )
                    Smax_weld = np.max( np.concatenate((S0_cell_flip.reshape(-1,1),S1_weld.reshape(-1,1)),axis=1), axis=1 )
                        
                    #SideA
                    Parts_link_Mis_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Mis_4T_Module[:n_nodes_1side,1:] = 1                                        
                    Parts_link_Mis_4T_Module[:n_nodes_1side,1+0] = S0_cell/Smax_cell                                        
                    #SideB
                    Parts_link_Mis_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Mis_4T_Module[n_nodes_1side:,1:] = 1
                    Parts_link_Mis_4T_Module[n_nodes_1side:,1+1] = S1_weld/Smax_weld 
                if item_temp0 == 'interface_3':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    temp1 = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp2 = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp3 = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp4 = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    S0_cell = 0.5*(temp1+temp2) * 0.5*(temp3+temp4)
                    S0_cell_flip = S0_cell.reshape(-1,self.nstack).T.reshape(-1)

                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    temp1 = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp2 = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp3 = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    temp4 = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    S1_weld = 0.5*(temp1+temp2) * 0.5*(temp3+temp4)
                    S1_weld_flip = S1_weld.reshape(self.nstack,-1).T.reshape(-1)
                    
                    Smax_cell = np.max( np.concatenate((S0_cell.reshape(-1,1),S1_weld_flip.reshape(-1,1)),axis=1), axis=1 )
                    Smax_weld = np.max( np.concatenate((S0_cell_flip.reshape(-1,1),S1_weld.reshape(-1,1)),axis=1), axis=1 )
                        
                    #SideA
                    Parts_link_Mis_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Mis_4T_Module[:n_nodes_1side,1:] = 1                                        
                    Parts_link_Mis_4T_Module[:n_nodes_1side,1+1] = S0_cell/Smax_cell                                        
                    #SideB
                    Parts_link_Mis_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Mis_4T_Module[n_nodes_1side:,1:] = 1
                    Parts_link_Mis_4T_Module[n_nodes_1side:,1+0] = S1_weld/Smax_weld 

                #-------------RouXc_4T_Module
                Parts_link_RouXc_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_RouXc_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_RouXc_4T_Module[:n_nodes_1side,1] = np.nan    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    RouXc_temp = params_C2M[item_temp0][item_temp1]['RouXc_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_RouXc_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_RouXc_4T_Module[n_nodes_1side:,1] = RouXc_temp[ind0_nodes_temp]
                if item_temp0 == 'interface_2' or item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    RouXc_temp = params_C2M[item_temp0][item_temp1]['RouXc_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_RouXc_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_RouXc_4T_Module[:n_nodes_1side,1] = RouXc_temp[ind0_nodes_temp]    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    RouXc_temp = params_C2M[item_temp0][item_temp1]['RouXc_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_RouXc_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_RouXc_4T_Module[n_nodes_1side:,1] = RouXc_temp[ind0_nodes_temp]
    
                #-------------delta_x1_4T_Module
                Parts_link_delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_x1_temp = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_x1_4T_Module[:n_nodes_1side,1] = delta_x1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_x1_temp = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_x1_4T_Module[n_nodes_1side:,1] = delta_x1_temp[ind0_nodes_temp]
    
                #-------------delta_x2_4T_Module
                Parts_link_delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_x2_temp = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_x2_4T_Module[:n_nodes_1side,1] = delta_x2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_x2_temp = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_x2_4T_Module[n_nodes_1side:,1] = delta_x2_temp[ind0_nodes_temp]
    
                #-------------delta_y1_4T_Module
                Parts_link_delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_y1_temp = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_y1_4T_Module[:n_nodes_1side,1] = delta_y1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_y1_temp = params_C2M[item_temp0][item_temp1]['delta_y1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_y1_4T_Module[n_nodes_1side:,1] = delta_y1_temp[ind0_nodes_temp]
    
                #-------------delta_y2_4T_Module
                Parts_link_delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_y2_temp = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_y2_4T_Module[:n_nodes_1side,1] = delta_y2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_y2_temp = params_C2M[item_temp0][item_temp1]['delta_y2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_y2_4T_Module[n_nodes_1side:,1] = delta_y2_temp[ind0_nodes_temp]
    
                #-------------delta_z1_4T_Module
                Parts_link_delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_z1_temp = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_z1_4T_Module[:n_nodes_1side,1] = delta_z1_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_z1_temp = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_z1_4T_Module[n_nodes_1side:,1] = delta_z1_temp[ind0_nodes_temp]
    
                #-------------delta_z2_4T_Module
                Parts_link_delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])
                #SideA
                item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                delta_z2_temp = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                Parts_link_delta_z2_4T_Module[:n_nodes_1side,1] = delta_z2_temp[ind0_nodes_temp]    
                #SideB
                item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                delta_z2_temp = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'].reshape(-1)
                ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                Parts_link_delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                Parts_link_delta_z2_4T_Module[n_nodes_1side:,1] = delta_z2_temp[ind0_nodes_temp]
                
                #-------------Delta_x1_4T_Module
                Parts_link_Delta_x1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_Delta_x1_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Delta_x1_4T_Module[:n_nodes_1side,1] = np.nan    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Delta_x1_temp = params_C2M[item_temp0][item_temp1]['Delta_x1_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_x1_4T_Module[n_nodes_1side:,1] = Delta_x1_temp[ind0_nodes_temp]
                if item_temp0 == 'interface_2':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    delta_x1_cell = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x1_cell_flip = delta_x1_cell.reshape(-1,self.nstack).T.reshape(-1)
                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_x2_weld = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x2_weld_flip = delta_x2_weld.reshape(self.nstack,-1).T.reshape(-1)

                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Parts_link_Delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_x1_4T_Module[:n_nodes_1side,1] = delta_x1_cell + delta_x2_weld_flip + self.cell_tab_L1
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Parts_link_Delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_x1_4T_Module[n_nodes_1side:,1] = delta_x1_cell_flip
                if item_temp0 == 'interface_3':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Delta_x1_cell = params_C2M[item_temp0][item_temp1]['Delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x2_cell = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x2_cell_flip = delta_x2_cell.reshape(-1,self.nstack).T.reshape(-1)
                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_x1_weld = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x2_weld = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]

                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Parts_link_Delta_x1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_x1_4T_Module[:n_nodes_1side,1] = Delta_x1_cell
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Parts_link_Delta_x1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_x1_4T_Module[n_nodes_1side:,1] = delta_x2_cell_flip + delta_x1_weld + self.cell_tab_L1
    
                #-------------Delta_x2_4T_Module
                Parts_link_Delta_x2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_Delta_x2_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Delta_x2_4T_Module[:n_nodes_1side,1] = np.nan    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Delta_x2_temp = params_C2M[item_temp0][item_temp1]['Delta_x2_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_x2_4T_Module[n_nodes_1side:,1] = Delta_x2_temp[ind0_nodes_temp]
                if item_temp0 == 'interface_2':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    delta_x1_cell = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x1_cell_flip = delta_x1_cell.reshape(-1,self.nstack).T.reshape(-1)
                    Delta_x2_cell = params_C2M[item_temp0][item_temp1]['Delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_x2_weld = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]

                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Parts_link_Delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_x2_4T_Module[:n_nodes_1side,1] = Delta_x2_cell
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Parts_link_Delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_x2_4T_Module[n_nodes_1side:,1] = delta_x1_cell_flip + delta_x2_weld + self.cell_tab_L1
                if item_temp0 == 'interface_3':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    delta_x2_cell = params_C2M[item_temp0][item_temp1]['delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_x1_weld = params_C2M[item_temp0][item_temp1]['delta_x1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    delta_x1_weld_flip = delta_x1_weld.reshape(self.nstack,-1).T.reshape(-1)
                    Delta_x2_weld = params_C2M[item_temp0][item_temp1]['Delta_x2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]

                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Parts_link_Delta_x2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_x2_4T_Module[:n_nodes_1side,1] = delta_x2_cell + delta_x1_weld_flip + self.cell_tab_L1
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Parts_link_Delta_x2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_x2_4T_Module[n_nodes_1side:,1] = Delta_x2_weld                    
                    
                #-------------Delta_y1_4T_Module
                Parts_link_Delta_y1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_Delta_y1_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Delta_y1_4T_Module[:n_nodes_1side,1] = np.nan    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Delta_y1_temp = params_C2M[item_temp0][item_temp1]['Delta_y1_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_y1_4T_Module[n_nodes_1side:,1] = Delta_y1_temp[ind0_nodes_temp]
                if item_temp0 == 'interface_2' or item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Delta_y1_temp = params_C2M[item_temp0][item_temp1]['Delta_y1_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_y1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_y1_4T_Module[:n_nodes_1side,1] = Delta_y1_temp[ind0_nodes_temp]    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Delta_y1_temp = params_C2M[item_temp0][item_temp1]['Delta_y1_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_y1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_y1_4T_Module[n_nodes_1side:,1] = Delta_y1_temp[ind0_nodes_temp]
    
                #-------------Delta_y2_4T_Module
                Parts_link_Delta_y2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA
                    Parts_link_Delta_y2_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Delta_y2_4T_Module[:n_nodes_1side,1] = np.nan    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Delta_y2_temp = params_C2M[item_temp0][item_temp1]['Delta_y2_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_y2_4T_Module[n_nodes_1side:,1] = Delta_y2_temp[ind0_nodes_temp]
                if item_temp0 == 'interface_2' or item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    Delta_y2_temp = params_C2M[item_temp0][item_temp1]['Delta_y2_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_y2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_y2_4T_Module[:n_nodes_1side,1] = Delta_y2_temp[ind0_nodes_temp]    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_3'     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_interface_1'    
                    Delta_y2_temp = params_C2M[item_temp0][item_temp1]['Delta_y2_4T_ALL'].reshape(-1)
                    ind0_nodes_temp = params_C2M[item_temp0][item_temp1][item_temp2]
                    Parts_link_Delta_y2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_y2_4T_Module[n_nodes_1side:,1] = Delta_y2_temp[ind0_nodes_temp]
    
                #-------------Delta_z1_4T_Module
                Parts_link_Delta_z1_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    #'ind0_Geo_right'
                    delta_z2_weld = params_C2M[item_temp0][item_temp1]['delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    Delta_z1_weld = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    delta_z1_tab = params_C2M[item_temp0][item_temp1]['delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]

                    #SideA
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = np.nan    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']         
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']      
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = delta_z2_weld + delta_z1_tab
                if item_temp0 == 'interface_2' or item_temp0 == 'interface_3':
                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    Delta_z1_cell = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z1_4T_Module[:n_nodes_1side,1] = Delta_z1_cell    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    
                    Delta_z1_weld = params_C2M[item_temp0][item_temp1]['Delta_z1_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z1_4T_Module[n_nodes_1side:,1] = Delta_z1_weld
    
                #-------------Delta_z2_4T_Module
                Parts_link_Delta_z2_4T_Module = np.nan * np.zeros([2*n_nodes_1side,2])

                if item_temp0 == 'interface_1' or item_temp0 == 'interface_4':
                    #SideA (prep for below)
                    #SideB (prep for below)
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']     #'part_1'
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']    #'ind0_Geo_right'
                    Delta_z2_tab = params_C2M[item_temp0][item_temp1]['Delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]

                    #SideA
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = np.nan
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = np.nan    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']         
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']      
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = Delta_z2_tab
                if item_temp0 == 'interface_2':
                    #prep for below
                    delta_z2_weld = params_C2M['interface_1']['part_2']['delta_z2_4T_ALL'][ params_C2M['interface_1']['part_2']['ind0_Geo_front_4T'] ]
                    delta_z1_tab = params_C2M['interface_1']['part_4']['delta_z1_4T_ALL'][ params_C2M['interface_1']['part_4']['ind0_Geo_link_4T'] ]

                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    Delta_z2_cell = params_C2M[item_temp0][item_temp1]['Delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = Delta_z2_cell    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']         
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']      
                    Delta_z2_weld = params_C2M[item_temp0][item_temp1]['Delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]                    
                    Delta_z2_weld[:(self.weld_n + self.nonweld_n)] = delta_z2_weld + delta_z1_tab
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = Delta_z2_weld
                if item_temp0 == 'interface_3':
                    #prep for below
                    delta_z2_weld = params_C2M['interface_4']['part_3']['delta_z2_4T_ALL'][ params_C2M['interface_4']['part_3']['ind0_Geo_front_4T'] ]
                    delta_z1_tab = params_C2M['interface_4']['part_5']['delta_z1_4T_ALL'][ params_C2M['interface_4']['part_5']['ind0_Geo_link_4T'] ]

                    #SideA
                    item_temp1 = self.interface_string[item_temp0]['SideA_part_id']     
                    item_temp2 = self.interface_string[item_temp0]['SideA_ind0_Geo']    
                    Delta_z2_cell = params_C2M[item_temp0][item_temp1]['Delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,0] = ind0_interfaces_Module_4T[:n_nodes_1side]
                    Parts_link_Delta_z2_4T_Module[:n_nodes_1side,1] = Delta_z2_cell    
                    #SideB
                    item_temp1 = self.interface_string[item_temp0]['SideB_part_id']         
                    item_temp2 = self.interface_string[item_temp0]['SideB_ind0_Geo']      
                    Delta_z2_weld = params_C2M[item_temp0][item_temp1]['Delta_z2_4T_ALL'][ params_C2M[item_temp0][item_temp1][item_temp2] ]                    
                    Delta_z2_weld[:(self.weld_n + self.nonweld_n)] = delta_z2_weld + delta_z1_tab
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,0] = ind0_interfaces_Module_4T[n_nodes_1side:]
                    Parts_link_Delta_z2_4T_Module[n_nodes_1side:,1] = Delta_z2_weld

                if i0 == 0:
                    ind0_interfaces_Module_4T_temp = ind0_interfaces_Module_4T.copy()
                    ind0_interfacesSideA_Module_4T_temp = ind0_interfacesSideA_Module_4T.copy()
                    ind0_interfacesSideB_Module_4T_temp = ind0_interfacesSideB_Module_4T.copy()
                    Parts_link_jx1_4T_Module_temp = Parts_link_jx1_4T_Module.copy()
                    Parts_link_jx2_4T_Module_temp = Parts_link_jx2_4T_Module.copy()
                    Parts_link_jy1_4T_Module_temp = Parts_link_jy1_4T_Module.copy()
                    Parts_link_jy2_4T_Module_temp = Parts_link_jy2_4T_Module.copy()
                    Parts_link_jz1_4T_Module_temp = Parts_link_jz1_4T_Module.copy()
                    Parts_link_jz2_4T_Module_temp = Parts_link_jz2_4T_Module.copy()
                    Parts_link_Lamda_4T_Module_temp = Parts_link_Lamda_4T_Module.copy()
                    Parts_link_Mis_4T_Module_temp = Parts_link_Mis_4T_Module.copy()
                    Parts_link_RouXc_4T_Module_temp = Parts_link_RouXc_4T_Module.copy()
                    Parts_link_delta_x1_4T_Module_temp = Parts_link_delta_x1_4T_Module.copy()
                    Parts_link_delta_x2_4T_Module_temp = Parts_link_delta_x2_4T_Module.copy()
                    Parts_link_delta_y1_4T_Module_temp = Parts_link_delta_y1_4T_Module.copy()
                    Parts_link_delta_y2_4T_Module_temp = Parts_link_delta_y2_4T_Module.copy()
                    Parts_link_delta_z1_4T_Module_temp = Parts_link_delta_z1_4T_Module.copy()
                    Parts_link_delta_z2_4T_Module_temp = Parts_link_delta_z2_4T_Module.copy()
                    Parts_link_Delta_x1_4T_Module_temp = Parts_link_Delta_x1_4T_Module.copy()
                    Parts_link_Delta_x2_4T_Module_temp = Parts_link_Delta_x2_4T_Module.copy()
                    Parts_link_Delta_y1_4T_Module_temp = Parts_link_Delta_y1_4T_Module.copy()
                    Parts_link_Delta_y2_4T_Module_temp = Parts_link_Delta_y2_4T_Module.copy()
                    Parts_link_Delta_z1_4T_Module_temp = Parts_link_Delta_z1_4T_Module.copy()
                    Parts_link_Delta_z2_4T_Module_temp = Parts_link_Delta_z2_4T_Module.copy()
                if i0 >= 1:
                    ind0_interfaces_Module_4T = np.append(ind0_interfaces_Module_4T_temp,ind0_interfaces_Module_4T,axis=0); ind0_interfaces_Module_4T_temp = ind0_interfaces_Module_4T
                    ind0_interfacesSideA_Module_4T = np.append(ind0_interfacesSideA_Module_4T_temp,ind0_interfacesSideA_Module_4T,axis=0); ind0_interfacesSideA_Module_4T_temp = ind0_interfacesSideA_Module_4T
                    ind0_interfacesSideB_Module_4T = np.append(ind0_interfacesSideB_Module_4T_temp,ind0_interfacesSideB_Module_4T,axis=0); ind0_interfacesSideB_Module_4T_temp = ind0_interfacesSideB_Module_4T
                    Parts_link_jx1_4T_Module = np.append(Parts_link_jx1_4T_Module_temp,Parts_link_jx1_4T_Module,axis=0); Parts_link_jx1_4T_Module_temp = Parts_link_jx1_4T_Module
                    Parts_link_jx2_4T_Module = np.append(Parts_link_jx2_4T_Module_temp,Parts_link_jx2_4T_Module,axis=0); Parts_link_jx2_4T_Module_temp = Parts_link_jx2_4T_Module
                    Parts_link_jy1_4T_Module = np.append(Parts_link_jy1_4T_Module_temp,Parts_link_jy1_4T_Module,axis=0); Parts_link_jy1_4T_Module_temp = Parts_link_jy1_4T_Module
                    Parts_link_jy2_4T_Module = np.append(Parts_link_jy2_4T_Module_temp,Parts_link_jy2_4T_Module,axis=0); Parts_link_jy2_4T_Module_temp = Parts_link_jy2_4T_Module
                    Parts_link_jz1_4T_Module = np.append(Parts_link_jz1_4T_Module_temp,Parts_link_jz1_4T_Module,axis=0); Parts_link_jz1_4T_Module_temp = Parts_link_jz1_4T_Module
                    Parts_link_jz2_4T_Module = np.append(Parts_link_jz2_4T_Module_temp,Parts_link_jz2_4T_Module,axis=0); Parts_link_jz2_4T_Module_temp = Parts_link_jz2_4T_Module
                    Parts_link_Lamda_4T_Module = np.append(Parts_link_Lamda_4T_Module_temp,Parts_link_Lamda_4T_Module,axis=0); Parts_link_Lamda_4T_Module_temp = Parts_link_Lamda_4T_Module
                    Parts_link_Mis_4T_Module = np.append(Parts_link_Mis_4T_Module_temp,Parts_link_Mis_4T_Module,axis=0); Parts_link_Mis_4T_Module_temp = Parts_link_Mis_4T_Module
                    Parts_link_RouXc_4T_Module = np.append(Parts_link_RouXc_4T_Module_temp,Parts_link_RouXc_4T_Module,axis=0); Parts_link_RouXc_4T_Module_temp = Parts_link_RouXc_4T_Module
                    Parts_link_delta_x1_4T_Module = np.append(Parts_link_delta_x1_4T_Module_temp,Parts_link_delta_x1_4T_Module,axis=0); Parts_link_delta_x1_4T_Module_temp = Parts_link_delta_x1_4T_Module
                    Parts_link_delta_x2_4T_Module = np.append(Parts_link_delta_x2_4T_Module_temp,Parts_link_delta_x2_4T_Module,axis=0); Parts_link_delta_x2_4T_Module_temp = Parts_link_delta_x2_4T_Module
                    Parts_link_delta_y1_4T_Module = np.append(Parts_link_delta_y1_4T_Module_temp,Parts_link_delta_y1_4T_Module,axis=0); Parts_link_delta_y1_4T_Module_temp = Parts_link_delta_y1_4T_Module
                    Parts_link_delta_y2_4T_Module = np.append(Parts_link_delta_y2_4T_Module_temp,Parts_link_delta_y2_4T_Module,axis=0); Parts_link_delta_y2_4T_Module_temp = Parts_link_delta_y2_4T_Module
                    Parts_link_delta_z1_4T_Module = np.append(Parts_link_delta_z1_4T_Module_temp,Parts_link_delta_z1_4T_Module,axis=0); Parts_link_delta_z1_4T_Module_temp = Parts_link_delta_z1_4T_Module
                    Parts_link_delta_z2_4T_Module = np.append(Parts_link_delta_z2_4T_Module_temp,Parts_link_delta_z2_4T_Module,axis=0); Parts_link_delta_z2_4T_Module_temp = Parts_link_delta_z2_4T_Module
                    Parts_link_Delta_x1_4T_Module = np.append(Parts_link_Delta_x1_4T_Module_temp,Parts_link_Delta_x1_4T_Module,axis=0); Parts_link_Delta_x1_4T_Module_temp = Parts_link_Delta_x1_4T_Module
                    Parts_link_Delta_x2_4T_Module = np.append(Parts_link_Delta_x2_4T_Module_temp,Parts_link_Delta_x2_4T_Module,axis=0); Parts_link_Delta_x2_4T_Module_temp = Parts_link_Delta_x2_4T_Module
                    Parts_link_Delta_y1_4T_Module = np.append(Parts_link_Delta_y1_4T_Module_temp,Parts_link_Delta_y1_4T_Module,axis=0); Parts_link_Delta_y1_4T_Module_temp = Parts_link_Delta_y1_4T_Module
                    Parts_link_Delta_y2_4T_Module = np.append(Parts_link_Delta_y2_4T_Module_temp,Parts_link_Delta_y2_4T_Module,axis=0); Parts_link_Delta_y2_4T_Module_temp = Parts_link_Delta_y2_4T_Module
                    Parts_link_Delta_z1_4T_Module = np.append(Parts_link_Delta_z1_4T_Module_temp,Parts_link_Delta_z1_4T_Module,axis=0); Parts_link_Delta_z1_4T_Module_temp = Parts_link_Delta_z1_4T_Module
                    Parts_link_Delta_z2_4T_Module = np.append(Parts_link_Delta_z2_4T_Module_temp,Parts_link_Delta_z2_4T_Module,axis=0); Parts_link_Delta_z2_4T_Module_temp = Parts_link_Delta_z2_4T_Module


        self.ind0_interfaces_Module_4T = ind0_interfaces_Module_4T
        self.ind0_interfacesSideA_Module_4T = ind0_interfacesSideA_Module_4T
        self.ind0_interfacesSideB_Module_4T = ind0_interfacesSideB_Module_4T
        self.Parts_link_jx1_4T_Module = Parts_link_jx1_4T_Module
        self.Parts_link_jx2_4T_Module = Parts_link_jx2_4T_Module
        self.Parts_link_jy1_4T_Module = Parts_link_jy1_4T_Module
        self.Parts_link_jy2_4T_Module = Parts_link_jy2_4T_Module
        self.Parts_link_jz1_4T_Module = Parts_link_jz1_4T_Module
        self.Parts_link_jz2_4T_Module = Parts_link_jz2_4T_Module
        self.Parts_link_Lamda_4T_Module = Parts_link_Lamda_4T_Module
        self.Parts_link_Mis_4T_Module = Parts_link_Mis_4T_Module
        self.Parts_link_RouXc_4T_Module = Parts_link_RouXc_4T_Module
        self.Parts_link_delta_x1_4T_Module = Parts_link_delta_x1_4T_Module
        self.Parts_link_delta_x2_4T_Module = Parts_link_delta_x2_4T_Module
        self.Parts_link_delta_y1_4T_Module = Parts_link_delta_y1_4T_Module
        self.Parts_link_delta_y2_4T_Module = Parts_link_delta_y2_4T_Module
        self.Parts_link_delta_z1_4T_Module = Parts_link_delta_z1_4T_Module
        self.Parts_link_delta_z2_4T_Module = Parts_link_delta_z2_4T_Module
        self.Parts_link_Delta_x1_4T_Module = Parts_link_Delta_x1_4T_Module
        self.Parts_link_Delta_x2_4T_Module = Parts_link_Delta_x2_4T_Module
        self.Parts_link_Delta_y1_4T_Module = Parts_link_Delta_y1_4T_Module
        self.Parts_link_Delta_y2_4T_Module = Parts_link_Delta_y2_4T_Module
        self.Parts_link_Delta_z1_4T_Module = Parts_link_Delta_z1_4T_Module
        self.Parts_link_Delta_z2_4T_Module = Parts_link_Delta_z2_4T_Module
        
        self.ind0_interfaces_Module_4T_Unique_local = np.argwhere(Parts_link_jx1_4T_Module[:,0] >= 0)[:,0]  #'ind0_interfaces_Module_4T_Unique_local' only meaningful for 'Pouch_weld_tab' where interface_1 SideA nodes is repeating with interface_2 SideB nodes
        self.Parts_link_ind0_jx1NaN_4T_local = np.argwhere(Parts_link_jx1_4T_Module[:,1] == -9999)[:,0]
        self.Parts_link_ind0_jx2NaN_4T_local = np.argwhere(Parts_link_jx2_4T_Module[:,1] == -9999)[:,0]
        self.Parts_link_ind0_jy1NaN_4T_local = np.argwhere(Parts_link_jy1_4T_Module[:,1] == -9999)[:,0]
        self.Parts_link_ind0_jy2NaN_4T_local = np.argwhere(Parts_link_jy2_4T_Module[:,1] == -9999)[:,0]
        self.Parts_link_ind0_jz1NaN_4T_local = np.argwhere(Parts_link_jz1_4T_Module[:,1] == -9999)[:,0]
        self.Parts_link_ind0_jz2NaN_4T_local = np.argwhere(Parts_link_jz2_4T_Module[:,1] == -9999)[:,0]
        self.Parts_link_ind0_jx1NonNaN_4T_local = np.argwhere(Parts_link_jx1_4T_Module[:,1] >= 0)[:,0]
        self.Parts_link_ind0_jx2NonNaN_4T_local = np.argwhere(Parts_link_jx2_4T_Module[:,1] >= 0)[:,0]
        self.Parts_link_ind0_jy1NonNaN_4T_local = np.argwhere(Parts_link_jy1_4T_Module[:,1] >= 0)[:,0]
        self.Parts_link_ind0_jy2NonNaN_4T_local = np.argwhere(Parts_link_jy2_4T_Module[:,1] >= 0)[:,0]
        self.Parts_link_ind0_jz1NonNaN_4T_local = np.argwhere(Parts_link_jz1_4T_Module[:,1] >= 0)[:,0]
        self.Parts_link_ind0_jz2NonNaN_4T_local = np.argwhere(Parts_link_jz2_4T_Module[:,1] >= 0)[:,0]
    #########################################################   
    ################## function for MatrixM #################
    #########################################################
    def fun_matrixM_4T(self,params_C2M):            #Example in this section is based on 2 cells: 2,2,1; 2,2,1
        #---------------------------  diagonally append MatrixCN(s) to form MatrixM_4T ---------------------------
        item = ip.status_Parts_name[0]
        MatrixM_4T = params_C2M[item]
        for item in ip.status_Parts_name[1:]:
            MatrixM_4T = block_diag( MatrixM_4T, params_C2M[item] )
        #--------------------------- clear the rows of interface nodes (ind0_interfaces_Module_4T) ---------------------------
        MatrixM_4T[self.ind0_interfaces_Module_4T] = 0
        #--------------------------------------fill in jx1, jx2, jy1, jy2, jz1, jz2 terms
        jx1_4T = self.Parts_link_jx1_4T_Module
        jx2_4T = self.Parts_link_jx2_4T_Module
        jy1_4T = self.Parts_link_jy1_4T_Module
        jy2_4T = self.Parts_link_jy2_4T_Module
        jz1_4T = self.Parts_link_jz1_4T_Module
        jz2_4T = self.Parts_link_jz2_4T_Module
        Lamda_4T = self.Parts_link_Lamda_4T_Module
        Mis_4T = self.Parts_link_Mis_4T_Module
        RouXc_4T = self.Parts_link_RouXc_4T_Module
        h_4T = self.h_4T_Module
        delta_x1_4T = self.Parts_link_delta_x1_4T_Module
        delta_x2_4T = self.Parts_link_delta_x2_4T_Module
        delta_y1_4T = self.Parts_link_delta_y1_4T_Module
        delta_y2_4T = self.Parts_link_delta_y2_4T_Module
        delta_z1_4T = self.Parts_link_delta_z1_4T_Module
        delta_z2_4T = self.Parts_link_delta_z2_4T_Module
        Delta_x1_4T = self.Parts_link_Delta_x1_4T_Module
        Delta_x2_4T = self.Parts_link_Delta_x2_4T_Module
        Delta_y1_4T = self.Parts_link_Delta_y1_4T_Module
        Delta_y2_4T = self.Parts_link_Delta_y2_4T_Module
        Delta_z1_4T = self.Parts_link_Delta_z1_4T_Module
        Delta_z2_4T = self.Parts_link_Delta_z2_4T_Module
        ind0_jx1NonNaN_4T_local = self.Parts_link_ind0_jx1NonNaN_4T_local
        ind0_jx2NonNaN_4T_local = self.Parts_link_ind0_jx2NonNaN_4T_local
        ind0_jy1NonNaN_4T_local = self.Parts_link_ind0_jy1NonNaN_4T_local
        ind0_jy2NonNaN_4T_local = self.Parts_link_ind0_jy2NonNaN_4T_local
        ind0_jz1NonNaN_4T_local = self.Parts_link_ind0_jz1NonNaN_4T_local
        ind0_jz2NonNaN_4T_local = self.Parts_link_ind0_jz2NonNaN_4T_local
        ind0_jx1NonNaN_4T = self.Parts_link_jx1_4T_Module[ind0_jx1NonNaN_4T_local,0]
        ind0_jx2NonNaN_4T = self.Parts_link_jx2_4T_Module[ind0_jx2NonNaN_4T_local,0]
        ind0_jy1NonNaN_4T = self.Parts_link_jy1_4T_Module[ind0_jy1NonNaN_4T_local,0]
        ind0_jy2NonNaN_4T = self.Parts_link_jy2_4T_Module[ind0_jy2NonNaN_4T_local,0]
        ind0_jz1NonNaN_4T = self.Parts_link_jz1_4T_Module[ind0_jz1NonNaN_4T_local,0]
        ind0_jz2NonNaN_4T = self.Parts_link_jz2_4T_Module[ind0_jz2NonNaN_4T_local,0]
        ind0_jx1NaN_4T_local = self.Parts_link_ind0_jx1NaN_4T_local
        ind0_jx2NaN_4T_local = self.Parts_link_ind0_jx2NaN_4T_local
        ind0_jy1NaN_4T_local = self.Parts_link_ind0_jy1NaN_4T_local
        ind0_jy2NaN_4T_local = self.Parts_link_ind0_jy2NaN_4T_local
        ind0_jz1NaN_4T_local = self.Parts_link_ind0_jz1NaN_4T_local
        ind0_jz2NaN_4T_local = self.Parts_link_ind0_jz2NaN_4T_local
        ind0_jx1NaN_4T = self.Parts_link_jx1_4T_Module[ind0_jx1NaN_4T_local,0]
        ind0_jx2NaN_4T = self.Parts_link_jx2_4T_Module[ind0_jx2NaN_4T_local,0]
        ind0_jy1NaN_4T = self.Parts_link_jy1_4T_Module[ind0_jy1NaN_4T_local,0]
        ind0_jy2NaN_4T = self.Parts_link_jy2_4T_Module[ind0_jy2NaN_4T_local,0]
        ind0_jz1NaN_4T = self.Parts_link_jz1_4T_Module[ind0_jz1NaN_4T_local,0]
        ind0_jz2NaN_4T = self.Parts_link_jz2_4T_Module[ind0_jz2NaN_4T_local,0]
        dt = self.dt

        MatrixM_4T[ind0_jx1NonNaN_4T,jx1_4T[ind0_jx1NonNaN_4T_local,1]-1] += -Lamda_4T[ind0_jx1NonNaN_4T_local,1+0]*Mis_4T[ind0_jx1NonNaN_4T_local,1+0]/RouXc_4T[ind0_jx1NonNaN_4T_local,1]*dt/(delta_x1_4T[ind0_jx1NonNaN_4T_local,1]+delta_x2_4T[ind0_jx1NonNaN_4T_local,1])/Delta_x1_4T[ind0_jx1NonNaN_4T_local,1]    #if ind0_jx1NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jx1NonNaN node , column of left neighbor node(jx1); if ind0_jx1NaN nodes case: elements are zero as initiated 
        MatrixM_4T[ind0_jx2NonNaN_4T,jx2_4T[ind0_jx2NonNaN_4T_local,1]-1] += -Lamda_4T[ind0_jx2NonNaN_4T_local,1+1]*Mis_4T[ind0_jx2NonNaN_4T_local,1+1]/RouXc_4T[ind0_jx2NonNaN_4T_local,1]*dt/(delta_x1_4T[ind0_jx2NonNaN_4T_local,1]+delta_x2_4T[ind0_jx2NonNaN_4T_local,1])/Delta_x2_4T[ind0_jx2NonNaN_4T_local,1]    #if ind0_jx2NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jx2NonNaN node , column of right neighbor node(jx2); if ind0_jx2NaN nodes case: elements are zero as initiated 
        MatrixM_4T[ind0_jy1NonNaN_4T,jy1_4T[ind0_jy1NonNaN_4T_local,1]-1] += -Lamda_4T[ind0_jy1NonNaN_4T_local,1+2]*Mis_4T[ind0_jy1NonNaN_4T_local,1+2]/RouXc_4T[ind0_jy1NonNaN_4T_local,1]*dt/(delta_y1_4T[ind0_jy1NonNaN_4T_local,1]+delta_y2_4T[ind0_jy1NonNaN_4T_local,1])/Delta_y1_4T[ind0_jy1NonNaN_4T_local,1]    #if ind0_jy1NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jy1NonNaN node , column of up neighbor node(jy1); if ind0_jy1NaN nodes case: elements are zero as initiated 
        MatrixM_4T[ind0_jy2NonNaN_4T,jy2_4T[ind0_jy2NonNaN_4T_local,1]-1] += -Lamda_4T[ind0_jy2NonNaN_4T_local,1+3]*Mis_4T[ind0_jy2NonNaN_4T_local,1+3]/RouXc_4T[ind0_jy2NonNaN_4T_local,1]*dt/(delta_y1_4T[ind0_jy2NonNaN_4T_local,1]+delta_y2_4T[ind0_jy2NonNaN_4T_local,1])/Delta_y2_4T[ind0_jy2NonNaN_4T_local,1]    #if ind0_jy2NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jy2NonNaN node , column of down neighbor node(jy2); if ind0_jy2NaN nodes case: elements are zero as initiated 
        MatrixM_4T[ind0_jz1NonNaN_4T,jz1_4T[ind0_jz1NonNaN_4T_local,1]-1] += -Lamda_4T[ind0_jz1NonNaN_4T_local,1+4]*Mis_4T[ind0_jz1NonNaN_4T_local,1+4]/RouXc_4T[ind0_jz1NonNaN_4T_local,1]*dt/(delta_z1_4T[ind0_jz1NonNaN_4T_local,1]+delta_z2_4T[ind0_jz1NonNaN_4T_local,1])/Delta_z1_4T[ind0_jz1NonNaN_4T_local,1]    #if ind0_jz1NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jz1NonNaN node , column of inner neighbor node(jz1); if ind0_jz1NaN nodes case: elements are zero as initiated 
        MatrixM_4T[ind0_jz2NonNaN_4T,jz2_4T[ind0_jz2NonNaN_4T_local,1]-1] += -Lamda_4T[ind0_jz2NonNaN_4T_local,1+5]*Mis_4T[ind0_jz2NonNaN_4T_local,1+5]/RouXc_4T[ind0_jz2NonNaN_4T_local,1]*dt/(delta_z1_4T[ind0_jz2NonNaN_4T_local,1]+delta_z2_4T[ind0_jz2NonNaN_4T_local,1])/Delta_z2_4T[ind0_jz2NonNaN_4T_local,1]    #if ind0_jz2NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jz2NonNaN node , column of outer neighbor node(jz2); if ind0_jz2NaN nodes case: elements are zero as initiated 
        #--------------------------------------fill in diagonal terms
        MatrixM_4T[ind0_jx1NaN_4T,ind0_jx1NaN_4T] += h_4T[ind0_jx1NaN_4T_local,1+0]*Mis_4T[ind0_jx1NaN_4T_local,1+0]/RouXc_4T[ind0_jx1NaN_4T_local,1]*dt/(delta_x1_4T[ind0_jx1NaN_4T_local,1]+delta_x2_4T[ind0_jx1NaN_4T_local,1])       #jx1 components in diagonal terms
        MatrixM_4T[ind0_jx1NonNaN_4T,ind0_jx1NonNaN_4T] += Lamda_4T[ind0_jx1NonNaN_4T_local,1+0]*Mis_4T[ind0_jx1NonNaN_4T_local,1+0]/RouXc_4T[ind0_jx1NonNaN_4T_local,1]*dt/(delta_x1_4T[ind0_jx1NonNaN_4T_local,1]+delta_x2_4T[ind0_jx1NonNaN_4T_local,1])/Delta_x1_4T[ind0_jx1NonNaN_4T_local,1]
    
        MatrixM_4T[ind0_jx2NaN_4T,ind0_jx2NaN_4T] += h_4T[ind0_jx2NaN_4T_local,1+1]*Mis_4T[ind0_jx2NaN_4T_local,1+1]/RouXc_4T[ind0_jx2NaN_4T_local,1]*dt/(delta_x1_4T[ind0_jx2NaN_4T_local,1]+delta_x2_4T[ind0_jx2NaN_4T_local,1])       #jx2 components in diagonal terms
        MatrixM_4T[ind0_jx2NonNaN_4T,ind0_jx2NonNaN_4T] += Lamda_4T[ind0_jx2NonNaN_4T_local,1+1]*Mis_4T[ind0_jx2NonNaN_4T_local,1+1]/RouXc_4T[ind0_jx2NonNaN_4T_local,1]*dt/(delta_x1_4T[ind0_jx2NonNaN_4T_local,1]+delta_x2_4T[ind0_jx2NonNaN_4T_local,1])/Delta_x2_4T[ind0_jx2NonNaN_4T_local,1]
    
        MatrixM_4T[ind0_jy1NaN_4T,ind0_jy1NaN_4T] += h_4T[ind0_jy1NaN_4T_local,1+2]*Mis_4T[ind0_jy1NaN_4T_local,1+2]/RouXc_4T[ind0_jy1NaN_4T_local,1]*dt/(delta_y1_4T[ind0_jy1NaN_4T_local,1]+delta_y2_4T[ind0_jy1NaN_4T_local,1])       #jy1 components in diagonal terms
        MatrixM_4T[ind0_jy1NonNaN_4T,ind0_jy1NonNaN_4T] += Lamda_4T[ind0_jy1NonNaN_4T_local,1+2]*Mis_4T[ind0_jy1NonNaN_4T_local,1+2]/RouXc_4T[ind0_jy1NonNaN_4T_local,1]*dt/(delta_y1_4T[ind0_jy1NonNaN_4T_local,1]+delta_y2_4T[ind0_jy1NonNaN_4T_local,1])/Delta_y1_4T[ind0_jy1NonNaN_4T_local,1]
    
        MatrixM_4T[ind0_jy2NaN_4T,ind0_jy2NaN_4T] += h_4T[ind0_jy2NaN_4T_local,1+3]*Mis_4T[ind0_jy2NaN_4T_local,1+3]/RouXc_4T[ind0_jy2NaN_4T_local,1]*dt/(delta_y1_4T[ind0_jy2NaN_4T_local,1]+delta_y2_4T[ind0_jy2NaN_4T_local,1])       #jy2 components in diagonal terms
        MatrixM_4T[ind0_jy2NonNaN_4T,ind0_jy2NonNaN_4T] += Lamda_4T[ind0_jy2NonNaN_4T_local,1+3]*Mis_4T[ind0_jy2NonNaN_4T_local,1+3]/RouXc_4T[ind0_jy2NonNaN_4T_local,1]*dt/(delta_y1_4T[ind0_jy2NonNaN_4T_local,1]+delta_y2_4T[ind0_jy2NonNaN_4T_local,1])/Delta_y2_4T[ind0_jy2NonNaN_4T_local,1]
    
        MatrixM_4T[ind0_jz1NaN_4T,ind0_jz1NaN_4T] += h_4T[ind0_jz1NaN_4T_local,1+4]*Mis_4T[ind0_jz1NaN_4T_local,1+4]/RouXc_4T[ind0_jz1NaN_4T_local,1]*dt/(delta_z1_4T[ind0_jz1NaN_4T_local,1]+delta_z2_4T[ind0_jz1NaN_4T_local,1])       #jz1 components in diagonal terms
        MatrixM_4T[ind0_jz1NonNaN_4T,ind0_jz1NonNaN_4T] += Lamda_4T[ind0_jz1NonNaN_4T_local,1+4]*Mis_4T[ind0_jz1NonNaN_4T_local,1+4]/RouXc_4T[ind0_jz1NonNaN_4T_local,1]*dt/(delta_z1_4T[ind0_jz1NonNaN_4T_local,1]+delta_z2_4T[ind0_jz1NonNaN_4T_local,1])/Delta_z1_4T[ind0_jz1NonNaN_4T_local,1]
    
        MatrixM_4T[ind0_jz2NaN_4T,ind0_jz2NaN_4T] += h_4T[ind0_jz2NaN_4T_local,1+5]*Mis_4T[ind0_jz2NaN_4T_local,1+5]/RouXc_4T[ind0_jz2NaN_4T_local,1]*dt/(delta_z1_4T[ind0_jz2NaN_4T_local,1]+delta_z2_4T[ind0_jz2NaN_4T_local,1])       #jz2 components in diagonal terms
        MatrixM_4T[ind0_jz2NonNaN_4T,ind0_jz2NonNaN_4T] += Lamda_4T[ind0_jz2NonNaN_4T_local,1+5]*Mis_4T[ind0_jz2NonNaN_4T_local,1+5]/RouXc_4T[ind0_jz2NonNaN_4T_local,1]*dt/(delta_z1_4T[ind0_jz2NonNaN_4T_local,1]+delta_z2_4T[ind0_jz2NonNaN_4T_local,1])/Delta_z2_4T[ind0_jz2NonNaN_4T_local,1]
    
        MatrixM_4T[self.ind0_interfaces_Module_4T,self.ind0_interfaces_Module_4T] += 1                                                                                               #"1" components in diagonal terms
        #======================================penalty on Temperature-constrained BC nodes (apply temperature BC)
        MatrixM_4T[self.ind0_BCtem_4T_Module,self.ind0_BCtem_4T_Module]=inf

        return MatrixM_4T
    #########################################################   
    ##########     function for VectorM_preTp     ###########
    #########################################################
    def fun_vectorM_preTp(self):          #VectorCN = VectorCN_preTp*Tp + VectorCN_conv_q;    VectorCN_preTp is very similar to MatrixCN, so form VectorCN based on MatrixCN
        VectorM_preTp=self.MatrixM_4T.copy()
        VectorM_preTp[np.arange(self.n_4T_Module),np.arange(self.n_4T_Module)] -= 1
        VectorM_preTp = -VectorM_preTp
        VectorM_preTp[np.arange(self.n_4T_Module),np.arange(self.n_4T_Module)] += 1
        return VectorM_preTp
    #########################################################   
    ###############    function for VectorM    ##############
    #########################################################
    def fun_vectorM_4T(self,params_C2M):               
        #---------------------------  diagonally append VectorCN(s) to form VectorM_4T ---------------------------
        item = ip.status_Parts_name[0]
        VectorM_4T = params_C2M[item]['VectorCN']
        for item in ip.status_Parts_name[1:]:
            VectorM_4T = np.append( VectorM_4T, params_C2M[item]['VectorCN'],axis=0 )
        #--------------------------- clear the rows of interface nodes (ind0_interfaces_Module_4T) ---------------------------
        VectorM_4T[self.ind0_interfaces_Module_4T] = 0
        #--------------------------- add non Tp term - conv
        Mis_4T = self.Parts_link_Mis_4T_Module
        RouXc_4T = self.Parts_link_RouXc_4T_Module
        h_4T = self.h_4T_Module
        Tconv_4T = self.Tconv_4T_Module
        delta_x1_4T = self.Parts_link_delta_x1_4T_Module
        delta_x2_4T = self.Parts_link_delta_x2_4T_Module
        delta_y1_4T = self.Parts_link_delta_y1_4T_Module
        delta_y2_4T = self.Parts_link_delta_y2_4T_Module
        delta_z1_4T = self.Parts_link_delta_z1_4T_Module
        delta_z2_4T = self.Parts_link_delta_z2_4T_Module
        ind0_jx1NaN_4T_local = self.Parts_link_ind0_jx1NaN_4T_local
        ind0_jx2NaN_4T_local = self.Parts_link_ind0_jx2NaN_4T_local
        ind0_jy1NaN_4T_local = self.Parts_link_ind0_jy1NaN_4T_local
        ind0_jy2NaN_4T_local = self.Parts_link_ind0_jy2NaN_4T_local
        ind0_jz1NaN_4T_local = self.Parts_link_ind0_jz1NaN_4T_local
        ind0_jz2NaN_4T_local = self.Parts_link_ind0_jz2NaN_4T_local
        ind0_jx1NaN_4T = self.Parts_link_jx1_4T_Module[ind0_jx1NaN_4T_local,0]
        ind0_jx2NaN_4T = self.Parts_link_jx2_4T_Module[ind0_jx2NaN_4T_local,0]
        ind0_jy1NaN_4T = self.Parts_link_jy1_4T_Module[ind0_jy1NaN_4T_local,0]
        ind0_jy2NaN_4T = self.Parts_link_jy2_4T_Module[ind0_jy2NaN_4T_local,0]
        ind0_jz1NaN_4T = self.Parts_link_jz1_4T_Module[ind0_jz1NaN_4T_local,0]
        ind0_jz2NaN_4T = self.Parts_link_jz2_4T_Module[ind0_jz2NaN_4T_local,0]
        dt = self.dt

        VectorM_conv_q=np.zeros([self.n_4T_Module,1])

        VectorM_conv_q[ind0_jx1NaN_4T,0] += h_4T[ind0_jx1NaN_4T_local,1+0]*Mis_4T[ind0_jx1NaN_4T_local,1+0]/RouXc_4T[ind0_jx1NaN_4T_local,1]*2*dt/(delta_x1_4T[ind0_jx1NaN_4T_local,1]+delta_x2_4T[ind0_jx1NaN_4T_local,1]) * Tconv_4T[ind0_jx1NaN_4T_local,1+0]                                                                                               #if ind0_jx1NaN nodes case: fill elements of the jx1 terms
        VectorM_conv_q[ind0_jx2NaN_4T,0] += h_4T[ind0_jx2NaN_4T_local,1+1]*Mis_4T[ind0_jx2NaN_4T_local,1+1]/RouXc_4T[ind0_jx2NaN_4T_local,1]*2*dt/(delta_x1_4T[ind0_jx2NaN_4T_local,1]+delta_x2_4T[ind0_jx2NaN_4T_local,1]) * Tconv_4T[ind0_jx2NaN_4T_local,1+1]                                                                                               #if ind0_jx2NaN nodes case: fill elements of the jx2 terms
        VectorM_conv_q[ind0_jy1NaN_4T,0] += h_4T[ind0_jy1NaN_4T_local,1+2]*Mis_4T[ind0_jy1NaN_4T_local,1+2]/RouXc_4T[ind0_jy1NaN_4T_local,1]*2*dt/(delta_y1_4T[ind0_jy1NaN_4T_local,1]+delta_y2_4T[ind0_jy1NaN_4T_local,1]) * Tconv_4T[ind0_jy1NaN_4T_local,1+2]                                                                                               #if ind0_jy1NaN nodes case: fill elements of the jy1 terms
        VectorM_conv_q[ind0_jy2NaN_4T,0] += h_4T[ind0_jy2NaN_4T_local,1+3]*Mis_4T[ind0_jy2NaN_4T_local,1+3]/RouXc_4T[ind0_jy2NaN_4T_local,1]*2*dt/(delta_y1_4T[ind0_jy2NaN_4T_local,1]+delta_y2_4T[ind0_jy2NaN_4T_local,1]) * Tconv_4T[ind0_jy2NaN_4T_local,1+3]                                                                                               #if ind0_jy2NaN nodes case: fill elements of the jy2 terms
        VectorM_conv_q[ind0_jz1NaN_4T,0] += h_4T[ind0_jz1NaN_4T_local,1+4]*Mis_4T[ind0_jz1NaN_4T_local,1+4]/RouXc_4T[ind0_jz1NaN_4T_local,1]*2*dt/(delta_z1_4T[ind0_jz1NaN_4T_local,1]+delta_z2_4T[ind0_jz1NaN_4T_local,1]) * Tconv_4T[ind0_jz1NaN_4T_local,1+4]                                                                                               #if ind0_jz1NaN nodes case: fill elements of the jz1 terms
        VectorM_conv_q[ind0_jz2NaN_4T,0] += h_4T[ind0_jz2NaN_4T_local,1+5]*Mis_4T[ind0_jz2NaN_4T_local,1+5]/RouXc_4T[ind0_jz2NaN_4T_local,1]*2*dt/(delta_z1_4T[ind0_jz2NaN_4T_local,1]+delta_z2_4T[ind0_jz2NaN_4T_local,1]) * Tconv_4T[ind0_jz2NaN_4T_local,1+5]                                                                                               #if ind0_jz2NaN nodes case: fill elements of the jz2 terms
        #--------------------------- add non Tp term - heat gen q
        item = ip.status_Parts_name[0]
        q_4T_Module = params_C2M[item]['q_4T_ALL']
        for item in ip.status_Parts_name[1:]:
            q_4T_Module = np.append( q_4T_Module, params_C2M[item]['q_4T_ALL'],axis=0 )
        
        q_4T = q_4T_Module[self.ind0_interfaces_Module_4T]
        
        VectorM_conv_q[self.ind0_interfaces_Module_4T[self.ind0_interfaces_Module_4T_Unique_local],0] += q_4T[self.ind0_interfaces_Module_4T_Unique_local,0]*dt/RouXc_4T[self.ind0_interfaces_Module_4T_Unique_local,1]                                                                                                                                                                                                     #heat gen components
        #--------------------------- form VectorM
        item = ip.status_Parts_name[0]
        T1_4T_Module = params_C2M[item]['T1_4T_ALL']
        T3_4T_Module = params_C2M[item]['T3_4T_ALL']
        for item in ip.status_Parts_name[1:]:
            T1_4T_Module = np.append( T1_4T_Module, params_C2M[item]['T1_4T_ALL'],axis=0 )
            T3_4T_Module = np.append( T3_4T_Module, params_C2M[item]['T3_4T_ALL'],axis=0 )
        VectorM_4T[self.ind0_interfaces_Module_4T]= self.VectorM_preTp[self.ind0_interfaces_Module_4T,:] .dot( T1_4T_Module ) + VectorM_conv_q[self.ind0_interfaces_Module_4T]
        #======================================penalty on Temperature-constrained BC nodes (apply temperature BC)
        VectorM_4T[self.ind0_BCtem_4T_Module,0]=(T3_4T_Module[self.ind0_BCtem_4T_Module,0]*inf)

        return VectorM_4T
    #########################################################   
    #########     function for Postprocessor     ############
    #########################################################
    if ip.status_FormFactor == 'Prismatic':
        def fun_Postprocessor(self,step,t_begin,t_end,params_C2M):        
            if self.status_BC_Module_4T == 'Prismatic_Cell1':
                plt.figure()
                plt.plot(self.t_record,self.T_avg_record-273.15,'bo')
                plt.title('Tavg for dual jellyroll')
                plt.figure()
                plt.plot(self.t_record,self.T_Delta_record,'bo')    
                plt.title('Tmax-Tmin for dual jellyroll')
    else:
        def fun_Postprocessor(self,step,t_begin,t_end):        
            if self.status_BC_Module_4T == 'Prismatic_Cell1':
                plt.figure()
                plt.plot(self.t_record,self.T_avg_record-273.15,'bo')
                plt.title('Tavg for dual jellyroll')
                plt.figure()
                plt.plot(self.t_record,self.T_Delta_record,'bo')    
                plt.title('Tmax-Tmin for dual jellyroll')
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
