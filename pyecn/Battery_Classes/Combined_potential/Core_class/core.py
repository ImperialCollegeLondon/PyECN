# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
from mayavi import mlab
import scipy.sparse.linalg
import scipy.sparse
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from moviepy.video.io.bindings import mplfig_to_npimage
import imageio
import time
import os, sys    #for rerun the script for PID FOPDT calibration
import scipy.io as sio 

import pyecn.parse_inputs as ip

if ip.status_visualization_method == 'plotly':
    from plotly.offline import plot
    import plotly.graph_objects as go
inf=1e10


##########################################################################################################################
################### ↓↓↓  Importing specific form factor classes to the core class ↓↓↓ ####################################
##########################################################################################################################

if ip.status_FormFactor=='Prismatic':
    from pyecn.Battery_Classes.Combined_potential.Form_factor_classes.prismatic import Prismatic
    Form_Factor = Prismatic
if ip.status_FormFactor=='Pouch':
    from pyecn.Battery_Classes.Combined_potential.Form_factor_classes.pouch import Pouch
    Form_Factor = Pouch
if ip.status_FormFactor=='Cylindrical':
    from pyecn.Battery_Classes.Combined_potential.Form_factor_classes.cylindrical import Cylindrical
    Form_Factor = Cylindrical
    
##########################################################################################################################
#################### ↑↑↑  Importing specific form factor classes to the core class ↑↑↑ ###################################
##########################################################################################################################
    

##########################################################################################################################
################################# ↓↓↓  Importing LUT classes to the core class ↓↓↓ #######################################
##########################################################################################################################

from pyecn.Battery_Classes.Combined_potential.LUT_class.LUT import  Read_LUTs

##########################################################################################################################
################################# ↑↑↑  Importing LUT classes to the core class ↑↑↑ #######################################
##########################################################################################################################


class Core(Form_Factor, Read_LUTs):
    def __init__(self, params_update, cell_ind0: int):
#        self.__dict__ = ip.__dict__.copy()       #copy the default inputs into this 'self'
#        self.__dict__.update(params_update)      #update the unique inputs for this 'self'               
        #---------copy attr in inputs.py into this class---------
        my_shelf = {}
        for key in dir(ip):
            if not key.startswith("__"):         #filter out internal attributes like __builtins__ etc
                my_shelf[key] = ip.__dict__[key]
        self.__dict__ = my_shelf.copy()
        self.__dict__.update(params_update)      #update the update(unique) inputs for this 'self'        
        self.cell_ind0 = cell_ind0
        #--------------------------------------------------------        
        #super().__init__(params_update)     #<-----Always place it here and not before my_shelf, this is used when you keep the instances of classes such as Pouch in that class itself and call them in this class (Point number 74 code tracker)

        
        ####################################
        ###CREATE class - self ATTRIBUTES###
        ####################################
        self.delta_An = self.delta_An_real * self.scalefactor_z
        self.delta_Ca = self.delta_Ca_real * self.scalefactor_z
        self.delta_Sep = self.delta_Sep_real * self.scalefactor_z
        self.delta_Al = self.delta_Al_real * self.scalefactor_z
        self.delta_Cu = self.delta_Cu_real * self.scalefactor_z
        self.delta_El = self.delta_El_real * self.scalefactor_z

        self.Lamda_El_x=(self.Lamda_An*self.delta_An_real + self.Lamda_Ca*self.delta_Ca_real + self.Lamda_Sep*self.delta_Sep_real)/self.delta_El_real;  self.Lamda_El_y=self.Lamda_El_x              #Electrode thermal conductivity in x and y direction  (in parallel)
        self.Lamda_El_z=self.delta_El_real/(self.delta_An_real/self.Lamda_An + self.delta_Sep_real/self.Lamda_Sep + self.delta_Ca_real/self.Lamda_Ca)                                      #Electrode thermal conductivity in z direction  (in series)     
        self.rou_El=self.rou_An*(self.delta_An/self.delta_El) + self.rou_Ca*(self.delta_Ca/self.delta_El) + self.rou_Sep*(self.delta_Sep/self.delta_El)
        self.rouXc_El=self.rou_An*self.c_An*(self.delta_An/self.delta_El) + self.rou_Ca*self.c_Ca*(self.delta_Ca/self.delta_El) + self.rou_Sep*self.c_Sep*(self.delta_Sep/self.delta_El)
        self.Alpha_El_x=self.Lamda_El_x/self.rouXc_El; self.Alpha_El_y=self.Alpha_El_x; self.Alpha_El_z=self.Lamda_El_z/self.rouXc_El                #α=λ/(ρ·c)
        self.Alpha_Sep=self.Lamda_Sep/self.rou_Sep/self.c_Sep
        
        if ip.status_FormFactor == 'Pouch':
            self.A_electrodes_real=self.Lx_electrodes_real*self.Ly_electrodes_real*(2*self.nstack_real-1)
            self.Lx=self.Lx_cell-self.Lx_cell/self.nx 
            self.Ly=self.Ly_cell-self.Ly_cell/self.ny
        
        if ip.status_FormFactor == 'Prismatic':
            self.theta_unit0 = 2*np.pi/self.nx_cylindrical* np.arange(self.nx_cylindrical)             #e.g. when nx_cylindrical=5, theta_unit0 is 0, π/5, 2π/5, 3π/5, 4π/5 
            self.ind0_SpiralandStripe_boundary1 = np.abs(self.theta_unit0-np.pi/2).argmin()       #0 index of theta_unit for cylindrical/pouch boundary1
            self.ind0_SpiralandStripe_boundary2 = np.abs(self.theta_unit0-np.pi*3/2).argmin()
            self.theta_unit0[ self.ind0_SpiralandStripe_boundary1 ] = np.pi/2                     #e.g. when nx_cylindrical=5, theta_unit0 is forced into 0, π/2, 2π/5, 3π/5, 3π/2
            self.theta_unit0[ self.ind0_SpiralandStripe_boundary2 ] = np.pi*3/2
            self.SpiralandStripe_Sep_s_real=self.fun_SpiralandStripefrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi,self.delta_El_real/2+self.delta_Al_real+self.delta_core_real,self.Lx_pouch,    self.nstack_real*2*np.pi,1 )
            self.SpiralandStripe_Sep_l_real=self.fun_SpiralandStripefrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi,self.delta_El_real/2+self.delta_Al_real+self.delta_core_real+self.delta_El_real+self.delta_Cu_real,self.Lx_pouch,  self.nstack_real*2*np.pi,1 )   #Seperator Sprial length i.e. shorter one and longer one; these are obtained by running "Spiral_Sep_s_real=fun_spiralfrom0(a0,b03,nstack_real*2*np.pi) and Spiral_Sep_l_real=fun_spiralfrom0(a0,b04,nstack_real*2*np.pi)" 
            self.A_electrodes_real=self.Ly_electrodes_real*(self.SpiralandStripe_Sep_s_real+self.SpiralandStripe_Sep_l_real)      #active electrodes length, width and area. These are for calculating elementary R and C 
        if ip.status_FormFactor == 'Cylindrical':
            self.Spiral_Sep_s_real = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_El_real/2+self.delta_Al_real+self.delta_core_real, self.nstack_real*2*np.pi ) 
            self.Spiral_Sep_l_real = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_El_real/2+self.delta_Al_real+self.delta_core_real+self.delta_El_real+self.delta_Cu_real,self.nstack_real*2*np.pi )   #Seperator Sprial length i.e. shorter one and longer one; these are obtained by running "Spiral_Sep_s_real=fun_spiralfrom0(a0,b03,nstack_real*2*np.pi) and Spiral_Sep_l_real=fun_spiralfrom0(a0,b04,nstack_real*2*np.pi)" 
            
            self.Spiral_Sep_s=self.fun_spiralfrom0( (2*self.delta_El+self.delta_Cu+self.delta_Al)/2/np.pi,self.delta_El/2+self.delta_Al+self.delta_core_real,self.nstack*2*np.pi )
            self.Spiral_Sep_l=self.fun_spiralfrom0( (2*self.delta_El+self.delta_Cu+self.delta_Al)/2/np.pi,self.delta_El/2+self.delta_Al+self.delta_core_real+self.delta_El+self.delta_Cu,self.nstack*2*np.pi )   #Seperator Sprial length i.e. shorter one and longer one; these are obtained by running "Spiral_Sep_s_real=fun_spiralfrom0(a0,b03,nlap_real*2*np.pi) and Spiral_Sep_l_real=fun_spiralfrom0(a0,b04,nlap_real*2*np.pi)" 
            
            self.A_electrodes_real = self.Ly_electrodes_real*(self.Spiral_Sep_s_real+self.Spiral_Sep_l_real)
        if ip.status_FormFactor == 'Cylindrical' or ip.status_FormFactor == 'Prismatic':
            self.LG = self.LG_Jellyroll-self.LG_Jellyroll/self.ny                        #Refer to ppt1 p242 for LG_Jellyroll, LG_Can and LG
            self.a0 = (2*self.delta_El+self.delta_Cu+self.delta_Al)/2/np.pi
            self.b0 = self.delta_El/2+self.delta_Al + self.delta_core_real                    #spiral geometry for center r=a0θ+b0      b01=delta_core_real+delta_Al/2     see notebook p21
            
        self.ne = self.nRC+1                                 #number of nodes in lumped ECN
        self.nz = 2*self.nstack+self.ne*(2*self.nstack-1)            #number of nodes in radial direction
        self.ntotal = self.nx*self.ny*self.nz
        self.nECN = (2*self.nstack-1)*self.nx*self.ny                    #number of lumped ECN, 36
        if ip.status_FormFactor == 'Pouch':
            self.nRAl = ((self.nx-1)*self.ny+(self.ny-1)*self.nx)*self.nstack; self.nRCu = self.nRAl  #number of CC(Al or Cu) resistances
            self.delta_cell = (self.delta_Al+self.delta_Cu+self.delta_El*2)*(self.nstack-1) +self.delta_Al+self.delta_Cu+self.delta_El                        #largest radius for the entire cell
            
            (
            self.node, self.xn, self.yn, self.zn, self.mat, self.xi, self.yi, self.zi, self.V, self.V_ele, self.Axy_ele, self.Al, self.Cu, self.Elb, self.Elr, 
            self.jx1, self.jx2, self.jy1, self.jy2, self.jz1, self.jz2, self.ind0_jx1, self.ind0_jx2, self.ind0_jy1, self.ind0_jy2, self.ind0_jz1, self.ind0_jz2,
            self.ind0_jx1NaN, self.ind0_jx2NaN, self.ind0_jy1NaN, self.ind0_jy2NaN, self.ind0_jz1NaN, self.ind0_jz2NaN, self.ind0_jx1NonNaN, self.ind0_jx2NonNaN, self.ind0_jy1NonNaN, self.ind0_jy2NonNaN, self.ind0_jz1NonNaN, self.ind0_jz2NonNaN  
            ) = self.fun_matrix1()
        if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical': 
            self.nRAl = (self.nx*(self.nstack-1)+(self.nx-1))*self.ny + self.nx*self.nstack*(self.ny-1); self.nRCu = self.nRAl  #number of CC(Al or Cu) resistances: (nx*(nstack-1)+(nx-1))*ny is the number in horizontal , nx*nstack*(ny-1) is the number in vertical
            self.b01 = self.b0-self.delta_El/2-self.delta_Al/2; self.b02 = self.b0+self.delta_El/2+self.delta_Al/2; self.b03 = self.b0; self.b04 = self.b0+self.delta_El+self.delta_Cu; self.b0_Sep = self.delta_core_real
            self.delta_cell = (self.delta_Al+self.delta_Cu+self.delta_El*2)*self.nstack+self.delta_Al+self.delta_Cu+self.delta_El + self.delta_core_real                        #largest radius for the entire cell
            
            (
            self.node, self.ax, self.ra, self.an, self.lap, self.theta, self.mat, self.xi, self.yi, self.zi, self.V, self.V_ele, self.Lx_ele, self.Axy_ele, self.node_ele, self.Al, self.Cu, self.Elb, self.Elr,                                              
            self.jx1, self.jx2, self.jy1, self.jy2, self.jz1, self.jz2, self.ind0_jx1, self.ind0_jx2, self.ind0_jy1, self.ind0_jy2, self.ind0_jz1, self.ind0_jz2,                                                                                             
            self.ind0_jx1NaN, self.ind0_jx2NaN, self.ind0_jy1NaN, self.ind0_jy2NaN, self.ind0_jz1NaN, self.ind0_jz2NaN, self.ind0_jx1NonNaN, self.ind0_jx2NonNaN, self.ind0_jy1NonNaN, self.ind0_jy2NonNaN, self.ind0_jz1NonNaN, self.ind0_jz2NonNaN         
            ) = self.fun_matrix1()                                                                                                                                                                                                                           
        self.AlCu=np.append(self.Al,self.Cu)
        self.El=np.sort(np.append(self.Elb, self.Elr))
        self.nCC=np.size(self.Al) + np.size(self.Cu)
        self.A_electrodes_eff = np.sum(self.Axy_ele*self.scalefactor_z)                   #see ppt1 p276

        nRC_backup=self.nRC; ne_backup=self.ne; nz_backup=self.nz; ntotal_backup=self.ntotal  #to calculate Matrix1 for Thermal model, nRC should be 0 temporarily, but later should be changed back, 
        self.nRC=0; self.ne=self.nRC+1; self.nz=2*self.nstack+self.ne*(2*self.nstack-1); self.ntotal=self.nx*self.ny*self.nz          #\likewise for variables related with nRC. nRC_backup used for recovering nRC later
        if ip.status_FormFactor == 'Pouch':
            
            (
            self.node_4T, self.xn_4T, self.yn_4T, self.zn_4T, self.mat_4T, self.xi_4T, self.yi_4T, self.zi_4T, self.V_4T, self.V_ele_4T, self.Axy_ele_4T, self.Al_4T, self.Cu_4T, self.Elb_4T, self.Elr_4T, 
            self.jx1_4T, self.jx2_4T, self.jy1_4T, self.jy2_4T, self.jz1_4T, self.jz2_4T, self.ind0_jx1_4T, self.ind0_jx2_4T, self.ind0_jy1_4T, self.ind0_jy2_4T, self.ind0_jz1_4T, self.ind0_jz2_4T,
            self.ind0_jx1NaN_4T, self.ind0_jx2NaN_4T, self.ind0_jy1NaN_4T, self.ind0_jy2NaN_4T, self.ind0_jz1NaN_4T, self.ind0_jz2NaN_4T, self.ind0_jx1NonNaN_4T, self.ind0_jx2NonNaN_4T, self.ind0_jy1NonNaN_4T, self.ind0_jy2NonNaN_4T, self.ind0_jz1NonNaN_4T, self.ind0_jz2NonNaN_4T  
            ) = self.fun_matrix1() 
        if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':
            (
            self.node_4T, self.ax_4T, self.ra_4T, self.an_4T, self.lap_4T, self.theta_4T, self.mat_4T, self.xi_4T, self.yi_4T, self.zi_4T,                                  
            self.V_4T, self.V_ele_4T, self.Lx_ele_4T, self.Axy_ele_4T, self.node_ele_4T, self.Al_4T, self.Cu_4T, self.Elb_4T, self.Elr_4T,                                  
            self.jx1_4T, self.jx2_4T, self.jy1_4T, self.jy2_4T, self.jz1_4T, self.jz2_4T,                                                                                   
            self.ind0_jx1_4T, self.ind0_jx2_4T, self.ind0_jy1_4T, self.ind0_jy2_4T, self.ind0_jz1_4T, self.ind0_jz2_4T,                                                     
            self.ind0_jx1NaN_4T, self.ind0_jx2NaN_4T, self.ind0_jy1NaN_4T, self.ind0_jy2NaN_4T, self.ind0_jz1NaN_4T, self.ind0_jz2NaN_4T,                                   
            self.ind0_jx1NonNaN_4T, self.ind0_jx2NonNaN_4T, self.ind0_jy1NonNaN_4T, self.ind0_jy2NonNaN_4T, self.ind0_jz1NonNaN_4T, self.ind0_jz2NonNaN_4T                 
            ) = self.fun_matrix1()                                                                                                                        
        self.ntotal_4T=np.size(self.node_4T); self.nz_4T=2*self.nstack+self.ne*(2*self.nstack-1)
        self.El_4T=np.sort(np.append(self.Elb_4T, self.Elr_4T))
        self.nRC=nRC_backup; self.ne=ne_backup; self.nz=nz_backup; self.ntotal=ntotal_backup
        
        self.List_node2ele=self.fun_GetList_node2ele()           #made for El nodes
        self.List_node2ele_4T=self.fun_GetList_node2ele_4T()     #made for El nodes
        self.List_ele2node=self.fun_GetList_ele2node()           #made for El nodes
        self.List_ele2node_4T=self.fun_GetList_ele2node_4T()     #made for El nodes
        self.List_node2node_E2T=self.fun_GetList_node2node_E2T() #made for CC nodes
        self.List_node2node_T2E=self.fun_GetList_node2node_T2E() #made for CC nodes
        ###   electrical BC   ###
        self.fun_get_Geo()
        if ip.status_FormFactor == 'Pouch':
            self.node_positive_0ind=self.ind0_Geo_left_Al              #positive node 0index
            self.node_negative_0ind=self.ind0_Geo_right_Cu             #negative node 0index
        
        if ip.status_FormFactor == 'Prismatic':
            self.node_positive_0ind=self.ind0_Geo_top2_10_102_110                                     #positive node: line  ind0=[1,9,101,109]
            self.node_negative_0ind=self.ind0_Geo_top55_57_155_157                                    #negative node: line  ind0=[54,56,154,156]

        if ip.status_FormFactor == 'Cylindrical': 
            self.n_top_Al=np.size(self.ind0_Geo_top_Al) 
            self.node_positive_0ind=self.ind0_Geo_top_Al[np.array([self.n_top_Al*1/3]).astype(int)]                                            #positive node: ind0=[0]  
            self.node_negative_0ind=self.ind0_Geo_bottom[-1:]                                        #negative node: line  ind0=[191]             
            if ip.status_electrical_tab == 'Tabless_virtual':
                self.node_positive_0ind=self.ind0_Geo_top_Al
                self.node_negative_0ind=self.ind0_Geo_bottom_Cu
        
        self.nPos=np.size(self.node_positive_0ind); self.nNeg=np.size(self.node_negative_0ind); self.ntab=self.nPos+self.nNeg
        self.node_positive_0ind_4T=self.List_node2node_E2T[self.node_positive_0ind,0]
        self.node_negative_0ind_4T=self.List_node2node_E2T[self.node_negative_0ind,0]

        #--------------------------------------------------------------------------preparation for modification on MatrixC and VectorI (preprocessor)
        print('\nrunning preprocessor...\n')
        ### getting initial voltage potential, MatrixC and I ###
        self.fun_IC()
        self.U_pndiff_plot=np.nan*np.zeros([self.nt+1])   #for plotting positive negative voltage difference
        self.U_pndiff_plot[0]=self.Uini[self.node_positive_0ind[0]]-self.Uini[self.node_negative_0ind[0]]
        self.I0_record=np.nan*np.zeros([self.nt+1])       #record I0
        self.I0_record[0]=self.status_discharge * self.Capacity_rated0/3600*self.C_rate
        #coulomb_ele=np.zeros([nECN,1])  #(for discharge)  coulomb counting for last nECN variables in unknown vector U; form is similar to Ei_pair
        self.Vratio_ele=self.V_ele/np.sum(self.V_ele)  #Vratio is elementary volume ratio, in the form of 1,2...nECN
        self.coulomb_ele_rated0=self.Capacity_rated0*self.Vratio_ele; self.coulomb_ele_rated=self.coulomb_ele_rated0; self.Capacity_rated=self.Capacity_rated0  #coulomb0_ele_rated is the EoL rated elementary capacity, in the form of 1,2...nECN;  coulomb0_ele_rated0 is the initial BoL elementary capacity
        self.coulomb_ele=self.Capacity0*self.Vratio_ele             #coulomb_ele here is the initial elementary capacity, in the form of 1,2...nECN
        self.Coulomb_Counting_As=np.zeros([self.nt+1,1])       #overall coulomb counting, in the form of 1,2...nt
        self.Charge_Throughput_As=np.zeros([self.nt+1,1])      #overall coulomb counting, in the form of 1,2...nt
        self.SoC=np.nan*np.zeros([self.nt+1])                  #SoC for entire cell
        if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical' :
            self.nAddCore_4T=0                                 #For status_ThermalBC_Core=='BCFix',no core nodes are added. For status_ThermalBC_Core=='SepFill' and 'InsuFill', core nodes are added
        self.plot_type= ip.status_plot_type_preprocess                                #For fun_plot(), differentiating instant plotting and replay plotting
        ### getting initial T ###
        if self.status_Model=='EandT' or self.status_Model=='E':
            self.fun_pre_matrixC()        
        if ip.status_FormFactor == 'Pouch':
            self.n_4T_ALL=self.ntotal_4T
            self.node_4T_ALL=self.node_4T 
        if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':
            if self.status_ThermalBC_Core=='SepFill':
                self.Alpha_Fill=self.Alpha_Sep
                self.nAddCore_4T=self.ny
                self.n_4T_ALL=self.ntotal_4T+self.nAddCore_4T; self.node_4T_4SepFill=np.arange(self.ntotal_4T+self.nAddCore_4T)+1
                if ip.status_FormFactor == 'Prismatic':
                    self.theta0=2*np.pi*(1-1/2/self.nx); self.S_SpiralandLine=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Sep*self.theta0**2+3*self.b0_Sep**2*self.theta0)/6 + self.Lx_pouch*( self.fun_SpiralandStripefrom0(self.a0,self.b0_Sep,self.Lx_pouch,  np.pi/2,self.ind0_SpiralandStripe_boundary1+1) + self.fun_SpiralandStripefrom0(self.a0,self.b0_Sep,self.Lx_pouch,  np.pi*3/2,self.ind0_SpiralandStripe_boundary2+1+2*self.nx_pouch) )   #prep S_spiral for Sep nodes
                else:
                    self.theta0=2*np.pi*(1-1/2/self.nx); self.S_spiral=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Sep*self.theta0**2+3*self.b0_Sep**2*self.theta0)/6   #prep S_spiral for Sep nodes 
            if self.status_ThermalBC_Core=='InsuFill':
                self.Alpha_Fill=0
                self.status_ThermalBC_Core='SepFill'
                self.nAddCore_4T=self.ny
                self.n_4T_ALL=self.ntotal_4T+self.nAddCore_4T; self.node_4T_4SepFill=np.arange(self.ntotal_4T+self.nAddCore_4T)+1
                if ip.status_FormFactor == 'Prismatic':
                    self.theta0=2*np.pi*(1-1/2/self.nx); self.S_SpiralandLine=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Sep*self.theta0**2+3*self.b0_Sep**2*self.theta0)/6 + self.Lx_pouch*( self.fun_SpiralandStripefrom0(self.a0,self.b0_Sep,self.Lx_pouch,  np.pi/2,self.ind0_SpiralandStripe_boundary1+1) + self.fun_SpiralandStripefrom0(self.a0,self.b0_Sep,self.Lx_pouch,  np.pi*3/2,self.ind0_SpiralandStripe_boundary2+1+2*self.nx_pouch) )   #prep S_spiral for Sep nodes
                else:
                    self.theta0=2*np.pi*(1-1/2/self.nx); self.S_spiral=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Sep*self.theta0**2+3*self.b0_Sep**2*self.theta0)/6   #prep S_spiral for Sep nodes
            if self.status_ThermalBC_Core=='SepAir':
                self.Alpha_Fill=self.Alpha_Sep
                self.nAddCore_4T=self.ny
                self.n_4T_ALL=self.ntotal_4T+self.nAddCore_4T; self.node_4T_4SepFill=np.arange(self.ntotal_4T+self.nAddCore_4T)+1
                self.theta0=2*np.pi*(1-1/2/self.nx)
                self.b0_Air=self.n_Air*self.b0_Sep
                if ip.status_FormFactor == 'Prismatic':
                    self.S_SpiralandLine_air=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Air*self.theta0**2+3*self.b0_Air**2*self.theta0)/6 + self.Lx_pouch*( self.fun_SpiralandStripefrom0(self.a0,self.b0_Air,self.Lx_pouch,  np.pi/2,self.ind0_SpiralandStripe_boundary1+1) + self.fun_SpiralandStripefrom0(self.a0,self.b0_Air,self.Lx_pouch,  np.pi*3/2,self.ind0_SpiralandStripe_boundary2+1+2*self.nx_pouch) )
                    self.S_SpiralandLine=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Sep*self.theta0**2+3*self.b0_Sep**2*self.theta0)/6 + self.Lx_pouch*( self.fun_SpiralandStripefrom0(self.a0,self.b0_Sep,self.Lx_pouch,  np.pi/2,self.ind0_SpiralandStripe_boundary1+1) + self.fun_SpiralandStripefrom0(self.a0,self.b0_Sep,self.Lx_pouch,  np.pi*3/2,self.ind0_SpiralandStripe_boundary2+1+2*self.nx_pouch) ) - self.S_SpiralandLine_air               #prep S_spiral for Sep nodes
                else:
                    self.S_spiral_air=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Air*self.theta0**2+3*self.b0_Air**2*self.theta0)/6
                    self.S_spiral=(self.a0**2*self.theta0**3+3*self.a0*self.b0_Sep*self.theta0**2+3*self.b0_Sep**2*self.theta0)/6 - self.S_spiral_air               #prep S_spiral for Sep nodes 
            if ip.status_FormFactor == 'Cylindrical':
                if self.status_ThermalPatition_Can=='Yes':

                    self.node_Can_4T, self.ax_Can_4T, self.ra_Can_4T, self.an_Can_4T, self.theta_Can_4T, self.mat_Can_4T, self.xi_Can_4T, self.yi_Can_4T, self.zi_Can_4T, self.Can_4T,                                                  \
                    self.jx1_Can_4T, self.jx2_Can_4T, self.jy1_Can_4T, self.jy2_Can_4T, self.jz1_Can_4T, self.jz2_Can_4T, self.jxz_Can_4T,                                                                                              \
                    self.ind0_jx1NaN_Can_4T, self.ind0_jx2NaN_Can_4T, self.ind0_jy1NaN_Can_4T, self.ind0_jy2NaN_Can_4T, self.ind0_jz1NaN_Can_4T, self.ind0_jz2NaN_Can_4T, self.ind0_jxzNaN_Can_4T,                                      \
                    self.ind0_jx1NonNaN_Can_4T, self.ind0_jx2NonNaN_Can_4T, self.ind0_jy1NonNaN_Can_4T, self.ind0_jy2NonNaN_Can_4T, self.ind0_jz1NonNaN_Can_4T, self.ind0_jz2NonNaN_Can_4T, self.ind0_jxzNonNaN_Can_4T,                 \
                    self.nCanTop_4T, self.nCanSurface_4T, self.nCanBottom_4T, self.nCan_4T                                                                                                                                              \
                    =                                                                                                                                                                                                                   \
                    self.fun_matrix1_Can_4T()
                    
                    self.n_4T_ALL=self.ntotal_4T+self.nAddCore_4T+self.nCan_4T;    self.node_4T_ALL=np.arange(self.ntotal_4T+self.nAddCore_4T+self.nCan_4T)+1
                    self.Alpha_Can_Base=self.Lamda_Can/self.rou_Can/self.c_Can_Base
        if self.status_Model=='EandT' or self.status_Model=='T':
            self.fun_get_Geo_4T()
            if ip.status_FormFactor == 'Cylindrical':
                if self.status_ThermalPatition_Can=='Yes':
                    self.fun_get_Geo_Can_4T()
            self.fun_pre_Thermal()
            if self.status_TemBC_smoothening=='Yes':
                self.T_cooling_smoothened=self.T_cooling + (self.T_initial-self.T_cooling)/np.exp(self.smoothening_stiffness * 0)
            if ip.status_FormFactor == 'Cylindrical':
                if self.status_Module_4T == 'Yes' and self.status_BC_Module_4T == 'Ribbon_cooling':
                    self.S_cooling_nx = np.unique(self.an_Can_4T[self.ind0_Geo_Can_S_cooling_node38_4T]).size
                    self.S_cooling_ny = np.unique(self.ax_Can_4T[self.ind0_Geo_Can_S_cooling_node38_4T]).size
                    l_temp = 2*np.pi*self.zi_Can_4T[self.ind0_Geo_Can_node34to37_4T[0]]  #perimeter of metal can outmost circle
                    self.S_cooling_size_x_model = self.S_cooling_nx * l_temp/self.nx 
                    self.S_cooling_size_y_model = self.S_cooling_ny * self.LG_Jellyroll/self.ny 
    #                print('S-shaped cooling thermal BC is applied.\n')
    #                print('Input rectangle in x,y dimension is: %.2fm x %.2fm.\nIn the current model, rectangle x,y dimension is: %.2fm x %.2fm;\nnumber of included nodes in x,y dimension is: %d x %d\n '%(self.S_cooling_size_x,self.S_cooling_size_y,self.S_cooling_size_x_model,self.S_cooling_size_y_model,self.S_cooling_nx,self.S_cooling_ny))
    #                temp1=input('press y to continue, n to break:')
    #                if temp1 != 'y':
    #                    raise Exception('exit')
            self.Tini_4T_ALL=self.fun_IC_4T()                                                  #apply Thermal initial condition
            if self.status_Thermal_solver=='CN':
                self.MatrixCN=self.fun_MatrixCN()
                if self.status_linsolver_T=='BandSparse':
                    [self.length_MatrixCN, self.ind0_l, self.ind0_u, self.ind0_r_expand, self.ind0_c_expand]=self.fun_band_matrix_precompute(self.MatrixCN)    #for BandSparse linear equations solver, diagonal ordered form is needed
                self.VectorCN_preTp=self.fun_VectorCN_preTp()
        else:
            self.fun_get_Geo_4T()
            if ip.status_FormFactor == 'Cylindrical':
                if self.status_ThermalPatition_Can=='Yes':
                    self.fun_get_Geo_Can_4T()
            self.fun_pre_Thermal()     #this is only for plotting, in fun_plot()
            self.Tini_4T_ALL=self.T_fixed* np.ones([self.n_4T_ALL,1])
        if self.status_Thermal_solver=='Explicit':                                      #if explicit solver, use stability requirement to check time step 
            self.fun_explicit_stability_check()        
        if self.status_PopFig_or_SaveGIF_instant=='Fig' or self.status_PopFig_or_SaveGIF_instant=='GIF':
            fig1, axs_fig1=plt.subplots(nrows=2, ncols=1)
            fig2, axs_fig2=plt.subplots(nrows=2, ncols=1)  
            fig3, axs_fig3=plt.subplots(nrows=2, ncols=1)  
            fig4, axs_fig4=plt.subplots(nrows=2, ncols=1)  
            if self.status_Model=='EandT'or self.status_Model=='E':
                fig5, axs_fig5=plt.subplots(nrows=2, ncols=1)  
                fig6, axs_fig6=plt.subplots(nrows=2, ncols=1)
                fig7, axs_fig7=plt.subplots(nrows=2, ncols=1)
            if self.status_Model=='EandT'or self.status_Model=='T':
                fig8, axs_fig8=plt.subplots()
                fig9, axs_fig9=plt.subplots()
            self.frames1=[]; self.frames2=[]; self.frames3=[]; self.frames4=[]; self.frames5=[]; self.frames6=[]; self.frames7=[]; self.frames8=[]; self.frames9=[]
        # if self.status_plot_P_cooling == 'Yes':
        #     self.P_cooling_record = np.nan*np.zeros([self.nt+1])
        if self.status_Model=='EandT':
            self.t_record=self.dt*np.arange(self.nt+1)                                                                                                       #record time(contain the t=0 point), node voltage potential and node temperature for postprocessor 
            self.V_record=np.nan*np.zeros([self.ntotal,self.nt+1])
            self.I_ele_record=np.nan*np.zeros([self.nECN,self.nt+1]) 
            self.q_4T_record=np.nan*np.zeros([self.ntotal_4T,self.nt+1])                                                                                     #record heat generation of each node
            self.T_record=np.nan*np.zeros([self.n_4T_ALL,self.nt+1]); self.T_record[:,0]=self.T_initial                                                                
            self.SoC_ele_record=np.nan*np.zeros([self.nECN,self.nt+1]) 
            self.T_avg_record=np.nan*np.zeros([self.nt+1]); self.T_SD_record=np.nan*np.zeros([self.nt+1]); self.T_Delta_record = np.nan*np.zeros([self.nt+1])                                                         #record node temperature average and SD for postprocessor          
            if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':
                self.T_core_Sep_Taz3pos_record=np.nan*np.zeros([self.nt+1,3])                                                                               #record temperature of Taz thermocouple 3 positions: top, medium and bottom                                                                       
                                                                                                  #record SoC of each element
            self.SoC_Cell_record=np.nan*np.zeros([self.nt+1])                                                                                           #record SoC of whole cell
            self.I_ele1_record=np.nan*np.zeros([self.nt+1]); self.IRi_ele1_record=np.nan*np.zeros([self.nt+1,3]); self.Ri_ele1_record=np.nan*np.zeros([self.nt+1,3]); self.SoC_ele1_record=np.nan*np.zeros([self.nt+1])   #record I, IRi, Ri and SoC of the first element 
        elif self.status_Model=='E':
            self.t_record=self.dt*np.arange(self.nt+1) 
            self.V_record=np.zeros([self.ntotal,self.nt+1])
            self.I_ele_record=np.nan*np.zeros([self.nECN,self.nt+1])                                                                                                  #record time(contain the t=0 point) and node voltage potential for postprocessor
            self.q_4T_record=np.nan*np.zeros([self.ntotal_4T,self.nt+1])                                                                                     #record heat generation of each node
            if ip.status_FormFactor == 'Pouch':
                self.T_record=self.T_fixed*np.ones([self.ntotal_4T,self.nt+1]) 
            else:
              self.T_record=self.T_fixed*np.ones([self.n_4T_ALL,self.nt+1]) 
              self.T_core_Sep_Taz3pos_record=np.nan*np.zeros([self.nt+1,3])                                                                               #record temperature of Taz thermocouple 3 positions: top, medium and bottom
            self.SoC_ele_record=np.nan*np.zeros([self.nECN,self.nt+1])                                                                                       #record SoC of each element
            self.SoC_Cell_record=np.nan*np.zeros([self.nt+1])                                                                                           #record SoC of whole cell
            self.I_ele1_record=np.nan*np.zeros([self.nt+1]); self.IRi_ele1_record=np.nan*np.zeros([self.nt+1,3]); self.Ri_ele1_record=np.nan*np.zeros([self.nt+1,3]); self.SoC_ele1_record=np.nan*np.zeros([self.nt+1])   #record I, IRi, Ri and SoC of the first element 
        else:  #i.e. status_Model=='T'
            self.t_record=self.dt*np.arange(self.nt+1) 
            if ip.status_FormFactor == 'Pouch':
                self.T_record=np.nan*np.zeros([self.ntotal_4T,self.nt+1])
                self.T_record[:,0]=self.T_initial                                                                #record time(contain the t=0 point) and node temperature for postprocessor
            else:
              self.T_record=np.nan*np.zeros([self.n_4T_ALL,self.nt+1])
              self.T_record[:,0]=self.T_initial                                                                #record time(contain the t=0 point) and node temperature for postprocessor  
            self.T_avg_record=np.nan*np.zeros([self.nt+1]); self.T_SD_record=np.nan*np.zeros([self.nt+1]); self.T_Delta_record = np.nan*np.zeros([self.nt+1])                                                         #record node temperature average and SD for postprocessor
            self.T_core_Sep_Taz3pos_record=np.nan*np.zeros([self.nt+1,3])                                                                               #record temperature of Taz thermocouple 3 positions: top, medium and bottom
        if self.status_ECN_method=='Neo' and (self.status_Model=='EandT' or self.status_Model=='E'):
            self.ind0_CC_neo=np.arange(self.nCC)
            self.List_Neo2General=self.fun_GetList_Neo2General()
            self.List_General2Neo=self.fun_GetList_General2Neo()                                                                                        #if used for electrical model, unknown vector length is reduced. Mapping from present to before (General) is established as a list                         
            self.IRi_1=np.zeros([self.nECN,self.nRC])                                                                                                        #prepare this for fun_I_neo
            self.MatrixC_NoCenter_neo=self.fun_matrixC_NoCenter_neo()                                                                                   #prepare for MatrixC_neo; MatrixC_NoCenter_neo is the same as MatrixC_neo except the central square to be filled in step loop                                                                                                        
            self.I_NoCenter_neo=self.fun_I_NoCenter_neo()                                                                                               #prepare for I_neo; I_NoCenter_neo is the same as I_neo except the central box to be filled in step loop                                                                                                        
        if self.status_Echeck=='Yes':
            self.Egen_Total_record=np.nan*np.zeros([self.nt+1]); self.Egen_Total_record[0]=0                               #accumulated heat source relative to initial time
            self.Eext_Total_BCconv_record=np.nan*np.zeros([self.nt+1]); self.Eext_Total_BCconv_record[0]=0                 #accumulated external work relative to initial time;  (-):heat lose  (+): heat absorption
            self.Eint_Delta_record=np.nan*np.zeros([self.nt+1]); self.Eint_Delta_record[0]=0                               #internal energy change relative to initial time;  (-):T decreasing  (+): T increasing
            self.Ebalance_record=np.nan*np.zeros([self.nt+1]); self.Ebalance_record[0]=0                                   #energy balance:  Egen_Total_record+Eext_Total_BCconv_record-Eint_Delta_record  
        else:
            self.ncycle=1
        if ip.status_FormFactor == 'Pouch':
            self.rou_c_V_weights=np.zeros([self.ntotal_4T,1])                                        #prep for fun_weighted_avg_and_std
            self.rou_c_V_weights[self.Al_4T,0]=self.rou_Al*self.c_Al*self.V_stencil_4T_ALL[self.Al_4T]                                 #prep for fun_weighted_avg_and_std
            self.rou_c_V_weights[self.Cu_4T,0]=self.rou_Cu*self.c_Al*self.V_stencil_4T_ALL[self.Cu_4T]
            self.rou_c_V_weights[self.Elb_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elb_4T]
            self.rou_c_V_weights[self.Elr_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elr_4T]
        if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':
            if self.status_ThermalBC_Core=='SepFill':
                self.rou_c_V_weights=np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                #prep for fun_weighted_avg_and_std 
                self.rou_c_V_weights[self.Al_4T,0]=self.rou_Al*self.c_Al*self.V_stencil_4T_ALL[self.Al_4T]                                            #prep for fun_weighted_avg_and_std 
                self.rou_c_V_weights[self.Cu_4T,0]=self.rou_Cu*self.c_Al*self.V_stencil_4T_ALL[self.Cu_4T]
                self.rou_c_V_weights[self.Elb_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elb_4T]
                self.rou_c_V_weights[self.Elr_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elr_4T]
                self.rou_c_V_weights[self.ind0_Geo_core_AddSep_4T_4SepFill,0]=self.rou_Sep*self.c_Sep*self.V_stencil_4T_ALL[self.ind0_Geo_core_AddSep_4T_4SepFill]
            if self.status_ThermalBC_Core=='InsuFill':
                self.rou_c_V_weights=np.zeros([self.ntotal_4T,1])                                                            #prep for fun_weighted_avg_and_std
                self.rou_c_V_weights[self.Al_4T,0]=self.rou_Al*self.c_Al*self.V_stencil_4T_ALL[self.Al_4T]                                                     #prep for fun_weighted_avg_and_std
                self.rou_c_V_weights[self.Cu_4T,0]=self.rou_Cu*self.c_Al*self.V_stencil_4T_ALL[self.Cu_4T]
                self.rou_c_V_weights[self.Elb_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elb_4T]
                self.rou_c_V_weights[self.Elr_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elr_4T]
            if self.status_ThermalBC_Core=='SepAir':
                self.rou_c_V_weights=np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                #prep for fun_weighted_avg_and_std 
                self.rou_c_V_weights[self.Al_4T,0]=self.rou_Al*self.c_Al*self.V_stencil_4T_ALL[self.Al_4T]                                            #prep for fun_weighted_avg_and_std 
                self.rou_c_V_weights[self.Cu_4T,0]=self.rou_Cu*self.c_Al*self.V_stencil_4T_ALL[self.Cu_4T]
                self.rou_c_V_weights[self.Elb_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elb_4T]
                self.rou_c_V_weights[self.Elr_4T,0]=self.rouXc_El*self.V_stencil_4T_ALL[self.Elr_4T]
                self.rou_c_V_weights[self.ind0_Geo_core_AddSep_4T_4SepFill,0]=self.rou_Sep*self.c_Sep*self.V_stencil_4T_ALL[self.ind0_Geo_core_AddSep_4T_4SepFill]
        if ip.status_get_unecessary_info == 'Yes':
            for item in ip.status_Cells_name:
                self.fun_bulk_4T()                                                                                #ρ_bulk, ρc_bulk, c_bulk
                #self.fun_weight_and_energy_density()
        if ip.status_FormFactor == 'Pouch':
            self.status_layer_focus_Al=1; self.status_layer_focus_Cu=self.nstack; self.status_layer_focus_Elb=1; self.status_layer_focus_Elr=1 #plot this layer unit (Al,Elb,Cu,Elr)
            



           
              
    #########################################################   
    ################## function for MatrixC #################
    ######################################################### 
    def fun_matrixC(self):
#        if self.status_Count=='Yes':
#            global duration_R0_interp, duration_Ri_interp, duration_Ci_interp
        if self.status_CC=='Yes':
            MatrixC=np.zeros([self.ntotal+self.nECN+self.ntab+1,self.ntotal+self.nECN+self.ntab+1])
        else: 
            MatrixC=np.zeros([self.ntotal+self.nECN+1,self.ntotal+self.nECN+1])                          
        #--------------------------- populate RAl(1/RAl) and RCu(1/RCu) elements in MatrixC
        MatrixC[self.RAl_pair[:,0].astype(int), self.RAl_pair[:,1].astype(int)] = 1/self.RAl_pair[:,2]
        MatrixC[self.RCu_pair[:,0].astype(int), self.RCu_pair[:,1].astype(int)] = 1/self.RCu_pair[:,2]
        #--------------------------- populate R0(1/R0) elements in MatrixC
        ind_ele=self.List_node2ele[self.R0_pair[:,1].astype(int)].reshape(-1)    #ind_ele is the element 0index corresponding to nodes in R0_pair[:,1]
        ts_R0_interp=time.time() if self.status_Count=='Yes' else []
        if self.status_EandT_coupling=='two-way':
            self.T_ele=self.T1_4T_ALL[ self.List_ele2node_4T[ind_ele] ].reshape(-1,1)
        else:
            self.T_ele=self.T_fixed * np.ones([np.size(ind_ele),1])
        if self.status_LUTinterp=='Interp':
            self.R0_pair[:,2]=( self.fun_R0_Interped(self.SoC_ele[ind_ele],self.T_ele)*self.delta_El/self.V_ele[ind_ele]    ).reshape(-1)                                                          
        if self.status_LUTinterp=='Fitting':
            self.R0_pair[:,2]=( self.fun_R0_Fitted(self.SoC_ele[ind_ele],self.T_ele)*self.delta_El/self.V_ele[ind_ele]    ).reshape(-1)                                                                      
        te_R0_interp=time.time() if self.status_Count=='Yes' else []; self.duration_R0_interp=te_R0_interp-ts_R0_interp if self.status_Count=='Yes' else []
        MatrixC[self.R0_pair[:,0].astype(int), self.R0_pair[:,1].astype(int)] = 1/self.R0_pair[:,2]     
        #--------------------------- populate Ri(1/Ri+Ci/dt) elements in MatrixC    
        ind_ele_RCform=self.List_node2ele[self.RC_pair[:,1].astype(int)].reshape(-1)    #ind_ele_RCform is the element 0index corresponding to nodes in RC_pair[:,1]
        if self.status_EandT_coupling=='two-way':
            self.T_ele_RCform=self.T1_4T_ALL[ self.List_ele2node_4T[ind_ele_RCform] ].reshape(-1,1)
        else:
            self.T_ele_RCform=self.T_fixed * np.ones([np.size(ind_ele_RCform),1]) 
        ts_Ri_interp=time.time() if self.status_Count=='Yes' else [] 
        if self.status_LUTinterp=='Interp':                  
            self.RC_pair[:,2]=( self.fun_Ri_Interped(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))*self.delta_El/self.V_ele[ind_ele_RCform]    ).reshape(-1)   
        if self.status_LUTinterp=='Fitting':                  
            self.RC_pair[:,2]=( self.fun_Ri_Fitted(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))*self.delta_El/self.V_ele[ind_ele_RCform]    ).reshape(-1)   
        te_Ri_interp=time.time() if self.status_Count=='Yes' else []; self.duration_Ri_interp=te_Ri_interp-ts_Ri_interp if self.status_Count=='Yes' else []
        ts_Ci_interp=time.time() if self.status_Count=='Yes' else [] 
        if self.status_LUTinterp=='Interp':
            self.RC_pair[:,3]=( self.fun_Ci_Interped(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))/self.delta_El*self.V_ele[ind_ele_RCform]    ).reshape(-1)    
        if self.status_LUTinterp=='Fitting':
            self.RC_pair[:,3]=( self.fun_Ci_Fitted(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))/self.delta_El*self.V_ele[ind_ele_RCform]    ).reshape(-1)    
        te_Ci_interp=time.time() if self.status_Count=='Yes' else []; self.duration_Ci_interp=te_Ci_interp-ts_Ci_interp if self.status_Count=='Yes' else []                                 
        MatrixC[self.RC_pair[:,0].astype(int), self.RC_pair[:,1].astype(int)] = 1/self.RC_pair[:,2]+self.RC_pair[:,3]/self.dt           #pop in 1/R0 elements in MatrixC
        #--------------------------- asymmetrically form topleft area ---------------------------
        MatrixC[:self.ntotal,:self.ntotal]=np.triu(MatrixC[:self.ntotal,:self.ntotal],1).T + MatrixC[:self.ntotal,:self.ntotal]              #upper and lower filled in topleft area of MatrixC
        np.fill_diagonal(MatrixC[:self.ntotal,:self.ntotal], -np.sum(MatrixC[:self.ntotal,:self.ntotal],axis=1))                   #diagonal filled in topleft area of MatrixC 
        #--------------------------- form centralleft and topcentral area ---------------------------         
        MatrixC[self.ind0_EleIsElb+self.ntotal,self.Ei_pair[self.ind0_EleIsElb,0].astype(int)]=1;  MatrixC[self.ind0_EleIsElb+self.ntotal,self.Ei_pair[self.ind0_EleIsElb,1].astype(int)]=-1    #1. case of Elb: if the left node from the Ei pair is blue, the node is positive, put 1 in RectangleC2
        MatrixC[self.ind0_EleIsElr+self.ntotal,self.Ei_pair[self.ind0_EleIsElr,0].astype(int)]=-1; MatrixC[self.ind0_EleIsElr+self.ntotal,self.Ei_pair[self.ind0_EleIsElr,1].astype(int)]=1     #2. case of Elr: if the left node from the Ei pair is red, the node is negative, put -1 in RectangleC2
        MatrixC[:self.ntotal,self.ntotal:(self.ntotal+self.nECN)]=MatrixC[self.ntotal:(self.ntotal+self.nECN),:self.ntotal].T
        if self.status_CC=='Yes':
        #--------------------------- form bottomleft area --------------------------- 
            MatrixC[(self.ntotal+self.nECN):(self.ntotal+self.nECN+self.nPos-1),self.node_positive_0ind[0]] = -1                                 #i.g. node_positive_0ind=[2,3] and node_negative_0ind=[6,11]. nPos=2 and nNeg=2. This line is to assign MatrixC elements in [2] column (or 3rd column, node3) to be -1 
            MatrixC[np.arange((self.ntotal+self.nECN),(self.ntotal+self.nECN+self.nPos-1)),self.node_positive_0ind[1:]] = 1                                 #This line is to assign MatrixC elements in [3] column (or 4th column, node4) to be 1 
            MatrixC[(self.ntotal+self.nECN+self.nPos-1):(self.ntotal+self.nECN+self.nPos-1+self.nNeg-1),self.node_negative_0ind[0]] = -1                   #i.g. node_positive_0ind=[2,3] and node_negative_0ind=[6,11]. nPos=2 and nNeg=2. This line is to assign MatrixC elements in [6] column (or 7rd column, node7) to be -1 
            MatrixC[np.arange((self.ntotal+self.nECN+self.nPos-1),(self.ntotal+self.nECN+self.nPos-1+self.nNeg-1)),self.node_negative_0ind[1:]] = 1                   #This line is to assign MatrixC elements in [11] column (or 12th column, node12) to be 1 
        #--------------------------- form bottomright area --------------------------- 
            MatrixC[-3,-(self.nPos+self.nNeg)-1:-self.nNeg-1] = 1                                                                     #third last row in MatrixC, KCL for positive nodes
            MatrixC[-2,-self.nNeg-1:-1] = -1                                                                                #second last in MatrixC, KCL for negative nodes
            MatrixC[-3,-1] = -1
            MatrixC[-2,-1] = 1
        #--------------------------- form topright area --------------------------- 
            MatrixC[self.node_positive_0ind,np.arange(-(self.nPos+self.nNeg),-self.nNeg)-1] = -1                                      
            MatrixC[self.node_negative_0ind,np.arange(-self.nNeg,0)-1] = 1                                                                    
        
        #--------------------penalty on negative nodes------------------
        MatrixC[self.node_negative_0ind[0],self.node_negative_0ind[0]]=inf
    
        if self.status_IVmode==0:      #current control mode
            MatrixC[-1,-1]=inf
        if self.status_IVmode==1:      #voltage control mode
            MatrixC[-1,self.node_positive_0ind[0]]=inf
        return MatrixC

    #########################################################   
    ##################    function for I    #################
    #########################################################
    
    def fun_I(self):
#        if self.status_Count=='Yes':
#            global duration_OCV_interp
        if self.status_CC=='Yes':
            I=np.zeros([self.ntotal+self.nECN+self.ntab+1])
        else:
            I=np.zeros([self.ntotal+self.nECN+1])
        #--------------------------- form top area ---------------------------
        i00=self.RC_pair[:,0].astype(int); j00=self.RC_pair[:,1].astype(int); Ci=self.RC_pair[:,3]   #RC case: two linking nodes (0index) and their capacitance Ci
        I[i00]=I[i00]+(self.U1[j00,0]-self.U1[i00,0])*Ci/self.dt
        I[j00]=I[j00]+(self.U1[i00,0]-self.U1[j00,0])*Ci/self.dt
        #--------------------------- form central area ---------------------------
        ts_OCV_interp=time.time() if self.status_Count=='Yes' else []
        if self.status_LUTinterp=='Interp':
            if not all(self.SoC_ele<self.LUT_SoC[1])*all(self.SoC_ele>self.LUT_SoC[-2])*all(self.T_ele>self.LUT_T[1])*all(self.T_ele<self.LUT_T[-2]):     #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
                print('PyECN warning: cut-off extrapolation is used')  
            OCV=self.fun_OCV_Interped(self.SoC_ele,self.T_ele)
        if self.status_LUTinterp=='Fitting':
            if not all(self.SoC_ele<self.LUT_SoC[1])*all(self.SoC_ele>self.LUT_SoC[-2])*all(self.T_ele>self.LUT_T[1])*all(self.T_ele<self.LUT_T[-2]):     #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
                print('PyECN warning: cut-off saturation (SoC and T axises) is used')  
            self.SoC_ele[self.SoC_ele < self.LUT_SoC[-2]]=self.LUT_SoC[-2]; self.SoC_ele[self.SoC_ele > self.LUT_SoC[1]]=self.LUT_SoC[1]    #Saturation used for x axis (SoC)
            self.T_ele[self.T_ele < self.LUT_T[1]]=self.LUT_T[1]; self.T_ele[self.T_ele > self.LUT_T[-2]]=self.LUT_T[-2]                    #Saturation used for y axis (T)
            OCV=self.fun_OCV_Fitted(self.SoC_ele,self.T_ele)
        self.Ei_pair[:,2]=OCV.reshape(self.nECN)                                           
        te_OCV_interp=time.time() if self.status_Count=='Yes' else []
        self.duration_OCV_interp=te_OCV_interp-ts_OCV_interp if self.status_Count=='Yes' else [] 
        I[self.ntotal:(self.ntotal+self.nECN)]=self.Ei_pair[:,2]     #OCV terms in down part of I vector, getting directly from Ei_pair
        #--------------------penalty on negative nodes------------------
        I[self.node_negative_0ind[0]]=inf*self.V_neg_ref
        if self.status_IVmode==0:      #current control mode
            I[-1]=inf*self.I_ext
        if self.status_IVmode==1:      #voltage control mode
            I[-1]=inf*self.V_ext
        return I
    #########################################################   
    ###  function for modifying CandI for status_CC='No'  ###
    #########################################################
    def fun_modifyCandI4NoCC(self):
#        global MatrixC, VectorI
        for i0 in self.AlCu:     #i0 is the CC 0index and also row for MatrixC
            self.MatrixC[i0]=0
            if self.mat[i0]==1:    #the CC node is positive
                self.MatrixC[i0,self.node_positive_0ind[0]]=1; self.MatrixC[i0,i0]=-1
                self.VectorI[i0]=0
            if self.mat[i0]==2:    #the CC node is negative
                self.MatrixC[i0,i0]=1
                self.VectorI[i0]=self.V_neg_ref
        self.MatrixC[self.node_positive_0ind[0]]=0; self.MatrixC[self.node_positive_0ind[0],-self.nECN-1:-1]=1      #for positive node, i1+i2+...+inECN - I0 = 0
        self.MatrixC[self.node_positive_0ind[0],-1]=-1
    #########################################################   
    ###############   function for Uini_neo   ###############
    #########################################################
    def fun_Uini_neo(self,step):
#        global MatrixC_neo, VectorI_neo
        self.MatrixC_neo=self.fun_matrixC_neo()             
        self.fun_IRi_neo(step)
        self.VectorI_neo=self.fun_I_neo()
        if self.status_CC=='No':   #if CC resistance is not considered, modify the MatrixC and VectorI for the CC node 0index
            self.fun_modifyCandI4NoCC_neo()
        #---------------------------------------------------------------------------
        if self.status_linsolver_E=='Sparse':
            self.Uini_neo=scipy.sparse.linalg.spsolve(self.MatrixC_neo,self.VectorI_neo) 
        else:
            self.Uini_neo=np.linalg.solve(self.MatrixC_neo,self.VectorI_neo)
        self.V_record[ self.List_Neo2General[self.ind0_CC_neo,0],step ]=self.Uini_neo[self.ind0_CC_neo]  
        self.I_ele_record[:,step]=self.Uini_neo[self.nCC:(self.nCC+self.nECN)]
        return self.Uini_neo.reshape([-1,1])
        #return Uini_neo
    #########################################################   
    ############ function for MatrixC_NoCenter_neo ##########
    #########################################################
    def fun_matrixC_NoCenter_neo(self):
        if self.status_CC=='Yes':
            MatrixC_NoCenter_neo=np.zeros([self.nCC+self.nECN+self.ntab+1,self.nCC+self.nECN+self.ntab+1])
        else: 
            MatrixC_NoCenter_neo=np.zeros([self.nCC+self.nECN+1,self.nCC+self.nECN+1])                          
        #--------------------------- form topleft area ---------------------------
        MatrixC_NoCenter_neo[ self.List_General2Neo[self.RAl_pair[:,0].astype(int),0], self.List_General2Neo[self.RAl_pair[:,1].astype(int),0] ] = 1/self.RAl_pair[:,2]  #populate RAl(1/RAl) elements in MatrixC_NoCenter_neo
        MatrixC_NoCenter_neo[ self.List_General2Neo[self.RCu_pair[:,0].astype(int),0], self.List_General2Neo[self.RCu_pair[:,1].astype(int),0] ] = 1/self.RCu_pair[:,2]  #populate RCu(1/RCu) elements in MatrixC_NoCenter_neo
        #--------------------------- asymmetrically form topleft area ---------------------------
        MatrixC_NoCenter_neo[:self.nCC,:self.nCC]=np.triu(MatrixC_NoCenter_neo[:self.nCC,:self.nCC],1).T + MatrixC_NoCenter_neo[:self.nCC,:self.nCC]              #upper and lower filled in topleft area of MatrixC
        np.fill_diagonal(MatrixC_NoCenter_neo[:self.nCC,:self.nCC], -np.sum(MatrixC_NoCenter_neo[:self.nCC,:self.nCC],axis=1))                 #diagonal filled in topleft area of MatrixC 
        #--------------------------- form centralleft and topcentral area ---------------------------        
        MatrixC_NoCenter_neo[ self.ind0_EleIsElb+self.nCC,self.List_General2Neo[self.ind0_AlOfElb,0] ]=1;  MatrixC_NoCenter_neo[ self.ind0_EleIsElb+self.nCC,self.List_General2Neo[self.ind0_CuOfElb,0] ]=-1    #1. case of Elb: if the node is Al (positive), put 1 in RectangleC2
        MatrixC_NoCenter_neo[ self.ind0_EleIsElr+self.nCC,self.List_General2Neo[self.ind0_CuOfElr,0] ]=-1; MatrixC_NoCenter_neo[ self.ind0_EleIsElr+self.nCC,self.List_General2Neo[self.ind0_AlOfElr,0] ]=1     #2. case of Elr: if the node is Cu (negative), put -1 in RectangleC2
        MatrixC_NoCenter_neo[:self.nCC,self.nCC:(self.nCC+self.nECN)]=MatrixC_NoCenter_neo[self.nCC:(self.nCC+self.nECN),:self.nCC].T   
        #--------------------------- form bottomleft area --------------------------- 
        if self.status_CC=='Yes':
            MatrixC_NoCenter_neo[ (self.nCC+self.nECN):(self.nCC+self.nECN+self.nPos-1),self.List_General2Neo[self.node_positive_0ind[0],0] ] = -1                                            #i.g. node_positive_0ind=[2,3] and node_negative_0ind=[6,11]. nPos=2 and nNeg=2. This line is to assign MatrixC elements in [2] column (or 3rd column, node3) to be -1 
            MatrixC_NoCenter_neo[ np.arange((self.nCC+self.nECN),(self.nCC+self.nECN+self.nPos-1)),self.List_General2Neo[self.node_positive_0ind[1:],0] ] = 1                                 #This line is to assign MatrixC elements in [3] column (or 4th column, node4) to be 1 
            MatrixC_NoCenter_neo[ (self.nCC+self.nECN+self.nPos-1):(self.nCC+self.nECN+self.nPos-1+self.nNeg-1),self.List_General2Neo[self.node_negative_0ind[0],0] ] = -1                    #i.g. node_positive_0ind=[2,3] and node_negative_0ind=[6,11]. nPos=2 and nNeg=2. This line is to assign MatrixC elements in [6] column (or 7rd column, node7) to be -1 
            MatrixC_NoCenter_neo[ np.arange((self.nCC+self.nECN+self.nPos-1),(self.nCC+self.nECN+self.nPos-1+self.nNeg-1)),self.List_General2Neo[self.node_negative_0ind[1:],0] ] = 1         #This line is to assign MatrixC elements in [11] column (or 12th column, node12) to be 1 
        #--------------------------- form bottomright area --------------------------- 
            MatrixC_NoCenter_neo[-3,-(self.nPos+self.nNeg)-1:-self.nNeg-1] = 1                                                                     #third last row in MatrixC, KCL for positive nodes
            MatrixC_NoCenter_neo[-2,-self.nNeg-1:-1] = -1                                                                                #second last in MatrixC, KCL for negative nodes
            MatrixC_NoCenter_neo[-3,-1] = -1
            MatrixC_NoCenter_neo[-2,-1] = 1
        #--------------------------- form topright area --------------------------- 
            MatrixC_NoCenter_neo[self.List_General2Neo[self.node_positive_0ind,0],np.arange(-(self.nPos+self.nNeg),-self.nNeg)-1] = -1                                      
            MatrixC_NoCenter_neo[self.List_General2Neo[self.node_negative_0ind,0],np.arange(-self.nNeg,0)-1] = 1                                                                        
        #--------------------penalty on negative nodes------------------
        MatrixC_NoCenter_neo[self.List_General2Neo[ self.node_negative_0ind[0],0],self.List_General2Neo[self.node_negative_0ind[0],0] ]=inf
    
        return MatrixC_NoCenter_neo
    #########################################################   
    ################ function for MatrixC_neo ###############
    #########################################################
    def fun_matrixC_neo(self):
        MatrixC_neo=self.MatrixC_NoCenter_neo.copy()
#        if self.status_Count=='Yes':
#            global duration_R0_interp                         
        #--------------------------- form central area ---------------------------    
        ind_ele=self.List_node2ele[self.R0_pair[:,1].astype(int)].reshape(-1)    #ind_ele is the element 0index corresponding to nodes in R0_pair[:,1]
        ts_R0_interp=time.time() if self.status_Count=='Yes' else []
        if self.status_EandT_coupling=='two-way':
            self.T_ele=self.T1_4T_ALL[ self.List_ele2node_4T[ind_ele] ].reshape(-1,1)
        else:
            self.T_ele=self.T_fixed * np.ones([np.size(ind_ele),1])
        if self.status_LUTinterp=='Interp':
            self.R0_pair[:,2]=( self.fun_R0_Interped(self.SoC_ele[ind_ele],self.T_ele)*self.delta_El/self.V_ele[ind_ele]    ).reshape(-1)                                                          
        if self.status_LUTinterp=='Fitting':
            self.R0_pair[:,2]=( self.fun_R0_Fitted(self.SoC_ele[ind_ele],self.T_ele)*self.delta_El/self.V_ele[ind_ele]    ).reshape(-1)
        te_R0_interp=time.time() if self.status_Count=='Yes' else []; self.duration_R0_interp=te_R0_interp-ts_R0_interp if self.status_Count=='Yes' else []             
        MatrixC_neo[ np.arange(self.nECN)+self.nCC, np.arange(self.nECN)+self.nCC ] = self.R0_pair[:,2]                                                                            
    
        if self.status_IVmode==0:      #current control mode
            MatrixC_neo[-1,-1]=inf
        if self.status_IVmode==1:      #voltage control mode
            MatrixC_neo[-1,self.List_General2Neo[ self.node_positive_0ind[0],0]]=inf
        return MatrixC_neo
    #########################################################   
    ###########    function for I_NoCenter_neo    ###########
    #########################################################
    def fun_I_NoCenter_neo(self):
        if self.status_CC=='Yes':
            I_NoCenter_neo=np.zeros([self.nCC+self.nECN+self.ntab+1])
        else:
            I_NoCenter_neo=np.zeros([self.nCC+self.nECN+1])           
        #--------------------penalty on negative nodes------------------
        I_NoCenter_neo[self.List_General2Neo[ self.node_negative_0ind[0],0] ]=inf*self.V_neg_ref
        return I_NoCenter_neo
    #########################################################   
    ###############    function for IRi_neo    ##############
    #########################################################
    def fun_IRi_neo(self,step):
#        global IRi_3
        if step == 0:
            self.IRi_3 = np.zeros([self.nECN,self.nRC])
        else:
            self.Ii_1=self.U1_neo[self.nCC:(self.nCC+self.nECN)]; self.IRi_3=self.Ii_1 + (self.IRi_1-self.Ii_1)*np.exp(-self.dt/(self.Ri_ele*self.Ci_ele))  
    #########################################################   
    ################    function for I_neo    ###############
    #########################################################
    def fun_I_neo(self):
#        if self.status_Count=='Yes':
#            global duration_OCV_interp                         
#        global Ri_ele, Ci_ele
        I_neo=self.I_NoCenter_neo.copy()
#        if self.status_Count=='Yes':
#            global duration_Ri_interp, duration_Ci_interp 
        #--------------------------- form central area ---------------------------
            #--------------------------- prepare Ri,Ci    
        ind_ele_RCform=self.List_node2ele[self.RC_pair[:,1].astype(int)].reshape(-1)    #ind_ele_RCform is the element 0index corresponding to nodes in RC_pair[:,1]
        if self.status_EandT_coupling=='two-way':
            self.T_ele_RCform=self.T1_4T_ALL[ self.List_ele2node_4T[ind_ele_RCform] ].reshape(-1,1)
        else:
            self.T_ele_RCform=self.T_fixed * np.ones([np.size(ind_ele_RCform),1]) 
        ts_Ri_interp=time.time() if self.status_Count=='Yes' else []                   
        if self.status_LUTinterp=='Interp':
            self.RC_pair[:,2]=( self.fun_Ri_Interped(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))*self.delta_El/self.V_ele[ind_ele_RCform]    ).reshape(-1)   
        if self.status_LUTinterp=='Fitting':
            self.RC_pair[:,2]=( self.fun_Ri_Fitted(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))*self.delta_El/self.V_ele[ind_ele_RCform]    ).reshape(-1)   
        te_Ri_interp=time.time() if self.status_Count=='Yes' else []; self.duration_Ri_interp=te_Ri_interp-ts_Ri_interp if self.status_Count=='Yes' else []
        ts_Ci_interp=time.time() if self.status_Count=='Yes' else [] 
        if self.status_LUTinterp=='Interp':
            self.RC_pair[:,3]=( self.fun_Ci_Interped(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))/self.delta_El*self.V_ele[ind_ele_RCform]    ).reshape(-1)    
        if self.status_LUTinterp=='Fitting':
            self.RC_pair[:,3]=( self.fun_Ci_Fitted(self.SoC_ele[ind_ele_RCform],self.T_ele_RCform,self.RC_pair[:,4].astype(int))/self.delta_El*self.V_ele[ind_ele_RCform]    ).reshape(-1)    
        te_Ci_interp=time.time() if self.status_Count=='Yes' else []; self.duration_Ci_interp=te_Ci_interp-ts_Ci_interp if self.status_Count=='Yes' else []                                 
        self.Ri_ele=self.RC_pair[self.ind0_ele_RC_pair,2]; self.Ci_ele=self.RC_pair[self.ind0_ele_RC_pair,3]
    
        ts_OCV_interp=time.time() if self.status_Count=='Yes' else []
        if self.status_LUTinterp=='Interp':
            if not all(self.SoC_ele<self.LUT_SoC[1])*all(self.SoC_ele>self.LUT_SoC[-2])*all(self.T_ele>self.LUT_T[1])*all(self.T_ele<self.LUT_T[-2]):     #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
                print('PyECN warning: cut-off extrapolation is used')  
            OCV=self.fun_OCV_Interped(self.SoC_ele,self.T_ele)
        if self.status_LUTinterp=='Fitting':
            if not all(self.SoC_ele<self.LUT_SoC[1])*all(self.SoC_ele>self.LUT_SoC[-2])*all(self.T_ele>self.LUT_T[1])*all(self.T_ele<self.LUT_T[-2]):     #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
                print('PyECN warning: cut-off saturation (SoC and T axises) is used')  
            self.SoC_ele[self.SoC_ele < self.LUT_SoC[-2]]=self.LUT_SoC[-2]; self.SoC_ele[self.SoC_ele > self.LUT_SoC[1]]=self.LUT_SoC[1]    #Saturation used for x axis (SoC)
            self.T_ele[self.T_ele < self.LUT_T[1]]=self.LUT_T[1]; self.T_ele[self.T_ele > self.LUT_T[-2]]=self.LUT_T[-2]                    #Saturation used for y axis (T)
            OCV=self.fun_OCV_Fitted(self.SoC_ele,self.T_ele)
        self.Ei_pair[:,2]=OCV.reshape(self.nECN)                                          
        te_OCV_interp=time.time() if self.status_Count=='Yes' else []
        self.duration_OCV_interp=te_OCV_interp-ts_OCV_interp if self.status_Count=='Yes' else []     
        I_neo[self.nCC:(self.nCC+self.nECN)]=self.Ei_pair[:,2]-np.sum(self.IRi_3*self.Ri_ele,axis=1)    #if step=0, IRi_3 is 0 as assigned in line4032: IRi_3=np.zeros([nECN,nRC])   
        #--------------------penalty on negative nodes------------------
        if self.status_IVmode==0:      #current control mode
            I_neo[-1]=inf*self.I_ext
        if self.status_IVmode==1:      #voltage control mode
            I_neo[-1]=inf*self.V_ext
        return I_neo
    #########################################################   
    ###  function for modifying CandI for status_CC='No'  ###
    #########################################################
    def fun_modifyCandI4NoCC_neo(self):
#        global MatrixC_neo, VectorI_neo
        for i0 in self.AlCu:     #i0 is the CC 0index and also row for MatrixC
            self.MatrixC_neo[ self.List_General2Neo[i0,0] ]=0
            if self.mat[i0]==1:    #the CC node is positive
                self.MatrixC_neo[ self.List_General2Neo[i0,0],self.List_General2Neo[self.node_positive_0ind[0],0] ]=1; self.MatrixC_neo[ self.List_General2Neo[i0,0],self.List_General2Neo[i0,0] ]=-1
                self.VectorI_neo[ self.List_General2Neo[i0,0] ]=0
            if self.mat[i0]==2:    #the CC node is negative
                self.MatrixC_neo[ self.List_General2Neo[i0,0],self.List_General2Neo[i0,0] ]=1
                self.VectorI_neo[ self.List_General2Neo[i0,0] ]=self.V_neg_ref
        self.MatrixC_neo[ self.List_General2Neo[self.node_positive_0ind[0],0] ]=0; self.MatrixC_neo[ self.List_General2Neo[self.node_positive_0ind[0],0],-self.nECN-1:-1]=1      #for positive node, i1+i2+...+inECN - I0 = 0
        self.MatrixC_neo[ self.List_General2Neo[self.node_positive_0ind[0],0],-1]=-1
    #########################################################   
    ### function for updating varables in the end of loop ###
    #########################################################
    def fun_update_neo(self):
#        global U1_neo, U3_neo, IRi_1, IRi_3 
        self.U1_neo=self.U3_neo
        self.IRi_1=self.IRi_3
    #########################################################   
    ###############     function for Uini     ###############
    #########################################################
 
    def fun_Uini(self,step):
#        global U1, Uini, MatrixC, VectorI
        self.U1=self.Uini.copy()
        self.MatrixC=self.fun_matrixC()             
        self.VectorI=self.fun_I()
        if self.status_CC=='No':   #if CC resistance is not considered, modify the MatrixC and VectorI for the CC node 0index
            self.fun_modifyCandI4NoCC()
        #---------------------------------------------------------------------------
        if self.status_linsolver_E=='Sparse':
            Uini=scipy.sparse.linalg.spsolve(self.MatrixC,self.VectorI) 
        else:
            Uini=np.linalg.solve(self.MatrixC,self.VectorI)
        self.V_record[:,step]=Uini[:self.ntotal]; self.I_ele_record[:,step]=Uini[self.ntotal:(self.ntotal+self.nECN)]
        return Uini.reshape([-1,1])
    
    #########################################################   
    #####   function for Electrical initial condition   #####
    #########################################################

    def fun_IC(self):
#        global Uini, V_neg_ref, OCV_0, Capacity0, status_discharge
    #------------------------------- input window --------------------------------
        status_discharge=ip.status_discharge; soc_initial=ip.soc_initial                    #for discharge  
    #    status_discharge=-1; soc_initial=0.04                  #for charge
    #-----------------------------------------------------------------------------
        
        self.fun_read_SoCandT()
        self.fun_read_OCV()                               #LUT_SoC, LUT_OCV_PerA are generated here      
        self.fun_read_dVdT()                              #LUT_SoC, LUT_dVdT_PerA are generated here
        self.fun_read_RsCs()                              #LUT_SoC, LUT_T, LUT_R0_PerA,LUT_Ri_PerA and LUT_Ci_PerA are generated here

        V_neg_ref = 0                                         #set negative tab node as voltage potential reference; set reference voltage potential value here 
        OCV_0=self.fun_OCV_Interped( np.array([soc_initial]),np.array([self.T_initial]) )
        Capacity0=self.Capacity_rated0 * soc_initial                #for discharge   cell pos/neg voltage and initial cell capacity    
    
        if self.status_CC=='Yes':
            Uini=np.zeros([self.ntotal+self.nECN+self.ntab,1])
        else:
            Uini=np.zeros([self.ntotal+self.nECN,1])
        
        Uini[self.Al]=V_neg_ref + OCV_0
        Uini[self.Cu]=V_neg_ref
        Uini[self.Elb]=V_neg_ref + OCV_0
        Uini[self.Elr]=V_neg_ref
        (
        self.Uini, self.V_neg_ref, self.OCV_0, self.Capacity0, self.status_discharge, self.soc_initial         
        )=(                                                                                    
        Uini, V_neg_ref, OCV_0, Capacity0, status_discharge, soc_initial
        )


    #########################################################   
    ################## function for OCV-SoC #################
    #########################################################
    def fun_OCV_Interped(self, SoC,T):              #input SoC of each ECN element, as vector SoC; output OCV as vector OCV;  SoC, T and OCV are from Ei_pair, in the form of 1,2...nECN  
    #=================================self written interpolation (slow)
    #    n0=np.size(SoC);  OCV=np.zeros([n0,1])
    #    for i0 in np.arange(n0):
    #        if SoC[i0]>LUT_SoC[1]*1.01 or SoC[i0]<LUT_SoC[-2]*1.01 or T[i0]<LUT_T[1]*1.01 or T[i0]>LUT_T[-2]*1.01:     #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
    #            print('PyECN warning: OCV cut-off extrapolation is used for 0index=%d / (%d ECN elements)'%(i0,n0))  
    #        T_ind_1=max(np.where(LUT_T< T[i0])[0]); T_ind_2=min(np.where(LUT_T>= T[i0])[0])             #T_ind_1 is exclusive left limit, T_ind_2 is inclusive right limit
    #        soc_ind_1=max(np.where(LUT_SoC> SoC[i0])[0]); soc_ind_2=min(np.where(LUT_SoC<=SoC[i0])[0])  #soc_ind_1 is exclusive left limit, soc_ind_2 is inclusive right limit
    #        y_1=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_OCV_PerA[T_ind_1,soc_ind_1,0],LUT_OCV_PerA[T_ind_2,soc_ind_1,0])   #interpolate T at fixed soc_ind_1
    #        y_2=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_OCV_PerA[T_ind_1,soc_ind_2,0],LUT_OCV_PerA[T_ind_2,soc_ind_2,0])   #interpolate T at fixed soc_ind_2    
    #        OCV[i0]=fun_interpolate(SoC[i0],LUT_SoC[soc_ind_1],LUT_SoC[soc_ind_2],y_1,y_2)
    
    #=================================scipy interpolation function (faster)
        n0=np.size(SoC);  OCV=np.zeros([n0,1])
        f = interpolate.interp2d(self.LUT_SoC, self.LUT_T, self.LUT_OCV_PerA.reshape(self.nT+2,self.nSoC+2), kind=self.status_scipy_interpkind)
        for i0 in np.arange(n0):
            OCV[i0]=f(SoC[i0],T[i0])
        if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
            OCV[OCV < self.LUT_OCV_Cell_min]=self.LUT_OCV_Cell_min; OCV[OCV > self.LUT_OCV_Cell_max]=self.LUT_OCV_Cell_max
        return OCV
    def fun_OCV_Fitted(self,SoC,T):              #input SoC of each ECN element, as vector SoC; output OCV as vector OCV;  SoC, T and OCV are from Ei_pair, in the form of 1,2...nECN  
        if self.status_Eparam=='Cylindrical_Cell1':
            SoC_scaled=SoC                                              #for Fitting expression, all inputs are scaled into x~:(0,1): x~ = (x-x_min)/(x_max-x_min); x=x~ *(x_max-_min) +x_min
            T_scaled=(T-self.LUT_T_min)/(self.LUT_T_max-self.LUT_T_min)
            OCV_scaled=self.fun_OCV_fitted_scaled_expression_Cell(SoC_scaled,T_scaled)
            OCV_scaled[OCV_scaled < 0]=0; OCV_scaled[OCV_scaled > 1]=1                #Saturation used
            OCV= OCV_scaled*(self.LUT_OCV_Cell_max-self.LUT_OCV_Cell_min)+self.LUT_OCV_Cell_min
        return OCV
    def fun_dOCVdSoC_Interped(self,SoC,T):              #input SoC of each ECN element, as vector SoC; output dOCV/dSoC as vector dOCVdSoC;  SoC, T and OCV are from Ei_pair, in the form of 1,2...nECN 
    #=================================scipy interpolation function (faster)
        n0=np.size(SoC);  dOCVdSoC=np.zeros([n0,1])
        f = interpolate.interp2d(self.LUT_SoC_for_dOCVdSoC, self.LUT_T, self.LUT_dOCVdSoC_PerA.reshape(self.nT+2,self.nSoC+2), kind=self.status_scipy_interpkind)
        for i0 in np.arange(n0):
            dOCVdSoC[i0]=f(SoC[i0],T[i0])
        if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
            dOCVdSoC[dOCVdSoC < self.LUT_dOCVdSoC_Cell_min]=self.LUT_dOCVdSoC_Cell_min; dOCVdSoC[dOCVdSoC > self.LUT_dOCVdSoC_Cell_max]=self.LUT_dOCVdSoC_Cell_max
        return dOCVdSoC
    #########################################################   
    ################# function for R0-SoC,T #################
    #########################################################
    def fun_R0_Interped(self,SoC,T):      #input SoC and T of each ECN element, both as scalar; output R0 as a scalar. R0 unit is same as LUT_R0_PerA, i.e. El thickness (91μm) per area has R0 Ohm  
    #=================================self written interpolation (slow)
    #    n0=np.size(SoC);  R0=np.zeros([n0,1])
    #    for i0 in np.arange(n0):
    #        T_ind_1=max(np.where(LUT_T< T[i0])[0]); T_ind_2=min(np.where(LUT_T>= T[i0])[0])             #T_ind_1 is exclusive left limit, T_ind_2 is inclusive right limit
    #        soc_ind_1=max(np.where(LUT_SoC> SoC[i0])[0]); soc_ind_2=min(np.where(LUT_SoC<=SoC[i0])[0])  #soc_ind_1 is exclusive left limit, soc_ind_2 is inclusive right limit
    #        y_1=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_R0_PerA[T_ind_1,soc_ind_1,0],LUT_R0_PerA[T_ind_2,soc_ind_1,0])   #interpolate T at fixed soc_ind_1
    #        y_2=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_R0_PerA[T_ind_1,soc_ind_2,0],LUT_R0_PerA[T_ind_2,soc_ind_2,0])   #interpolate T at fixed soc_ind_2    
    #        R0[i0]=fun_interpolate(SoC[i0],LUT_SoC[soc_ind_1],LUT_SoC[soc_ind_2],y_1,y_2)
    
    #=================================scipy interpolation function (faster)
        n0=np.size(SoC);  R0=np.zeros([n0,1])
        f = interpolate.interp2d(self.LUT_SoC, self.LUT_T, self.LUT_R0_PerA.reshape(self.nT+2,self.nSoC+2), kind=self.status_scipy_interpkind)
        for i0 in np.arange(n0):
            R0[i0]=f(SoC[i0],T[i0])
    
    #    x=LUT_SoC; y=LUT_T; z=LUT_R0_PerA.reshape(nT+2,nSoC+2)
    #    spl=RectBivariateSpline(x, y, z)
    #    R0=spl(SoC.reshape(-1),T.reshape(-1),grid=False)    
    #    print(1)
        if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
            R0[R0 < self.LUT_R0_PerA_min]=self.LUT_R0_PerA_min; R0[R0 > self.LUT_R0_PerA_max]=self.LUT_R0_PerA_max
        return R0
    def fun_R0_Fitted(self,SoC,T):              #input SoC of each ECN element, as vector SoC; output R0 as vector R0;  SoC and OCV are from Ei_pair, in the form of 1,2...nECN  
        if self.status_Eparam=='Cylindrical_Cell1':
            SoC_scaled=SoC                                              #for Fitting expression, all inputs are scaled into x~:(0,1): x~ = (x-x_min)/(x_max-x_min); x=x~ *(x_max-_min) +x_min
            T_scaled=(T-self.LUT_T_min)/(self.LUT_T_max-self.LUT_T_min)
            R0_scaled_Cell=self.fun_R0_fitted_scaled_expression_Cell(SoC_scaled,T_scaled)
            R0_scaled_Cell[R0_scaled_Cell < 0]=0; R0_scaled_Cell[R0_scaled_Cell > 1]=1                #Saturation used
            R0_Cell= R0_scaled_Cell*(self.LUT_R0_Cell_max-self.LUT_R0_Cell_min)+self.LUT_R0_Cell_min
            #-----------------------------for Fitting, R0 needs to be transformed from whole cell data into per thickness per area to be used for element
            R0=R0_Cell *self.A_electrodes_real
            R0=R0/self.scalefactor_z
        return R0
    #########################################################   
    ################# function for Ri-SoC,T #################
    #########################################################
    def fun_Ri_Interped(self,SoC,T,indRC):      #input SoC, T and indRC(for example, 1,2,3 when nRC=3)of each ECN element, both as vector; output Ri as a vector. Ri unit is same as LUT_Ri_PerA, i.e. El thickness (91μm) per area has Ri Ohm 
    #=================================self written interpolation (slow)
    #    n0=np.size(SoC);  Ri=np.zeros([n0,1])
    #    for i0 in np.arange(n0):
    #        T_ind_1=max(np.where(LUT_T< T[i0])[0]); T_ind_2=min(np.where(LUT_T>= T[i0])[0])             #T_ind_1 is exclusive left limit, T_ind_2 is inclusive right limit
    #        soc_ind_1=max(np.where(LUT_SoC> SoC[i0])[0]); soc_ind_2=min(np.where(LUT_SoC<=SoC[i0])[0])  #soc_ind_1 is exclusive left limit, soc_ind_2 is inclusive right limit
    #        y_1=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_Ri_PerA[T_ind_1,soc_ind_1,indRC[i0]-1],LUT_Ri_PerA[T_ind_2,soc_ind_1,indRC[i0]-1])   #interpolate T at fixed soc_ind_1
    #        y_2=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_Ri_PerA[T_ind_1,soc_ind_2,indRC[i0]-1],LUT_Ri_PerA[T_ind_2,soc_ind_2,indRC[i0]-1])   #interpolate T at fixed soc_ind_2
    #        Ri[i0]=fun_interpolate(SoC[i0],LUT_SoC[soc_ind_1],LUT_SoC[soc_ind_2],y_1,y_2)
    
    #=================================scipy interpolation function (faster)    
        Ri=np.zeros([self.nECN*self.nRC,1]); Ri_ele_temp=np.zeros([self.nECN,self.nRC])
        for i0 in np.arange(self.nRC):
            f = interpolate.interp2d(self.LUT_SoC, self.LUT_T, self.LUT_Ri_PerA[:,:,i0], kind=self.status_scipy_interpkind)
            for i00 in np.arange(self.nECN):
                temp=f(SoC[self.ind0_ele_RC_pair[i00,i0],0],T[self.ind0_ele_RC_pair[i00,i0],0])
                Ri_ele_temp[i00,i0]=temp
    
        if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
            for i0 in np.arange(self.nRC):
                temp=Ri_ele_temp[:,i0]
                temp[temp < self.LUT_Ri_PerA_min[i0]]=self.LUT_Ri_PerA_min[i0]; temp[temp > self.LUT_Ri_PerA_max[i0]]=self.LUT_Ri_PerA_max[i0]
                Ri_ele_temp[:,i0]=temp
        Ri[self.ind0_ele_RC_pair,0]=Ri_ele_temp
        return Ri
    
    def fun_Ri_Fitted(self,SoC,T,indRC):      #input SoC, T and indRC(for example, 1,2,3 when nRC=3)of each ECN element, both as vector; output Ri as a vector. Ri unit is same as LUT_Ri_PerA, i.e. El thickness (91μm) per area has Ri Ohm 
#        global Ri_record4Ci_Fit       #for later use in fun_Ci_Fitted
        Ri=np.zeros([self.nECN*self.nRC,1])
        if self.status_Eparam=='Cylindrical_Cell1':
            SoC_scaled=SoC                                              #for Fitting expression, all inputs are scaled into x~:(0,1): x~ = (x-x_min)/(x_max-x_min); x=x~ *(x_max-_min) +x_min
            T_scaled=(T-self.LUT_T_min)/(self.LUT_T_max-self.LUT_T_min)
            #-----------------------------calculate through fitted surface
            for i0 in np.arange(self.nRC):    
                if i0 == 0:
                    SoC1_scaled=SoC_scaled[self.ind0_ele_RC_pair[:,0]]; T1_scaled=T_scaled[self.ind0_ele_RC_pair[:,0]]
                    R1_scaled_Cell=self.fun_R1_fitted_scaled_expression_Cell(SoC1_scaled,T1_scaled)
                    R1_scaled_Cell[R1_scaled_Cell < 0]=0; R1_scaled_Cell[R1_scaled_Cell > 1]=1                #Saturation used
                    R1_Cell= R1_scaled_Cell*(self.LUT_Ri_Cell_max[0]-self.LUT_Ri_Cell_min[0])+self.LUT_Ri_Cell_min[0]          #back to value before normalization
                    Ri[self.ind0_ele_RC_pair[:,0]]=R1_Cell                                          #write R1(nECN,1) into Ri(nECN*nRC,1)
                elif i0 == 1:
                    SoC2_scaled=SoC_scaled[self.ind0_ele_RC_pair[:,1]]; T2_scaled=T_scaled[self.ind0_ele_RC_pair[:,1]]
                    R2_scaled_Cell=self.fun_R2_fitted_scaled_expression_Cell(SoC2_scaled,T2_scaled)
                    R2_scaled_Cell[R2_scaled_Cell < 0]=0; R2_scaled_Cell[R2_scaled_Cell > 1]=1                #Saturation used
                    R2_Cell= R2_scaled_Cell*(self.LUT_Ri_Cell_max[1]-self.LUT_Ri_Cell_min[1])+self.LUT_Ri_Cell_min[1]          #back to value before normalization
                    Ri[self.ind0_ele_RC_pair[:,1]]=R2_Cell                                          #write R2(nECN,1) into Ri(nECN*nRC,1)
                else:
                    SoC3_scaled=SoC_scaled[self.ind0_ele_RC_pair[:,2]]; T3_scaled=T_scaled[self.ind0_ele_RC_pair[:,2]]
                    R3_scaled_Cell=self.fun_R3_fitted_scaled_expression_Cell(SoC3_scaled,T3_scaled)
                    R3_scaled_Cell[R3_scaled_Cell < 0]=0; R3_scaled_Cell[R3_scaled_Cell > 1]=1                #Saturation used
                    R3_Cell= R3_scaled_Cell*(self.LUT_Ri_Cell_max[2]-self.LUT_Ri_Cell_min[2])+self.LUT_Ri_Cell_min[2]          #back to value before normalization
                    Ri[self.ind0_ele_RC_pair[:,2]]=R3_Cell                                          #write R3(nECN,1) into Ri(nECN*nRC,1)
            #-----------------------------for Fitting, Ri needs to be transformed from whole cell data into per thickness per area to be used for element
            Ri=Ri *self.A_electrodes_real
            Ri=Ri/self.scalefactor_z
        self.Ri_record4Ci_Fit=Ri
        return Ri
    #########################################################   
    ################# function for Ci-SoC,T #################
    #########################################################
    def fun_Ci_Interped(self,SoC,T,indRC):      #input SoC, T and indRC(for example, 1,2,3 when nRC=3)of each ECN element, both as vector; output Ci as a vector. Ci unit is same as LUT_Ci_PerA, i.e. El thickness (91μm) per area has Ci F  
    #=================================self written interpolation (slow)
    #    n0=np.size(SoC);  Ci=np.zeros([n0,1])
    #    for i0 in np.arange(n0):
    #        T_ind_1=max(np.where(LUT_T< T[i0])[0]); T_ind_2=min(np.where(LUT_T>= T[i0])[0])             #T_ind_1 is exclusive left limit, T_ind_2 is inclusive right limit
    #        soc_ind_1=max(np.where(LUT_SoC> SoC[i0])[0]); soc_ind_2=min(np.where(LUT_SoC<=SoC[i0])[0])  #soc_ind_1 is exclusive left limit, soc_ind_2 is inclusive right limit
    #        y_1=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_Ci_PerA[T_ind_1,soc_ind_1,indRC[i0]-1],LUT_Ci_PerA[T_ind_2,soc_ind_1,indRC[i0]-1])   #interpolate T at fixed soc_ind_1
    #        y_2=fun_interpolate(T[i0],LUT_T[T_ind_1],LUT_T[T_ind_2],LUT_Ci_PerA[T_ind_1,soc_ind_2,indRC[i0]-1],LUT_Ci_PerA[T_ind_2,soc_ind_2,indRC[i0]-1])   #interpolate T at fixed soc_ind_2
    #        Ci[i0]=fun_interpolate(SoC[i0],LUT_SoC[soc_ind_1],LUT_SoC[soc_ind_2],y_1,y_2)    
    
    #=================================scipy interpolation function (faster)
        Ci=np.zeros([self.nECN*self.nRC,1]); Ci_ele_temp=np.zeros([self.nECN,self.nRC])
        for i0 in np.arange(self.nRC):
            f = interpolate.interp2d(self.LUT_SoC, self.LUT_T, self.LUT_Ci_PerA[:,:,i0], kind=self.status_scipy_interpkind)
            for i00 in np.arange(self.nECN):
                temp=f(SoC[self.ind0_ele_RC_pair[i00,i0],0],T[self.ind0_ele_RC_pair[i00,i0],0])
                Ci_ele_temp[i00,i0]=temp
    
        if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
            for i0 in np.arange(self.nRC):
                temp=Ci_ele_temp[:,i0]
                temp[temp < self.LUT_Ci_PerA_min[i0]]=self.LUT_Ci_PerA_min[i0]; temp[temp > self.LUT_Ci_PerA_max[i0]]=self.LUT_Ci_PerA_max[i0]
                Ci_ele_temp[:,i0]=temp
        Ci[self.ind0_ele_RC_pair,0]=Ci_ele_temp
        return Ci
    def fun_Ci_Fitted(self,SoC,T,indRC):      #input SoC, T and indRC(for example, 1,2,3 when nRC=3)of each ECN element, both as vector; output Ri as a vector. Ri unit is same as LUT_Ri_PerA, i.e. El thickness (91μm) per area has Ri Ohm 
        Ci=np.zeros([self.nECN*self.nRC,1])
        Ri_Cell=self.Ri_record4Ci_Fit/self.A_electrodes_real*self.scalefactor_z     #transfer from per thickness per area into whole cell data
        if self.status_Eparam=='Cylindrical_Cell1':
            SoC_scaled=SoC                                              #for Fitting expression, all inputs are scaled into x~:(0,1): x~ = (x-x_min)/(x_max-x_min); x=x~ *(x_max-_min) +x_min
            T_scaled=(T-self.LUT_T_min)/(self.LUT_T_max-self.LUT_T_min)
            #-----------------------------calculate through fitted surface
            for i0 in np.arange(self.nRC):    
                if i0 == 0:
                    SoC1_scaled=SoC_scaled[self.ind0_ele_RC_pair[:,0]]; T1_scaled=T_scaled[self.ind0_ele_RC_pair[:,0]]
                    R1_Cell=Ri_Cell[self.ind0_ele_RC_pair[:,0]]
                    C1_Cell=self.fun_R1C1_fitted_unscaled_expression_Cell(SoC1_scaled,T1_scaled)/R1_Cell
                    Ci[self.ind0_ele_RC_pair[:,0]]=C1_Cell                                          #write C1(nECN,1) into Ci(nECN*nRC,1)
                    C1_Cell[C1_Cell < self.LUT_Ci_Cell_min[0]]=self.LUT_Ci_Cell_min[0]; C1_Cell[C1_Cell > self.LUT_Ci_Cell_max[0]]=self.LUT_Ci_Cell_max[0]                        #Saturation used
                elif i0 == 1:
                    SoC2_scaled=SoC_scaled[self.ind0_ele_RC_pair[:,1]]; T2_scaled=T_scaled[self.ind0_ele_RC_pair[:,1]]
                    R2_Cell=Ri_Cell[self.ind0_ele_RC_pair[:,1]]
                    C2_Cell=self.fun_R2C2_fitted_unscaled_expression_Cell(SoC2_scaled,T2_scaled)/R2_Cell
                    Ci[self.ind0_ele_RC_pair[:,1]]=C2_Cell                                          #write C2(nECN,1) into Ci(nECN*nRC,1)
                    C2_Cell[C2_Cell < self.LUT_Ci_Cell_min[1]]=self.LUT_Ci_Cell_min[1]; C2_Cell[C2_Cell > self.LUT_Ci_Cell_max[1]]=self.LUT_Ci_Cell_max[1]                        #Saturation used
                else:
                    SoC3_scaled=SoC_scaled[self.ind0_ele_RC_pair[:,2]]; T3_scaled=T_scaled[self.ind0_ele_RC_pair[:,2]]
                    R3_Cell=Ri_Cell[self.ind0_ele_RC_pair[:,2]]
                    C3_Cell=self.fun_R3C3_fitted_unscaled_expression_Cell(SoC3_scaled,T3_scaled) /R3_Cell
                    Ci[self.ind0_ele_RC_pair[:,2]]=C3_Cell                                          #write C3(nECN,1) into Ci(nECN*nRC,1)
                    C3_Cell[C3_Cell < self.LUT_Ci_Cell_min[2]]=self.LUT_Ci_Cell_min[2]; C3_Cell[C3_Cell > self.LUT_Ci_Cell_max[2]]=self.LUT_Ci_Cell_max[2]                        #Saturation used
            #-----------------------------for Fitting, Ci needs to be transformed from whole cell data into per thickness per area to be used for element
            Ci=Ci /self.A_electrodes_real
            Ci=Ci*self.scalefactor_z
        return Ci
    #########################################################   
    ################## function for dVdT-SoC ################
    #########################################################
    def fun_dVdT_Interped(self,SoC):              #input SoC of each ECN element, as vector SoC; output dVdT as vector dVdT; dVdT unit is same as LUT_dVdT_PerA, i.e. per volume has dVdT V/K; SoC and dVdT are from Ei_pair, in the form of 1,2...nECN  
    #=================================self written interpolation (slow)
    #    dVdT=np.zeros([nECN,1]) 
    #    ind_l=np.zeros([nECN],dtype=int); ind_r=np.zeros([nECN],dtype=int)
    #    for i0 in np.arange(nECN):    #i0 is the 0index for ECN element
    #        if SoC[i0]>LUT_SoC_entropy[1]*1.01 or SoC[i0]<LUT_SoC_entropy[-2]*1.01:           #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
    #            print('PyECN warning: dVdT cut-off extrapolation is used for 0index=%d / (%d ECN elements)'%(i0,nECN))
    #        ind_l[i0]=max(np.where(LUT_SoC_entropy> SoC[i0])[0])  #LUT_SoC: 0.99,0.98...0.00, ind_l is exclusive left limit
    #        ind_r[i0]=min(np.where(LUT_SoC_entropy<=SoC[i0])[0])  #ind_r is inclusive right limit
    #    dVdT=fun_interpolate(SoC,LUT_SoC_entropy[ind_l],LUT_SoC_entropy[ind_r],LUT_dVdT_PerA[ind_l],LUT_dVdT_PerA[ind_r])  #conduct interpolation
    
    #=================================scipy interpolation function (faster)
        f = interpolate.interp1d(self.LUT_SoC_entropy.reshape(-1), self.LUT_dVdT_PerA.reshape(-1), kind=self.status_scipy_interpkind)
        dVdT=f(SoC)
        if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
            dVdT[dVdT < self.LUT_dVdT_Cell_min]=self.LUT_dVdT_Cell_min; dVdT[dVdT > self.LUT_dVdT_Cell_max]=self.LUT_dVdT_Cell_max
        return dVdT
    def fun_dVdT_Fitted(self,SoC):              #input SoC of each ECN element, as vector SoC; output dVdT as vector dVdT; dVdT unit is same as LUT_dVdT_PerA, i.e. per volume has dVdT V/K; SoC and dVdT are from Ei_pair, in the form of 1,2...nECN  
        if self.status_Eparam=='Cylindrical_Cell1':
            SoC_scaled=SoC                                              #for Fitting expression, all inputs are scaled into x~:(0,1): x~ = (x-x_min)/(x_max-x_min); x=x~ *(x_max-_min) +x_min
            dVdT_scaled=self.fun_dVdT_fitted_scaled_expression_Cell(SoC_scaled)
            dVdT_scaled[dVdT_scaled < 0]=0; dVdT_scaled[dVdT_scaled > 1]=1                #Saturation used
            dVdT= dVdT_scaled*(self.LUT_dVdT_Cell_max-self.LUT_dVdT_Cell_min)+self.LUT_dVdT_Cell_min
        return dVdT
    #########################################################   
    ############### function for LUT plotting ###############
    #########################################################
    def fun_LUTplot(self):
    #=================================plot OCV    
        plt.figure()
        if self.status_LUTinterp=='Interp':
            SoC=np.linspace(-0.1, 1.1, 100);   T=np.linspace(self.LUT_T[1,0]-self.LUT_T[1,0]/100, self.LUT_T[-2,0]+self.LUT_T[-2,0]/100, 100)
            X,Y = np.meshgrid(SoC,T)
            f = interpolate.interp2d(self.LUT_SoC[1:-1], self.LUT_T[1:-1], self.LUT_OCV_PerA[1:-1,1:-1,0], kind=self.status_scipy_interpkind)
            OCV=f(SoC,T)
            if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
                OCV[OCV < self.LUT_OCV_Cell_min]=self.LUT_OCV_Cell_min; OCV[OCV > self.LUT_OCV_Cell_max]=self.LUT_OCV_Cell_max;                            #saturation condition, out-of-range value (especially minus) is not physical
        if self.status_LUTinterp=='Fitting':
            SoC=np.linspace(self.LUT_SoC[-2], self.LUT_SoC[1], 100);   T=np.linspace(self.LUT_T[1], self.LUT_T[-2], 100)
            X,Y = np.meshgrid(SoC,T)
            OCV=self.fun_OCV_Fitted(X,Y)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, OCV, rstride=1, cstride=1,cmap='winter', edgecolor='none',alpha=0.5)          #plot interpolated or fitted surface
        
        X_LUT,Y_LUT = np.meshgrid(self.LUT_SoC[1:-1],self.LUT_T[1:-1])
        x= X_LUT.reshape(-1); y=Y_LUT.reshape(-1); z=self.LUT_OCV_PerA[1:-1,1:-1,0].reshape(-1) 
        ax.scatter3D(x, y, z,color='r',depthshade=False)                                                    #plot raw LUT scattered points
        
        ax.set_xlabel('SoC'); ax.set_ylabel('Temperature [K]'); ax.set_zlabel('OCV [V]')
        if self.status_LUTinterp=='Interp':
            ax.set_title('%s cell OCV Interpolation, kind=%s'%(self.status_Eparam,self.status_scipy_interpkind))
        if self.status_LUTinterp=='Fitting':
            ax.set_title('%s cell OCV Fitting'%(self.status_Eparam))
    #=================================plot R0    
        plt.figure()
        if self.status_LUTinterp=='Interp':
            SoC=np.linspace(-0.1, 1.1, 100);   T=np.linspace(self.LUT_T[1,0]-self.LUT_T[1,0]/100, self.LUT_T[-2,0]+self.LUT_T[-2,0]/100, 100)
            X,Y = np.meshgrid(SoC,T)
            f = interpolate.interp2d(self.LUT_SoC[1:-1], self.LUT_T[1:-1], self.LUT_R0_PerA[1:-1,1:-1,0], kind=self.status_scipy_interpkind)
            R0=f(SoC,T)
            if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
                R0[R0 < self.LUT_R0_PerA_min]=self.LUT_R0_PerA_min; R0[R0 > self.LUT_R0_PerA_max]=self.LUT_R0_PerA_max;                                   #saturation condition, out-of-range value (especially minus) is not physical
        if self.status_LUTinterp=='Fitting':
            SoC=np.linspace(self.LUT_SoC[-2], self.LUT_SoC[1], 100);   T=np.linspace(self.LUT_T[1], self.LUT_T[-2], 100)
            X,Y = np.meshgrid(SoC,T)
            R0=self.fun_R0_Fitted(X,Y)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, R0, rstride=1, cstride=1,cmap='winter', edgecolor='none',alpha=0.5)   #plot interpolated or fitted surface
       
        X_LUT,Y_LUT = np.meshgrid(self.LUT_SoC[1:-1],self.LUT_T[1:-1])
        x= X_LUT.reshape(-1); y=Y_LUT.reshape(-1); z=self.LUT_R0_PerA[1:-1,1:-1,0].reshape(-1) 
        ax.scatter3D(x, y, z,color='r',depthshade=False)                                                    #plot raw LUT scattered points
        
        ax.set_xlabel('SoC'); ax.set_ylabel('Temperature [K]'); ax.set_zlabel('R0\n for unit area for element\n[Ω·m2]')
        if self.status_LUTinterp=='Interp':
            ax.set_title('%s cell R0 Interpolation, kind=%s'%(self.status_Eparam,self.status_scipy_interpkind))
        if self.status_LUTinterp=='Fitting':
            ax.set_title('%s cell R0 Fitting'%(self.status_Eparam))
    #=================================plot Ri       
        for i0 in np.arange(self.nRC):
            plt.figure()
            if self.status_LUTinterp=='Interp':
                SoC=np.linspace(-0.1, 1.1, 100);   T=np.linspace(self.LUT_T[1,0]-self.LUT_T[1,0]/100, self.LUT_T[-2,0]+self.LUT_T[-2,0]/100, 100)
                X,Y = np.meshgrid(SoC,T)
                f = interpolate.interp2d(self.LUT_SoC[1:-1], self.LUT_T[1:-1], self.LUT_Ri_PerA[1:-1,1:-1,i0], kind=self.status_scipy_interpkind)
                Ri=f(SoC,T)
                if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
                    Ri[Ri < self.LUT_Ri_PerA_min[i0]]=self.LUT_Ri_PerA_min[i0]; Ri[Ri > self.LUT_Ri_PerA_max[i0]]=self.LUT_Ri_PerA_max[i0];                                   #saturation condition, out-of-range value (especially minus) is not physical
            if self.status_LUTinterp=='Fitting':
                SoC=np.linspace(self.LUT_SoC[-2], self.LUT_SoC[1], 100);   T=np.linspace(self.LUT_T[1], self.LUT_T[-2], 100)
                X,Y = np.meshgrid(SoC,T)
                X_scaled=X;          Y_scaled=(Y-self.LUT_T_min)/(self.LUT_T_max-self.LUT_T_min)
                if i0 == 0:
                    Ri_scaled=self.fun_R1_fitted_scaled_expression_Cell(X_scaled,Y_scaled)
                    Ri_scaled[Ri_scaled < 0]=0; Ri_scaled[Ri_scaled > 1]=1                #Saturation used
                    Ri= Ri_scaled*(self.LUT_Ri_Cell_max[0]-self.LUT_Ri_Cell_min[0])+self.LUT_Ri_Cell_min[0]
                elif i0 == 1:
                    Ri_scaled=self.fun_R2_fitted_scaled_expression_Cell(X_scaled,Y_scaled)
                    Ri_scaled[Ri_scaled < 0]=0; Ri_scaled[Ri_scaled > 1]=1                #Saturation used
                    Ri= Ri_scaled*(self.LUT_Ri_Cell_max[1]-self.LUT_Ri_Cell_min[1])+self.LUT_Ri_Cell_min[1]
                else:
                    Ri_scaled=self.fun_R3_fitted_scaled_expression_Cell(X_scaled,Y_scaled)
                    Ri_scaled[Ri_scaled < 0]=0; Ri_scaled[Ri_scaled > 1]=1                #Saturation used
                    Ri= Ri_scaled*(self.LUT_Ri_Cell_max[2]-self.LUT_Ri_Cell_min[2])+self.LUT_Ri_Cell_min[2]
                Ri=Ri *self.A_electrodes_real
                Ri=Ri/self.scalefactor_z
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, Ri, rstride=1, cstride=1,cmap='winter', edgecolor='none',alpha=0.5)           #plot interpolated or fitted surface
       
            X_LUT,Y_LUT = np.meshgrid(self.LUT_SoC[1:-1],self.LUT_T[1:-1])
            x= X_LUT.reshape(-1); y=Y_LUT.reshape(-1); z=self.LUT_Ri_PerA[1:-1,1:-1,i0].reshape(-1) 
            ax.scatter3D(x, y, z,color='r',depthshade=False)                                                    #plot raw LUT scattered points
        
            ax.set_xlabel('SoC'); ax.set_ylabel('Temperature [K]'); ax.set_zlabel('R%d\n for unit area for element\n[Ω·m2]'%(i0+1))
            if self.status_LUTinterp=='Interp':
                ax.set_title('%s cell R%d Interpolation, kind=%s'%(self.status_Eparam,i0+1,self.status_scipy_interpkind))
            if self.status_LUTinterp=='Fitting':
                ax.set_title('%s cell R%d Fitting'%(self.status_Eparam,i0+1))
    #=================================plot Ci       
        for i0 in np.arange(self.nRC):
            plt.figure()
            if self.status_LUTinterp=='Interp':
                SoC=np.linspace(-0.1, 1.1, 100);   T=np.linspace(self.LUT_T[1,0]-self.LUT_T[1,0]/100, self.LUT_T[-2,0]+self.LUT_T[-2,0]/100, 100)
                X,Y = np.meshgrid(SoC,T)
                f = interpolate.interp2d(self.LUT_SoC[1:-1], self.LUT_T[1:-1], self.LUT_Ci_PerA[1:-1,1:-1,i0], kind=self.status_scipy_interpkind)
                Ci=f(SoC,T)
                if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
                    Ci[Ci < self.LUT_Ci_PerA_min[i0]]=self.LUT_Ci_PerA_min[i0]; Ci[Ci > self.LUT_Ci_PerA_max[i0]]=self.LUT_Ci_PerA_max[i0];                                   #saturation condition, out-of-range value (especially minus) is not physical
            if self.status_LUTinterp=='Fitting':
                SoC=np.linspace(self.LUT_SoC[-2], self.LUT_SoC[1], 100);   T=np.linspace(self.LUT_T[1], self.LUT_T[-2], 100)
                X,Y = np.meshgrid(SoC,T)
                X_scaled=X;          Y_scaled=(Y-self.LUT_T_min)/(self.LUT_T_max-self.LUT_T_min)
                if i0 == 0:
                    Ri_scaled_Cell=self.fun_R1_fitted_scaled_expression_Cell(X_scaled,Y_scaled)
                    Ri_Cell= Ri_scaled_Cell*(self.LUT_Ri_Cell_max[0]-self.LUT_Ri_Cell_min[0])+self.LUT_Ri_Cell_min[0]
                    Ci_Cell=self.fun_R1C1_fitted_unscaled_expression_Cell(X_scaled,Y_scaled)/Ri_Cell
                    Ci_Cell[Ci_Cell < self.LUT_Ci_Cell_min[0]]=self.LUT_Ci_Cell_min[0]; Ci_Cell[Ci_Cell > self.LUT_Ci_Cell_max[0]]=self.LUT_Ci_Cell_max[0]                        #Saturation used
                elif i0 == 1:
                    Ri_scaled_Cell=self.fun_R2_fitted_scaled_expression_Cell(X_scaled,Y_scaled)
                    Ri_Cell= Ri_scaled_Cell*(self.LUT_Ri_Cell_max[1]-self.LUT_Ri_Cell_min[1])+self.LUT_Ri_Cell_min[1]
                    Ci_Cell=self.fun_R2C2_fitted_unscaled_expression_Cell(X_scaled,Y_scaled)/Ri_Cell
                    Ci_Cell[Ci_Cell < self.LUT_Ci_Cell_min[1]]=self.LUT_Ci_Cell_min[1]; Ci_Cell[Ci_Cell > self.LUT_Ci_Cell_max[1]]=self.LUT_Ci_Cell_max[1]                        #Saturation used
                else:
                    Ri_scaled_Cell=self.fun_R3_fitted_scaled_expression_Cell(X_scaled,Y_scaled)
                    Ri_Cell= Ri_scaled_Cell*(self.LUT_Ri_Cell_max[2]-self.LUT_Ri_Cell_min[2])+self.LUT_Ri_Cell_min[2]
                    Ci_Cell=self.fun_R3C3_fitted_unscaled_expression_Cell(X_scaled,Y_scaled)/Ri_Cell
                    Ci_Cell[Ci_Cell < self.LUT_Ci_Cell_min[2]]=self.LUT_Ci_Cell_min[2]; Ci_Cell[Ci_Cell > self.LUT_Ci_Cell_max[2]]=self.LUT_Ci_Cell_max[2]                        #Saturation used
                Ci=Ci_Cell /self.A_electrodes_real
                Ci=Ci*self.scalefactor_z
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, Ci, rstride=1, cstride=1,cmap='winter', edgecolor='none',alpha=0.5)           #plot interpolated or fitted surface
       
            X_LUT,Y_LUT = np.meshgrid(self.LUT_SoC[1:-1],self.LUT_T[1:-1])
            x= X_LUT.reshape(-1); y=Y_LUT.reshape(-1); z=self.LUT_Ci_PerA[1:-1,1:-1,i0].reshape(-1) 
            ax.scatter3D(x, y, z,color='r',depthshade=False)                                                    #plot raw LUT scattered points
        
            ax.set_xlabel('SoC'); ax.set_ylabel('Temperature [K]'); ax.set_zlabel('C%d\n for unit area for element\n[F/m2]'%(i0+1))
            if self.status_LUTinterp=='Interp':
                ax.set_title('%s cell C%d Interpolation, kind=%s'%(self.status_Eparam,i0+1,self.status_scipy_interpkind))
            if self.status_LUTinterp=='Fitting':
                ax.set_title('%s cell C%d Fitting'%(self.status_Eparam,i0+1))
    #=================================plot dVdT       
        plt.figure()
        if self.status_LUTinterp=='Interp':
            SoC=np.linspace(-0.1, 1.1, 100)
            f = interpolate.interp1d(self.LUT_SoC_entropy.reshape(-1), self.LUT_dVdT_PerA.reshape(-1), kind=self.status_scipy_interpkind)
            dVdT=f(SoC)
            if self.status_scipy_interpkind != 'linear':                              #Saturation is used when cubic interpolation is used. Because runge phenomenon may happen causing minus or unreasonably high value
                dVdT[dVdT < self.LUT_dVdT_Cell_min]=self.LUT_dVdT_Cell_min; dVdT[dVdT > self.LUT_dVdT_Cell_max]=self.LUT_dVdT_Cell_max                                   #saturation condition, out-of-range value (especially minus) is not physical
        if self.status_LUTinterp=='Fitting':
            SoC=np.linspace(self.LUT_SoC_entropy[-2], self.LUT_SoC_entropy[1], 100)
            dVdT=self.fun_dVdT_Fitted(SoC)
        plt.plot(SoC,dVdT,'b')                                                                               #plot interpolated or fitted curve
        plt.plot(self.LUT_SoC_entropy[1:-1],self.LUT_dVdT_PerA[1:-1],'ro')                                                    #plot raw LUT scattered points
        plt.xlabel('SoC'); plt.ylabel('dV/dT [V/K]') 
        if self.status_LUTinterp=='Interp':
            plt.title('Pouch_Cell1 cell dV/dT %s, kind=%s'%(self.status_LUTinterp,self.status_scipy_interpkind))
        if self.status_LUTinterp=='Fitting':
            plt.title('Pouch_Cell1 cell dV/dT Fitting')
    #########################################################   
    ############### function for interpolation ##############
    #########################################################
    def fun_interpolate(x,x_1,x_2,y_1,y_2):      #input x, exclusive left limit x_1, inclusive right limit x_2, y_1 resulting from x_1, y_2 resulting from x_2, 
                                                 #output interplated y    inputs can be vector (or scalar), output can be vector (or scalar)  
        y=(x-x_1)*(y_2-y_1)/(x_2-x_1)+y_1
        return y
    #########################################################   
    ######   function for Thermal initial condition   #######
    #########################################################
    def fun_IC_4T(self):   
#    if status_ThermalBC_Core=='BCFix':
#        Tini_4T=np.zeros([ntotal_4T,1])
#        Tini_4T[ind0_Domain_4T_4BCFix]=T_initial
        if ip.status_FormFactor == 'Pouch' or self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
            self.Tini_4T_ALL=self.T_initial * np.ones([self.n_4T_ALL,1])
            global step 
            step=0; self.fun_BC_4T_ALL()   #step=0 is only for situation of status_Can_Scheme=='ReadBCTem'. In this situation, step is used in Table_T_center[step]. In other situations e.g. status_Can_Scheme=='AllTem', step is not used
            self.Tini_4T_ALL[self.ind0_BCtem_ALL]=self.T3_4T_ALL[self.ind0_BCtem_ALL]  
        return self.Tini_4T_ALL
    #########################################################   
    ########### functions for checking thermal BC ###########
    #########################################################
    def fun_BC_4T_check(self):                      
        Input=np.zeros([self.n_4T_ALL,6])            
        Standard=np.zeros([self.n_4T_ALL,6]) 
        if ip.status_FormFactor == 'Pouch':  #refer to ppt p203
            Standard[self.ind0_jx1NaN_4T,0]=1                   #these nodes in x1 direction should be input
            Standard[self.ind0_jx2NaN_4T,1]=1                   #these nodes in x2 direction should be input
            Standard[self.ind0_jy1NaN_4T,2]=1                   #...
            Standard[self.ind0_jy2NaN_4T,3]=1
            Standard[self.ind0_jz1NaN_4T,4]=1
            Standard[self.ind0_jz2NaN_4T,5]=1           
        if ip.status_FormFactor == 'Prismatic': #refer to ppt p205,   Note Prismatic & status_ThermalPatition_Can=='No' case of Cylindrical have have lines for 
            Standard[np.array([],dtype=int),0]                                         =1                   #these nodes in x1 direction should be input
            Standard[self.ind0_Geo_finalcrosssectionWithEnds_4T_4SepFill,1]            =1                   #these nodes in x2 direction should be input
            Standard[self.ind0_Geo_top_4T_4SepFill,2]                                  =1                   #...
            Standard[self.ind0_Geo_bottom_4T_4SepFill,3]                               =1
            Standard[np.array([],dtype=int),4]                                         =1
            Standard[np.concatenate((self.ind0_Geo_top_edge_outer_4T_4SepFill,self.ind0_Geo_surface_4T_4SepFill,self.ind0_Geo_bottom_edge_outer_4T_4SepFill)),5]=1    
        if ip.status_FormFactor == 'Cylindrical' and self.status_ThermalPatition_Can=='No': #refer to ppt p205
            Standard[np.array([],dtype=int),0]                                         =1                   #these nodes in x1 direction should be input
            Standard[self.ind0_Geo_finalcrosssectionWithEnds_4T_4SepFill,1]            =1                   #these nodes in x2 direction should be input
            Standard[self.ind0_Geo_top_4T_4SepFill,2]                                  =1                   #...
            Standard[self.ind0_Geo_bottom_4T_4SepFill,3]                               =1
            Standard[np.array([],dtype=int),4]                                         =1
            Standard[np.concatenate((self.ind0_Geo_top_edge_outer_4T_4SepFill,self.ind0_Geo_surface_4T_4SepFill,self.ind0_Geo_bottom_edge_outer_4T_4SepFill)),5]=1 
        else:
            Standard[np.array([],dtype=int),0]                                         =1                   #these nodes in x1 direction should be input
            Standard[np.array([],dtype=int),1]                                         =1                   #these nodes in x2 direction should be input
            Standard[self.ind0_Geo_Can_top_4T +(self.ntotal_4T+self.nAddCore_4T),2]    =1                   #...
            Standard[self.ind0_Geo_Can_bottom_4T +(self.ntotal_4T+self.nAddCore_4T),3] =1
            Standard[np.array([],dtype=int),4]                                         =1
            Standard[np.concatenate((self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T+self.nAddCore_4T), 
                                     self.ind0_Geo_Can_surface_4T +(self.ntotal_4T+self.nAddCore_4T),
                                     self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T+self.nAddCore_4T)
                                     )),5]                                             =1    
        Input[np.isnan(self.h_4T_ALL)]=0; Input[~np.isnan(self.h_4T_ALL)]=1    #change NaN elements in h_4T to be 0, input elements to be 1
        Input[self.ind0_BCtem_ALL]=1                                  #if BC is temperature constrained for node, no need to check this node
        Outcome=Input*Standard
        InputError=np.argwhere(~(Outcome == Standard))
        if len(InputError)==0:
            print('thermal BC input is passed')
            temp1=input('press y to continue, n to break:')
            if temp1 != 'y':
                raise Exception('exit')
        else:
            print('\nnode ind0 number:', InputError[:,0] )
            print('\nin direction(0~5):', InputError[:,1])
            print('\nis missing Thermal BC inputs(h or T), please check fun_BC_4T_ALL()')
            input('please press any key +Enter to break:')
            raise Exception('exit')
    #########################################################   
    ############## functions for getting Lists ##############
    #########################################################
    def fun_GetList_node2ele(self):        #return element 0index respective of all nodes, with expection of CC node; in Electrical node frame
        List_node2ele=np.zeros([self.ntotal,1],dtype=int)
        for i0 in np.arange(self.ntotal):
            if self.mat[i0] <=2:
                List_node2ele[i0]=-9999
            else:
                if ip.status_FormFactor == 'Pouch':
                    n_before=self.nx*(2*self.nstack-1)*(self.yn[i0]-1) + self.nx*((self.zn[i0]-1)//(self.ne+1)) + (self.xn[i0]-1)  #calculate how many elements before this element
                if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':
                    n_before=self.nx*(2*self.nstack-1)*(self.ax[i0]-1) + self.nx*((self.ra[i0]-1)//(self.ne+1)) + (self.an[i0]-1)  #calculate how many elements before this element
                List_node2ele[i0]=n_before   #for example, if there are 2 elements before, this element is the 3rd element, 0index is 2
        return List_node2ele
    def fun_GetList_node2ele_4T(self):     #return element 0index respective of all nodes, made for El nodes; in Thermal node frame
        List_node2ele_4T=np.zeros([self.ntotal_4T,1],dtype=int)
        for i0 in np.arange(self.ntotal_4T):
            if self.mat_4T[i0] <=2:
                List_node2ele_4T[i0]=-9999
            else:
                if ip.status_FormFactor == 'Pouch':
                    n_before=self.nx*(2*self.nstack-1)*(self.yn_4T[i0]-1) + self.nx*((self.zn_4T[i0]-1)//2) + (self.xn_4T[i0]-1)  #calculate how many elements before this element
                if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':    
                    n_before=self.nx*(2*self.nstack-1)*(self.ax_4T[i0]-1) + self.nx*((self.ra_4T[i0]-1)//2) + (self.an_4T[i0]-1)  #calculate how many elements before this element
                List_node2ele_4T[i0]=n_before   #for example, if there are 2 elements before, this element is the 3rd element, 0index is 2
        return List_node2ele_4T
    def fun_GetList_ele2node(self):        #return node 0indexes respective of all elements, made for El nodes; in Electrical node frame
        List_ele2node=np.zeros([self.nECN,self.ne],dtype=int)  
        indnode=np.arange(self.ntotal).reshape(-1,1); indele=self.List_node2ele
        for i0 in np.arange(self.nECN):
            List_ele2node[i0] = indnode[np.where(indele==i0)]
        return List_ele2node

    def fun_GetList_ele2node_4T(self):     #return node 0indexes respective of all elements, made for El nodes; in Thermal node frame
        List_ele2node_4T=np.zeros([self.nECN,1],dtype=int)  
        indnode=np.arange(self.ntotal_4T).reshape(-1,1); indele=self.List_node2ele_4T
        for i0 in np.arange(self.nECN):
            List_ele2node_4T[i0] = indnode[np.where(indele==i0)]
        return List_ele2node_4T
    
    def fun_GetList_node2node_E2T(self):   #return node 0indexes respective of all nodes, made for CC nodes; 
        List_node2node_E2T=np.zeros([self.ntotal,1],dtype=int)
        for i0 in np.arange(self.ntotal):
            if self.mat[i0] >= 3:      
                List_node2node_E2T[i0]=-9999  #for El node, no calculation, put NAN
            else:
                if ip.status_FormFactor == 'Pouch':
                    if self.zn[i0]!=self.nz:      #not the nodes in the backmost surface
                        indele=self.List_node2ele[self.ind0_jz1[i0]]
                        indnode_4T=self.List_ele2node_4T[indele]
                        List_node2node_E2T[i0]=self.jz2_4T[indnode_4T]-1

                    else:              #for nodes in the backmost surface
                        indele=self.List_node2ele[self.ind0_jz2[i0]]
                        indnode_4T=self.List_ele2node_4T[indele]
                        List_node2node_E2T[i0]=self.jz1_4T[indnode_4T]-1
                if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':        
                    if self.ra[i0]!=self.nz:    #not the node in the outmost layer
                        indele=self.List_node2ele[self.jz2[i0]-1]
                        indnode_4T=self.List_ele2node_4T[indele]
                        List_node2node_E2T[i0]=self.jz1_4T[indnode_4T]-1
                    
                    else:              #for node in the outmost layer
                        indele=self.List_node2ele[self.jz1[i0]-1]
                        indnode_4T=self.List_ele2node_4T[indele]
                        List_node2node_E2T[i0]=self.jz2_4T[indnode_4T]-1
        return List_node2node_E2T
    def fun_GetList_node2node_T2E(self):   #return node 0indexes respective of all nodes, made for CC nodes; 
        List_node2node_T2E=np.zeros([self.ntotal_4T,1],dtype=int)
        for i0 in np.arange(self.ntotal_4T):
            if self.mat_4T[i0] >= 3:      
                List_node2node_T2E[i0]=-9999  #for El node, no calculation, put NAN
            else:
                if ip.status_FormFactor == 'Pouch':
                    if self.zn_4T[i0]!=self.nz_4T:    #not the nodes in the backmost surface
                        indele=self.List_node2ele_4T[self.ind0_jz1_4T[i0]]
                        indnode=self.List_ele2node[indele,0]   #select the 0 column
                        List_node2node_T2E[i0]=self.jz2[indnode]-1
                    else:              #for nodes in the backmost surface
                        indele=self.List_node2ele_4T[self.ind0_jz2_4T[i0]]
                        indnode=self.List_ele2node[indele,-1]  #select the last column
                        List_node2node_T2E[i0]=self.jz1[indnode]-1
                if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':    
                    if self.ra_4T[i0]!=self.nz_4T:    #not the node in the outmost layer
                        indele=self.List_node2ele_4T[self.jz2_4T[i0]-1]
                        indnode=self.List_ele2node[indele,0]   #select the 0 column
                        List_node2node_T2E[i0]=self.jz1[indnode]-1
                    else:              #for node in the outmost layer
                        indele=self.List_node2ele_4T[self.jz1_4T[i0]-1]
                        indnode=self.List_ele2node[indele,-1]  #select the last column
                        List_node2node_T2E[i0]=self.jz2[indnode]-1
        return List_node2node_T2E
    #########################################################   
    ############## functions for getting Lists ##############
    #########################################################
    def fun_GetList_Neo2General(self):    
        temp=np.append(self.Al,self.Cu)
        List_Neo2General=temp[np.argsort(temp)]
        return List_Neo2General.reshape(-1,1)
        #return List_Neo2General
    def fun_GetList_General2Neo(self):
        List_General2Neo=-9999*np.ones([self.ntotal,1],dtype=int)
        for i0 in np.arange(self.ntotal):
            if i0 in self.List_Neo2General:
                List_General2Neo[i0]=np.where(self.List_Neo2General==i0)[0]
        return List_General2Neo
    #########################################################   
    ######       function for electrical Geometry        #######
    ######################################################### 
    def fun_get_Geo(self):
#        global ind0_Geo_left, ind0_Geo_right
#        global ind0_Geo_left_Al, ind0_Geo_left_Cu, ind0_Geo_right_Al, ind0_Geo_right_Cu
#        global ind0_Geo_left_Al_OneThird, ind0_Geo_left_Cu_TwoThird
        if ip.status_FormFactor == 'Pouch':
            ind0_Geo_left=np.where(self.xn==1)[0]                       #left nodes i.g. ind0=[0,15,30,45,48,63,78,93,96,111,126,141]
            ind0_Geo_right=np.where(self.xn==self.nx)[0]                     #right nodes i.g. ind0=[2,17,32,47,50,65,80,95,98,113,128,143]
            
            ind0_Geo_left_Al=np.array([x for x in ind0_Geo_left if self.mat[x]==1 ],dtype=int)      #left Al nodes (Al tab) i.g. ind0=[0,30,48,78,96,126]
            ind0_Geo_left_Cu=np.array([x for x in ind0_Geo_left if self.mat[x]==2 ],dtype=int)      #left Cu nodes (Cu tab) i.g. ind0=[15,45,63,93,111,141]
            ind0_Geo_right_Al=np.array([x for x in ind0_Geo_right if self.mat[x]==1 ],dtype=int)    #right Al nodes (Cu tab) i.g. ind0=[2,32,50,80,98,128]
            ind0_Geo_right_Cu=np.array([x for x in ind0_Geo_right if self.mat[x]==2 ],dtype=int)    #right Cu nodes (Cu tab) i.g. ind0=[17,47,65,95,113,143]
        
            ind0_Geo_left_Al_OneThird=np.array([x for x in ind0_Geo_left_Al if self.yn[x]==int(self.ny/3) ],dtype=int)
            ind0_Geo_left_Cu_TwoThird=np.array([x for x in ind0_Geo_left_Cu if self.yn[x]==self.ny-int(self.ny/3)+1 ],dtype=int)

            (
            self.ind0_Geo_left, self.ind0_Geo_right,
            self.ind0_Geo_left_Al, self.ind0_Geo_left_Cu, self.ind0_Geo_right_Al, self.ind0_Geo_right_Cu,
            self.ind0_Geo_left_Al_OneThird, self.ind0_Geo_left_Cu_TwoThird
            )=(                                                          
            ind0_Geo_left, ind0_Geo_right,
            ind0_Geo_left_Al, ind0_Geo_left_Cu, ind0_Geo_right_Al, ind0_Geo_right_Cu,
            ind0_Geo_left_Al_OneThird, ind0_Geo_left_Cu_TwoThird
            )
        if ip.status_FormFactor == 'Prismatic':
            ind0_Geo_top=np.where(self.ax==1)[0]                                                              #top nodes i.g. ind0=[0~63]
            ind0_Geo_bottom=np.where(self.ax==self.ny)[0]
            ind0_Geo_top_Al=np.array([x for x in ind0_Geo_top if self.mat[x]==1 ],dtype=int)                  #top Al nodes
            ind0_Geo_top_Cu=np.array([x for x in ind0_Geo_top if self.mat[x]==2 ],dtype=int)                  #top Cu nodes
            ind0_Geo_bottom_Al=np.array([x for x in ind0_Geo_bottom if self.mat[x]==1 ],dtype=int)            #bottom Al nodes
            ind0_Geo_bottom_Cu=np.array([x for x in ind0_Geo_bottom if self.mat[x]==2 ],dtype=int)            #bottom Cu nodes        
            ind0_Geo_top2_10_102_110=np.array([x for x in ind0_Geo_top_Al if self.an[x]==self.ind0_SpiralandStripe_boundary1+1 or self.an[x]==self.ind0_SpiralandStripe_boundary2+1+2*self.nx_pouch ],dtype=int)              #top Al nodes in line i.g. ind0=[3,43]
            ind0_Geo_top55_57_155_157=np.array([x for x in ind0_Geo_top_Cu if self.an[x]==self.ind0_SpiralandStripe_boundary1+1+self.nx_pouch or self.an[x]==self.ind0_SpiralandStripe_boundary2+1+self.nx_pouch ],dtype=int)     #bottom Cu nodes in line i.g. ind0=[151,191]
            ind0_Geo_top2_102=np.array([x for x in ind0_Geo_top2_10_102_110 if self.an[x]==self.ind0_SpiralandStripe_boundary1+1 ],dtype=int)              
            ind0_Geo_top55_155=np.array([x for x in ind0_Geo_top55_57_155_157 if self.an[x]==self.ind0_SpiralandStripe_boundary1+1+self.nx_pouch ],dtype=int)     
            (
            self.ind0_Geo_top, self.ind0_Geo_bottom,
            self.ind0_Geo_top_Al, self.ind0_Geo_top_Cu,
            self.ind0_Geo_bottom_Al, self.ind0_Geo_bottom_Cu,
            self.ind0_Geo_top2_10_102_110, self.ind0_Geo_top55_57_155_157,
            self.ind0_Geo_top2_102, self.ind0_Geo_top55_155
            )=(
            ind0_Geo_top, ind0_Geo_bottom,
            ind0_Geo_top_Al, ind0_Geo_top_Cu,
            ind0_Geo_bottom_Al, ind0_Geo_bottom_Cu,
            ind0_Geo_top2_10_102_110, ind0_Geo_top55_57_155_157,
            ind0_Geo_top2_102, ind0_Geo_top55_155
            )
        if ip.status_FormFactor == 'Cylindrical':
            ind0_Geo_top=np.where(self.ax==1)[0]                                                              #top nodes i.g. ind0=[0~63]
            ind0_Geo_bottom=np.where(self.ax==self.ny)[0]
            ind0_Geo_top_Al=np.array([x for x in ind0_Geo_top if self.mat[x]==1 ],dtype=int)                  #top Al nodes (Al tab) i.g. ind0=[0,1,2,3, 40,41,42,43]
            ind0_Geo_bottom_Cu=np.array([x for x in ind0_Geo_bottom if self.mat[x]==2 ],dtype=int)            #bottom Cu nodes (Cu tab) i.g. ind0=[148,149,150,151, 188,189,190,191]        
            ind0_Geo_top4_44=np.array([x for x in ind0_Geo_top_Al if self.an[x]==self.nx ],dtype=int)              #top Al nodes in line i.g. ind0=[3,43]
            ind0_Geo_bottom152_192=np.array([x for x in ind0_Geo_bottom_Cu if self.an[x]==self.nx ],dtype=int)     #bottom Cu nodes in line i.g. ind0=[151,191]        
            ind0temp1=np.array([x for x in ind0_Geo_top if self.ra[x]==self.nz ],dtype=int)
            ind0_Geo_top64=np.array([x for x in ind0temp1 if self.an[x]==self.nx ],dtype=int)                      #top rightmost node i.g. ind0=[63]
            (
            self.ind0_Geo_top, self.ind0_Geo_bottom,                  
            self.ind0_Geo_top_Al, self.ind0_Geo_bottom_Cu,             
            self.ind0_Geo_top4_44, self.ind0_Geo_bottom152_192,        
            self.ind0_Geo_top64                                        
            )=(                                                          
            ind0_Geo_top, ind0_Geo_bottom,                             
            ind0_Geo_top_Al, ind0_Geo_bottom_Cu,                       
            ind0_Geo_top4_44, ind0_Geo_bottom152_192,                  
            ind0_Geo_top64
            )


   
   
    #########################################################   
    ############### function for energy check ###############
    #########################################################
    def fun_Echeck(self,step):
#        global S_stencil_4T_ALL
        self.Egen_Total_record[step] = np.sum(self.q_4T_ALL[:,0]*self.V_stencil_4T_ALL)*self.dt + self.Egen_Total_record[step-1]
    
        self.S_stencil_4T_ALL=np.nan*np.zeros([self.n_4T_ALL,6])             # node plane area (in 6 directions) used in thermal stencil; S_Stencil_4T_ALL: ((δy1+δy2)/2*(δz1+δz2)/2,(δy1+δy2)/2*δz1+δz2)/2, (δx1+δx2)/2*(δz1+δz2)/2,(δx1+δx2)/2*(δz1+δz2)/2, (δx1+δx2)/2*(δy1+δy2)/2,(δx1+δx2)/2*(δy1+δy2)/2), shape is (87,1)
        self.S_stencil_4T_ALL[:,0]=0.5*self.delta_xyz_4T_ALL[:,2] * 0.5*self.delta_xyz_4T_ALL[:,4]
        self.S_stencil_4T_ALL[:,1]=self.S_stencil_4T_ALL[:,0]
        self.S_stencil_4T_ALL[:,2]=0.5*self.delta_xyz_4T_ALL[:,0] * 0.5*self.delta_xyz_4T_ALL[:,4]
        self.S_stencil_4T_ALL[:,3]=self.S_stencil_4T_ALL[:,2]
        self.S_stencil_4T_ALL[:,4]=0.5*self.delta_xyz_4T_ALL[:,0] * 0.5*self.delta_xyz_4T_ALL[:,2]
        self.S_stencil_4T_ALL[:,5]=self.S_stencil_4T_ALL[:,4]
         
        if ip.status_FormFactor == 'Prismatic' or ip.status_FormFactor == 'Cylindrical':
        #--only for cylindrical and prismatic
            self.S_stencil_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2] = self.S_Irreg_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]
            self.S_stencil_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3] = self.S_Irreg_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]

        self.Eext_Total_BCconv_record[step] = np.nansum( self.h_4T_ALL * self.S_stencil_4T_ALL * -(self.T_record[:,step-1].reshape(-1,1)-self.T_cooling) ) *self.dt + self.Eext_Total_BCconv_record[step-1]
        
        self.Eint_Delta_record[step] = np.sum(self.RouXc_4T_ALL[:,0]*self.V_stencil_4T_ALL*(self.T_record[:,step]-self.T_initial))
    
        self.Ebalance_record[step]=self.Egen_Total_record[step]+self.Eext_Total_BCconv_record[step]-self.Eint_Delta_record[step] 
    #########################################################   
    ############### function for Electrical BC ##############
    #########################################################
    def fun_BC(self,step):    
#        global I_ext, V_ext, status_IVmode      #I_ext is current load (current control), V_ext is positive voltage (voltage control). status_IVmode: 0 for current control, 1 for voltage control   
        #===============================without PID control===============================
        if self.status_PID=='No':   
            #---------------constant current mode---------------
            self.status_IVmode=ip.status_IVmode
            #---------------------------------------------------
            self.I_ext = self.status_discharge * self.Capacity_rated0/3600*self.C_rate                  #for discharge, status_discharge=1; for charge, status_discharge=-1
                #----------------------user defined discharge----------------------
            if hasattr(self,'Table_I_ext'):   #if current is already loaded from external file
                self.I_ext=self.Table_I_ext[step]                                                                      
            #---------------CC-CV mode
            if self.status_IVmode==0:
                self.I_ext = self.status_discharge * self.Capacity_rated0/3600*self.C_rate                  #for discharge, status_discharge=1; for charge, status_discharge=-1
            elif self.status_IVmode==1:
                self.V_ext = self.V_highlimit_single
        #=============================== with PID control ================================
    #########################################################   
    ###########       function for heat gen       ###########
    #########################################################
    def fun_HeatGen_4T(self):              #return heat gen vector q_4T   
        q_4T_ALL=np.zeros([self.n_4T_ALL,1])
        #------------------calculate the CC node heat gen term----------------
        if self.status_CC=='Yes':      #if CC='No', no heat gen added to CC node 
            RAl_pair_4T=self.RAl_pair.copy()
            RAl_pair_4T[:,0]=self.List_node2node_E2T[self.RAl_pair[:,0].astype(int)].reshape(self.nRAl)   
            RAl_pair_4T[:,1]=self.List_node2node_E2T[self.RAl_pair[:,1].astype(int)].reshape(self.nRAl)
            RCu_pair_4T=self.RCu_pair.copy()
            RCu_pair_4T[:,0]=self.List_node2node_E2T[self.RCu_pair[:,0].astype(int)].reshape(self.nRCu) 
            RCu_pair_4T[:,1]=self.List_node2node_E2T[self.RCu_pair[:,1].astype(int)].reshape(self.nRCu)
            for i0 in np.arange(self.nRAl):   #calculate Al node heat gen; loop over RAl_pair_4T
                Al_focus_4T=RAl_pair_4T[i0,0].astype(int); Al_focus=self.RAl_pair[i0,0].astype(int) #Al node to add heat gen into
                Al_side=self.RAl_pair[i0,1].astype(int)
                RAl=RAl_pair_4T[i0,2]
                q_4T_ALL[Al_focus_4T]=q_4T_ALL[Al_focus_4T] + (self.U3[Al_focus]-self.U3[Al_side])**2 /RAl  #heat gen is U^2/R
            for i0 in np.arange(self.nRCu):   #calculate Cu node heat gen; loop over RCu_pair_4T
                Cu_focus_4T=RCu_pair_4T[i0,0].astype(int); Cu_focus=self.RCu_pair[i0,0].astype(int) #Cu node to add heat gen into
                Cu_side=self.RCu_pair[i0,1].astype(int)
                RCu=RCu_pair_4T[i0,2]
                q_4T_ALL[Cu_focus_4T]=q_4T_ALL[Cu_focus_4T] + (self.U3[Cu_focus]-self.U3[Cu_side])**2 /RCu  #heat gen is U^2/R
        #------------------calculate the El node heat gen term------------------
        R0_pair_4T=self.R0_pair.copy()
        R0_pair_4T[:,0]=self.List_node2node_E2T[self.R0_pair[:,0].astype(int)].reshape(self.nECN)
        R0_pair_4T[:,1]=self.List_ele2node_4T[self.List_node2ele[self.R0_pair[:,1].astype(int)]].reshape(self.nECN)   
        El_focus_4T=R0_pair_4T[:,1].astype(int) #El node to add heat gen into
        El_1=self.R0_pair[:,0].astype(int); El_2=self.R0_pair[:,1].astype(int)
        q_4T_ALL[El_focus_4T]=q_4T_ALL[El_focus_4T] + (self.U3[El_1]-self.U3[El_2])**2/ R0_pair_4T[:,2].reshape(-1,1) 
        if self.nRC != 0:     #if nRC=0, no heat gen added to El node     
            RC_pair_4T=self.RC_pair.copy()
            RC_pair_4T[:,0]=self.List_ele2node_4T[self.List_node2ele[self.RC_pair[:,0].astype(int)]].reshape(self.nRC*self.nECN)    #for RC_pair_4T, 0th and 1st coulmn are the same
            RC_pair_4T[:,1]=RC_pair_4T[:,0]
            El_focus_4T=RC_pair_4T[self.ind0_ele_RC_pair_4T,0].astype(int) #El node to add heat gen into
            El_1=self.RC_pair[self.ind0_ele_RC_pair,0].astype(int); El_2=self.RC_pair[self.ind0_ele_RC_pair,1].astype(int)        
            q_4T_ALL[El_focus_4T]=q_4T_ALL[El_focus_4T] + np.sum((self.U3[El_1,0]-self.U3[El_2,0])**2/ self.RC_pair[self.ind0_ele_RC_pair,2],axis=1).reshape(-1,1)   
            if ip.status_FormFactor == 'Cylindrical':
                if self.status_Cap_heatgen == 'Yes':
                    V_ratio = self.V_stencil_4T_ALL[ self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T + self.nAddCore_4T)] / np.sum(self.V_stencil_4T_ALL[ self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T + self.nAddCore_4T)])
                    q_4T_ALL[self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T + self.nAddCore_4T),0] += (self.Rcap * (self.I0_record[step]**2)) * V_ratio                  
        return q_4T_ALL
    #########################################################   
    ###########   function for heat gen for Neo   ###########
    #########################################################
    def fun_HeatGen_neo_4T(self):              #return heat gen vector q_4T   
        q_4T_ALL=np.zeros([self.n_4T_ALL,1])
        #------------------calculate the CC node heat gen term----------------
        if self.status_CC=='Yes':      #if CC='No', no heat gen added to CC node 
            RAl_pair_4T=self.RAl_pair.copy()
            RAl_pair_4T[:,0]=self.List_node2node_E2T[self.RAl_pair[:,0].astype(int)].reshape(self.nRAl)   
            RAl_pair_4T[:,1]=self.List_node2node_E2T[self.RAl_pair[:,1].astype(int)].reshape(self.nRAl)
            RCu_pair_4T=self.RCu_pair.copy()
            RCu_pair_4T[:,0]=self.List_node2node_E2T[self.RCu_pair[:,0].astype(int)].reshape(self.nRCu) 
            RCu_pair_4T[:,1]=self.List_node2node_E2T[self.RCu_pair[:,1].astype(int)].reshape(self.nRCu)
            for i0 in np.arange(self.nRAl):   #calculate Al node heat gen; loop over RAl_pair_4T
                Al_focus_4T=RAl_pair_4T[i0,0].astype(int); Al_focus=self.RAl_pair[i0,0].astype(int) #Al node to add heat gen into
                Al_side=self.RAl_pair[i0,1].astype(int)
                RAl=RAl_pair_4T[i0,2]
                q_4T_ALL[Al_focus_4T]=q_4T_ALL[Al_focus_4T] + (self.U3[Al_focus]-self.U3[Al_side])**2 /RAl  #heat gen is U^2/R
            for i0 in np.arange(self.nRCu):   #calculate Cu node heat gen; loop over RCu_pair_4T
                Cu_focus_4T=RCu_pair_4T[i0,0].astype(int); Cu_focus=self.RCu_pair[i0,0].astype(int) #Cu node to add heat gen into
                Cu_side=self.RCu_pair[i0,1].astype(int)
                RCu=RCu_pair_4T[i0,2]
                q_4T_ALL[Cu_focus_4T]=q_4T_ALL[Cu_focus_4T] + (self.U3[Cu_focus]-self.U3[Cu_side])**2 /RCu  #heat gen is U^2/R
        #------------------calculate the El node heat gen term------------------
        R0_pair_4T=self.R0_pair.copy()
        R0_pair_4T[:,0]=self.List_node2node_E2T[self.R0_pair[:,0].astype(int)].reshape(self.nECN)
        R0_pair_4T[:,1]=self.List_ele2node_4T[self.List_node2ele[self.R0_pair[:,1].astype(int)]].reshape(self.nECN)   
        El_focus_4T=R0_pair_4T[:,1].astype(int) #El node to add heat gen into
        q_4T_ALL[El_focus_4T]=q_4T_ALL[El_focus_4T] + (self.Ii_3)**2 * R0_pair_4T[:,2].reshape(-1,1)
        if self.nRC != 0:     #if nRC=0, no heat gen added to El node     
            RC_pair_4T=self.RC_pair.copy()
            RC_pair_4T[:,0]=self.List_ele2node_4T[self.List_node2ele[self.RC_pair[:,0].astype(int)]].reshape(self.nRC*self.nECN)    #for RC_pair_4T, 0th and 1st coulmn are the same
            RC_pair_4T[:,1]=RC_pair_4T[:,0]
            El_focus_4T=RC_pair_4T[self.ind0_ele_RC_pair_4T,0].astype(int) #El node to add heat gen into
            q_4T_ALL[El_focus_4T]=q_4T_ALL[El_focus_4T] + np.sum(self.IRi_3**2*self.Ri_ele,axis=1).reshape(-1,1)           
            if ip.status_FormFactor == 'Cylindrical':
                if self.status_Cap_heatgen == 'Yes':
                    V_ratio = self.V_stencil_4T_ALL[ self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T + self.nAddCore_4T)] / np.sum(self.V_stencil_4T_ALL[ self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T + self.nAddCore_4T)])
                    q_4T_ALL[self.ind0_Geo_Can_node30to33_4T +(self.ntotal_4T + self.nAddCore_4T),0] += (self.Rcap * (self.I0_record[step]**2)) * V_ratio                  
        return q_4T_ALL
    #########################################################   
    ###########        function for entropy       ###########
    #########################################################
    def fun_Entropy_4T(self):
        Ei_pair_4T=self.Ei_pair.copy()
        Ei_pair_4T[:,0]=self.List_ele2node_4T[self.List_node2ele[self.Ei_pair[:,0].astype(int)]].reshape(self.nECN)    #for Ei_pair_4T, 0th and 1st coulmn are the same
        Ei_pair_4T[:,1]=Ei_pair_4T[:,0]
            
        El_focus_4T=Ei_pair_4T[:,0].astype(int) #El node to add entropy into
        if self.status_LUTinterp=='Interp':
            if not all(self.SoC_ele<self.LUT_SoC_entropy[1])*all(self.SoC_ele>self.LUT_SoC_entropy[-2]):     #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
                print('PyECN warning: cut-off extrapolation is used for dVdT')  
            dVdT=self.fun_dVdT_Interped(self.SoC_ele)
        if self.status_LUTinterp=='Fitting':
            if not all(self.SoC_ele<self.LUT_SoC_entropy[1])*all(self.SoC_ele>self.LUT_SoC_entropy[-2]):     #indicator showing whether extrapolation is used; LUT_SoC[0]=-inf, LUT_SoC[1]=1
                print('PyECN warning: cut-off saturation (SoC axis) is used for dVdT')  
            dVdT=self.fun_dVdT_Fitted(self.SoC_ele)
        self.q_4T_ALL[El_focus_4T]=self.q_4T_ALL[El_focus_4T] + self.T1_4T_ALL[El_focus_4T]*self.U3[self.ntotal:(self.ntotal+self.nECN)]*dVdT  
        return self.q_4T_ALL   
    #########################################################   
    ###########     function for Thermal model    ###########
    #########################################################
    def fun_Thermal(self, T1_4T_ALL,T3_4T_ALL, ind0_BCtem_ALL, ind0_BCtem_others_ALL, h_4T_ALL, Tconv_4T_ALL, ind0_BCconv_ALL, ind0_BCconv_others_ALL):     #return node temperature vector T_4T (in Thermal node framework)        
    #================================================================explicit solver
        if self.status_Thermal_solver == 'Explicit':
            if ip.status_FormFactor == 'Pouch':
                #----------------------------------------Core BC；separator(added), calculate using different (stencil) equations: half El half Sep, refer to Notebook P55 
                T_copy=T3_4T_ALL.copy()
                T3_4T_ALL=np.nan*np.zeros([self.n_4T_ALL,1])              #initialization of T3_4T_ALL
                Stencil_4T_ALL=np.zeros([self.n_4T_ALL,6])         #initialization of Stencil_4T_ALL
                #======================================calculate nodes except Sep nodes (i.e. 1~84)
                Stencil_4T_ALL[self.ind0_jx1NaN_4T_ALL,0] += h_4T_ALL[self.ind0_jx1NaN_4T_ALL,0] * (Tconv_4T_ALL[self.ind0_jx1NaN_4T_ALL,0]-T1_4T_ALL[self.ind0_jx1NaN_4T_ALL,0])                                                                                          #fill in convection terms for x1 plane
                Stencil_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0] / self.Delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL] * (T1_4T_ALL[self.ind0_jx1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL],0]-T1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0])       #fill in conduction terms for x1 plane
        
                Stencil_4T_ALL[self.ind0_jx2NaN_4T_ALL,1] += h_4T_ALL[self.ind0_jx2NaN_4T_ALL,1] * (Tconv_4T_ALL[self.ind0_jx2NaN_4T_ALL,1]-T1_4T_ALL[self.ind0_jx2NaN_4T_ALL,0])                                                                                          #fill in convection terms for x2 plane
                Stencil_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,1] += self.Lamda_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,1] / self.Delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL] * (T1_4T_ALL[self.ind0_jx2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL],0]-T1_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,0])       #fill in conduction terms for x2 plane
        
                Stencil_4T_ALL[self.ind0_jy1NaN_4T_ALL,2] += h_4T_ALL[self.ind0_jy1NaN_4T_ALL,2] * (Tconv_4T_ALL[self.ind0_jy1NaN_4T_ALL,2]-T1_4T_ALL[self.ind0_jy1NaN_4T_ALL,0])                                                                                          #fill in convection terms for y1 plane
                Stencil_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,2] += self.Lamda_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,2] / self.Delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL] * (T1_4T_ALL[self.ind0_jy1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL],0]-T1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,0])       #fill in conduction terms for y1 plane
        
                Stencil_4T_ALL[self.ind0_jy2NaN_4T_ALL,3] += h_4T_ALL[self.ind0_jy2NaN_4T_ALL,3] * (Tconv_4T_ALL[self.ind0_jy2NaN_4T_ALL,3]-T1_4T_ALL[self.ind0_jy2NaN_4T_ALL,0])                                                                                          #fill in convection terms for y2 plane
                Stencil_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,3] += self.Lamda_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,3] / self.Delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL] * (T1_4T_ALL[self.ind0_jy2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL],0]-T1_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,0])       #fill in conduction terms for y2 plane
        
                Stencil_4T_ALL[self.ind0_jz1NaN_4T_ALL,4] += h_4T_ALL[self.ind0_jz1NaN_4T_ALL,4] * (Tconv_4T_ALL[self.ind0_jz1NaN_4T_ALL,4]-T1_4T_ALL[self.ind0_jz1NaN_4T_ALL,0])                                                                                          #fill in convection terms for z1 plane
                Stencil_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,4] += self.Lamda_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,4] / self.Delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL] * (T1_4T_ALL[self.ind0_jz1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL],0]-T1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,0])       #fill in conduction terms for z1 plane
        
                Stencil_4T_ALL[self.ind0_jz2NaN_4T_ALL,5] += h_4T_ALL[self.ind0_jz2NaN_4T_ALL,5] * (Tconv_4T_ALL[self.ind0_jz2NaN_4T_ALL,5]-T1_4T_ALL[self.ind0_jz2NaN_4T_ALL,0])                                                                                          #fill in convection terms for z1 plane
                Stencil_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,5] += self.Lamda_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,5] / self.Delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL] * (T1_4T_ALL[self.ind0_jz2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL],0]-T1_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,0])       #fill in conduction terms for z2 plane
                    
                StencilSum_4T_ALL=(Stencil_4T_ALL / self.delta_xyz_4T_ALL) .dot( np.ones([6,1])) *2*self.dt/self.RouXc_4T_ALL     #sum along axis-1 of Stencil_4T_ALL, whose shape is (87,6) before, and (87,1) after execution of this line
        
                T3_4T_ALL[:self.n_4T_ALL]=T1_4T_ALL[:self.n_4T_ALL] + StencilSum_4T_ALL[:self.n_4T_ALL] + self.q_4T_ALL*self.dt/self.RouXc_4T_ALL[:self.n_4T_ALL]
                T3_4T_ALL[ind0_BCtem_ALL]=T_copy[ind0_BCtem_ALL]                                                            #replace temperature-constrained nodes in T3_4T_ALL (i.e. NaN) with temperature BC stored before (i.e. T_copy)
        
            if ip.status_FormFactor == 'Cylindrical' or ip.status_FormFactor == 'Prismatic':     
                    
    #        #----------------------------------------Core BC: temperature is fixed
    #        if status_ThermalBC_Core=='BCFix':           #2 kinds of Domain                 
    #            T3_4T[ind0_Domain_CC_4T]=T1_4T[ind0_Domain_CC_4T]                                                                                                                                                                                                                                                                                                                                                          \
    #                           + Alpha_CC*dt/(Delta_x1_4T[ind0_Domain_CC_4T]) * ( (T1_4T[ind0_Domain_CC_4T_jx2]-T1_4T[ind0_Domain_CC_4T])/Delta_x2_4T[ind0_Domain_CC_4T]-(T1_4T[ind0_Domain_CC_4T]-T1_4T[ind0_Domain_CC_4T_jx1])/Delta_x1_4T[ind0_Domain_CC_4T] )                                                                                                                                                          \
    #                           + Alpha_CC*dt/(Delta_y1_4T[ind0_Domain_CC_4T]) * ( (T1_4T[ind0_Domain_CC_4T_jy2]-T1_4T[ind0_Domain_CC_4T])/Delta_y2_4T[ind0_Domain_CC_4T]-(T1_4T[ind0_Domain_CC_4T]-T1_4T[ind0_Domain_CC_4T_jy1])/Delta_y1_4T[ind0_Domain_CC_4T] )                                                                                                                                                          \
    #                           + Alpha_hybrid_El_AlCu_CC*dt/(Delta_z1_4T[ind0_Domain_CC_4T]) * ( (T1_4T[ind0_Domain_CC_4T_jz2]-T1_4T[ind0_Domain_CC_4T])/Delta_z2_4T[ind0_Domain_CC_4T]-(T1_4T[ind0_Domain_CC_4T]-T1_4T[ind0_Domain_CC_4T_jz1])/Delta_z1_4T[ind0_Domain_CC_4T] )                                                                                                                                           \
    #                           + q_4T[ind0_Domain_CC_4T]*dt/rou_CC/c_CC
    #                     
    #            T3_4T[ind0_Domain_El_4T]=T1_4T[ind0_Domain_El_4T]                                                                                                                                                                                                                                                                                                                                                          \
    #                           + Alpha_El_x*dt/(Delta_x1_4T[ind0_Domain_El_4T]) * ( (T1_4T[ind0_Domain_El_4T_jx2]-T1_4T[ind0_Domain_El_4T])/Delta_x2_4T[ind0_Domain_El_4T]-(T1_4T[ind0_Domain_El_4T]-T1_4T[ind0_Domain_El_4T_jx1])/Delta_x1_4T[ind0_Domain_El_4T] )                                                                                                                                                        \
    #                           + Alpha_El_y*dt/(Delta_y1_4T[ind0_Domain_El_4T]) * ( (T1_4T[ind0_Domain_El_4T_jy2]-T1_4T[ind0_Domain_El_4T])/Delta_y2_4T[ind0_Domain_El_4T]-(T1_4T[ind0_Domain_El_4T]-T1_4T[ind0_Domain_El_4T_jy1])/Delta_y1_4T[ind0_Domain_El_4T] )                                                                                                                                                        \
    #                           + Alpha_El_z*dt/(Delta_z1_4T[ind0_Domain_El_4T]) * ( (T1_4T[ind0_Domain_El_4T_jz2]-T1_4T[ind0_Domain_El_4T])/Delta_z2_4T[ind0_Domain_El_4T]-(T1_4T[ind0_Domain_El_4T]-T1_4T[ind0_Domain_El_4T_jz1])/Delta_z1_4T[ind0_Domain_El_4T] )                                                                                                                                                        \
    #                           + q_4T[ind0_Domain_El_4T]*dt/rouXc_El
            #----------------------------------------Core BC；separator(added), calculate using different (stencil) equations: half El half Sep, refer to Notebook P55
                if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':  
                    T_copy=T3_4T_ALL.copy()
                    T3_4T_ALL=np.nan*np.zeros([self.n_4T_ALL,1])          #initialization of T3_4T
                    Stencil_4T_ALL=np.zeros([self.n_4T_ALL,6])         #initialization of Stencil_4T
                    #======================================calculate nodes suitable for Regular stencil; e.g. nodes except Sep nodes (i.e. 1~84)
                    Stencil_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0] += h_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0] * (Tconv_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0]-T1_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0])                                                                              #fill in convection terms for x1 plane
                    Stencil_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0] / self.Delta_x1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL] * (T1_4T_ALL[self.jx1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0])      #fill in conduction terms for x1 plane
        
                    Stencil_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1] += h_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1] * (Tconv_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1]-T1_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,0])                                                                              #fill in convection terms for x2 plane
                    Stencil_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,1] += self.Lamda_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,1] / self.Delta_x2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL] * (T1_4T_ALL[self.jx2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,0])      #fill in conduction terms for x2 plane
        
                    Stencil_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2] += h_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2] * (Tconv_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2]-T1_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,0])                                                                              #fill in convection terms for y1 plane
                    Stencil_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,2] += self.Lamda_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,2] / self.Delta_y1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL] * (T1_4T_ALL[self.jy1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,0])      #fill in conduction terms for y1 plane
        
                    Stencil_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3] += h_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3] * (Tconv_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3]-T1_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,0])                                                                              #fill in convection terms for y2 plane
                    Stencil_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,3] += self.Lamda_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,3] / self.Delta_y2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL] * (T1_4T_ALL[self.jy2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,0])      #fill in conduction terms for y2 plane
        
                    Stencil_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4] += h_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4] * (Tconv_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4]-T1_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,0])                                                                              #fill in convection terms for z1 plane
                    Stencil_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,4] += self.Lamda_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,4] / self.Delta_z1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL] * (T1_4T_ALL[self.jz1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,0])      #fill in conduction terms for z1 plane
        
                    Stencil_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5] += h_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5] * (Tconv_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5]-T1_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,0])                                                                              #fill in convection terms for z1 plane
                    Stencil_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,5] += self.Lamda_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,5] / self.Delta_z2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL] * (T1_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,0]) * (self.delta_x1_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1]+self.delta_x2_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1])/(self.delta_x1_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL])      #fill in conduction terms for z2 plane
                    
                    StencilSum_4T_ALL=(Stencil_4T_ALL / self.delta_xyz_4T_ALL) .dot( np.ones([6,1])) *2*self.dt/self.RouXc_4T_ALL     #sum along axis-1 of Stencil_4T, whose shape is (87,6) before, and (87,1) after execution of this line
        
                    T3_4T_ALL[self.Reg_ind0_jxzNaN_4T_ALL]=T1_4T_ALL[self.Reg_ind0_jxzNaN_4T_ALL] + StencilSum_4T_ALL[self.Reg_ind0_jxzNaN_4T_ALL] + self.q_4T_ALL[self.Reg_ind0_jxzNaN_4T_ALL]*self.dt/self.RouXc_4T_ALL[self.Reg_ind0_jxzNaN_4T_ALL]
                    if self.status_Tab_ThermalPath=='Yes' and ip.status_FormFactor == 'Cylindrical':    #if thermal tabs are considered, add heat transfer by tab to e.g.node1 of Regular stencil, only in y direction
                        T3_4T_ALL[self.node_positive_0ind_4T,0] += self.Lamda_tab/self.RouXc_4T_ALL[self.node_positive_0ind_4T,0]*8*self.A_tab*self.dt/self.L_tab/(self.delta_xyz_4T_ALL[self.node_positive_0ind_4T,0]*self.delta_xyz_4T_ALL[self.node_positive_0ind_4T,2]*self.delta_xyz_4T_ALL[self.node_positive_0ind_4T,4]) * (T1_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0]-T1_4T_ALL[self.node_positive_0ind_4T,0])
                        T3_4T_ALL[self.node_negative_0ind_4T,0] += self.Lamda_tab/self.RouXc_4T_ALL[self.node_negative_0ind_4T,0]*8*self.A_tab*self.dt/self.L_tab/(self.delta_xyz_4T_ALL[self.node_negative_0ind_4T,0]*self.delta_xyz_4T_ALL[self.node_negative_0ind_4T,2]*self.delta_xyz_4T_ALL[self.node_negative_0ind_4T,4]) * (T1_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0]-T1_4T_ALL[self.node_negative_0ind_4T,0])
                    #======================================calculate nodes suitable for Irregular stencil; e.g. Sep nodes (i.e. 85~87)            
                    StencilSum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]=0        #initialize the Irregular stencil, e.g. the last 3 rows in StencilSum_4T_ALL
        
                    StencilSum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] += ( self.Alpha_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]*self.dt/self.S_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]/2* ( (self.delta_x1_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1]+self.delta_x2_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1])                                                                                       \
                                                                                * (T1_4T_ALL[:,0][self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1]-T1_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] )/self.Delta_z1_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1] ) )  .dot( np.ones([self.nx,1]) )            
                    
                    StencilSum_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0] += h_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]) * (Tconv_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]-T1_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0])
                    StencilSum_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]) * (T1_4T_ALL[self.jy1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0]) /self.Delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]
        
                    StencilSum_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0] += h_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]) * (Tconv_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]-T1_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0])
                    StencilSum_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]) * (T1_4T_ALL[self.jy2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0]) /self.Delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]
        
                    T3_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]=T1_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] + StencilSum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] + self.q_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]*self.dt/self.RouXc_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]
                    if self.status_Tab_ThermalPath=='Yes' and ip.status_FormFactor == 'Cylindrical':    #if thermal tabs are considered, add heat transfer by tab to e.g.node88 of Irregular stencil, only in y direction
                        T3_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0] += np.sum( self.Lamda_tab/self.RouXc_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0]*self.A_tab*2*self.dt/self.L_tab/self.S_Irreg_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0]/(self.delta_y1_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T)]+self.delta_y2_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T)]) *(T1_4T_ALL[self.node_positive_0ind_4T,0]-T1_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0]) )
                        T3_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0] += np.sum( self.Lamda_tab/self.RouXc_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0]*self.A_tab*2*self.dt/self.L_tab/self.S_Irreg_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0]/(self.delta_y1_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T)]+self.delta_y2_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T)]) *(T1_4T_ALL[self.node_negative_0ind_4T,0]-T1_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0]) )
                    #======================================apply temperature BC
                    T3_4T_ALL[ind0_BCtem_ALL]=T_copy[ind0_BCtem_ALL]                                                           #replace temperature-constrained nodes in T3_4T (i.e. NaN) with temperature BC stored before (i.e. T_copy)                     
    #================================================================implicit Crank-Nicolson solver            
        if self.status_Thermal_solver == 'CN':              
#            self.VectorCN=self.fun_VectorCN()
#            if self.status_TemBC_VectorCN_check == 'Yes':
#                self.fun_implicit_TemBC_VectorCN_check()  #for Temperature constrained BC (Dirichlet BC), i.g. node 36 has initial 30°C, node 40 is suddenly assigned with 20°C, λΔT/Δz could numerically cause a large number that the heat is extracted unreasonably large.  In order to avoid this, Vector_CN is used as an indicator
            
            if self.status_linsolver_T=='BandSparse':
                top=np.zeros([self.length_MatrixCN,self.length_MatrixCN]); bottom=np.zeros([self.length_MatrixCN,self.length_MatrixCN]); MatrixCN_expand=np.concatenate((top,self.MatrixCN,bottom))
                ab=MatrixCN_expand[self.ind0_r_expand,self.ind0_c_expand]
                T3_4T_ALL=scipy.linalg.solve_banded((self.ind0_l,self.ind0_u),ab,self.VectorCN)
            elif self.status_linsolver_T=='Sparse':
                T3_4T_ALL=scipy.sparse.linalg.spsolve(self.MatrixCN,self.VectorCN).reshape(-1,1)
            else:  #i.e. status_linsolver_T=='General'
                T3_4T_ALL=np.linalg.solve(self.MatrixCN,self.VectorCN)
        return T3_4T_ALL                                                                                                       
    #########################################################   
    ##### function for CN solver Tem BC stability check #####
    #########################################################
    def fun_implicit_TemBC_VectorCN_check(self):    #for Temperature constrained BC (Dirichlet BC), i.g. node 36 has initial 30°C, node 40 is suddenly assigned with 20°C, λΔT/Δz could numerically cause a large number that the heat is extracted unreasonably large.  In order to avoid this, Vector_CN is used as an indicator
        if ip.status_FormFactor == 'Pouch' or self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
            ind0_temp1=np.where(self.VectorCN[self.ind0_BCtem_others_ALL] >= self.TemCheck_ceil)[0]
            ind0_temp2=np.where(self.VectorCN[self.ind0_BCtem_others_ALL] <= self.TemCheck_floor)[0]
            ind0_illnode=np.concatenate((self.ind0_BCtem_others_ALL[ind0_temp1],self.ind0_BCtem_others_ALL[ind0_temp2]))
            if ind0_illnode.size !=0:
                print('\nrequired by implicit solver Dirichlet Temperature BC, ill ind0 node for VectorCN is:',ind0_illnode )
                temp1=input('press y to continue, n to break:')
                if temp1 != 'y':
                    raise Exception('exit')
    #########################################################   
    ##### function for explicit solver stability check ######
    #########################################################
    def fun_explicit_stability_check(self):        #note that for dynamic BC, especially dynamic convection h, stability should be checked stability, as stability expression is relevant with h
        Stability_4T_ALL=np.zeros([self.n_4T_ALL,6])         #initialization of Stencil_4T
        if ip.status_FormFactor == 'Pouch':
            #--------------------------------------calculate nodes except Sep nodes (i.e. 1~63)    
            Stability_4T_ALL[self.ind0_jx1NaN_4T_ALL,0] += self.h_4T_ALL[self.ind0_jx1NaN_4T_ALL,0]                                                                                          #similar to fun_Thermal() explicit solver case
            Stability_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0] / self.Delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]
        
            Stability_4T_ALL[self.ind0_jx2NaN_4T_ALL,1] += self.h_4T_ALL[self.ind0_jx2NaN_4T_ALL,1]
            Stability_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,1] += self.Lamda_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,1] / self.Delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]
        
            Stability_4T_ALL[self.ind0_jy1NaN_4T_ALL,2] += self.h_4T_ALL[self.ind0_jy1NaN_4T_ALL,2]
            Stability_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,2] += self.Lamda_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,2] / self.Delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]
        
            Stability_4T_ALL[self.ind0_jy2NaN_4T_ALL,3] += self.h_4T_ALL[self.ind0_jy2NaN_4T_ALL,3]
            Stability_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,3] += self.Lamda_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,3] / self.Delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]
        
            Stability_4T_ALL[self.ind0_jz1NaN_4T_ALL,4] += self.h_4T_ALL[self.ind0_jz1NaN_4T_ALL,4]
            Stability_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,4] += self.Lamda_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,4] / self.Delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]
        
            Stability_4T_ALL[self.ind0_jz2NaN_4T_ALL,5] += self.h_4T_ALL[self.ind0_jz2NaN_4T_ALL,5]
            Stability_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,5] += self.Lamda_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,5] / self.Delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]
                
            StabilitySum_4T_ALL=( (Stability_4T_ALL / self.delta_xyz_4T) .dot( np.ones([6,1])) *2/self.RouXc_4T )**(-1)                      #sum along axis-1 of Stencil_4T, whose shape is (63,6) before, and (63,1) after execution of this line
        
        if ip.status_FormFactor == 'Cylindrical' or ip.status_FormFactor == 'Prismatic':
            if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
                
                #--------------------------------------calculate nodes suitable for Regular stencil; e.g. nodes except Sep nodes (i.e. 1~84)    
                Stability_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0] += self.h_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0]                                                                                          #similar to fun_Thermal() explicit solver case
                Stability_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0] / self.Delta_x1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL]
        
                Stability_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1] += self.h_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1]
                Stability_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,1] += self.Lamda_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,1] / self.Delta_x2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL]
        
                Stability_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2] += self.h_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2]
                Stability_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,2] += self.Lamda_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,2] / self.Delta_y1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL]
        
                Stability_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3] += self.h_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3]
                Stability_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,3] += self.Lamda_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,3] / self.Delta_y2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL]
        
                Stability_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4] += self.h_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4]
                Stability_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,4] += self.Lamda_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,4] / self.Delta_z1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL]
        
                Stability_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5] += self.h_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5]
                Stability_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,5] += self.Lamda_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,5] / self.Delta_z2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL] * (self.delta_x1_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1]+self.delta_x2_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1])/(self.delta_x1_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL])
                
                StabilitySum_4T_ALL=( (Stability_4T_ALL / self.delta_xyz_4T_ALL) .dot( np.ones([6,1])) *2/self.RouXc_4T_ALL )**(-1)                      #sum along axis-1 of Stencil_4T, whose shape is (87,6) before, and (87,1) after execution of this line
                
                if ip.status_FormFactor == 'Cylindrical' and self.status_Tab_ThermalPath=='Yes':    #if thermal tabs are considered, add heat transfer by tab to e.g.node1 of Regular stencil, only in y direction
                    StabilitySum_4T_ALL[self.node_positive_0ind_4T,0]=(   (StabilitySum_4T_ALL[self.node_positive_0ind_4T,0])**(-1) + self.Lamda_tab/self.RouXc_4T_ALL[self.node_positive_0ind_4T,0]*8*self.A_tab/self.L_tab/(self.delta_xyz_4T_ALL[self.node_positive_0ind_4T,0]*self.delta_xyz_4T_ALL[self.node_positive_0ind_4T,2]*self.delta_xyz_4T_ALL[self.node_positive_0ind_4T,4])   )**(-1)
                    StabilitySum_4T_ALL[self.node_negative_0ind_4T,0]=(   (StabilitySum_4T_ALL[self.node_negative_0ind_4T,0])**(-1) + self.Lamda_tab/self.RouXc_4T_ALL[self.node_negative_0ind_4T,0]*8*self.A_tab/self.L_tab/(self.delta_xyz_4T_ALL[self.node_negative_0ind_4T,0]*self.delta_xyz_4T_ALL[self.node_negative_0ind_4T,2]*self.delta_xyz_4T_ALL[self.node_negative_0ind_4T,4])   )**(-1)
                #--------------------------------------calculate nodes suitable for Irregular stencil; e.g. Sep nodes (i.e. 85~87)
                StabilitySum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]=0        #initialize the Irregular stencil, e.g. the last 3 rows in StabilitySum_4T_ALL
        
                StabilitySum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] += ( self.Alpha_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]/self.S_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]/2* ( self.delta_x1_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1]+self.delta_x2_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1]                                                                                            \
                                                                            / self.Delta_z1_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1] ) )  .dot( np.ones([self.nx,1]) )            
                    
                StabilitySum_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0] += self.h_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0]*2/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL])
                StabilitySum_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0]*2/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]) /self.Delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]
        
                StabilitySum_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0] += self.h_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0]*2/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL])
                StabilitySum_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0]*2/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]) /self.Delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]
                
                if ip.status_FormFactor == 'Cylindrical' and self.status_Tab_ThermalPath=='Yes':    #if thermal tabs are considered, add heat transfer by tab to e.g.node88 of Irregular stencil, only in y direction
                    StabilitySum_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0] += self.Lamda_tab/self.RouXc_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0]*self.A_tab*2/self.L_tab/self.S_Irreg_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T),0]/(self.delta_y1_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T)]+self.delta_y2_4T_ALL[self.ind0_Geo_Can_node1_4T +(self.ntotal_4T+self.nAddCore_4T)])
                    StabilitySum_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0] += self.Lamda_tab/self.RouXc_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0]*self.A_tab*2/self.L_tab/self.S_Irreg_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T),0]/(self.delta_y1_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T)]+self.delta_y2_4T_ALL[self.ind0_Geo_Can_node46_4T +(self.ntotal_4T+self.nAddCore_4T)])
        ind0temp=np.argwhere(np.isnan(StabilitySum_4T_ALL[:,0])).reshape(-1);   StabilitySum_4T_ALL[ind0temp]=inf                                #replace NaN with inf (i.e.1e20)
        dt_min_suggest=np.min(StabilitySum_4T_ALL)
        
        ind0_bottleneck=np.where(StabilitySum_4T_ALL==dt_min_suggest)[0]
        
        print('\nrequired by explicit solver stablization and based on the initial BC, dt should be smaller than: %f s\nlimited by node_4T 0index:'%(dt_min_suggest), ind0_bottleneck )
        print('dt right now is %f s'%(self.dt))
        temp1=input('press y to continue, n to break:')
        if temp1 != 'y':
            raise Exception('exit')
                               
    #########################################################   
    ###########       function for MatrixCN       ###########
    #########################################################
    
    #########################################################   
    # function for band matrix dimension and diagonal form  #
    #########################################################
    def fun_band_matrix_precompute(self, MatrixCN):                                                                                           #i.g. A=np.array([[1,2,0,0,0],[2,3,4,0,0],[3,1,5,0,0],[0,7,5,8,0],[0,0,0,0,1]]), length=5
        #---------------get band matrix dimension: ind0_l and ind0_u---------------
        length=MatrixCN.shape[0]
        counter_l=-length+1; counter_u=length-1
        for i0 in np.linspace(-length+1,-1,length-1,dtype=int):                                                                         #i.g. for A, i0: -4,-3,-2,-1,0
            if (np.diagonal(MatrixCN,i0)==np.zeros([length+i0])).all():
                counter_l=counter_l+1
            else:
                break
        for i0 in np.linspace(length-1,1,length-1,dtype=int):                                                                           #i.g. for A, i0: 4,3,2,1,0
            if (np.diagonal(MatrixCN,i0)==np.zeros([length-i0])).all():
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
    ###########       function for VectorCN       ###########
    #########################################################
    def fun_VectorCN_preTp(self):          #VectorCN = VectorCN_preTp*Tp + VectorCN_conv_q;    VectorCN_preTp is very similar to MatrixCN, so form VectorCN based on MatrixCN
        VectorCN_preTp=self.MatrixCN.copy()
        VectorCN_preTp[np.arange(self.n_4T_ALL),np.arange(self.n_4T_ALL)] -= 1
        VectorCN_preTp = -VectorCN_preTp
        VectorCN_preTp[np.arange(self.n_4T_ALL),np.arange(self.n_4T_ALL)] += 1
        return VectorCN_preTp
    #########################################################   
    ###########       function for VectorCN       ###########
    #########################################################

    #########################################################   
    ###########         function for plot         ###########
    #########################################################
    def fun_plot(self,step, status_climit_vector):
        if ip.status_FormFactor == 'Pouch':
            n_col_layers=self.nx*self.nstack
            vec_col_focus_Al=np.arange(self.nx)+self.nx*(self.status_layer_focus_Al-1)
            vec_col_focus_Cu=np.arange(self.nx)+self.nx*(self.status_layer_focus_Cu-1)
            vec_col_focus_Elb=np.arange(self.nx)+self.nx*(self.status_layer_focus_Elb-1)            
            ind0_Al_4T=self.Al_4T.reshape(-1,n_col_layers)[:,vec_col_focus_Al]
            ind0_Cu_4T=self.Cu_4T.reshape(-1,n_col_layers)[:,vec_col_focus_Cu]
            ind0_Elb_4T=self.Elb_4T.reshape(-1,n_col_layers)[:,vec_col_focus_Elb]
            if self.nstack > 1:
                vec_col_focus_Elr=np.arange(self.nx)+self.nx*(self.status_layer_focus_Elr-1)
                ind0_Elr_4T=self.Elr_4T.reshape(-1,self.nx*(self.nstack-1))[:,vec_col_focus_Elr]
        #====================================================Fig. 1,2 temperature plotting for Current Collectors and Electrodes   
            array_h_Al_4T=self.xi_4T[ind0_Al_4T];   array_v_Al_4T=self.yi_4T[ind0_Al_4T];   array_c_T_Al_4T=self.T_record[:,step][ind0_Al_4T]-273.15
            array_h_Cu_4T=self.xi_4T[ind0_Cu_4T];   array_v_Cu_4T=self.yi_4T[ind0_Cu_4T];   array_c_T_Cu_4T=self.T_record[:,step][ind0_Cu_4T]-273.15
            array_h_Elb_4T=self.xi_4T[ind0_Elb_4T]; array_v_Elb_4T=self.yi_4T[ind0_Elb_4T]; array_c_T_Elb_4T=self.T_record[:,step][ind0_Elb_4T]-273.15            
            if self.nstack > 1:
                array_h_Elr_4T=self.xi_4T[ind0_Elr_4T]; array_v_Elr_4T=self.yi_4T[ind0_Elr_4T]; array_c_T_Elr_4T=self.T_record[:,step][ind0_Elr_4T]-273.15
            #------------------------plot temperature T for Current Collectors    
            self.ax1=self.axs_fig1[0]    
            self.ax1.set_title('Al')                                                                                 #add title
            climit_vector=status_climit_vector
            self.ax1.contourf(array_h_Al_4T/array_h_Al_4T.max()*self.Lx_electrodes_real,array_v_Al_4T/array_v_Al_4T.max()*self.Ly_electrodes_real,array_c_T_Al_4T,climit_vector,cmap="RdBu_r",extend='both')   #plot contour
            #self.ax1.contourf(array_h_Al_4T,array_v_Al_4T,array_c_T_Al_4T,climit_vector,cmap="RdBu_r",extend='both')   #plot contour
            #fig.colorbar(surf1)                                                                                #add colorbar
            #    ax1.scatter(array_h_Sep,array_v_Sep,facecolors='w',edgecolors='k')
        if ip.status_FormFactor == 'Cylindrical' or ip.status_FormFactor == 'Prismatic':
            if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
                n_v=self.ny; n_h_Al=int(np.size(self.Al_4T)/n_v); n_h_Cu=n_h_Al; n_h_Elb=int(np.size(self.Elb_4T)/n_v); n_h_Elr=int(np.size(self.Elr_4T)/n_v)
                ind0_Al_4T=self.Al_4T.reshape(n_v,n_h_Al)
                ind0_Cu_4T=self.Cu_4T.reshape(n_v,n_h_Cu)
                ind0_Elb_4T=self.Elb_4T.reshape(n_v,n_h_Elb)
                if self.nstack > 1:
                    ind0_Elr_4T=self.Elr_4T.reshape(n_v,n_h_Elr)      
        #====================================================Fig. 1,2 temperature plotting for Current Collectors and Electrodes       
                #------------------------plot temperature T for Current Collectors
                array_h_Al_4T=self.xi_4T[ind0_Al_4T];   array_v_Al_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Al_4T]);   array_c_T_Al_4T=self.T_record[:,step][ind0_Al_4T]-273.15
                array_h_Cu_4T=self.xi_4T[ind0_Cu_4T];   array_v_Cu_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Cu_4T]);   array_c_T_Cu_4T=self.T_record[:,step][ind0_Cu_4T]-273.15
                array_h_Elb_4T=self.xi_4T[ind0_Elb_4T]; array_v_Elb_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Elb_4T]); array_c_T_Elb_4T=self.T_record[:,step][ind0_Elb_4T]-273.15                
                if self.nstack > 1:
                    array_h_Elr_4T=self.xi_4T[ind0_Elr_4T]; array_v_Elr_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Elr_4T]); array_c_T_Elr_4T=self.T_record[:,step][ind0_Elr_4T]-273.15                
                array_h_Sep_4T=(array_h_Al_4T[:,0]-self.b01).reshape(-1,1); array_v_Sep=array_v_Al_4T[:,0].reshape(-1,1)
                array_h_SepAl_4T=np.append(array_h_Sep_4T,array_h_Al_4T,axis=1); array_v_SepAl_4T=np.append(array_v_Sep,array_v_Al_4T,axis=1); array_c_SepAl_4T=np.append(self.T_record[:,step].reshape(-1,1)[self.ind0_Geo_core_AddSep_4T_4SepFill]-273.15,array_c_T_Al_4T,axis=1)         
                self.ax1=self.axs_fig1[0]    
                self.ax1.set_title('Al')
                if ip.status_FormFactor == 'Cylindrical':
                    self.ax1.contourf(array_h_SepAl_4T *(self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_SepAl_4T,array_c_SepAl_4T,status_climit_vector,cmap="RdBu_r")                                                                                 #add title                
                elif ip.status_FormFactor == "Prismatic":
                    self.ax1.contourf(array_h_SepAl_4T *(self.SpiralandStripe_Sep_s_real),array_v_SepAl_4T,array_c_SepAl_4T,status_climit_vector,cmap="RdBu_r")                                                                                 #add title                
                else:
                    self.ax1.contourf(array_h_SepAl_4T,array_v_SepAl_4T,array_c_SepAl_4T, status_climit_vector,cmap="RdBu_r",extend='both')   #plot contour                
                #fig.colorbar(surf1)                                                                                #add colorbar   
                self.ax1.scatter(array_h_Sep_4T,array_v_Sep,facecolors='w',edgecolors='k') 
        if ip.status_FormFactor == 'Pouch' or self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':    
            self.ax2=self.axs_fig1[1]
            self.ax2.set_title('Cu')                                                                                 #add title
            
            #surf2=self.ax2.contourf(array_h_Cu_4T,array_v_Cu_4T,array_c_T_Cu_4T,self.status_climit_vector,cmap="RdBu_r",extend='both')      #plot contour
            climit_vector=status_climit_vector
            if ip.status_FormFactor == 'Cylindrical':
                surf2=self.ax2.contourf(array_h_Cu_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Cu_4T,array_c_T_Cu_4T,status_climit_vector,cmap="RdBu_r")      #plot contour
            elif ip.status_FormFactor == "Prismatic":
                surf2=self.ax2.contourf(array_h_Cu_4T * (self.SpiralandStripe_Sep_s_real),array_v_Cu_4T,array_c_T_Cu_4T,status_climit_vector,cmap="RdBu_r")      #plot contour
            else:
                surf2=self.ax2.contourf(array_h_Cu_4T/array_h_Cu_4T.max()*self.Lx_electrodes_real,array_v_Cu_4T/array_v_Cu_4T.max()*self.Ly_electrodes_real,array_c_T_Cu_4T,climit_vector,cmap="RdBu_r",extend='both')      #plot contour
            
            surf2.cmap.set_under('cyan'); surf2.cmap.set_over('yellow')
            cb_ax = self.fig1.add_axes([0.83, 0.1, 0.02, 0.8])                                                       #add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
            self.fig1.colorbar(surf2, cax=cb_ax)                                                                     #add colorbar    
            self.fig1.subplots_adjust(right=0.8,hspace=0.3)
            if ip.status_FormFactor == 'Cylindrical':
                self.ax1.scatter(array_h_Sep_4T,array_v_Sep,facecolors='w',edgecolors='k')            
            
            if self.status_PlotNode=='Yes':
                self.ax1.scatter(array_h_Al_4T,array_v_Al_4T,c='k')
                self.ax2.scatter(array_h_Cu_4T,array_v_Cu_4T,c='k')
            if self.plot_type=='instant':    #for instant plotting (before postprocessor)                                                            
                self.fig1.suptitle('T [°C], step=%i(%.4f s), Current Collector'%(step,step*self.dt))  
                if self.status_PopFig_or_SaveGIF_instant =='Fig':
                    plt.pause(0.01)                                                                             #if no pause, plt.show and plt.clf would not let see the plot                                                             
                if self.status_PopFig_or_SaveGIF_instant =='GIF':
                    im=mplfig_to_npimage(self.fig1)    #convert matplot figure to numpy image
                    self.frames1.append(im)
            else: #plot_type=='replay'   for replay plotting (within postprocessor)
                #self.fig1.suptitle('replay, T [°C], step=%i(%.4f s), Current Collector'%(step,step*self.dt))
                self.fig1.suptitle('T [°C] of Current Collectors at half of cell thickness, t=%.1f s'%(step*self.dt))
                if self.status_PopFig_or_SaveGIF_replay =='Fig':
                    plt.pause(0.01)                                                                             #if no pause, plt.show and plt.clf would not let see the plot                                                             
                if self.status_PopFig_or_SaveGIF_replay =='GIF':
                    im=mplfig_to_npimage(self.fig1)    #convert matplot figure to numpy image
                    self.frames1.append(im)
            #------------------------plot temperature T for Electrodes
            if self.nstack > 1:
                cmin=np.min(np.append(array_c_T_Elb_4T,array_c_T_Elr_4T)); cmax=np.max(np.append(array_c_T_Elb_4T,array_c_T_Elr_4T)); climit_vector=np.linspace(cmin,cmax,self.status_levels) 
            else:
                cmin=np.min(array_c_T_Elb_4T); cmax=np.max(array_c_T_Elb_4T); climit_vector=np.linspace(cmin,cmax,self.status_levels)                 

            self.ax1=self.axs_fig2[0]    
            #self.ax1.set_title('Elb')
            if ip.status_FormFactor == 'Cylindrical':
                self.ax1.set_title('Long layer')                                                                                #add title       
                self.ax1.contourf(array_h_Elb_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Elb_4T,array_c_T_Elb_4T,climit_vector,cmap="RdBu_r")         #plot contour
            else:
                self.ax1.set_title('Cathode')                                                                                #add title       
                #self.ax1.contourf(array_h_Elb_4T,array_v_Elb_4T,array_c_T_Elb_4T,self.status_climit_vector,cmap="RdBu_r",extend='both')         #plot contour
                if ip.status_FormFactor == "Prismatic":
                    self.ax1.contourf(array_h_Elb_4T * (self.SpiralandStripe_Sep_s_real),array_v_Elb_4T,array_c_T_Elb_4T,climit_vector,cmap="RdBu_r")         #plot contour
                else:
                    self.ax1.contourf(array_h_Elb_4T/array_h_Elb_4T.max()*self.Lx_electrodes_real,array_v_Elb_4T/array_v_Elb_4T.max()*self.Ly_electrodes_real,array_c_T_Elb_4T,status_climit_vector,cmap="RdBu_r",extend='both')         #plot contour
            #fig.colorbar(surf1)                                                                                #add colorbar                                                                         
            if self.nstack > 1:
                self.ax2=self.axs_fig2[1]
                if ip.status_FormFactor == 'Cylindrical':
                    self.ax2.set_title('Short layer')                                                                                #add title
                    surf2=self.ax2.contourf(array_h_Elr_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Elr_4T,array_c_T_Elr_4T,climit_vector,cmap="RdBu_r")   #plot contour
                else:
                #self.ax2.set_title('Elr')                                                                                #add title
                    self.ax2.set_title('Anode')
                    #surf2=self.ax2.contourf(array_h_Elr_4T,array_v_Elr_4T,array_c_T_Elr_4T,self.status_climit_vector,cmap="RdBu_r",extend='both')   #plot contour
                    if ip.status_FormFactor == "Prismatic":
                        surf2=self.ax2.contourf(array_h_Elr_4T * (self.SpiralandStripe_Sep_s_real),array_v_Elr_4T,array_c_T_Elr_4T,climit_vector,cmap="RdBu_r")   #plot contour
                    else:
                        surf2=self.ax2.contourf(array_h_Elr_4T/array_h_Elr_4T.max()*self.Lx_electrodes_real,array_v_Elr_4T/array_v_Elr_4T.max()*self.Ly_electrodes_real,array_c_T_Elr_4T,status_climit_vector,cmap="RdBu_r",extend='both')   #plot contour
                surf2.cmap.set_under('cyan'); surf2.cmap.set_over('yellow')
                cb_ax = self.fig2.add_axes([0.83, 0.1, 0.02, 0.8])                                                       #add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
                self.fig2.colorbar(surf2, cax=cb_ax)                                                                     #add colorbar    
                self.fig2.subplots_adjust(right=0.8,hspace=0.3)
            if self.status_PlotNode=='Yes':
                self.ax1.scatter(array_h_Elb_4T,array_v_Elb_4T,c='k')
                if self.nstack > 1:
                    self.ax2.scatter(array_h_Elr_4T,array_v_Elr_4T,c='k')
            if self.plot_type=='instant':    #for instant plotting (before postprocessor)
                #self.fig2.suptitle('replay, T [°C], step=%i(%.4f s), Electrode'%(step,step*self.dt))
                self.fig2.suptitle('T [°C], step=%i(%.4f s), Electrode'%(step,step*self.dt))
                if self.status_PopFig_or_SaveGIF_instant =='Fig':
                    plt.pause(0.01)  
                if self.status_PopFig_or_SaveGIF_instant =='GIF':                                                    #if no pause, plt.show and plt.clf would not let see the plot                                                             
                    im=mplfig_to_npimage(self.fig2)                                                                  #convert matplot figure to numpy image
                    self.frames2.append(im) 
            else: #plot_type=='replay'   for replay plotting (within postprocessor)
                #self.fig2.suptitle('replay, T [°C], step=%i(%.4f s), Electrode'%(step,step*self.dt))
                self.fig2.suptitle('T [°C] of Electrodes at half of cell thickness, t=%.1f s'%(step*self.dt))
                if self.status_PopFig_or_SaveGIF_replay =='Fig':
                    plt.pause(0.01)  
                if self.status_PopFig_or_SaveGIF_replay =='GIF':                                                     #if no pause, plt.show and plt.clf would not let see the plot                                                             
                    im=mplfig_to_npimage(self.fig2)                                                                  #convert matplot figure to numpy image
                    self.frames2.append(im)
    #====================================================Fig. 3,4 heat gen plotting for Current Collectors and Electrodes       
            #------------------------plot heat gen q for Current Collectors
            if self.status_Model=='EandT':
                
                if ip.status_FormFactor == 'Pouch':
                    array_h_Al_4T=self.xi_4T[ind0_Al_4T];      array_v_Al_4T=self.yi_4T[ind0_Al_4T];      array_c_q_Al_4T=self.q_4T_record[:,step][ind0_Al_4T]
                    array_h_Cu_4T=self.xi_4T[ind0_Cu_4T];      array_v_Cu_4T=self.yi_4T[ind0_Cu_4T];      array_c_q_Cu_4T=self.q_4T_record[:,step][ind0_Cu_4T]
                    array_h_Elb_4T=self.xi_4T[ind0_Elb_4T];    array_v_Elb_4T=self.yi_4T[ind0_Elb_4T];    array_c_q_Elb_4T=self.q_4T_record[:,step][ind0_Elb_4T]
                    if self.nstack > 1:
                        array_h_Elr_4T=self.xi_4T[ind0_Elr_4T];    array_v_Elr_4T=self.yi_4T[ind0_Elr_4T];    array_c_q_Elr_4T=self.q_4T_record[:,step][ind0_Elr_4T]
                      
                
                if ip.status_FormFactor == 'Cylindrical' or ip.status_FormFactor == 'Prismatic':
                    array_h_Al_4T=self.xi_4T[ind0_Al_4T];   array_v_Al_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Al_4T]);   array_c_q_Al_4T=self.q_4T_record[:,step][ind0_Al_4T]
                    array_h_Cu_4T=self.xi_4T[ind0_Cu_4T];   array_v_Cu_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Cu_4T]);   array_c_q_Cu_4T=self.q_4T_record[:,step][ind0_Cu_4T]
                    array_h_Elb_4T=self.xi_4T[ind0_Elb_4T]; array_v_Elb_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Elb_4T]); array_c_q_Elb_4T=self.q_4T_record[:,step][ind0_Elb_4T]
                    if self.nstack > 1:
                        array_h_Elr_4T=self.xi_4T[ind0_Elr_4T]; array_v_Elr_4T=(self.LG_Jellyroll-self.yi_4T[ind0_Elr_4T]); array_c_q_Elr_4T=self.q_4T_record[:,step][ind0_Elr_4T]
                    array_h_Sep_4T=(array_h_Al_4T[:,0]-self.b01).reshape(-1,1); array_v_Sep=array_v_Al_4T[:,0].reshape(-1,1)
                
                
                self.ax1=self.axs_fig3[0]    
                self.ax1.set_title('Al')                                                                                 #add title
                #surf1=self.ax1.contourf(array_h_Al_4T,array_v_Al_4T,array_c_q_Al_4T,self.status_levels,cmap="Reds")   #plot contour        
                if ip.status_FormFactor == 'Cylindrical':
                    surf1=self.ax1.contourf(array_h_Al_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Al_4T,array_c_q_Al_4T,self.status_levels,cmap="Reds")           #plot contour
                elif ip.status_FormFactor == "Prismatic":
                    surf1=self.ax1.contourf(array_h_Al_4T * (self.SpiralandStripe_Sep_s_real),array_v_Al_4T,array_c_q_Al_4T,self.status_levels,cmap="Reds")           #plot contour        
                else:
                    surf1=self.ax1.contourf(array_h_Al_4T/array_h_Al_4T.max()*self.Lx_electrodes_real,array_v_Al_4T/array_v_Al_4T.max()*self.Ly_electrodes_real,array_c_q_Al_4T,self.status_levels,cmap="Reds")   #plot contour
                #fig.colorbar(surf1)                                                                                #add colorbar                                                                         
                self.ax2=self.axs_fig3[1]
                self.ax2.set_title('Cu')                                                                                 #add title
                #surf2=self.ax2.contourf(array_h_Cu_4T,array_v_Cu_4T,array_c_q_Cu_4T,self.status_levels,cmap="Reds")      #plot contour
                if ip.status_FormFactor == 'Cylindrical':
                    surf2=self.ax2.contourf(array_h_Cu_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Cu_4T,array_c_q_Cu_4T,self.status_levels,cmap="Reds")           #plot contour
                elif ip.status_FormFactor == "Prismatic":
                    surf2=self.ax2.contourf(array_h_Cu_4T * (self.SpiralandStripe_Sep_s_real),array_v_Cu_4T,array_c_q_Cu_4T,self.status_levels,cmap="Reds")           #plot contour
                else:
                    surf2=self.ax2.contourf(array_h_Cu_4T/array_h_Cu_4T.max()*self.Lx_electrodes_real,array_v_Cu_4T/array_v_Cu_4T.max()*self.Ly_electrodes_real,array_c_q_Cu_4T,self.status_levels,cmap="Reds")      #plot contour
                if ip.status_FormFactor == 'Pouch':
                    #cb_ax1 = self.fig3.add_axes([0.83, 0.1, 0.02, 0.8])                                                     #add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
                    cb_ax2 = self.fig3.add_axes([0.83, 0.1, 0.02, 0.8])                                                     #add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
                    #    ax1.scatter(array_h_Sep,array_v_Sep,facecolors='w',edgecolors='k')
                if ip.status_FormFactor == 'Prismatic':
                    #cb_ax1 = self.fig3.add_axes([0.83, 0.57, 0.02, 0.3])                                                     #add an axes, lower left corner in [0.83, 0.57] measured in figure coordinate with axes width 0.02 and height 0.3
                    cb_ax2 = self.fig3.add_axes([0.83, 0.13, 0.02, 0.3])                                                     #add an axes, lower left corner in [0.83, 0.13] measured in figure coordinate with axes width 0.02 and height 0.3
                    self.ax1.scatter(array_h_Sep_4T,array_v_Sep,facecolors='w',edgecolors='k')
                if ip.status_FormFactor == 'Cylindrical':
                    cb_ax1 = self.fig3.add_axes([0.83, 0.57, 0.02, 0.3])                                                     #add an axes, lower left corner in [0.83, 0.57] measured in figure coordinate with axes width 0.02 and height 0.3
                    cb_ax2 = self.fig3.add_axes([0.83, 0.13, 0.02, 0.3])                                                     #add an axes, lower left corner in [0.83, 0.13] measured in figure coordinate with axes width 0.02 and height 0.3
                    self.fig3.colorbar(surf1,cax=cb_ax1)                                                                     #add colorbar 
                    self.ax1.scatter(array_h_Sep_4T,array_v_Sep,facecolors='w',edgecolors='k')
                
                #self.fig3.colorbar(surf1,cax=cb_ax1)                                                                     #add colorbar 
                self.fig3.colorbar(surf2,cax=cb_ax2)                                                                     #add colorbar 
                self.fig3.subplots_adjust(right=0.8,hspace=0.3)
                if self.status_PlotNode=='Yes':
                    self.ax1.scatter(array_h_Al_4T,array_v_Al_4T,c='k')
                    self.ax2.scatter(array_h_Cu_4T,array_v_Cu_4T,c='k')
                if self.plot_type=='instant':    #for instant plotting (before postprocessor)                                                            
                    #self.fig3.suptitle('q [J/s/m3], step=%i(%.4f s), Current Collector'%(step,step*self.dt))  
                    self.fig3.suptitle('Net Heat Flux [J/s/m3] of Current Collectors at half of cell thickness, t=%.1f s'%(step*self.dt))  
                    if self.status_PopFig_or_SaveGIF_instant =='Fig':
                        plt.pause(0.01)                                                                             #if no pause, plt.show and plt.clf would not let see the plot                                                             
                    if self.status_PopFig_or_SaveGIF_instant =='GIF':
                        im=mplfig_to_npimage(self.fig3)    #convert matplot figure to numpy image
                        self.frames3.append(im)
                else: #plot_type=='replay'   for replay plotting (within postprocessor)
                    #self.fig3.suptitle('replay, q [J/s/m3], step=%i(%.4f s), Current Collector'%(step,step*self.dt))
                    self.fig3.suptitle('Net Heat Flux [J/s/m3] of Current Collectors at half of cell thickness, t=%.1f s'%(step*self.dt))
                    if self.status_PopFig_or_SaveGIF_replay =='Fig':
                        plt.pause(0.01)                                                                             #if no pause, plt.show and plt.clf would not let see the plot                                                             
                    if self.status_PopFig_or_SaveGIF_replay =='GIF':
                        im=mplfig_to_npimage(self.fig3)    #convert matplot figure to numpy image
                        self.frames3.append(im)
            #------------------------plot heat gen q for Electrodes
                self.ax1=self.axs_fig4[0]    
                #self.ax1.set_title('Elb')                                                                                #add title           
                if ip.status_FormFactor == 'Cylindrical':
                    self.ax1.set_title('Long layer') 
                else:
                    self.ax1.set_title('Cathode')
                if self.nstack > 1:
                    cmin=np.min(np.append(array_c_q_Elb_4T,array_c_q_Elr_4T)); cmax=np.max(np.append(array_c_q_Elb_4T,array_c_q_Elr_4T)); climit_vector=np.linspace(cmin,cmax,self.status_levels) 
                else:
                    cmin=np.min(array_c_q_Elb_4T); cmax=np.max(array_c_q_Elb_4T); climit_vector=np.linspace(cmin,cmax,self.status_levels)                 
                #self.ax1.contourf(array_h_Elb_4T,array_v_Elb_4T,array_c_q_Elb_4T,climit_vector,cmap="Reds")         #plot contour
                if ip.status_FormFactor == 'Cylindrical':
                    self.ax1.contourf(array_h_Elb_4T,array_v_Elb_4T,array_c_q_Elb_4T,climit_vector,cmap="Reds")              #plot contour
                elif ip.status_FormFactor == "Prismatic":
                    self.ax1.contourf(array_h_Elb_4T,array_v_Elb_4T,array_c_q_Elb_4T,climit_vector,cmap="Reds")              #plot contour
                else:
                    self.ax1.contourf(array_h_Elb_4T/array_h_Elb_4T.max()*self.Lx_electrodes_real,array_v_Elb_4T/array_v_Elb_4T.max()*self.Ly_electrodes_real,array_c_q_Elb_4T,climit_vector,cmap="Reds")         #plot contour
                
                #fig.colorbar(surf1)                                                                                #add colorbar                                                                         
                if self.nstack > 1:
                    self.ax2=self.axs_fig4[1]
                    #self.ax2.set_title('Elr') 
                    if ip.status_FormFactor == 'Cylindrical':
                        self.ax2.set_title('Short layer')                                                                                #add title
                        surf2=self.ax2.contourf(array_h_Elr_4T,array_v_Elr_4T,array_c_q_Elr_4T,climit_vector,cmap="Reds")        #plot contour
                    else:                                                         #add title
                        if ip.status_FormFactor == "Prismatic":
                            surf2=self.ax2.contourf(array_h_Elr_4T,array_v_Elr_4T,array_c_q_Elr_4T,climit_vector,cmap="Reds")        #plot contour
                        else:
                            self.ax2.set_title('Anode')
                            #surf2=self.ax2.contourf(array_h_Elr_4T,array_v_Elr_4T,array_c_q_Elr_4T,climit_vector,cmap="Reds")   #plot contour
                            surf2=self.ax2.contourf(array_h_Elr_4T/array_h_Elr_4T.max()*self.Lx_electrodes_real,array_v_Elr_4T/array_v_Elr_4T.max()*self.Ly_electrodes_real,array_c_q_Elr_4T,climit_vector,cmap="Reds")   #plot contour
                    cb_ax = self.fig4.add_axes([0.83, 0.1, 0.02, 0.8])                                                       #add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
                    self.fig4.colorbar(surf2, cax=cb_ax)                                                                     #add colorbar    
                    self.fig4.subplots_adjust(right=0.8,hspace=0.3)
                if self.status_PlotNode=='Yes':
                    self.ax1.scatter(array_h_Elb_4T,array_v_Elb_4T,c='k')
                    if self.nstack > 1:
                        self.ax2.scatter(array_h_Elr_4T,array_v_Elr_4T,c='k')
                if self.plot_type=='instant':    #for instant plotting (before postprocessor)
                    self.fig4.suptitle('replay, q [J/s/m3], step=%i(%.4f s), Electrode'%(step,step*self.dt))
                    if self.status_PopFig_or_SaveGIF_instant =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_instant =='GIF':                                                    #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig4)                                                                  #convert matplot figure to numpy image
                        self.frames4.append(im) 
                else: #plot_type=='replay'   for replay plotting (within postprocessor)
                    #self.fig4.suptitle('replay, q [J/s/m3], step=%i(%.4f s), Electrode'%(step,step*self.dt))
                    self.fig4.suptitle('Net Heat Flux [J/s/m3] of Electrodes at half of cell thickness, t=%.1f s'%(step*self.dt))
                    if self.status_PopFig_or_SaveGIF_replay =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_replay =='GIF':                                                     #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig4)                                                                  #convert matplot figure to numpy image
                        self.frames4.append(im)    
    #====================================================Fig. 5,6 current density and SoC plotting for Electrodes
            if self.status_Model=='EandT'or self.status_Model=='E':
                ind0_ele_Elb_4T=self.List_node2ele_4T[ind0_Elb_4T,0]
                if self.nstack > 1:
                    ind0_ele_Elr_4T=self.List_node2ele_4T[ind0_Elr_4T,0]    
            #------------------------plot current density rouI for Electrodes
                array_c_rouI_Elb_4T=self.I_ele_record[:,step][ind0_ele_Elb_4T]/(self.Axy_ele[ind0_ele_Elb_4T,0]*self.scalefactor_z)
                if self.nstack > 1:
                    array_c_rouI_Elr_4T=self.I_ele_record[:,step][ind0_ele_Elr_4T]/(self.Axy_ele[ind0_ele_Elr_4T,0]*self.scalefactor_z)
                self.ax1=self.axs_fig5[0]    
                #self.ax1.set_title('Elb')
                if ip.status_FormFactor == 'Cylindrical':
                    self.ax1.set_title('Long layer')
                else:
                    self.ax1.set_title('Cathode')                                                                                #add title
                if self.nstack > 1:
                    cmin=np.min(np.append(array_c_rouI_Elb_4T,array_c_rouI_Elr_4T)); cmax=np.max(np.append(array_c_rouI_Elb_4T,array_c_rouI_Elr_4T)); climit_vector=np.linspace(cmin,cmax,self.status_levels) 
                else:
                    cmin=np.min(array_c_rouI_Elb_4T); cmax=np.max(array_c_rouI_Elb_4T); climit_vector=np.linspace(cmin,cmax,self.status_levels)                 
                if abs(cmax-cmin) <= 1e-10:
                    climit_vector=self.status_levels
                #self.ax1.contourf(array_h_Elb_4T,array_v_Elb_4T,array_c_rouI_Elb_4T,climit_vector,cmap="Reds")           #plot contour
                if ip.status_FormFactor == 'Cylindrical':
                    self.ax1.contourf(array_h_Elb_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Elb_4T,array_c_rouI_Elb_4T,climit_vector,cmap="Reds")           #plot contour
                elif ip.status_FormFactor == "Prismatic":
                    self.ax1.contourf(array_h_Elb_4T * (self.SpiralandStripe_Sep_s_real),array_v_Elb_4T,array_c_rouI_Elb_4T,climit_vector,cmap="Reds")           #plot contour
                else:
                    self.ax1.contourf(array_h_Elb_4T/array_h_Elb_4T.max()*self.Lx_electrodes_real,array_v_Elb_4T/array_v_Elb_4T.max()*self.Ly_electrodes_real,array_c_rouI_Elb_4T,climit_vector,cmap="Reds")                         #plot contour
                #fig.colorbar(surf1)                                                                                #add colorbar                                                                         
                if self.nstack > 1:
                    self.ax2=self.axs_fig5[1]
                    #self.ax2.set_title('Elr')                                                                                #add title
                    if ip.status_FormFactor == 'Cylindrical':
                        self.ax2.set_title('Short layer')                                                                                #add title
                        surf2=self.ax2.contourf(array_h_Elr_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Elr_4T,array_c_rouI_Elr_4T,climit_vector,cmap="Reds")     #plot contour
                    else:
                        self.ax2.set_title('Anode')
                        #surf2=self.ax2.contourf(array_h_Elr_4T,array_v_Elr_4T,array_c_rouI_Elr_4T,climit_vector,cmap="Reds")     #plot contour
                        if ip.status_FormFactor == "Prismatic":
                            surf2=self.ax2.contourf(array_h_Elr_4T * (self.SpiralandStripe_Sep_s_real),array_v_Elr_4T,array_c_rouI_Elr_4T,climit_vector,cmap="Reds")     #plot contour
                        else:
                            surf2=self.ax2.contourf(array_h_Elr_4T/array_h_Elr_4T.max()*self.Lx_electrodes_real,array_v_Elr_4T/array_v_Elr_4T.max()*self.Ly_electrodes_real,array_c_rouI_Elr_4T,climit_vector,cmap="Reds")                   #plot contour
                    cb_ax = self.fig5.add_axes([0.83, 0.1, 0.02, 0.8])                                                       #add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
                    self.fig5.colorbar(surf2, cax=cb_ax)                                                                     #add colorbar    
                    self.fig5.subplots_adjust(right=0.8,hspace=0.3)
                if self.status_PlotNode=='Yes':
                    self.ax1.scatter(array_h_Elb_4T,array_v_Elb_4T,c='k')
                    if self.nstack > 1:
                        self.ax2.scatter(array_h_Elr_4T,array_v_Elr_4T,c='k')
                if self.plot_type=='instant':    #for instant plotting (before postprocessor)
                    self.fig5.suptitle('I density [A/m2], step=%i(%.4f s), Electrode'%(step,step*self.dt))  
                    
                    if self.status_PopFig_or_SaveGIF_instant =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_instant =='GIF':                                                    #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig5)                                                                  #convert matplot figure to numpy image
                        self.frames5.append(im) 
                else: #plot_type=='replay'   for replay plotting (within postprocessor)
                    #self.fig5.suptitle('replay, I density [A/m2], step=%i(%.4f s), Electrode'%(step,step*self.dt))
                    self.fig5.suptitle('I density [A/m2] of Electrodes at half of cell thickness, t=%.1f s'%(step*self.dt))
                    if self.status_PopFig_or_SaveGIF_replay =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_replay =='GIF':                                                     #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig5)                                                                  #convert matplot figure to numpy image
                        self.frames5.append(im)    
            #------------------------plot SoC for Electrodes
               # array_c_SoC_Elb_4T=self.SoC_ele_record[:,step][ind0_ele_Elb_4T]
                if ip.status_FormFactor == 'Cylindrical':
                    array_c_SoC_Elb_4T=self.SoC_ele_record[:,step][ind0_ele_Elb_4T]
                    if self.nstack > 1:
                        array_c_SoC_Elr_4T=self.SoC_ele_record[:,step][ind0_ele_Elr_4T]
                    self.ax1=self.axs_fig6[0]    
                    self.ax1.set_title('Long layer')                                                                                    #add title
                else:
                    array_c_SoC_Elb_4T=self.SoC_ele_record[:,step][ind0_ele_Elb_4T]/(self.Axy_ele[ind0_ele_Elb_4T,0]*self.scalefactor_z)
                    if self.nstack > 1:
                        #array_c_SoC_Elr_4T=self.SoC_ele_record[:,step][ind0_ele_Elr_4T]
                        array_c_SoC_Elr_4T=self.SoC_ele_record[:,step][ind0_ele_Elr_4T]/(self.Axy_ele[ind0_ele_Elr_4T,0]*self.scalefactor_z)
                    self.ax1=self.axs_fig6[0]    
                    #self.ax1.set_title('Elb')                                                                                    #add title
                    self.ax1.set_title('Cathode')
                if self.nstack > 1:
                    cmin=np.min(np.append(array_c_SoC_Elb_4T,array_c_SoC_Elr_4T)); cmax=np.max(np.append(array_c_SoC_Elb_4T,array_c_SoC_Elr_4T)); climit_vector=np.linspace(cmin,cmax,self.status_levels) 
                else:
                    cmin=np.min(array_c_SoC_Elb_4T); cmax=np.max(array_c_SoC_Elb_4T); climit_vector=np.linspace(cmin,cmax,self.status_levels)                 
                if abs(cmax-cmin) <= 1e-10:
                    climit_vector=self.status_levels
                if ip.status_FormFactor == 'Cylindrical':
                    self.ax1.contourf(array_h_Elb_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Elb_4T,array_c_SoC_Elb_4T,climit_vector,cmap="Greens")              #plot contour
                elif ip.status_FormFactor == "Prismatic":
                    self.ax1.contourf(array_h_Elb_4T * (self.SpiralandStripe_Sep_s_real),array_v_Elb_4T,array_c_SoC_Elb_4T,climit_vector,cmap="Greens")              #plot contour
                else:
                    #self.ax1.contourf(array_h_Elb_4T,array_v_Elb_4T,array_c_SoC_Elb_4T,climit_vector,cmap="Greens")                         #plot contour
                    self.ax1.contourf(array_h_Elb_4T/array_h_Elb_4T.max()*self.Lx_electrodes_real,array_v_Elb_4T/array_v_Elb_4T.max()*self.Ly_electrodes_real,array_c_SoC_Elb_4T,climit_vector,cmap="Greens")                         #plot contour
                #fig.colorbar(surf1)                                                                                #add colorbar                                                                         
                if self.nstack > 1:
                    self.ax2=self.axs_fig6[1]
                    if ip.status_FormFactor == 'Cylindrical':
                        self.ax2.set_title('Short layer')                                                                                #add title
                        surf2=self.ax2.contourf(array_h_Elr_4T * (self.Spiral_Sep_s_real/self.Spiral_Sep_s),array_v_Elr_4T,array_c_SoC_Elr_4T,climit_vector,cmap="Greens")    #plot contour
                    else:
                        #self.ax2.set_title('Elr')                                                                                #add title
                        self.ax2.set_title('Anode')
                        #surf2=self.ax2.contourf(array_h_Elr_4T,array_v_Elr_4T,array_c_SoC_Elr_4T,climit_vector,cmap="Greens")                   #plot contour
                        if ip.status_FormFactor == "Prismatic":
                            surf2=self.ax2.contourf(array_h_Elr_4T * (self.SpiralandStripe_Sep_s_real),array_v_Elr_4T,array_c_SoC_Elr_4T,climit_vector,cmap="Greens")    #plot contour
                        else:
                            surf2=self.ax2.contourf(array_h_Elr_4T/array_h_Elr_4T.max()*self.Lx_electrodes_real,array_v_Elr_4T/array_v_Elr_4T.max()*self.Ly_electrodes_real,array_c_SoC_Elr_4T,climit_vector,cmap="Greens")                   #plot contour
                    cb_ax = self.fig6.add_axes([0.83, 0.1, 0.02, 0.8])                                                       #add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
                    self.fig6.colorbar(surf2, cax=cb_ax)                                                                     #add colorbar    
                    self.fig6.subplots_adjust(right=0.8,hspace=0.3)
                if self.status_PlotNode=='Yes':
                    self.ax1.scatter(array_h_Elb_4T,array_v_Elb_4T,c='k')
                    if self.nstack > 1:
                        self.ax2.scatter(array_h_Elr_4T,array_v_Elr_4T,c='k')
                if self.plot_type=='instant':    #for instant plotting (before postprocessor)
                    self.fig6.suptitle('SoC, step=%i(%.4f s), Electrode'%(step,step*self.dt))  
                    if self.status_PopFig_or_SaveGIF_instant =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_instant =='GIF':                                                    #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig6)                                                                  #convert matplot figure to numpy image
                        self.frames6.append(im) 
                else: #plot_type=='replay'   for replay plotting (within postprocessor)
                   # self.fig6.suptitle('replay, SoC, step=%i(%.4f s), Electrode'%(step,step*self.dt))
                    self.fig6.suptitle('SoC of Electrodes at half of cell thickness, t=%.1f s'%(step*self.dt))
                    if self.status_PopFig_or_SaveGIF_replay =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_replay =='GIF':                                                     #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig6)                                                                  #convert matplot figure to numpy image
                        self.frames6.append(im)    
    #====================================================Fig. 7 voltage potential for Current Collectors
                array_c_V_Al_4T=self.V_record[:,step][self.List_node2node_T2E[ind0_Al_4T,0]]
                array_c_V_Cu_4T=self.V_record[:,step][self.List_node2node_T2E[ind0_Cu_4T,0]]
                #------------------------plot voltage potential V for Current Collectors
                self.ax1=self.axs_fig7[0]    
                self.ax1.set_title('Al')                                                                                 #add title
                #surf1=self.ax1.contourf(array_h_Al_4T,array_v_Al_4T,array_c_V_Al_4T,self.status_levels,cmap="Purples")        #plot contour
                if ip.status_FormFactor == 'Cylindrical':
                    surf1=self.ax1.contourf(array_h_Al_4T,array_v_Al_4T,array_c_V_Al_4T,self.status_levels,cmap="Purples")        #plot contour
                elif ip.status_FormFactor == "Prismatic":
                    surf1=self.ax1.contourf(array_h_Al_4T,array_v_Al_4T,array_c_V_Al_4T,self.status_levels,cmap="Purples")        #plot contour
                else:
                    surf1=self.ax1.contourf(array_h_Al_4T/array_h_Al_4T.max()*self.Lx_electrodes_real,array_v_Al_4T/array_v_Al_4T.max()*self.Ly_electrodes_real,array_c_V_Al_4T,self.status_levels,cmap="Purples")        #plot contour
                #fig.colorbar(surf1)                                                                                #add colorbar                                                                         
                self.ax2=self.axs_fig7[1]
                self.ax2.set_title('Cu')                                                                                 #add title
                #surf2=self.ax2.contourf(array_h_Cu_4T,array_v_Cu_4T,array_c_V_Cu_4T,self.status_levels,cmap="Purples")        #plot contour
                if ip.status_FormFactor == 'Cylindrical':
                    surf2=self.ax2.contourf(array_h_Cu_4T,array_v_Cu_4T,array_c_V_Cu_4T,self.status_levels,cmap="Purples")        #plot contour
                elif ip.status_FormFactor == "Prismatic":
                    surf2=self.ax2.contourf(array_h_Cu_4T,array_v_Cu_4T,array_c_V_Cu_4T,self.status_levels,cmap="Purples")        #plot contour
                else:
                    surf2=self.ax2.contourf(array_h_Cu_4T/array_h_Cu_4T.max()*self.Lx_electrodes_real,array_v_Cu_4T/array_v_Cu_4T.max()*self.Ly_electrodes_real,array_c_V_Cu_4T,self.status_levels,cmap="Purples")        #plot contour
                cb_ax1 = self.fig7.add_axes([0.83, 0.57, 0.02, 0.3])                                                     #add an axes, lower left corner in [0.83, 0.57] measured in figure coordinate with axes width 0.02 and height 0.3
                cb_ax2 = self.fig7.add_axes([0.83, 0.13, 0.02, 0.3])                                                     #add an axes, lower left corner in [0.83, 0.13] measured in figure coordinate with axes width 0.02 and height 0.3
                self.fig7.colorbar(surf1,cax=cb_ax1)                                                                     #add colorbar 
                self.fig7.colorbar(surf2,cax=cb_ax2)                                                                     #add colorbar 
                self.fig7.subplots_adjust(right=0.8,hspace=0.3)
                if ip.status_FormFactor == 'Cylindrical' or ip.status_FormFactor == 'Prismatic':
                    self.ax1.scatter(array_h_Sep_4T,array_v_Sep,facecolors='w',edgecolors='k')
                if self.status_PlotNode=='Yes':
                    self.ax1.scatter(array_h_Al_4T,array_v_Al_4T,c='k')
                    self.ax2.scatter(array_h_Cu_4T,array_v_Cu_4T,c='k')
                if self.plot_type=='instant':    #for instant plotting (before postprocessor)                                                            
                    #self.fig7.suptitle('Voltage [V], step=%i(%.4f s), Current Collector'%(step,step*self.dt))  
                    self.fig7.suptitle('Voltage, step=%i(%.4f s) [V], Current Collector'%(step,step*self.dt))  
                    if self.status_PopFig_or_SaveGIF_instant =='Fig':
                        plt.pause(0.01)                                                                             #if no pause, plt.show and plt.clf would not let see the plot                                                             
                    if self.status_PopFig_or_SaveGIF_instant =='GIF':
                        im=mplfig_to_npimage(self.fig7)    #convert matplot figure to numpy image
                        self.frames7.append(im)
                else: #plot_type=='replay'   for replay plotting (within postprocessor)
                    #self.fig7.suptitle('replay, Voltage [V], step=%i(%.4f s), Current Collector'%(step,step*self.dt))
                    self.fig7.suptitle('V of Current Collectors at half of cell thickness, t=%.1f s'%(step*self.dt))
                    if self.status_PopFig_or_SaveGIF_replay =='Fig':
                        plt.pause(0.01)                                                                             #if no pause, plt.show and plt.clf would not let see the plot                                                             
                    if self.status_PopFig_or_SaveGIF_replay =='GIF':
                        im=mplfig_to_npimage(self.fig7)    #convert matplot figure to numpy image
                        self.frames7.append(im)
            if self.status_Model=='EandT'or self.status_Model=='T':
    #====================================================Fig. 8,9 cooling rate, T distribution SD for Electrodes
                vector_t=self.t_record
            #------------------------plot cooling rate for Electrodes         
                self.ax1=self.axs_fig8        
                vector_y1_Elb=self.T_avg_record-273.15
                #self.ax1.plot(vector_t,vector_y1_Elb,'bo') 
                # ax1.plot(vector_t,vector_y1_Elb,'k-')
                self.ax1.plot(self.Charge_Throughput_As[:step+1]/3600,vector_y1_Elb[:step+1],'k-')
                self.ax1.set_xlabel('Charge throughput [Ah]')
                self.ax1.set_ylabel('Volume-average temperture [°C]')
                ax2 = self.ax1.twinx()
                ax2.plot(self.Charge_Throughput_As[:step+1]/3600,self.T_Delta_record[:step+1],'b-')
                ax2.set_ylabel('Tmax-Tmin [°C]',color='b')
                ax2.tick_params(axis='y', labelcolor='b',color='b')
                ax2.spines['right'].set_color('b')
                ax2.set_ylim([-1,21])
                
                if self.plot_type=='instant':    #for instant plotting (before postprocessor)
                    #self.fig8.suptitle('T_avg, step=%i(%.4f s), Electrode'%(step,step*self.dt)) 
                    self.fig8.suptitle('T_avg [°C] of cell with time [s], step=%i(%.4f s), Electrode'%(step,step*self.dt)) 
                    if self.status_PopFig_or_SaveGIF_instant =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_instant =='GIF':                                                    #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig8)                                                                  #convert matplot figure to numpy image
                        self.frames8.append(im) 
                else: #plot_type=='replay'   for replay plotting (within postprocessor)
                    #self.fig8.suptitle('replay, T_avg, step=%i(%.4f s), Electrode'%(step,step*self.dt))
                    self.fig8.suptitle('T_avg [°C] of cell with time [s]')
                    if self.status_PopFig_or_SaveGIF_replay =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_replay =='GIF':                                                     #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig8)                                                                  #convert matplot figure to numpy image
                        self.frames8.append(im)
            #------------------------plot T distribution for Electrodes
                self.ax1=self.axs_fig9        
                vector_y2_Elb=self.T_SD_record
                self.ax1.plot(vector_t,vector_y2_Elb,'b-')    
                if self.plot_type=='instant':    #for instant plotting (before postprocessor)
                    #self.fig9.suptitle('T_SD, step=%i(%.4f s), Electrode'%(step,step*self.dt)) 
                    self.fig9.suptitle('T_SD [°C] of cell with time [s]') 
                    if self.status_PopFig_or_SaveGIF_instant =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_instant =='GIF':                                                    #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig9)                                                                  #convert matplot figure to numpy image
                        self.frames9.append(im) 
                else: #plot_type=='replay'   for replay plotting (within postprocessor)
                    #self.fig9.suptitle('replay, T_SD, step=%i(%.4f s), Electrode'%(step,step*self.dt))
                    self.fig9.suptitle('T_SD [°C] of cell with time [s]')
                    if self.status_PopFig_or_SaveGIF_replay =='Fig':
                        plt.pause(0.01)  
                    if self.status_PopFig_or_SaveGIF_replay =='GIF':                                                     #if no pause, plt.show and plt.clf would not let see the plot                                                             
                        im=mplfig_to_npimage(self.fig9)                                                                  #convert matplot figure to numpy image
                        self.frames9.append(im)
    
    #########################################################   
    #########     function for Postprocessor     ############
    #########################################################
    def fun_Postprocessor(self,step,t_begin,t_end,status_climit_vector): 
        #------------------------------------------------------------------------------------------------------------------------------postprocessor
        self.plot_type= ip.status_plot_type_postprocess                           #For fun_plot(), differentiating instant plotting and replay plotting
        if ip.status_PopFig_or_SaveGIF_replay=='Fig' or ip.status_PopFig_or_SaveGIF_replay=='GIF' and ip.status_fig1to9 == 'Yes':
            self.fig1, self.axs_fig1=plt.subplots(nrows=2, ncols=1)
            self.fig2, self.axs_fig2=plt.subplots(nrows=2, ncols=1)  
            if ip.status_Model=='EandT':
                self.fig3, self.axs_fig3=plt.subplots(nrows=2, ncols=1)
                self.fig4, self.axs_fig4=plt.subplots(nrows=2, ncols=1)
            if ip.status_Model=='EandT'or ip.status_Model=='E':
                self.fig5, self.axs_fig5=plt.subplots(nrows=2, ncols=1)
                self.fig6, self.axs_fig6=plt.subplots(nrows=2, ncols=1)
                self.fig7, self.axs_fig7=plt.subplots(nrows=2, ncols=1)
            if ip.status_Model=='EandT'or ip.status_Model=='T':
                self.fig8, self.axs_fig8=plt.subplots()
                self.fig9, self.axs_fig9=plt.subplots()
            self.frames1=[]; self.frames2=[]; self.frames3=[]; self.frames4=[]; self.frames5=[]; self.frames6=[]; self.frames7=[]; self.frames8=[]; self.frames9=[]
            for step_plot in np.arange(1,step+1):   #step: 1,2...nt   loop for time    
                if step_plot%ip.status_GIFdownsample_num==0:                #downsample frames of GIF
                    self.fun_plot(step, status_climit_vector); 
                if step_plot != step:
                    print('\r    generating step=%d out of %d steps'%(step_plot, step),end='')      #by default end is '\n'. To print the outcome in the same line, 1. remove '\n'; 2. '\r' to put the output always in front
                else:
                    print('\r    generating step=%d out of %d steps'%(step_plot, step),end='')      #by default end is '\n'. To print the outcome in the same line, 1. remove '\n'; 2. '\r' to put the output always in front
                    print('\n')
        if ip.status_PopFig_or_SaveGIF_replay == 'GIF' and ip.status_fig1to9 == 'Yes':
            imageio.mimsave('11.png', self.frames1, 'GIF', duration=0.2)      #save frames into a GIF
            imageio.mimsave('12.png', self.frames2, 'GIF', duration=0.2)
            if ip.status_Model=='EandT':
                imageio.mimsave('13.png', self.frames3, 'GIF', duration=0.2)      #save frames into a GIF
                imageio.mimsave('14.png', self.frames4, 'GIF', duration=0.2)
            if ip.status_Model=='EandT'or ip.status_Model=='E':
                imageio.mimsave('15.png', self.frames5, 'GIF', duration=0.2)
                imageio.mimsave('16.png', self.frames6, 'GIF', duration=0.2)
                imageio.mimsave('17.png', self.frames7, 'GIF', duration=0.2)
            if ip.status_Model=='EandT'or ip.status_Model=='T':
                imageio.mimsave('18.png', self.frames8, 'GIF', duration=0.2)
                imageio.mimsave('19.png', self.frames9, 'GIF', duration=0.2)        
        if ip.status_Model=='EandT'or ip.status_Model=='E':    
            plt.figure(11)
            plt.subplot(2,1,1)
            plt.plot(self.Charge_Throughput_As/3600,self.U_pndiff_plot,'r-')    
            plt.ylabel('Terminal voltage [V]') 
            plt.subplot(2,1,2)
            plt.plot(self.Charge_Throughput_As/3600,self.I0_record,'b-')
            plt.xlabel('Charge throughput [Ah]'); plt.ylabel('Terminal current [A]') 
            if self.status_Echeck=='Yes':        
                plt.figure()
                plt.plot(self.t_record,self.Egen_Total_record,'y')
                plt.plot(self.t_record,self.Eext_Total_BCconv_record,'g')
                plt.plot(self.t_record,self.Eint_Delta_record,'k')
                plt.plot(self.t_record,self.Ebalance_record,'r')
                plt.xlabel('Time [s]');     plt.ylabel('Model energy [J]'); plt.title('Model energy conservation check')    
                plt.legend(('Egen_Total','Eext_Total_BCconv','Eint_Delta','E_balance'))        
        if self.status_PlotLUT=='Yes':
            self.fun_LUTplot()    
        print('postprocessor done\n')
        print('all done :)\n')            
        print('Notes:\n')  
        print('total time elapsed: %f (s) or %.2f (min) or %.2f (h), finishing time: %s\n' %(t_end-t_begin, (t_end-t_begin)/60, (t_end-t_begin)/3600, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()) ))
        print('Electrodes plate area (in Equivalent Circuit Network model):\n%.4f (m2) in the current model (A_electrodes_eff)\n%.4f (m2) if no lumping conducted (A_electrodes_real))\ndifference in percentage is %.2f %% \n'%(self.A_electrodes_eff,self.A_electrodes_real,(self.A_electrodes_eff-self.A_electrodes_real)/self.A_electrodes_real*100 ))      #see ppt1 p276 
        # if ip.status_FormFactor == 'Pouch' or ip.status_FormFactor == 'Cylindrical':
        #     print('Energy density for the model with casing is %.2f Wh/kg\n'%(self.EnergyDensity/3600))    
        if self.scalefactor_z!=1 and self.status_CC=='Yes':
            print('now salefactor_z≠1 and status_CC=Yes\nthere might be error in heat gen and related consequences\nwhen lumping is used and current collector resistance is considered\n')
        if self.status_Model=='EandT' or self.status_Model=='T':
            print('convective h should be fixed for now if any conv BC\n')  
        if hasattr(self,'Table_I_ext'):    
            print('current from external file is used as input. nt is %d\n'%self.nt)    
        if self.status_Echeck=='Yes':
            print('Heat conservation is checked currently only for convection BC, and not for T constrained BC.\n')           
        plt.show()
    #########################################################   
    #########     function for Visualization     ############
    #########################################################
  
         
    
    
    
    
    
    
    
    
