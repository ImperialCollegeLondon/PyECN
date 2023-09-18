# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mayavi import mlab
import scipy.sparse.linalg
import scipy.sparse
from moviepy.video.io.bindings import mplfig_to_npimage
import imageio
import time
import os, sys    #for rerun the script for PID FOPDT calibration

import pyecn.parse_inputs as ip

inf=1e10

class Ribbon:
    def __init__(self, params_update):
#        self.__dict__ = ip.__dict__.copy()       #copy the default inputs into this 'self'
#        self.__dict__.update(params_update)      #update the unique inputs for this 'self'        
        #---------copy attr in inputs.py into this class---------
        my_shelf = {}
        for key in dir(ip):
            if not key.startswith("__"):         #filter out internal attributes like __builtins__ etc
                my_shelf[key] = ip.__dict__[key]
        self.__dict__ = my_shelf.copy()
        self.__dict__.update(params_update)      #update the update(unique) inputs for this 'self'        
        #--------------------------------------------------------        
        ####################################
        ###CREATE class - self ATTRIBUTES###
        ####################################
        self.rouXc_ribbon=self.rou_ribbon*self.c_ribbon

        self.S_cooling_nx = params_update['S_cooling_nx']
        S_cooling_ny = params_update['S_cooling_ny']
        S_cooling_cell_spacing = params_update['S_cooling_cell_spacing']
        S_cooling_ribbon_addlength = params_update['S_cooling_ribbon_addlength']
        dx = params_update['dx_ribbon']
        dy = params_update['dy_ribbon']
        self.nx_spacing = round(S_cooling_cell_spacing/dx); self.nx_addlength =round(S_cooling_ribbon_addlength/dx)
        self.nx = 1 + (self.nx_spacing+1) * self.status_Cells_num_Module + self.nx_addlength
        self.ny = S_cooling_ny
        self.nz = self.nz_ribbon
        self.Lx_ribbon = self.nx * dx
        self.Ly_ribbon = self.ny * dy
        self.Lz_ribbon = params_update['Lz_ribbon']
        self.Lx = self.Lx_ribbon-self.Lx_ribbon/self.nx 
        self.Ly = self.Ly_ribbon-self.Ly_ribbon/self.ny
        self.Lz = self.Lz_ribbon-self.Lz_ribbon/self.nz
        self.delta_ribbon = self.Lz_ribbon/self.nz
        self.ntotal_4T = self.nx*self.ny*self.nz

        (
        self.node_4T, self.xn_4T, self.yn_4T, self.zn_4T, self.mat_4T, self.xi_4T, self.yi_4T, self.zi_4T, self.V_4T,
        self.jx1_4T, self.jx2_4T, self.jy1_4T, self.jy2_4T, self.jz1_4T, self.jz2_4T, self.ind0_jx1_4T, self.ind0_jx2_4T, self.ind0_jy1_4T, self.ind0_jy2_4T, self.ind0_jz1_4T, self.ind0_jz2_4T,
        self.ind0_jx1NaN_4T, self.ind0_jx2NaN_4T, self.ind0_jy1NaN_4T, self.ind0_jy2NaN_4T, self.ind0_jz1NaN_4T, self.ind0_jz2NaN_4T, self.ind0_jx1NonNaN_4T, self.ind0_jx2NonNaN_4T, self.ind0_jy1NonNaN_4T, self.ind0_jy2NonNaN_4T, self.ind0_jz1NonNaN_4T, self.ind0_jz2NonNaN_4T  
        ) = self.fun_matrix1()                             
        

            
        #--------------------------------------------------------------------------preparation for modification on MatrixC and VectorI (preprocessor)
        print('\nrunning preprocessor...\n')
        self.n_4T_ALL=self.ntotal_4T
        self.node_4T_ALL=self.node_4T
        self.rou_c_V_weights=self.rou_ribbon*self.c_ribbon*self.V_4T             #prep for fun_weighted_avg_and_std
        self.fun_get_Geo_4T()
        self.fun_pre_Thermal()
        if self.status_TemBC_smoothening=='Yes':
            self.T_cooling_smoothened=self.T_cooling + (self.T_initial-self.T_cooling)/np.exp(self.smoothening_stiffness * 0)
        self.Tini_4T_ALL=self.fun_IC_4T()                                                  #apply Thermal initial condition
        if self.status_Thermal_solver=='CN':
            self.MatrixCN=self.fun_MatrixCN()
            if self.status_linsolver_T=='BandSparse':
                [self.length_MatrixCN, self.ind0_l, self.ind0_u, self.ind0_r_expand, self.ind0_c_expand]=self.fun_band_matrix_precompute(self.MatrixCN)    #for BandSparse linear equations solver, diagonal ordered form is needed
            self.VectorCN_preTp=self.fun_VectorCN_preTp()

        self.t_record=self.dt*np.arange(self.nt+1) 
        self.T_record=np.nan*np.zeros([self.ntotal_4T,self.nt+1]); self.T_record[:,0]=self.T_initial                                                                #record time(contain the t=0 point) and node temperature for postprocessor
        self.T_avg_record=np.nan*np.zeros([self.nt+1]); self.T_SD_record=np.nan*np.zeros([self.nt+1])                                                         #record node temperature average and SD for postprocessor
    #########################################################   
    ################## function for matrix1 #################
    #########################################################
    def fun_matrix1(self):
        node=np.arange(1, self.ntotal_4T+1)
        xn_unit=np.linspace(1,self.nx,self.nx,dtype=int)     #repeating node number for xn
        yn_unit=np.linspace(1,self.ny,self.ny,dtype=int)     #repeating node number for yn
        zn_unit=np.linspace(1,self.nz,self.nz,dtype=int)     #repeating node number for zn
        xn=np.tile(xn_unit,self.ny*self.nz)
        yn=np.repeat(yn_unit,self.nx*self.nz) 
        zn=np.repeat(zn_unit,self.nx); zn=np.tile(zn,self.ny)
          
        mat=np.ones([self.ntotal_4T])
        
        xi=np.zeros(self.ntotal_4T); yi=np.zeros(self.ntotal_4T); zi=np.zeros(self.ntotal_4T)
        xi=(xn-1)*self.Lx/(self.nx-1)
        yi=(yn-1)*self.Ly/(self.ny-1)    
        zi=(zn-1)*self.Lz/(self.nz-1)    
    
        jx1=np.zeros(self.ntotal_4T,dtype=int)            #initialize left-neighbor node number in x direction
        jx2=np.zeros(self.ntotal_4T,dtype=int)            #initialize right-neighbor node number in x direction
        jy1=np.zeros(self.ntotal_4T,dtype=int)            #initialize up-neighbor node number in y direction
        jy2=np.zeros(self.ntotal_4T,dtype=int)            #initialize down-neighbor node number in y direction
        jz1=np.zeros(self.ntotal_4T,dtype=int)            #initialize inner-neighbor node number in z direction
        jz2=np.zeros(self.ntotal_4T,dtype=int)            #initialize outer-neighbor node number in z direction
        
        #---------------------------- neighbor node number ------------------------------
        for i in (node-1):
            if xn[i]==1:                                  #for leftmost nodes(xn=1), no left neighbor
                jx1[i]=np.array([-9999])        
            else:
                jx1[i]=node[i]-1                          #for node[i], left-neighbor number jx1[i] is node[i]-1
    
        for i in (node-1):
            if xn[i]==self.nx:                                 #for rightmost nodes (xn=nx), no right neighbor
                jx2[i]=np.array([-9999])           
            else:
                jx2[i]=node[i]+1                          #for node[i], right-neighbor number jx2[i] is node[i]+1
    
        for i in (node-1):
            if yn[i]==1:
                jy1[i]=np.array([-9999])                  #for upmost nodes (yn=1), no up-neighbor number jy1[i] 
            else:
                jy1[i]=node[i]-self.nx*self.nz                    
        for i in (node-1):
            if yn[i]==self.ny:
                jy2[i]=np.array([-9999])                  #for downmost nodes (yn=ny), no down-neighbor number jy2[i]
            else:
                jy2[i]=node[i]+self.nx*self.nz
                
        for i in (node-1):
            if zn[i]==self.nz:
                jz1[i]=np.array([-9999])                  #for node[i] in the frontmost nodes (zn=1), no inner-neighbor number jz1[i]
            else:
                jz1[i]=node[i]+self.nx
        for i in (node-1):
            if zn[i]==1:
                jz2[i]=np.array([-9999])                  #for node[i] in the backmost nodes (zn=1), no outer-neighbor number jz2[i]
            else:
                jz2[i]=node[i]-self.nx    
        #---------------------------- for jx1, jx2 etc., find NaN and NonNaN element 0-index ------------------------------
        ind0_jx1NaN=np.where(jx1==-9999)[0]               #find the ind0 of the NaN elements in jx1
        ind0_jx2NaN=np.where(jx2==-9999)[0]
        ind0_jy1NaN=np.where(jy1==-9999)[0]
        ind0_jy2NaN=np.where(jy2==-9999)[0]
        ind0_jz1NaN=np.where(jz1==-9999)[0]
        ind0_jz2NaN=np.where(jz2==-9999)[0]
        ind0_jx1NonNaN=np.where(jx1!=-9999)[0]            #find the ind0 of the NonNaN elements in jx1
        ind0_jx2NonNaN=np.where(jx2!=-9999)[0]
        ind0_jy1NonNaN=np.where(jy1!=-9999)[0]
        ind0_jy2NonNaN=np.where(jy2!=-9999)[0]
        ind0_jz1NonNaN=np.where(jz1!=-9999)[0]
        ind0_jz2NonNaN=np.where(jz2!=-9999)[0]
        #---------------------------- jx1, jx2 etc. transformed into 0-index ------------------------------        
        ind0_jx1=jx1-1; ind0_jx1[ind0_jx1NaN]=np.array([-9999])
        ind0_jx2=jx2-1; ind0_jx2[ind0_jx2NaN]=np.array([-9999])
        ind0_jy1=jy1-1; ind0_jy1[ind0_jy1NaN]=np.array([-9999])
        ind0_jy2=jy2-1; ind0_jy2[ind0_jy2NaN]=np.array([-9999])
        ind0_jz1=jz1-1; ind0_jz1[ind0_jz1NaN]=np.array([-9999])
        ind0_jz2=jz2-1; ind0_jz2[ind0_jz2NaN]=np.array([-9999])
        #---------------------------- volume ---------------------------------
        V = (self.Lx *self.Ly *self.Lz)/(self.nx *self.ny *self.nz) * np.ones([self.ntotal_4T,1])  #V is volume of each node, in the form of 1,2...ntotal. When two nodes belong to the same element, they have the same volume i.e. the elementary volume
    
        return node, xn, yn, zn, mat, xi, yi, zi, V,                                                                                                                 \
               jx1, jx2, jy1, jy2, jz1, jz2, ind0_jx1, ind0_jx2, ind0_jy1, ind0_jy2, ind0_jz1, ind0_jz2,                                                                                       \
               ind0_jx1NaN, ind0_jx2NaN, ind0_jy1NaN, ind0_jy2NaN, ind0_jz1NaN, ind0_jz2NaN, ind0_jx1NonNaN, ind0_jx2NonNaN, ind0_jy1NonNaN, ind0_jy2NonNaN, ind0_jz1NonNaN, ind0_jz2NonNaN 
    #########################################################   
    #  functions for weighted avg & std of nodes temperature#
    #########################################################
    def fun_weighted_avg_and_std(self):
        weight_avg=np.average(self.T3_4T_ALL,weights=self.rou_c_V_weights)
        weight_std=np.sqrt( np.average( (self.T3_4T_ALL-weight_avg)**2,weights=self.rou_c_V_weights ) )
        return (weight_avg,weight_std)
    #########################################################   
    ######   function for Thermal initial condition   #######
    #########################################################
    def fun_IC_4T(self):   
        self.Tini_4T_ALL=np.zeros([self.n_4T_ALL,1])
        self.Tini_4T_ALL[:]=self.T_initial                     
#        global step 
        self.step=0; self.fun_BC_4T_ALL()   #step=0 is only for situation of status_Can_Scheme=='ReadBCTem'. In this situation, step is used in Table_T_center_BC[step]. In other situations e.g. status_Can_Scheme=='AllTem', step is not used
        self.Tini_4T_ALL[self.ind0_BCtem_ALL]=self.T3_4T_ALL[self.ind0_BCtem_ALL]  
        return self.Tini_4T_ALL
    #########################################################   
    ###########      function for Thermal BC      ###########
    #########################################################
    def fun_BC_4T_ALL(self):         # output vector T_4T (after constraining temperature on BC nodes)
#        global T3_4T_ALL, ind0_BCtem_ALL, ind0_BCtem_others_ALL, h_4T_ALL, Tconv_4T_ALL, ind0_BCconv_ALL, ind0_BCconv_others_ALL
    
        if self.status_TabSurface_Scheme=='AllConv':
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
    
            #-----------------------------get all constrained node number
            ind0_BCtem=[]                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int) 
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])
            
            h_4T[ self.ind0_Geo_left_4T,0 ]= self.h_ribbon                                                                                                                  
                     
            h_4T[ self.ind0_Geo_right_4T,1 ]= self.h_ribbon                                                                                    
           
            h_4T[ self.ind0_Geo_top_4T,2 ]= self.h_ribbon                                                                                      

            h_4T[ self.ind0_Geo_bottom_4T,3 ]= self.h_ribbon                                                                                      

            h_4T[ self.ind0_Geo_back_4T,4 ]= self.h_ribbon                                                                            

            h_4T[ self.ind0_Geo_front_4T,5 ]= self.h_ribbon                                                                                 
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                 
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling_ribbon                                                                               
                             
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_cooling_ribbon                                                                              
           
            Tconv_4T[ self.ind0_Geo_top_4T,2 ]= self.T_cooling_ribbon                                                                                        

            Tconv_4T[ self.ind0_Geo_bottom_4T,3 ]= self.T_cooling_ribbon                                                                                             

            Tconv_4T[ self.ind0_Geo_back_4T,4 ]= self.T_cooling_ribbon                                                                                           

            Tconv_4T[ self.ind0_Geo_front_4T,5 ]= self.T_cooling_ribbon                                                                                               
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top_4T,self.ind0_Geo_bottom_4T,self.ind0_Geo_back_4T,self.ind0_Geo_front_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                                      #all the other nodes    
        
        if self.status_TabSurface_Scheme=='Single_Tab_Cooling':
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
            T3_4T[self.ind0_Geo_right_4T]=self.T_cooling_ribbon                                                                                              #BC on right nodes             
    
            #-----------------------------get all constrained node number
            ind0_BCtem=self.ind0_Geo_right_4T                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int) 
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])
            
            h_4T[ self.ind0_Geo_left_4T,0 ]= self.h_ribbon                                                                                                                  
                                
            h_4T[ self.ind0_Geo_top_4T,2 ]= self.h_ribbon                                                                                      

            h_4T[ self.ind0_Geo_bottom_4T,3 ]= self.h_ribbon                                                                                      

            h_4T[ self.ind0_Geo_back_4T,4 ]= self.h_ribbon                                                                            

            h_4T[ self.ind0_Geo_front_4T,5 ]= self.h_ribbon                                                                                 
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                 
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling_ribbon                                                                               
                                        
            Tconv_4T[ self.ind0_Geo_top_4T,2 ]= self.T_cooling_ribbon                                                                                        

            Tconv_4T[ self.ind0_Geo_bottom_4T,3 ]= self.T_cooling_ribbon                                                                                             

            Tconv_4T[ self.ind0_Geo_back_4T,4 ]= self.T_cooling_ribbon                                                                                           

            Tconv_4T[ self.ind0_Geo_front_4T,5 ]= self.T_cooling_ribbon                                                                                               
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_top_4T,self.ind0_Geo_bottom_4T,self.ind0_Geo_back_4T,self.ind0_Geo_front_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                                      #all the other nodes    

        self.T3_4T_ALL=T3_4T.copy()
        self.ind0_BCtem_ALL=ind0_BCtem.copy()
        self.ind0_BCtem_others_ALL=ind0_BCtem_others.copy()
        self.h_4T_ALL=h_4T.copy()
        self.Tconv_4T_ALL=Tconv_4T.copy()
        self.ind0_BCconv_ALL=ind0_BCconv.copy()
        self.ind0_BCconv_others_ALL=ind0_BCconv_others.copy()
    #########################################################   
    ######       function for thermal Geometry        #######
    ######################################################### 
    def fun_get_Geo_4T(self):
        ind0_Geo_top_4T=np.where(self.yn_4T==1)[0]                     #top nodes i.g. ind0=[0,1,2~20]
        ind0_Geo_bottom_4T=np.where(self.yn_4T==self.ny)[0]            #bottom nodes i.g. ind0=[42,43,44~62]
        ind0_Geo_left_4T=np.where(self.xn_4T==1)[0]                    #left surface nodes i.g. ind0=[0,3,6,9,12,15,18...21,24,27,30,33,36,39...42,45,48,51,54,57,60]
        ind0_Geo_right_4T=np.where(self.xn_4T==self.nx)[0]             #right surface nodes i.g. ind0=[2,5,8,11,14,17,20...23,26,29,32,35,38,41...44,47,50,53,56,59,62]
        ind0_Geo_front_4T=np.where(self.zn_4T==1)[0]                   #front surface nodes i.g. ind0=[0,1,2,21,22,23,42,43,44]
        ind0_Geo_back_4T=np.where(self.zn_4T==self.nz)[0]              #front surface nodes i.g. ind0=[18,19,20,39,40,41,60,61,62]
        
        for i0 in np.arange(self.status_Cells_num_Module):
            center_xn_temp = i0*(self.nx_spacing+1)+2
            n_half_temp = int((self.S_cooling_nx-1)/2)
            temp_1 = np.array([x for x in ind0_Geo_front_4T if self.xn_4T[x] >= 1 + center_xn_temp - n_half_temp and self.xn_4T[x] <= 1 + center_xn_temp + n_half_temp ],dtype=int)
            temp_1 = temp_1.reshape(self.ny,-1)
            temp_2 = np.roll( np.flip(np.arange(self.S_cooling_nx)),-n_half_temp )
            temp_3 = temp_1[:,temp_2].reshape(-1)
            item = f'ind0_Geo_interface_{i0+1}_ribbon'
            setattr(self,item,temp_3)

        (
        self.ind0_Geo_top_4T, 
        self.ind0_Geo_bottom_4T, 
        self.ind0_Geo_left_4T, 
        self.ind0_Geo_right_4T, 
        self.ind0_Geo_front_4T, 
        self.ind0_Geo_back_4T
        )=(
        ind0_Geo_top_4T, 
        ind0_Geo_bottom_4T, 
        ind0_Geo_left_4T, 
        ind0_Geo_right_4T, 
        ind0_Geo_front_4T, 
        ind0_Geo_back_4T
        )
    #########################################################   
    ### function for reading LUT_SoC,LUT_T,LUT_Ri_PerA,LUT_Ci_PerA ####
    #########################################################
    def fun_pre_Thermal(self):          
    #=======================================prep for stencil of node 1~63    
#        global Lamda_4T, RouXc_4T                                                                                           #for nodes 1~63
#        global Delta_x1_4T, Delta_x2_4T, Delta_y1_4T, Delta_y2_4T, Delta_z1_4T, Delta_z2_4T                                 #for nodes 1~63        
#        global delta_x1_4T, delta_x2_4T, delta_y1_4T, delta_y2_4T, delta_z1_4T, delta_z2_4T, delta_xyz_4T, V_stencil_4T     #for nodes 1~63        
        #-----------------------------------fill in Lamda_4T
        Lamda_4T=self.Lamda_ribbon * np.ones([self.ntotal_4T,6])                                                                                                        #Lamda_4T shape is (63,6). λ term in 6-node stencil
        #-----------------------------------fill in RouXc_4T
        RouXc_4T=self.rouXc_ribbon * np.ones([self.ntotal_4T,1])                                                                                                   #RouXc_4T shape is (63,1). ρc term in 6-node stencil
        #-----------------------------------fill in Delta_x1_4T
        Delta_x1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δx1 for each node
        Delta_x1_4T[self.ind0_jx1NonNaN_4T]=self.xi_4T[self.ind0_jx1NonNaN_4T] - self.xi_4T[self.jx1_4T[self.ind0_jx1NonNaN_4T]-1]
        Delta_x1_4T[self.ind0_jx1NaN_4T]=np.nan
    
        Delta_x2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δx2 for each node
        Delta_x2_4T[self.ind0_jx2NonNaN_4T]=self.xi_4T[self.jx2_4T[self.ind0_jx2NonNaN_4T]-1] - self.xi_4T[self.ind0_jx2NonNaN_4T]
        Delta_x2_4T[self.ind0_jx2NaN_4T]=np.nan
    
        Delta_y1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δy1 for each node
        Delta_y1_4T[self.ind0_jy1NonNaN_4T]=self.yi_4T[self.ind0_jy1NonNaN_4T] - self.yi_4T[self.jy1_4T[self.ind0_jy1NonNaN_4T]-1]
        Delta_y1_4T[self.ind0_jy1NaN_4T]=np.nan
    
        Delta_y2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δy2 for each node
        Delta_y2_4T[self.ind0_jy2NonNaN_4T]=self.yi_4T[self.jy2_4T[self.ind0_jy2NonNaN_4T]-1] - self.yi_4T[self.ind0_jy2NonNaN_4T]
        Delta_y2_4T[self.ind0_jy2NaN_4T]=np.nan
    
        Delta_z1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δz1 for each node
        Delta_z1_4T[self.ind0_jz1NonNaN_4T]=self.zi_4T[self.jz1_4T[self.ind0_jz1NonNaN_4T]-1] - self.zi_4T[self.ind0_jz1NonNaN_4T]
        Delta_z1_4T[self.ind0_jz1NaN_4T]=np.nan
    
        Delta_z2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δz2 for each node
        Delta_z2_4T[self.ind0_jz2NonNaN_4T]=self.zi_4T[self.ind0_jz2NonNaN_4T] - self.zi_4T[self.jz2_4T[self.ind0_jz2NonNaN_4T]-1]
        Delta_z2_4T[self.ind0_jz2NaN_4T]=np.nan
    
    
        delta_x1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get δx1 for each node
        delta_x1_4T[self.ind0_jx1NonNaN_4T]=self.xi_4T[self.ind0_jx1NonNaN_4T] - self.xi_4T[self.jx1_4T[self.ind0_jx1NonNaN_4T]-1]
        delta_x1_4T[self.ind0_jx1NaN_4T]=self.xi_4T[self.jx2_4T[self.ind0_jx1NaN_4T]-1]-self.xi_4T[self.ind0_jx1NaN_4T]
    
        delta_x2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get δx2 for each node
        delta_x2_4T[self.ind0_jx2NonNaN_4T]=self.xi_4T[self.jx2_4T[self.ind0_jx2NonNaN_4T]-1] - self.xi_4T[self.ind0_jx2NonNaN_4T]
        delta_x2_4T[self.ind0_jx2NaN_4T]=self.xi_4T[self.ind0_jx2NaN_4T]-self.xi_4T[self.jx1_4T[self.ind0_jx2NaN_4T]-1]
        
        delta_y1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get δy1 for each node
        delta_y1_4T[self.ind0_jy1NonNaN_4T]=self.yi_4T[self.ind0_jy1NonNaN_4T] - self.yi_4T[self.jy1_4T[self.ind0_jy1NonNaN_4T]-1]
        delta_y1_4T[self.ind0_jy1NaN_4T]=self.yi_4T[self.jy2_4T[self.ind0_jy1NaN_4T]-1]-self.yi_4T[self.ind0_jy1NaN_4T]
    
        delta_y2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get δy2 for each node
        delta_y2_4T[self.ind0_jy2NonNaN_4T]=self.yi_4T[self.jy2_4T[self.ind0_jy2NonNaN_4T]-1] - self.yi_4T[self.ind0_jy2NonNaN_4T]
        delta_y2_4T[self.ind0_jy2NaN_4T]=self.yi_4T[self.ind0_jy2NaN_4T]-self.yi_4T[self.jy1_4T[self.ind0_jy2NaN_4T]-1]

        delta_z1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get δz1 for each node
        delta_z1_4T[self.ind0_jz1NonNaN_4T]=self.zi_4T[self.jz1_4T[self.ind0_jz1NonNaN_4T]-1] - self.zi_4T[self.ind0_jz1NonNaN_4T]
        delta_z1_4T[self.ind0_jz1NaN_4T]=self.zi_4T[self.ind0_jz1NaN_4T]-self.zi_4T[self.jz2_4T[self.ind0_jz1NaN_4T]-1]
    
        delta_z2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get δz2 for each node
        delta_z2_4T[self.ind0_jz2NonNaN_4T]=self.zi_4T[self.ind0_jz2NonNaN_4T] - self.zi_4T[self.jz2_4T[self.ind0_jz2NonNaN_4T]-1]
        delta_z2_4T[self.ind0_jz2NaN_4T]=self.zi_4T[self.jz1_4T[self.ind0_jz2NaN_4T]-1]-self.zi_4T[self.ind0_jz2NaN_4T]
        
        delta_xyz_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                               #(δx1+δx2,δx1+δx2,δy1+δy2,δy1+δy2,2δz,2δz), shape is (63,1)
        delta_xyz_4T[:,0:2]=(delta_x1_4T+delta_x2_4T).reshape(-1,1) 
        delta_xyz_4T[:,2:4]=(delta_y1_4T+delta_y2_4T).reshape(-1,1)
        delta_xyz_4T[:,4:6]=(delta_z1_4T+delta_z2_4T).reshape(-1,1)
        V_stencil_4T=(delta_xyz_4T[:,0]/2) * (delta_xyz_4T[:,2]/2) * (delta_xyz_4T[:,4]/2)  #node volume used in thermal stencil delta_xyz_4T_ALL: (δx1+δx2,δx1+δx2,δy1+δy2,δy1+δy2,2δz,2δz), shape is (63,1)
        #=======================================update variables to be used in stencil (fun_Thermal())
        self.xi_4T_ALL=self.xi_4T.copy();  self.yi_4T_ALL=self.yi_4T.copy();  self.zi_4T_ALL=self.zi_4T.copy()
        self.jx1_4T_ALL=self.jx1_4T.copy();  self.jx2_4T_ALL=self.jx2_4T.copy();  self.jy1_4T_ALL=self.jy1_4T.copy();  self.jy2_4T_ALL=self.jy2_4T.copy();  self.jz1_4T_ALL=self.jz1_4T.copy();  self.jz2_4T_ALL=self.jz2_4T.copy()
        self.ind0_jx1NaN_4T_ALL=self.ind0_jx1NaN_4T.copy();        self.ind0_jx2NaN_4T_ALL=self.ind0_jx2NaN_4T.copy();        self.ind0_jy1NaN_4T_ALL=self.ind0_jy1NaN_4T.copy();        self.ind0_jy2NaN_4T_ALL=self.ind0_jy2NaN_4T.copy();        self.ind0_jz1NaN_4T_ALL=self.ind0_jz1NaN_4T.copy();        self.ind0_jz2NaN_4T_ALL=self.ind0_jz2NaN_4T.copy()
        self.ind0_jx1NonNaN_4T_ALL=self.ind0_jx1NonNaN_4T.copy();  self.ind0_jx2NonNaN_4T_ALL=self.ind0_jx2NonNaN_4T.copy();  self.ind0_jy1NonNaN_4T_ALL=self.ind0_jy1NonNaN_4T.copy();  self.ind0_jy2NonNaN_4T_ALL=self.ind0_jy2NonNaN_4T.copy();  self.ind0_jz1NonNaN_4T_ALL=self.ind0_jz1NonNaN_4T.copy();  self.ind0_jz2NonNaN_4T_ALL=self.ind0_jz2NonNaN_4T.copy()
        self.ind0_jx1_4T_ALL=self.ind0_jx1_4T.copy(); self.ind0_jx2_4T_ALL=self.ind0_jx2_4T.copy(); self.ind0_jy1_4T_ALL=self.ind0_jy1_4T.copy(); self.ind0_jy2_4T_ALL=self.ind0_jy2_4T.copy(); self.ind0_jz1_4T_ALL=self.ind0_jz1_4T.copy(); self.ind0_jz2_4T_ALL=self.ind0_jz2_4T.copy()
        self.mat_4T_ALL=self.mat_4T.copy()
        self.Lamda_4T_ALL=Lamda_4T.copy(); self.RouXc_4T_ALL=RouXc_4T.copy()                                                                                                                                                                                                  
        self.Delta_x1_4T_ALL=Delta_x1_4T.copy(); self.Delta_x2_4T_ALL=Delta_x2_4T.copy(); self.Delta_y1_4T_ALL=Delta_y1_4T.copy(); self.Delta_y2_4T_ALL=Delta_y2_4T.copy(); self.Delta_z1_4T_ALL=Delta_z1_4T.copy(); self.Delta_z2_4T_ALL=Delta_z2_4T.copy()                                                                                                  
        self.delta_x1_4T_ALL=delta_x1_4T.copy(); self.delta_x2_4T_ALL=delta_x2_4T.copy(); self.delta_y1_4T_ALL=delta_y1_4T.copy(); self.delta_y2_4T_ALL=delta_y2_4T.copy(); self.delta_z1_4T_ALL=delta_z1_4T.copy(); self.delta_z2_4T_ALL=delta_z2_4T.copy()
        self.delta_xyz_4T_ALL=delta_xyz_4T.copy()           
        self.V_stencil_4T_ALL=V_stencil_4T.copy()  #node volume used in thermal stencil delta_xyz_4T_ALL: (δx1+δx2,δx1+δx2,δy1+δy2,δy1+δy2,2δz,2δz), shape is (87,1)
    #########################################################   
    ###########     function for Thermal model    ###########
    #########################################################
    def fun_Thermal(self, T1_4T_ALL,T3_4T_ALL, ind0_BCtem_ALL, ind0_BCtem_others_ALL, h_4T_ALL, Tconv_4T_ALL, ind0_BCconv_ALL, ind0_BCconv_others_ALL):     #return node temperature vector T_4T (in Thermal node framework)        
    #================================================================explicit solver
        if self.status_Thermal_solver == 'Explicit':        
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
    ###########       function for MatrixCN       ###########
    #########################################################
    def fun_MatrixCN(self):
        MatrixCN=np.zeros([self.n_4T_ALL,self.n_4T_ALL])        
    #======================================calculate nodes except Sep nodes (i.e. 1~84)
        #--------------------------------------fill in jx1, jx2, jy1, jy2, jz1, jz2 terms
        MatrixCN[self.ind0_jx1NonNaN_4T_ALL,self.ind0_jx1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]] += -self.Lamda_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0]/self.RouXc_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx1NonNaN_4T_ALL])/self.Delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]    #if ind0_jx1NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jx1NonNaN node , column of left neighbor node(jx1); if ind0_jx1NaN nodes case: elements are zero as initiated 
        MatrixCN[self.ind0_jx2NonNaN_4T_ALL,self.ind0_jx2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]] += -self.Lamda_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,1]/self.RouXc_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL])/self.Delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]    #if ind0_jx2NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jx2NonNaN node , column of right neighbor node(jx2); if ind0_jx2NaN nodes case: elements are zero as initiated 
        MatrixCN[self.ind0_jy1NonNaN_4T_ALL,self.ind0_jy1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]] += -self.Lamda_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy1NonNaN_4T_ALL])/self.Delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]    #if ind0_jy1NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jy1NonNaN node , column of up neighbor node(jy1); if ind0_jy1NaN nodes case: elements are zero as initiated 
        MatrixCN[self.ind0_jy2NonNaN_4T_ALL,self.ind0_jy2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]] += -self.Lamda_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL])/self.Delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]    #if ind0_jy2NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jy2NonNaN node , column of down neighbor node(jy2); if ind0_jy2NaN nodes case: elements are zero as initiated 
        MatrixCN[self.ind0_jz1NonNaN_4T_ALL,self.ind0_jz1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]] += -self.Lamda_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,4]/self.RouXc_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz1NonNaN_4T_ALL])/self.Delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]    #if ind0_jz1NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jz1NonNaN node , column of inner neighbor node(jz1); if ind0_jz1NaN nodes case: elements are zero as initiated 
        MatrixCN[self.ind0_jz2NonNaN_4T_ALL,self.ind0_jz2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]] += -self.Lamda_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,5]/self.RouXc_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL])/self.Delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]    #if ind0_jz2NonNaN nodes case: fill elements of the MatrixCN: row of ind0_jz2NonNaN node , column of outer neighbor node(jz2); if ind0_jz2NaN nodes case: elements are zero as initiated 
        #--------------------------------------fill in diagonal terms
        MatrixCN[self.ind0_jx1NaN_4T_ALL,self.ind0_jx1NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jx1NaN_4T_ALL,0]/self.RouXc_4T_ALL[self.ind0_jx1NaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx1NaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx1NaN_4T_ALL])       #jx1 components in diagonal terms
        MatrixCN[self.ind0_jx1NonNaN_4T_ALL,self.ind0_jx1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0]/self.RouXc_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx1NonNaN_4T_ALL])/self.Delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jx2NaN_4T_ALL,self.ind0_jx2NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jx2NaN_4T_ALL,1]/self.RouXc_4T_ALL[self.ind0_jx2NaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx2NaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx2NaN_4T_ALL])       #jx2 components in diagonal terms
        MatrixCN[self.ind0_jx2NonNaN_4T_ALL,self.ind0_jx2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,1]/self.RouXc_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL])/self.Delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jy1NaN_4T_ALL,self.ind0_jy1NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.ind0_jy1NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy1NaN_4T_ALL])       #jy1 components in diagonal terms
        MatrixCN[self.ind0_jy1NonNaN_4T_ALL,self.ind0_jy1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy1NonNaN_4T_ALL])/self.Delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jy2NaN_4T_ALL,self.ind0_jy2NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.ind0_jy2NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy2NaN_4T_ALL])       #jy2 components in diagonal terms
        MatrixCN[self.ind0_jy2NonNaN_4T_ALL,self.ind0_jy2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL])/self.Delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jz1NaN_4T_ALL,self.ind0_jz1NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jz1NaN_4T_ALL,4]/self.RouXc_4T_ALL[self.ind0_jz1NaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz1NaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz1NaN_4T_ALL])       #jz1 components in diagonal terms
        MatrixCN[self.ind0_jz1NonNaN_4T_ALL,self.ind0_jz1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,4]/self.RouXc_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz1NonNaN_4T_ALL])/self.Delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jz2NaN_4T_ALL,self.ind0_jz2NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jz2NaN_4T_ALL,5]/self.RouXc_4T_ALL[self.ind0_jz2NaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz2NaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz2NaN_4T_ALL])       #jz2 components in diagonal terms
        MatrixCN[self.ind0_jz2NonNaN_4T_ALL,self.ind0_jz2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,5]/self.RouXc_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL])/self.Delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]
    
        MatrixCN[np.arange(self.n_4T_ALL),np.arange(self.n_4T_ALL)] += 1                                                                                               #"1" components in diagonal terms
        MatrixCN[self.ind0_BCtem_ALL,self.ind0_BCtem_ALL]=inf
           
        return MatrixCN
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
    def fun_VectorCN(self):                #VectorCN = VectorCN_preTp*Tp + VectorCN_conv_q;    VectorCN_preTp is very similar to MatrixCN, so form VectorCN based on MatrixCN
#        global VectorCN, VectorCN_conv_q
        VectorCN=np.zeros([self.n_4T_ALL,1])       
    #    VectorCN_preTp=np.zeros([n_4T_ALL,n_4T_ALL])
        self.VectorCN_conv_q=np.zeros([self.n_4T_ALL,1])
        #==================================================add Tp term        
    #    VectorCN_preTp=MatrixCN.copy()
    #    VectorCN_preTp[np.arange(n_4T_ALL),np.arange(n_4T_ALL)] -= 1
    #    VectorCN_preTp = -VectorCN_preTp
    #    VectorCN_preTp[np.arange(n_4T_ALL),np.arange(n_4T_ALL)] += 1
        #==================================================add non Tp term - conv         
        self.VectorCN_conv_q[self.ind0_jx1NaN_4T_ALL,0] += self.h_4T_ALL[self.ind0_jx1NaN_4T_ALL,0]/self.RouXc_4T_ALL[self.ind0_jx1NaN_4T_ALL,0]*2*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx1NaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx1NaN_4T_ALL]) * self.Tconv_4T_ALL[self.ind0_jx1NaN_4T_ALL,0]                                                                                               #if ind0_jx1NaN nodes case: fill elements of the jx1 terms
        self.VectorCN_conv_q[self.ind0_jx2NaN_4T_ALL,0] += self.h_4T_ALL[self.ind0_jx2NaN_4T_ALL,1]/self.RouXc_4T_ALL[self.ind0_jx2NaN_4T_ALL,0]*2*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx2NaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx2NaN_4T_ALL]) * self.Tconv_4T_ALL[self.ind0_jx2NaN_4T_ALL,1]                                                                                               #if ind0_jx2NaN nodes case: fill elements of the jx2 terms
        self.VectorCN_conv_q[self.ind0_jy1NaN_4T_ALL,0] += self.h_4T_ALL[self.ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.ind0_jy1NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy1NaN_4T_ALL]) * self.Tconv_4T_ALL[self.ind0_jy1NaN_4T_ALL,2]                                                                                               #if ind0_jy1NaN nodes case: fill elements of the jy1 terms
        self.VectorCN_conv_q[self.ind0_jy2NaN_4T_ALL,0] += self.h_4T_ALL[self.ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.ind0_jy2NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy2NaN_4T_ALL]) * self.Tconv_4T_ALL[self.ind0_jy2NaN_4T_ALL,3]                                                                                               #if ind0_jy2NaN nodes case: fill elements of the jy2 terms
        self.VectorCN_conv_q[self.ind0_jz1NaN_4T_ALL,0] += self.h_4T_ALL[self.ind0_jz1NaN_4T_ALL,4]/self.RouXc_4T_ALL[self.ind0_jz1NaN_4T_ALL,0]*2*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz1NaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz1NaN_4T_ALL]) * self.Tconv_4T_ALL[self.ind0_jz1NaN_4T_ALL,4]                                                                                               #if ind0_jz1NaN nodes case: fill elements of the jz1 terms
        self.VectorCN_conv_q[self.ind0_jz2NaN_4T_ALL,0] += self.h_4T_ALL[self.ind0_jz2NaN_4T_ALL,5]/self.RouXc_4T_ALL[self.ind0_jz2NaN_4T_ALL,0]*2*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz2NaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz2NaN_4T_ALL]) * self.Tconv_4T_ALL[self.ind0_jz2NaN_4T_ALL,5]                                                                                               #if ind0_jz2NaN nodes case: fill elements of the jz2 terms
        #==================================================add non Tp term - heat gen q         
        self.VectorCN_conv_q[:self.n_4T_ALL,0] += self.q_4T_ALL[:self.n_4T_ALL,0]*self.dt/self.RouXc_4T_ALL[:self.n_4T_ALL,0]                                                                                                                                                                                                     #heat gen components
    
        VectorCN= self.VectorCN_preTp .dot( self.T1_4T_ALL ) + self.VectorCN_conv_q
        #======================================penalty on Temperature-constrained BC nodes (apply temperature BC)
        VectorCN[self.ind0_BCtem_ALL,0]=(self.T3_4T_ALL[self.ind0_BCtem_ALL,0]*inf)
    
        return VectorCN
    #########################################################   
    #########     function for Postprocessor     ############
    #########################################################
    def fun_Postprocessor(self,step,t_begin,t_end):        
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
                    self.fun_plot(step); 
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
    
            if self.status_Echeck=='Yes':        
                plt.figure()
                plt.plot(self.t_record,self.Egen_Total_record,'y')
                plt.plot(self.t_record,self.Eext_Total_BCconv_record,'g')
                plt.plot(self.t_record,self.Eint_Delta_record,'k')
                plt.plot(self.t_record,self.Ebalance_record,'r')
                plt.xlabel('Time [s]');     plt.ylabel('Model energy [J]'); plt.title('Model energy conservation check')    
                plt.legend(('Egen_Total','Eext_Total_BCconv','Eint_Delta','E_balance'))
    
    
        print('postprocessor done\n')
        print('all done :)\n')    
        
        print('Notes:\n')  
        print('total time elapsed: %f (s) or %.2f (min) or %.2f (h), finishing time: %s\n' %(t_end-t_begin, (t_end-t_begin)/60, (t_end-t_begin)/3600, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()) ))
        if self.status_Model=='EandT' or self.status_Model=='T':
            print('convective h should be fixed for now if any conv BC\n')  
        if self.status_PID=='Yes':    
            print('PID mode is on.\n')
        if self.status_Echeck=='Yes':
            print('Heat conservation is checked currently only for convection BC, and not for T constrained BC.\n')    
    
    
    
    #########################################################   
    #########     function for Visualization     ############
    #########################################################
    
    def fun_mayavi_by_node(self, XYZ_Module, plot_Variable_ALL, vmin, vmax, title_string, colormap_string):        
        
#        self.plot_steps_available=np.where(~np.isnan(self.T_avg_record))[0]      #in cycling mode, there are NaN values in the last cycles. So here plot_steps_available is all the steps with non-NaN values 
#        self.plot_step=self.plot_steps_available[-1]                             #plot the last step from non-NaN steps 
        
#        self.plot_Variable_ALL=self.T_record[:,plot_step]-273.15  #variable to be visualized  
        
        self.X_Module = XYZ_Module[0]         #this cell's X location in Module
        self.Y_Module = XYZ_Module[1]
        self.Z_Module = XYZ_Module[2]
        rotate_angle = 0*np.pi/2                
        #===================================prepare X,Y,Z,C for visualization
        self.yi_plot_4T_ALL=self.yi_4T_ALL.copy()
        #-----------------------plot front surface
        self.ind0_plot_front=self.ind0_Geo_front_4T.reshape(self.ny,self.nx)
        self.X1=self.xi_4T_ALL[self.ind0_plot_front]
        self.Y1=self.zi_4T_ALL[self.ind0_plot_front]
        self.Z1=self.Ly-self.yi_4T_ALL[self.ind0_plot_front]
        self.C1=plot_Variable_ALL[self.ind0_plot_front]
        temp1 = self.X1.copy(); temp2 = self.Y1.copy()
        self.X1 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y1 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X1 += self.X_Module; self.Y1 += self.Y_Module; self.Z1 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot back surface
        self.ind0_plot_back=self.ind0_Geo_back_4T.reshape(self.ny,self.nx)
        self.X2=self.xi_4T_ALL[self.ind0_plot_back]
        self.Y2=self.zi_4T_ALL[self.ind0_plot_back]
        self.Z2=self.Ly-self.yi_4T_ALL[self.ind0_plot_back]
        temp1 = self.X2.copy(); temp2 = self.Y2.copy()
        self.X2 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y2 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.C2=plot_Variable_ALL[self.ind0_plot_back]
        self.X2 += self.X_Module; self.Y2 += self.Y_Module; self.Z2 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot left surface
        self.ind0_plot_left=self.ind0_Geo_left_4T.reshape(self.ny,self.nz)
        self.X3=self.xi_4T_ALL[self.ind0_plot_left]
        self.Y3=self.zi_4T_ALL[self.ind0_plot_left]
        self.Z3=self.Ly-self.yi_4T_ALL[self.ind0_plot_left]
        temp1 = self.X3.copy(); temp2 = self.Y3.copy()
        self.X3 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y3 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.C3=plot_Variable_ALL[self.ind0_plot_left]
        self.X3 += self.X_Module; self.Y3 += self.Y_Module; self.Z3 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot right surface
        self.ind0_plot_right=self.ind0_Geo_right_4T.reshape(self.ny,self.nz)
        self.X4=self.xi_4T_ALL[self.ind0_plot_right]
        self.Y4=self.zi_4T_ALL[self.ind0_plot_right]
        self.Z4=self.Ly-self.yi_4T_ALL[self.ind0_plot_right]
        temp1 = self.X4.copy(); temp2 = self.Y4.copy()
        self.X4 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y4 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.C4=plot_Variable_ALL[self.ind0_plot_right]
        self.X4 += self.X_Module; self.Y4 += self.Y_Module; self.Z4 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot top surface
        self.ind0_plot_top=self.ind0_Geo_top_4T.reshape(self.nz,self.nx)
        self.X5=self.xi_4T_ALL[self.ind0_plot_top]
        self.Y5=self.zi_4T_ALL[self.ind0_plot_top]
        self.Z5=self.Ly-self.yi_4T_ALL[self.ind0_plot_top]
        self.C5=plot_Variable_ALL[self.ind0_plot_top]
        temp1 = self.X5.copy(); temp2 = self.Y5.copy()
        self.X5 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y5 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X5 += self.X_Module; self.Y5 += self.Y_Module; self.Z5 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot bottom surface
        self.ind0_plot_bottom=self.ind0_Geo_bottom_4T.reshape(self.nz,self.nx)
        self.X6=self.xi_4T_ALL[self.ind0_plot_bottom]
        self.Y6=self.zi_4T_ALL[self.ind0_plot_bottom]
        self.Z6=self.Ly-self.yi_4T_ALL[self.ind0_plot_bottom]
        self.C6=plot_Variable_ALL[self.ind0_plot_bottom]
        temp1 = self.X6.copy(); temp2 = self.Y6.copy()
        self.X6 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y6 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X6 += self.X_Module; self.Y6 += self.Y_Module; self.Z6 += self.Z_Module    #Get cells' spatial locations in Module
            

        #===================================visualization
#       self.fig = mlab.figure(bgcolor=(1,1,1))    
#       self.vmin=self.plot_Variable_ALL.min();  self.vmax=self.plot_Variable_ALL.max()
        self.surf1 = mlab.mesh(self.X1, self.Y1, self.Z1, scalars=self.C1, colormap='coolwarm')
        self.surf2 = mlab.mesh(self.X2, self.Y2, self.Z2, scalars=self.C2, colormap='coolwarm')
        self.surf3 = mlab.mesh(self.X3, self.Y3, self.Z3, scalars=self.C3, colormap='coolwarm')
        self.surf4 = mlab.mesh(self.X4, self.Y4, self.Z4, scalars=self.C4, colormap='coolwarm')
        self.surf5 = mlab.mesh(self.X5, self.Y5, self.Z5, scalars=self.C5, colormap='coolwarm')
        self.surf6 = mlab.mesh(self.X6, self.Y6, self.Z6, scalars=self.C6, colormap='coolwarm')

        self.surf1.module_manager.scalar_lut_manager.use_default_range = False
        self.surf1.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf2.module_manager.scalar_lut_manager.use_default_range = False
        self.surf2.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf3.module_manager.scalar_lut_manager.use_default_range = False
        self.surf3.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf4.module_manager.scalar_lut_manager.use_default_range = False
        self.surf4.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf5.module_manager.scalar_lut_manager.use_default_range = False
        self.surf5.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf6.module_manager.scalar_lut_manager.use_default_range = False
        self.surf6.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.cb=mlab.colorbar(title='°C',orientation='vertical',label_fmt='%.2f',nb_labels=5)
        self.cb.scalar_bar.unconstrained_font_size = True
        self.cb.title_text_property.font_family = 'times'; self.cb.title_text_property.bold=False; self.cb.title_text_property.italic=False; self.cb.title_text_property.color=(0,0,0); self.cb.title_text_property.font_size=20
        self.cb.label_text_property.font_family = 'times'; self.cb.label_text_property.bold=True;  self.cb.label_text_property.italic=False; self.cb.label_text_property.color=(0,0,0); self.cb.label_text_property.font_size=15
    
    
    
    
    
    
    
    
    
    
    
