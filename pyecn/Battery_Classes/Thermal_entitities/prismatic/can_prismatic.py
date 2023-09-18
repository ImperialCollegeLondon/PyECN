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

class Can_Prismatic:
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
        self.rouXc_Can=self.rou_Can*self.c_Can
        self.R_cylinder = params_update['R_cylinder']
        self.dx = self.Lx_pouch/self.nx_pouch      #dx for node 11-30; for nodes at edge e.g. 1,9,17,25,33, dx is updated elsewhere in the code
        self.dy = self.LG_Jellyroll/self.ny        #dy for node 11-30; for nodes at edge e.g. 1,9,17,25,33, dx is updated elsewhere in the code
        self.dz = (4*self.R_cylinder + self.delta_Can_real)/(self.nz_Can-2)
        self.nx_cylinder = np.ceil((self.R_cylinder-0.5*self.dx)/self.dx).astype(int) + 1   #'+1' is for the node on the edge
        self.nx_Can = self.nx_pouch +1 + 2*self.nx_cylinder
        self.ny_Can = self.ny + 2
        self.ntotal_4T = self.nx_Can*self.ny_Can*self.nz_Can - (self.nx_Can-2)*(self.ny_Can-2)*(self.nz_Can-2)

        (
        self.node_4T, self.xn_4T, self.yn_4T, self.zn_4T, self.mat_4T, self.xi_4T, self.yi_4T, self.zi_4T, self.V_4T,
        self.jx1_4T, self.jx2_4T, self.jy1_4T, self.jy2_4T, self.jz1_4T, self.jz2_4T, self.ind0_jx1_4T, self.ind0_jx2_4T, self.ind0_jy1_4T, self.ind0_jy2_4T, self.ind0_jz1_4T, self.ind0_jz2_4T,
        self.ind0_jx1NaN_4T, self.ind0_jx2NaN_4T, self.ind0_jy1NaN_4T, self.ind0_jy2NaN_4T, self.ind0_jz1NaN_4T, self.ind0_jz2NaN_4T, self.ind0_jx1NonNaN_4T, self.ind0_jx2NonNaN_4T, self.ind0_jy1NonNaN_4T, self.ind0_jy2NonNaN_4T, self.ind0_jz1NonNaN_4T, self.ind0_jz2NonNaN_4T  
        ) = self.fun_matrix1()                             
            
        #--------------------------------------------------------------------------preparation for modification on MatrixC and VectorI (preprocessor)
        print('\nrunning preprocessor...\n')
        self.n_4T_ALL=self.ntotal_4T
        self.node_4T_ALL=self.node_4T
        self.rou_c_V_weights=self.rou_Can*self.c_Can*self.V_4T             #prep for fun_weighted_avg_and_std
        self.fun_get_Geo_4T()
        self.fun_pre_Thermal()
#        #---unique for can_prismatic---
#        self.ind0_jz1NaN_4T_ALL = np.array([x for x in self.ind0_jz1NaN_4T_ALL if not (self.zn_4T[x]==self.nz_Can and self.xn_4T[x]>=self.nx_cylinder+1 and self.xn_4T[x]<=self.nx_Can-self.nx_cylinder and self.yn_4T[x]>1 and self.yn_4T[x]<self.ny_Can) ],dtype=int)    #for node117-120,125-129,133-137, update ind0_jz1NaN as those nodes are linked now to jellyroll
#        self.ind0_jz1NonNaN_4T_ALL = np.array([x for x in np.arange(self.ntotal_4T) if x not in self.ind0_jz1NaN_4T_ALL],dtype=int)
#
#        self.ind0_jz2NaN_4T_ALL = np.array([x for x in self.ind0_jz2NaN_4T_ALL if not (self.zn_4T[x]==1 and self.xn_4T[x]>=self.nx_cylinder+1 and self.xn_4T[x]<=self.nx_Can-self.nx_cylinder and self.yn_4T[x]>1 and self.yn_4T[x]<self.ny_Can) ],dtype=int)              #for node11-14,19-22,27-30, update ind0_jz2NaN as those nodes are linked now to jellyroll
#        self.ind0_jz2NonNaN_4T_ALL = np.array([x for x in np.arange(self.ntotal_4T) if x not in self.ind0_jz2NaN_4T_ALL],dtype=int)
#        #------------------------------

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
        self.T_avg_record=np.nan*np.zeros([self.nt+1]); self.T_SD_record=np.nan*np.zeros([self.nt+1]); self.T_Delta_record = np.nan*np.zeros([self.nt+1])           #record node temperature average and SD for postprocessor

        self.S_stencil_4T_ALL=np.nan*np.zeros([self.ntotal_4T,6])             # node plane area (in 6 directions) used in thermal stencil; S_Stencil_4T_ALL: ((δy1+δy2)/2*(δz1+δz2)/2,(δy1+δy2)/2*δz1+δz2)/2, (δx1+δx2)/2*(δz1+δz2)/2,(δx1+δx2)/2*(δz1+δz2)/2, (δx1+δx2)/2*(δy1+δy2)/2,(δx1+δx2)/2*(δy1+δy2)/2), shape is (87,1)
        self.S_stencil_4T_ALL[:,0]=0.5*self.delta_xyz_4T_ALL[:,2] * 0.5*self.delta_xyz_4T_ALL[:,4]
        self.S_stencil_4T_ALL[:,1]=self.S_stencil_4T_ALL[:,0]
        self.S_stencil_4T_ALL[:,2]=0.5*self.delta_xyz_4T_ALL[:,0] * 0.5*self.delta_xyz_4T_ALL[:,4]
        self.S_stencil_4T_ALL[:,3]=self.S_stencil_4T_ALL[:,2]
        self.S_stencil_4T_ALL[:,4]=0.5*self.delta_xyz_4T_ALL[:,0] * 0.5*self.delta_xyz_4T_ALL[:,2]
        self.S_stencil_4T_ALL[:,5]=self.S_stencil_4T_ALL[:,4]
        self.S_AllConv = np.sum( self.S_stencil_4T_ALL[~np.isnan(self.h_4T_ALL)] )    
        self.Eext_Total_BCconv_record=np.nan*np.zeros([self.nt+1]); self.Eext_Total_BCconv_record[0]=0                 #accumulated external work relative to initial time;  (-):heat lose  (+): heat absorption
    #########################################################   
    ################## function for matrix1 #################
    #########################################################
    def fun_matrix1(self):

        #============================== metal block is implemented at first
        ntotal_4T_block = self.nx_Can*self.ny_Can*self.nz_Can        #nodes number of block 
        node_4T_block = np.arange(1, ntotal_4T_block+1)
        xn_unit=np.linspace(1,self.nx_Can,self.nx_Can,dtype=int)     #repeating node number for xn
        yn_unit=np.linspace(1,self.ny_Can,self.ny_Can,dtype=int)     #repeating node number for yn
        zn_unit=np.linspace(1,self.nz_Can,self.nz_Can,dtype=int)     #repeating node number for zn
        xn=np.tile(xn_unit,self.ny_Can*self.nz_Can)
        yn=np.repeat(np.tile(yn_unit,self.nz_Can),self.nx_Can)
        zn=np.repeat(zn_unit,self.nx_Can*self.ny_Can)
                
        xi=np.zeros(ntotal_4T_block); yi=np.zeros(ntotal_4T_block); zi=np.zeros(ntotal_4T_block)
        xi=(xn-1)*self.dx
        yi=(yn-1)*self.dy    
        zi=(zn-1)*self.dz    
        #due to edge thickness, xi, yi, zi needs to be modified
        xi[xn!=1] -= (0.5*self.dx - 0.5*self.delta_Can_real)           
        xi[xn==self.nx_Can] -= (0.5*self.dx - 0.5*self.delta_Can_real)         

        yi[yn!=1] -= (0.5*self.dy - 0.5*self.delta_Can_real)           
        yi[yn==self.ny_Can] -= (0.5*self.dy - 0.5*self.delta_Can_real)         

        zi[zn!=1] -= (0.5*self.dz - 0.5*self.delta_Can_real)           
        zi[zn==self.nz_Can] -= (0.5*self.dz - 0.5*self.delta_Can_real)         
        #---------------------------- volume ---------------------------------
        V_dx = self.dx*np.ones(ntotal_4T_block)
        V_dx[xn==1] = self.delta_Can_real
        V_dx[xn==self.nx_Can] = self.delta_Can_real
        V_dy = self.dy*np.ones(ntotal_4T_block)
        V_dy[yn==1] = self.delta_Can_real
        V_dy[yn==self.ny_Can] = self.delta_Can_real
        V_dz = self.dz*np.ones(ntotal_4T_block)
        V_dz[zn==1] = self.delta_Can_real
        V_dz[zn==self.nz_Can] = self.delta_Can_real
        V = V_dx * V_dy * V_dz
        #============================== deduct inner block from whole block to form Can
        node = np.arange(1, self.ntotal_4T+1)
        mat=np.ones([self.ntotal_4T])
        
        ind0_inner_block = []
        for i0 in node_4T_block-1:
            if (xn[i0]!= 1 and xn[i0]!= self.nx_Can) and (yn[i0]!= 1 and yn[i0]!= self.ny_Can) and (zn[i0]!= 1 and zn[i0]!= self.nz_Can):
                ind0_inner_block = np.append(ind0_inner_block,i0).astype(int)
        
        xn = np.delete(xn,ind0_inner_block)
        yn = np.delete(yn,ind0_inner_block)
        zn = np.delete(zn,ind0_inner_block)

        xi = np.delete(xi,ind0_inner_block)
        yi = np.delete(yi,ind0_inner_block)
        zi = np.delete(zi,ind0_inner_block)

    
        jx1=np.zeros(self.ntotal_4T,dtype=int)            #initialize left-neighbor node number in x direction
        jx2=np.zeros(self.ntotal_4T,dtype=int)            #initialize right-neighbor node number in x direction
        jy1=np.zeros(self.ntotal_4T,dtype=int)            #initialize up-neighbor node number in y direction
        jy2=np.zeros(self.ntotal_4T,dtype=int)            #initialize down-neighbor node number in y direction
        jz1=np.zeros(self.ntotal_4T,dtype=int)            #initialize inner-neighbor node number in z direction
        jz2=np.zeros(self.ntotal_4T,dtype=int)            #initialize outer-neighbor node number in z direction       

        for i0 in np.arange(self.ntotal_4T):
            if xn[i0]==1:                                  
                jx1[i0]=np.array([-9999])    
            elif xn[i0]==self.nx_Can and (yn[i0]>1 and yn[i0]<self.ny_Can) and (zn[i0]>1 and zn[i0]<self.nz_Can):     
                jx1[i0]=np.array([-9999])
            else:
                jx1[i0]=node[i0]-1                          
        for i0 in np.arange(self.ntotal_4T):
            if xn[i0]==self.nx_Can:                                  
                jx2[i0]=np.array([-9999])    
            elif xn[i0]==1 and (yn[i0]>1 and yn[i0]<self.ny_Can) and (zn[i0]>1 and zn[i0]<self.nz_Can):     
                jx2[i0]=np.array([-9999])
            else:
                jx2[i0]=node[i0]+1                          
        for i0 in np.arange(self.ntotal_4T):
            if yn[i0]==1:                                  
                jy1[i0]=np.array([-9999])    
            elif yn[i0]==self.ny_Can and (xn[i0]>1 and xn[i0]<self.nx_Can) and (zn[i0]>1 and zn[i0]<self.nz_Can):     
                jy1[i0]=np.array([-9999])
            else:
                ind0_temp = node[yn==yn[i0]-1] -1
                jy1[i0] = np.array([x for x in ind0_temp if (xn[x]==xn[i0] and zn[x]==zn[i0]) ],dtype=int) + 1
        for i0 in np.arange(self.ntotal_4T):
            if yn[i0]==self.ny_Can:                                  
                jy2[i0]=np.array([-9999])    
            elif yn[i0]==1 and (xn[i0]>1 and xn[i0]<self.nx_Can) and (zn[i0]>1 and zn[i0]<self.nz_Can):     
                jy2[i0]=np.array([-9999])
            else:
                ind0_temp = node[yn==yn[i0]+1] -1
                jy2[i0] = np.array([x for x in ind0_temp if (xn[x]==xn[i0] and zn[x]==zn[i0]) ],dtype=int) + 1
        for i0 in np.arange(self.ntotal_4T):
            if zn[i0]==1:                                  
                jz1[i0]=np.array([-9999])    
            #elif zn[i0]==self.ny_Can and (xn[i0]>1 and xn[i0]<self.nx_Can) and (yn[i0]>1 and yn[i0]<self.ny_Can):
            elif zn[i0]==self.nz_Can and (xn[i0]>1 and xn[i0]<self.nx_Can) and (yn[i0]>1 and yn[i0]<self.ny_Can):     
                jz1[i0]=np.array([-9999]) 
            else:
                ind0_temp = node[zn==zn[i0]-1] -1
                jz1[i0] = np.array([x for x in ind0_temp if (xn[x]==xn[i0] and yn[x]==yn[i0]) ],dtype=int) + 1
        for i0 in np.arange(self.ntotal_4T):
            if zn[i0]==self.nz_Can:                                  
                jz2[i0]=np.array([-9999])    
            elif zn[i0]==1 and (xn[i0]>1 and xn[i0]<self.nx_Can) and (yn[i0]>1 and yn[i0]<self.ny_Can):     
                jz2[i0]=np.array([-9999]) 
            else:
                ind0_temp = node[zn==zn[i0]+1] -1
                jz2[i0] = np.array([x for x in ind0_temp if (xn[x]==xn[i0] and yn[x]==yn[i0]) ],dtype=int) + 1
                
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
                
        V = np.delete(V,ind0_inner_block).reshape(-1,1)                

        return node, xn, yn, zn, mat, xi, yi, zi, V,                                                                                                                 \
               jx1, jx2, jy1, jy2, jz1, jz2, ind0_jx1, ind0_jx2, ind0_jy1, ind0_jy2, ind0_jz1, ind0_jz2,                                                                                       \
               ind0_jx1NaN, ind0_jx2NaN, ind0_jy1NaN, ind0_jy2NaN, ind0_jz1NaN, ind0_jz2NaN, ind0_jx1NonNaN, ind0_jx2NonNaN, ind0_jy1NonNaN, ind0_jy2NonNaN, ind0_jz1NonNaN, ind0_jz2NonNaN 
    #########################################################   
    #  functions for weighted avg & std of nodes temperature#
    #########################################################
    def fun_weighted_avg_and_std(self, T, rou_c_V_weights):
        weight_avg=np.average(self.T3_4T_ALL,weights=self.rou_c_V_weights)
        weight_std=np.sqrt( np.average( (self.T3_4T_ALL-weight_avg)**2,weights=self.rou_c_V_weights ) )
        T_Delta=np.max(T)-np.min(T)
        return (weight_avg,weight_std,T_Delta)
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
                       
            h_4T[ self.ind0_Geo_back_4T,0 ]= self.h_Can                                                                            
            h_4T[ self.ind0_Geo_back_4T,1 ]= self.h_inner_jellyroll                                                                            

            h_4T[ self.ind0_Geo_front_4T,1 ]= self.h_Can                                                                                 
            h_4T[ self.ind0_Geo_front_4T,0 ]= self.h_inner_jellyroll                                                                                 

            h_4T[ self.ind0_Geo_top_4T,2 ]= self.h_Can                                                                                      
            h_4T[ self.ind0_Geo_top_4T,3 ]= self.h_inner_jellyroll                                                                                      

            h_4T[ self.ind0_Geo_bottom_4T,3 ]= self.h_Can                                                                                      
            h_4T[ self.ind0_Geo_bottom_4T,2 ]= self.h_inner_jellyroll                                                                                       

            h_4T[ self.ind0_Geo_left_4T,4 ]= self.h_Can                                                                                                                  
            h_4T[ self.ind0_Geo_left_4T,5 ]= self.h_inner_jellyroll                                                                                                                  
                     
            h_4T[ self.ind0_Geo_right_4T,5 ]= self.h_Can                                                                                    
            h_4T[ self.ind0_Geo_right_4T,4 ]= self.h_inner_jellyroll                                                                                    
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                 

            Tconv_4T[ self.ind0_Geo_back_4T,0 ]= self.T_cooling                                                                                           
            Tconv_4T[ self.ind0_Geo_back_4T,1 ]= self.T_cooling                                                                                           

            Tconv_4T[ self.ind0_Geo_front_4T,1 ]= self.T_cooling                                                                                               
            Tconv_4T[ self.ind0_Geo_front_4T,0 ]= self.T_cooling                                                                                               
           
            Tconv_4T[ self.ind0_Geo_top_4T,2 ]= self.T_cooling                                                                                        
            Tconv_4T[ self.ind0_Geo_top_4T,3 ]= self.T_cooling                                                                                        

            Tconv_4T[ self.ind0_Geo_bottom_4T,3 ]= self.T_cooling                                                                                             
            Tconv_4T[ self.ind0_Geo_bottom_4T,2 ]= self.T_cooling                                                                                             
    
            Tconv_4T[ self.ind0_Geo_left_4T,4 ]= self.T_cooling                                                                               
            Tconv_4T[ self.ind0_Geo_left_4T,5 ]= self.T_cooling                                                                               
                             
            Tconv_4T[ self.ind0_Geo_right_4T,5 ]= self.T_cooling                                                                              
            Tconv_4T[ self.ind0_Geo_right_4T,4 ]= self.T_cooling                                                                              
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top_4T,self.ind0_Geo_bottom_4T,self.ind0_Geo_back_4T,self.ind0_Geo_front_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                                      #all the other nodes            
        if self.status_TabSurface_Scheme=='BaseCool_Prismatic_Cell1':
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
            T3_4T[self.ind0_Geo_bottom_4T]=self.T_cooling                                                                                              #BC on right nodes             
    
            #-----------------------------get all constrained node number
            ind0_BCtem=self.ind0_Geo_bottom_4T                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int) 
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])
                       
            h_4T[ self.ind0_Geo_back_4T,0 ]= self.h_Can                                                                            
            h_4T[ self.ind0_Geo_back_4T,1 ]= self.h_inner_jellyroll                                                                            

            h_4T[ self.ind0_Geo_front_4T,1 ]= self.h_Can                                                                                 
            h_4T[ self.ind0_Geo_front_4T,0 ]= self.h_inner_jellyroll                                                                                 

            h_4T[ self.ind0_Geo_top_4T,2 ]= self.h_Can                                                                                      
            h_4T[ self.ind0_Geo_top_4T,3 ]= self.h_inner_jellyroll                                                                                      

#            h_4T[ self.ind0_Geo_bottom_4T,3 ]= self.h_Can                                                                                      
#            h_4T[ self.ind0_Geo_bottom_4T,2 ]= self.h_inner_jellyroll                                                                                       

            h_4T[ self.ind0_Geo_left_4T,4 ]= self.h_Can                                                                                                                  
            h_4T[ self.ind0_Geo_left_4T,5 ]= self.h_inner_jellyroll                                                                                                                  
                     
            h_4T[ self.ind0_Geo_right_4T,5 ]= self.h_Can                                                                                    
            h_4T[ self.ind0_Geo_right_4T,4 ]= self.h_inner_jellyroll                                                                                    
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                 

            Tconv_4T[ self.ind0_Geo_back_4T,0 ]= self.T_cooling                                                                                           
            Tconv_4T[ self.ind0_Geo_back_4T,1 ]= self.T_cooling                                                                                           

            Tconv_4T[ self.ind0_Geo_front_4T,1 ]= self.T_cooling                                                                                               
            Tconv_4T[ self.ind0_Geo_front_4T,0 ]= self.T_cooling                                                                                               
           
            Tconv_4T[ self.ind0_Geo_top_4T,2 ]= self.T_cooling                                                                                        
            Tconv_4T[ self.ind0_Geo_top_4T,3 ]= self.T_cooling                                                                                        

#            Tconv_4T[ self.ind0_Geo_bottom_4T,3 ]= self.T_cooling                                                                                             
#            Tconv_4T[ self.ind0_Geo_bottom_4T,2 ]= self.T_cooling                                                                                             
    
            Tconv_4T[ self.ind0_Geo_left_4T,4 ]= self.T_cooling                                                                               
            Tconv_4T[ self.ind0_Geo_left_4T,5 ]= self.T_cooling                                                                               
                             
            Tconv_4T[ self.ind0_Geo_right_4T,5 ]= self.T_cooling                                                                              
            Tconv_4T[ self.ind0_Geo_right_4T,4 ]= self.T_cooling                                                                              
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top_4T,self.ind0_Geo_back_4T,self.ind0_Geo_front_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                                      #all the other nodes    
        if self.status_TabSurface_Scheme=='SingleSideCond':
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
            T3_4T[self.ind0_Geo_left_4T]=self.T_cooling                                                                                              #BC on right nodes             
    
            #-----------------------------get all constrained node number
            ind0_BCtem=self.ind0_Geo_left_4T                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int) 
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])
                       
            h_4T[ self.ind0_Geo_back_4T,0 ]= self.h_Can                                                                            
            h_4T[ self.ind0_Geo_back_4T,1 ]= self.h_inner_jellyroll                                                                            

            h_4T[ self.ind0_Geo_front_4T,1 ]= self.h_Can                                                                                 
            h_4T[ self.ind0_Geo_front_4T,0 ]= self.h_inner_jellyroll                                                                                 

            h_4T[ self.ind0_Geo_top_4T,2 ]= self.h_Can                                                                                      
            h_4T[ self.ind0_Geo_top_4T,3 ]= self.h_inner_jellyroll                                                                                      

            h_4T[ self.ind0_Geo_bottom_4T,3 ]= self.h_Can                                                                                      
            h_4T[ self.ind0_Geo_bottom_4T,2 ]= self.h_inner_jellyroll                                                                                       

#            h_4T[ self.ind0_Geo_left_4T,4 ]= self.h_Can                                                                                                                  
#            h_4T[ self.ind0_Geo_left_4T,5 ]= self.h_inner_jellyroll                                                                                                                  
                     
            h_4T[ self.ind0_Geo_right_4T,5 ]= self.h_Can                                                                                    
            h_4T[ self.ind0_Geo_right_4T,4 ]= self.h_inner_jellyroll                                                                                    
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                 

            Tconv_4T[ self.ind0_Geo_back_4T,0 ]= self.T_cooling                                                                                           
            Tconv_4T[ self.ind0_Geo_back_4T,1 ]= self.T_cooling                                                                                           

            Tconv_4T[ self.ind0_Geo_front_4T,1 ]= self.T_cooling                                                                                               
            Tconv_4T[ self.ind0_Geo_front_4T,0 ]= self.T_cooling                                                                                               
           
            Tconv_4T[ self.ind0_Geo_top_4T,2 ]= self.T_cooling                                                                                        
            Tconv_4T[ self.ind0_Geo_top_4T,3 ]= self.T_cooling                                                                                        

            Tconv_4T[ self.ind0_Geo_bottom_4T,3 ]= self.T_cooling                                                                                             
            Tconv_4T[ self.ind0_Geo_bottom_4T,2 ]= self.T_cooling                                                                                             
    
#            Tconv_4T[ self.ind0_Geo_left_4T,4 ]= self.T_cooling                                                                               
#            Tconv_4T[ self.ind0_Geo_left_4T,5 ]= self.T_cooling                                                                               
                             
            Tconv_4T[ self.ind0_Geo_right_4T,5 ]= self.T_cooling                                                                              
            Tconv_4T[ self.ind0_Geo_right_4T,4 ]= self.T_cooling                                                                              
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_right_4T,self.ind0_Geo_top_4T,self.ind0_Geo_bottom_4T,self.ind0_Geo_back_4T,self.ind0_Geo_front_4T))     #all the convection-constrained nodes
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
        ind0_Geo_top_4T=np.where(self.yn_4T==1)[0]                     
        ind0_Geo_bottom_4T=np.where(self.yn_4T==self.ny_Can)[0]            
        ind0_Geo_left_4T=np.where(self.zn_4T==1)[0]                    
        ind0_Geo_right_4T=np.where(self.zn_4T==self.nz_Can)[0]             
        ind0_Geo_front_4T=np.where(self.xn_4T==self.nx_Can)[0]                   
        ind0_Geo_back_4T=np.where(self.xn_4T==1)[0]
        ind0_Geo_Can_Probe20_4T = int(np.size(ind0_Geo_left_4T)/2)              
               
        temp_1 = np.array([x for x in ind0_Geo_left_4T if self.xn_4T[x] >= self.nx_cylinder+1 and self.xn_4T[x] <= self.nx_Can-self.nx_cylinder and self.yn_4T[x] >= 2 and self.yn_4T[x] <= self.ny_Can-1 ],dtype=int)
        item = 'ind0_Geo_interface_2_can'
        setattr(self,item,temp_1)

        temp_2 = np.array([x for x in ind0_Geo_right_4T if self.xn_4T[x] >= self.nx_cylinder+1 and self.xn_4T[x] <= self.nx_Can-self.nx_cylinder and self.yn_4T[x] >= 2 and self.yn_4T[x] <= self.ny_Can-1 ],dtype=int)
        item = 'ind0_Geo_interface_3_can'
        setattr(self,item,temp_2)
        
        (
        self.ind0_Geo_top_4T, 
        self.ind0_Geo_bottom_4T, 
        self.ind0_Geo_left_4T, 
        self.ind0_Geo_right_4T, 
        self.ind0_Geo_front_4T, 
        self.ind0_Geo_back_4T,
        self.ind0_Geo_Can_Probe20_4T
        )=(
        ind0_Geo_top_4T, 
        ind0_Geo_bottom_4T, 
        ind0_Geo_left_4T, 
        ind0_Geo_right_4T, 
        ind0_Geo_front_4T, 
        ind0_Geo_back_4T,
        ind0_Geo_Can_Probe20_4T
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
        Lamda_4T=self.Lamda_Can * np.ones([self.ntotal_4T,6])                                                                                                        #Lamda_4T shape is (63,6). λ term in 6-node stencil
        #-----------------------------------fill in RouXc_4T
        RouXc_4T=self.rouXc_Can * np.ones([self.ntotal_4T,1])                                                                                                   #RouXc_4T shape is (63,1). ρc term in 6-node stencil
        #-----------------------------------fill in Delta_x1_4T
        Delta_x1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δx1 for each node
        Delta_x1_4T[self.ind0_jx1NonNaN_4T]=self.xi_4T[self.ind0_jx1NonNaN_4T] - self.xi_4T[self.jx1_4T[self.ind0_jx1NonNaN_4T]-1]
        Delta_x1_4T[self.xn_4T==2] = self.dx
        Delta_x1_4T[self.xn_4T==self.nx_Can] = self.dx
        Delta_x1_4T[self.ind0_jx1NaN_4T]=np.nan
    
        Delta_x2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δx2 for each node
        Delta_x2_4T[self.ind0_jx2NonNaN_4T]=self.xi_4T[self.jx2_4T[self.ind0_jx2NonNaN_4T]-1] - self.xi_4T[self.ind0_jx2NonNaN_4T]
        Delta_x2_4T[self.xn_4T==1] = self.dx
        Delta_x2_4T[self.xn_4T==self.nx_Can-1] = self.dx
        Delta_x2_4T[self.ind0_jx2NaN_4T]=np.nan
    
        Delta_y1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δy1 for each node
        Delta_y1_4T[self.ind0_jy1NonNaN_4T]=self.yi_4T[self.ind0_jy1NonNaN_4T] - self.yi_4T[self.jy1_4T[self.ind0_jy1NonNaN_4T]-1]
        Delta_y1_4T[self.yn_4T==2] = self.dy
        Delta_y1_4T[self.yn_4T==self.ny_Can] = self.dy
        Delta_y1_4T[self.ind0_jy1NaN_4T]=np.nan
    
        Delta_y2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δy2 for each node
        Delta_y2_4T[self.ind0_jy2NonNaN_4T]=self.yi_4T[self.jy2_4T[self.ind0_jy2NonNaN_4T]-1] - self.yi_4T[self.ind0_jy2NonNaN_4T]
        Delta_y2_4T[self.yn_4T==1] = self.dy
        Delta_y2_4T[self.yn_4T==self.ny_Can-1] = self.dy
        Delta_y2_4T[self.ind0_jy2NaN_4T]=np.nan
    
        Delta_z1_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δz1 for each node
        Delta_z1_4T[self.ind0_jz1NonNaN_4T]=self.zi_4T[self.ind0_jz1NonNaN_4T] - self.zi_4T[self.jz1_4T[self.ind0_jz1NonNaN_4T]-1]
        Delta_z1_4T[self.zn_4T==2] = self.dz
        Delta_z1_4T[self.zn_4T==self.nz_Can] = self.dz
        Delta_z1_4T[self.ind0_jz1NaN_4T]=np.nan
    
        Delta_z2_4T=np.nan*np.zeros([self.ntotal_4T])                                                                                      #get Δz2 for each node
        Delta_z2_4T[self.ind0_jz2NonNaN_4T]=self.zi_4T[self.jz2_4T[self.ind0_jz2NonNaN_4T]-1] - self.zi_4T[self.ind0_jz2NonNaN_4T]
        Delta_z2_4T[self.zn_4T==1] = self.dz
        Delta_z2_4T[self.zn_4T==self.nz_Can-1] = self.dz
        Delta_z2_4T[self.ind0_jz2NaN_4T]=np.nan
        
        delta_x1_4T=Delta_x1_4T.copy()                                                                                                     #get δx1 for each node
        delta_x1_4T[self.xn_4T==1] = self.delta_Can_real
        delta_x1_4T[self.xn_4T==self.nx_Can] = self.delta_Can_real
    
        delta_x2_4T=Delta_x2_4T.copy()                                                                                                     #get δx2 for each node
        delta_x2_4T[self.xn_4T==1] = self.delta_Can_real
        delta_x2_4T[self.xn_4T==self.nx_Can] = self.delta_Can_real
        
        delta_y1_4T=Delta_y1_4T.copy()                                                                                                     #get δy1 for each node
        delta_y1_4T[self.yn_4T==1] = self.delta_Can_real
        delta_y1_4T[self.yn_4T==self.ny_Can] = self.delta_Can_real
    
        delta_y2_4T=Delta_y2_4T.copy()                                                                                                     #get δy2 for each node
        delta_y2_4T[self.yn_4T==1] = self.delta_Can_real
        delta_y2_4T[self.yn_4T==self.ny_Can] = self.delta_Can_real

        delta_z1_4T=Delta_z1_4T.copy()                                                                                                     #get δz1 for each node
        delta_z1_4T[self.zn_4T==1] = self.delta_Can_real
        delta_z1_4T[self.zn_4T==self.nz_Can] = self.delta_Can_real
    
        delta_z2_4T=Delta_z2_4T.copy()                                                                                                     #get δz2 for each node
        delta_z2_4T[self.zn_4T==1] = self.delta_Can_real
        delta_z2_4T[self.zn_4T==self.nz_Can] = self.delta_Can_real
        
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
    ############### function for energy check ###############
    #########################################################
    def fun_Echeck(self,step):
#        global S_stencil_4T_ALL
    
        self.Eext_Total_BCconv_record[step] = np.nansum( self.h_4T_ALL * self.S_stencil_4T_ALL * -(self.T3_4T_ALL.reshape(-1,1)-self.T_cooling) ) *self.dt + self.Eext_Total_BCconv_record[step-1]

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
                imageio.mimsave('15,png', self.frames5, 'GIF', duration=0.2)
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
        status_PlotSlide = 'No'
        if status_PlotSlide == 'Yes':
            xn_PlotSlice = int(self.nx_Can/2)
            zn_PlotSlice = int(self.nz_Can/2)
        
        self.X_Module = XYZ_Module[0]         #this cell's X location in Module
        self.Y_Module = XYZ_Module[1]
        self.Z_Module = XYZ_Module[2]
        rotate_angle = 0*np.pi/2                
        #===================================prepare X,Y,Z,C for visualization
        self.yi_plot_4T_ALL=self.yi_4T_ALL.copy()
        #-----------------------plot front surface
        self.ind0_plot_front=self.ind0_Geo_front_4T.reshape(self.nz_Can,self.ny_Can).T
        self.X1= self.zi_4T_ALL[self.ind0_plot_front]
        self.Y1= -self.xi_4T_ALL[self.ind0_plot_front]
        self.Z1= -self.yi_4T_ALL[self.ind0_plot_front]
        self.C1=plot_Variable_ALL[self.ind0_plot_front]
        temp1 = self.X1.copy(); temp2 = self.Y1.copy()
        self.X1 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y1 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X1 += self.X_Module; self.Y1 += self.Y_Module; self.Z1 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot back surface
        self.ind0_plot_back=self.ind0_Geo_back_4T.reshape(self.nz_Can,self.ny_Can).T
        if status_PlotSlide == 'Yes':
            self.ind0_plot_back = self.ind0_plot_back[:,:zn_PlotSlice]
        self.X2= self.zi_4T_ALL[self.ind0_plot_back]
        self.Y2= -self.xi_4T_ALL[self.ind0_plot_back]
        self.Z2= -self.yi_4T_ALL[self.ind0_plot_back]
        self.C2=plot_Variable_ALL[self.ind0_plot_back]
        temp1 = self.X2.copy(); temp2 = self.Y2.copy()
        self.X2 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y2 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X2 += self.X_Module; self.Y2 += self.Y_Module; self.Z2 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot left surface
        self.ind0_plot_left=self.ind0_Geo_left_4T.reshape(self.ny_Can,self.nx_Can)
        self.X3= self.zi_4T_ALL[self.ind0_plot_left]
        self.Y3= -self.xi_4T_ALL[self.ind0_plot_left]
        self.Z3= -self.yi_4T_ALL[self.ind0_plot_left]
        self.C3=plot_Variable_ALL[self.ind0_plot_left]
        temp1 = self.X3.copy(); temp2 = self.Y3.copy()
        self.X3 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y3 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X3 += self.X_Module; self.Y3 += self.Y_Module; self.Z3 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot right surface
        self.ind0_plot_right=self.ind0_Geo_right_4T.reshape(self.ny_Can,self.nx_Can)
        if status_PlotSlide == 'Yes':
            self.ind0_plot_right = self.ind0_plot_right[:,xn_PlotSlice:]
        self.X4= self.zi_4T_ALL[self.ind0_plot_right]
        self.Y4= -self.xi_4T_ALL[self.ind0_plot_right]
        self.Z4= -self.yi_4T_ALL[self.ind0_plot_right]
        self.C4=plot_Variable_ALL[self.ind0_plot_right]
        temp1 = self.X4.copy(); temp2 = self.Y4.copy()
        self.X4 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y4 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X4 += self.X_Module; self.Y4 += self.Y_Module; self.Z4 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot top surface
        self.ind0_plot_top=self.ind0_Geo_top_4T.reshape(self.nz_Can,self.nx_Can).T
        if status_PlotSlide == 'Yes':
            self.ind0_plot_top = self.ind0_plot_top[xn_PlotSlice:,:]
        self.X5= self.zi_4T_ALL[self.ind0_plot_top]
        self.Y5= -self.xi_4T_ALL[self.ind0_plot_top]
        self.Z5= -self.yi_4T_ALL[self.ind0_plot_top]
        self.C5=plot_Variable_ALL[self.ind0_plot_top]
        temp1 = self.X5.copy(); temp2 = self.Y5.copy()
        self.X5 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y5 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X5 += self.X_Module; self.Y5 += self.Y_Module; self.Z5 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot bottom surface
        self.ind0_plot_bottom=self.ind0_Geo_bottom_4T.reshape(self.nz_Can,self.nx_Can).T
        self.X6= self.zi_4T_ALL[self.ind0_plot_bottom]
        self.Y6= -self.xi_4T_ALL[self.ind0_plot_bottom]
        self.Z6= -self.yi_4T_ALL[self.ind0_plot_bottom]
        self.C6=plot_Variable_ALL[self.ind0_plot_bottom]
        temp1 = self.X6.copy(); temp2 = self.Y6.copy()
        self.X6 = temp1 * np.cos(rotate_angle) + temp2 * np.sin(rotate_angle)
        self.Y6 = temp2 * np.cos(rotate_angle) - temp1 * np.sin(rotate_angle)
        self.X6 += self.X_Module; self.Y6 += self.Y_Module; self.Z6 += self.Z_Module    #Get cells' spatial locations in Module
            

        #===================================visualization
#       self.fig = mlab.figure(bgcolor=(1,1,1))    
#       self.vmin=self.plot_Variable_ALL.min();  self.vmax=self.plot_Variable_ALL.max()
        self.surf1 = mlab.mesh(self.X1, self.Y1, self.Z1, scalars=self.C1, colormap='coolwarm', opacity=0.5)
        self.surf2 = mlab.mesh(self.X2, self.Y2, self.Z2, scalars=self.C2, colormap='coolwarm', opacity=0.5)
        self.surf3 = mlab.mesh(self.X3, self.Y3, self.Z3, scalars=self.C3, colormap='coolwarm', opacity=0.5)
        self.surf4 = mlab.mesh(self.X4, self.Y4, self.Z4, scalars=self.C4, colormap='coolwarm', opacity=0.5)
        self.surf5 = mlab.mesh(self.X5, self.Y5, self.Z5, scalars=self.C5, colormap='coolwarm', opacity=0.5)
        self.surf6 = mlab.mesh(self.X6, self.Y6, self.Z6, scalars=self.C6, colormap='coolwarm', opacity=0.5)

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
    
    
    
    
    
    
    
    
    
    
    
