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

class Pouch():

          
    #########################################################   
    ################## function for matrix1 #################
    #########################################################
    def fun_matrix1(self):
        node=np.arange(1, self.ntotal+1)
        xn_unit=np.linspace(1,self.nx,self.nx,dtype=int)     #repeating node number for xn
        yn_unit=np.linspace(1,self.ny,self.ny,dtype=int)     #repeating node number for yn
        zn_unit=np.linspace(1,self.nz,self.nz,dtype=int)     #repeating node number for zn
        xn=np.tile(xn_unit,self.ny*self.nz)
        yn=np.repeat(yn_unit,self.nx*self.nz) 
        zn=np.repeat(zn_unit,self.nx); zn=np.tile(zn,self.ny)
          
        mat_unit=np.array([1,3,2,4])                      #repeating node number for mat
        mat_tmp1=np.repeat(mat_unit,[self.nx,self.ne*self.nx,self.nx,self.ne*self.nx])  #within one layer Al,El_b,Cu,El_r [1,3,2,4]
        mat_tmp2=np.tile(mat_tmp1,self.nstack)                 #within two layers Al,El_b,Cu,El_r [1,3,2,4,1,3,2,4]  yn=1
        mat_tmp3=mat_tmp2[:(self.nz*self.nx)]                       #within two layers after trimming Al,El_b,Cu,El_r [1,3,2,4,1,3,2]  yn=1
        mat=np.tile(mat_tmp3,self.ny)
    
        Al=np.where(mat==1)[0]; Cu=np.where(mat==2)[0]; Elb=np.where(mat==3)[0]; Elr=np.where(mat==4)[0]   #for case of 192 nodes, ind from 0 to 191
    
        xi=np.zeros(self.ntotal); yi=np.zeros(self.ntotal); zi=np.zeros(self.ntotal)
        xi=(xn-1)*self.Lx/(self.nx-1)
        yi=(yn-1)*self.Ly/(self.ny-1)    
        zi[Al]=(zn[Al]-1)//(self.ne+1) * (self.delta_El+self.delta_Al/2+self.delta_Cu/2) + self.delta_Al/2
        zi[Cu]=(zn[Cu]-1)//(self.ne+1) * (self.delta_El+self.delta_Al/2+self.delta_Cu/2) + self.delta_Al/2
        zi[Elb]=(zn[Elb]-1)//(self.ne+1) * (self.delta_El+self.delta_Al/2+self.delta_Cu/2) + self.delta_Al/2+self.delta_El/2 + self.delta_Al/2
        zi[Elr]=(zn[Elr]-1)//(self.ne+1) * (self.delta_El+self.delta_Al/2+self.delta_Cu/2) + self.delta_Cu/2+self.delta_El/2 + self.delta_Al/2
    
        jx1=np.zeros(self.ntotal,dtype=int)            #initialize left-neighbor node number in x direction
        jx2=np.zeros(self.ntotal,dtype=int)            #initialize right-neighbor node number in x direction
        jy1=np.zeros(self.ntotal,dtype=int)            #initialize up-neighbor node number in y direction
        jy2=np.zeros(self.ntotal,dtype=int)            #initialize down-neighbor node number in y direction
        jz1=np.zeros(self.ntotal,dtype=int)            #initialize inner-neighbor node number in z direction
        jz2=np.zeros(self.ntotal,dtype=int)            #initialize outer-neighbor node number in z direction
        
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
        V=np.zeros([self.ntotal,1])  #V is volume of each node, in the form of 1,2...ntotal. When two nodes belong to the same element, they have the same volume i.e. the elementary volume
        Sxy0=(self.Lx/(self.nx-1)) * (self.Ly/(self.ny-1))
        V[Al]  = Sxy0 * self.delta_Al
        V[Cu]  = Sxy0 * self.delta_Cu
        V[Elb] = Sxy0 * self.delta_El
        V[Elr] = Sxy0 * self.delta_El
        
        V_ele= Sxy0*self.delta_El*np.ones([self.nECN,1])    #V_ele is volume of each ECN element, in the form of 1,2...nECN 
        Axy_ele=Sxy0*np.ones([self.nECN,1])            #Axy_ele is Electrodes cross section area of each ECN element, in the form of 1,2...nECN            
        return node, xn, yn, zn, mat, xi, yi, zi, V, V_ele, Axy_ele, Al, Cu, Elb, Elr,                                                                                                                  \
               jx1, jx2, jy1, jy2, jz1, jz2, ind0_jx1, ind0_jx2, ind0_jy1, ind0_jy2, ind0_jz1, ind0_jz2,                                                                                       \
               ind0_jx1NaN, ind0_jx2NaN, ind0_jy1NaN, ind0_jy2NaN, ind0_jz1NaN, ind0_jz2NaN, ind0_jx1NonNaN, ind0_jx2NonNaN, ind0_jy1NonNaN, ind0_jy2NonNaN, ind0_jz1NonNaN, ind0_jz2NonNaN 
    #########################################################   
    ######## function for ρ_bulk, ρc_bulk and c_bulk ########
    ########      λx_bulk, λy_bulk and λz_bulk       ########
    #########################################################
    def fun_bulk_4T(self):
#        global rou_bulk, c_bulk, rouXc_bulk, V_sum
        self.V_sum=np.sum(self.V_stencil_4T_ALL)
        self.rou_bulk=np.sum(self.rou_Al*(self.V_stencil_4T_ALL[self.Al_4T]/self.V_sum)) + np.sum(self.rou_Cu*(self.V_stencil_4T_ALL[self.Cu_4T]/self.V_sum)) + np.sum(self.rou_El*(self.V_stencil_4T_ALL[self.Elb_4T]/self.V_sum)) + np.sum(self.rou_El*(self.V_stencil_4T_ALL[self.Elr_4T]/self.V_sum))
        self.rouXc_bulk=np.sum(self.rou_Al*self.c_Al*(self.V_stencil_4T_ALL[self.Al_4T]/self.V_sum)) + np.sum(self.rou_Cu*self.c_Cu*(self.V_stencil_4T_ALL[self.Cu_4T]/self.V_sum)) + np.sum(self.rouXc_El*(self.V_stencil_4T_ALL[self.Elb_4T]/self.V_sum)) + np.sum(self.rouXc_El*(self.V_stencil_4T_ALL[self.Elr_4T]/self.V_sum))
        self.c_bulk=self.rouXc_bulk/self.rou_bulk
#        global Lamda_bulk_x, Lamda_bulk_y, Lamda_bulk_z
        self.Lamda_bulk_x=(self.Lamda_Cu*self.delta_Cu_real+2*self.Lamda_El_x*self.delta_El_real+self.Lamda_Al*self.delta_Al_real)/(self.delta_Cu_real+2*self.delta_El_real+self.delta_Al_real)
        self.Lamda_bulk_y=self.Lamda_bulk_x
        self.Lamda_bulk_z=(self.delta_Cu_real+2*self.delta_El_real+self.delta_Al_real)/(self.delta_Cu_real/self.Lamda_Cu+2*self.delta_El_real/self.Lamda_El_z+self.delta_Al_real/self.Lamda_Al)
    #########################################################   
    ######## function for weight and energy density #########
    #########################################################
    def fun_weight_and_energy_density(self): 
#        global weight_Jellyroll, weight_Casing, weight_Total, EnergyDensity     
        self.weight_Al = self.Lx_electrodes_real*self.Ly_electrodes_real*self.delta_Al_real*self.rou_Al*self.nstack_real
        self.weight_Cu = self.Lx_electrodes_real*self.Ly_electrodes_real*self.delta_Cu_real*self.rou_Cu*self.nstack_real
        self.weight_Elb = self.Lx_electrodes_real*self.Ly_electrodes_real*self.nstack_real* ( self.delta_Ca_real*self.rou_Ca + self.delta_Sep_real*self.rou_Sep + self.delta_An_real*self.rou_An )
        self.weight_Elr = self.Lx_electrodes_real*self.Ly_electrodes_real*(self.nstack_real-1)* ( self.delta_Ca_real*self.rou_Ca + self.delta_Sep_real*self.rou_Sep + self.delta_An_real*self.rou_An )
    
        self.weight_Jellyroll = self.weight_Al + self.weight_Cu + self.weight_Elb + self.weight_Elr
    
        self.weight_Casing = 2 * self.Lx_cell*self.Ly_cell*( self.SpecSheet_Casing_delta_Polyamide*self.SpecSheet_Casing_rou_Polyamide + self.SpecSheet_Casing_delta_Al*self.SpecSheet_Casing_rou_Al + self.SpecSheet_Casing_delta_Polypropylene*self.SpecSheet_Casing_rou_Polypropylene)    
    
        self.weight_Total = self.weight_Jellyroll + self.weight_Casing
    
        self.EnergyDensity = self.SpecSheet_Energy/self.weight_Total
    #########################################################   
    #  functions for weighted avg & std of nodes temperature#
    #########################################################
    def fun_weighted_avg_and_std(self, T, rou_c_V_weights):
        weight_avg=np.average(self.T3_4T_ALL,weights=self.rou_c_V_weights)
        weight_std=np.sqrt( np.average( (self.T3_4T_ALL-weight_avg)**2,weights=self.rou_c_V_weights ) )
        T_Delta=np.max(self.T3_4T_ALL)-np.min(self.T3_4T_ALL)
        return (weight_avg,weight_std,T_Delta)
    #########################################################   
    ##################  function for pairs  #################
    #########################################################
    def fun_pre_matrixC(self):        #aim of this part is to get all _pair
#        global RAl_pair, RCu_pair, R0_pair, RC_pair, Ei_pair
        self.RAl_pair=np.zeros([self.nRAl,3]); counter0=0       #RAl case: two linking nodes (0index). For example, [[0,1,RAl],[1,2,RAl],[2,3,RAl],[3,40,RAl],[20,21,RAl]...]
                                                           #means node1,2 is RAl; node2,3 is RAl; node3,4 is RAl; node4,41 is RAl; node21,22 is RAl
                                                           #Note: first number is no larger than second number: [node1,node2,R0], node1 <= node2
        self.RCu_pair=np.zeros([self.nRCu,3]); counter1=0       #RCu case: two linking nodes (0index). For example, [[20,21,RAl],[21,22,RAl],[22,23,RAl],[23,60,RAl],[60,61,RAl]...]
                                                           #means node21,22 is RCu; nod22,23 is RCu; node23,24 is RCu; node24,61 is RCu; node61,62 is RCu
                                                           #Note: first number is no larger than second number: [node1,node2,R0], node1 <= node2                                                      
        temp1=np.arange(0,self.nRC+2); temp2=np.roll(temp1,-1); temp3=np.append(temp1,temp2)
        ECN_pair_lib=np.reshape(temp3,(2,self.nRC+2)).T    #for 3 RCs, node link cases: array([[0,1],[1,2],[2,3],[3,4],[4,0]]) for R0, R1, R2, R3 and Ei respectively     
        self.R0_pair=np.zeros([self.nECN,3]); counter2=0             #R0 case: two linking nodes (0index). For example, [[0,4,R0],[1,5,R0],[2,6,R0],[3,7,R0],[20,24,R0]...]
                                                           #means node1,5 is R0; node2,6 is R0; node3,7 is R0; node4,8 is R0; node21,26 is R0
                                                           #Note: first number is no larger than second number: [node1,node2,R0], node1 <= node2
        self.RC_pair=np.zeros([self.nRC*self.nECN,5]); counter3=0         #RC case: two linking nodes (0index). For example, [[4,8,R1,C1,indRC],[5,9,R1,C1,indRC],[6,10,R1,C1,indRC],[7,11,R1,C1,indRC],[8,12,R2,C2,indRC]...]
                                                           #means node5,9 is R1,C1; node6,10 is R1,C1; node7,11 is R1,C1; node8,12 is R1,C1; node9,13 is R2,C2   
                                                           #Note: first number is no larger than second number: [node1,node2,R1,C1,indRC], node1 <= node2                                             
        self.Ei_pair=np.zeros([self.nECN,3]); counter4=0             #Ei case: two linking nodes (0index). For example, [[16,20,Ei],[17,21,Ei],[18,22,Ei],[19,23,Ei],[36,40,Ei]...]
                                                           #means node17,21 is Ei; node18,22 is Ei; node19,23 is Ei; node20,24 is Ei; node37,41 is Ei
                                                           #Note: first number is no larger than second number: [node1,node2,Ei], node1 <= node2
        #----------------------getting MatrixC1-----------------------
        for i0 in (self.node-1):      
            for j0 in range(i0+1,self.ntotal):
                isneighbor = ((self.node[i0]==self.jx1[j0]) or (self.node[i0]==self.jx2[j0]) or (self.node[i0]==self.jy1[j0]) or (self.node[i0]==self.jy2[j0]) or (self.node[i0]==self.jz1[j0]) or (self.node[i0]==self.jz2[j0]))
    
                if isneighbor:        # for elements in upper triangle and non-zero elements in MatrixC
                    if self.mat[i0]==self.mat[j0] and (self.mat[i0]==1 or self.mat[i0]==2):    #i. RAl or RCu case           
                        if self.mat[i0]==1:
                            delta=self.delta_Al; Conductivity=self.Conductivity_Al
                        if self.mat[i0]==2:
                            delta=self.delta_Cu; Conductivity=self.Conductivity_Cu
                        
                        if self.yi[i0]==self.yi[j0]:                     #1. horizontal resistance     yi[j0]>yi[i0] because j0 is larger than i0
                            L=self.Lx/(self.nx-1)    
                            A=delta*self.Ly/(self.ny-1)
                            R=L/A/Conductivity           # ※ difference between Cylindrical code version and Pouch code version, refer to p217 ECM52 and p125                 
                        else:                                  #2. vertical resistance     yi[j0]>yi[i0] because j0 is larger than i0
                            L=self.Ly/(self.ny-1)
                            A=delta*self.Lx/(self.nx-1)
                            R=L/A/Conductivity                   
                        if self.mat[i0]==1:
                            self.RAl_pair[counter0,0]=i0; self.RAl_pair[counter0,1]=j0; self.RAl_pair[counter0,2]=R; counter0=counter0+1
                        if self.mat[i0]==2:
                            self.RCu_pair[counter1,0]=i0; self.RCu_pair[counter1,1]=j0; self.RCu_pair[counter1,2]=R; counter1=counter1+1
                        
                    elif self.xn[i0]==self.xn[j0] and self.yn[i0]==self.yn[j0]:                #ii. Elb and Elr case
                        temp4=(self.zn[i0]-1)%(self.ne+1); temp5=(self.zn[j0]-1)%(self.ne+1)            #no. in lumped RCs. For example, temp3 and temp4 can be 0,1,2,3,4                                 
                        indRC=np.where((ECN_pair_lib==[temp4,temp5]).all(1))[0]             #find the row index of link case [temp1,temp2] in ECN_pair_lib    np.where((ECN_pair_lib==[temp4,temp5]).all(1)) returns tuple, so add [0] in the end
                        if indRC==0:        #1. case of R0
                            self.R0_pair[counter2,0]=i0; self.R0_pair[counter2,1]=j0; counter2=counter2+1
                        elif indRC==self.nRC+1:  #2. case of Ei
                            self.Ei_pair[counter4,0]=i0; self.Ei_pair[counter4,1]=j0; counter4=counter4+1          #for later use in computing RectangleC2                        
                        else:               #3. case of R123 and C123                    
                            self.RC_pair[counter3,0]=i0; self.RC_pair[counter3,1]=j0; self.RC_pair[counter3,4]=indRC; counter3=counter3+1
    
#        global ind0_EleIsElb, ind0_EleIsElr, ind0_AlOfElb, ind0_CuOfElb, ind0_AlOfElr, ind0_CuOfElr, ind0_ele_RC_pair, ind0_ele_RC_pair_4T
        self.ind0_EleIsElb = np.where(self.mat[self.Ei_pair[:,0].astype(int)]==3)[0];   self.ind0_EleIsElr = np.where(self.mat[self.Ei_pair[:,0].astype(int)]==4)[0]
        self.ind0_AlOfElb = self.jz2[ self.List_ele2node[self.ind0_EleIsElb][:,0] ]-1                    #for all Elb elements (in the increasing order), get ind0 of Al node 
        self.ind0_CuOfElb = self.jz1[ self.List_ele2node[self.ind0_EleIsElb][:,-1] ]-1                   #for all Elb elements (in the increasing order), get ind0 of Cu node 
        self.ind0_AlOfElr = self.jz1[ self.List_ele2node[self.ind0_EleIsElr][:,-1] ]-1                   #for all Elr elements (in the increasing order), get ind0 of Al node 
        self.ind0_CuOfElr = self.jz2[ self.List_ele2node[self.ind0_EleIsElr][:,0] ]-1                    #for all Elr elements (in the increasing order), get ind0 of Cu node 
        self.ind0_ele_RC_pair = np.concatenate(np.split(np.arange(self.nECN*self.nRC).reshape(-1,self.nx).T,self.nECN/self.nx,1))                                         #prepare this for MatrixC_neo and I_neo. RC_pair is not in the sequence of elements, ind0_ele_RC_pair gives the ind0 in element sequence    notebook p62                     
        if self.nRC != 0:
            self.ind0_ele_RC_pair_4T = self.ind0_ele_RC_pair[:,0]                  

    #########################################################   
    ###########      function for Thermal BC      ###########
    #########################################################
    def fun_BC_4T_ALL(self):         # output vector T_4T (after constraining temperature on BC nodes)
#        global T3_4T_ALL, ind0_BCtem_ALL, ind0_BCtem_others_ALL, h_4T_ALL, Tconv_4T_ALL, ind0_BCconv_ALL, ind0_BCconv_others_ALL
        if self.status_TabSurface_Scheme=='TabConv_SurTem':     
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                                  #T_4T initialization, shape is (63,1)        
    
            if self.status_TemBC_smoothening=='No':
                T3_4T[self.ind0_Geo_top2_5_8_11_14_17_20_4T]=35+273.15                                                                                                 #BC on left surface nodes 2,5,8,11,14,17,20
                T3_4T[self.ind0_Geo_bottom44_47_50_53_56_59_62_4T]=35+273.15                                                                                           #BC on right surface nodes 44,47,50,53,56,59,62
                T3_4T[self.ind0_Geo_front23_4T]=self.T_cooling                                                                                                              #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back41_4T]=45+273.15                                                                                                               #BC on back surface nodes 41
            else:
                T3_4T[self.ind0_Geo_top2_5_8_11_14_17_20_4T]=self.T_cooling_smoothened                                                                                      #BC on left surface nodes 2,5,8,11,14,17,20
                T3_4T[self.ind0_Geo_bottom44_47_50_53_56_59_62_4T]=self.T_cooling_smoothened                                                                                #BC on right surface nodes 44,47,50,53,56,59,62
                T3_4T[self.ind0_Geo_front23_4T]=self.T_cooling_smoothened                                                                                                   #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back41_4T]=self.T_cooling_smoothened                                                                                                    #BC on back surface nodes 41            
            #-----------------------------get all constrained node number
            ind0_BCtem=np.concatenate((self.ind0_Geo_top2_5_8_11_14_17_20_4T, self.ind0_Geo_bottom44_47_50_53_56_59_62_4T, self.ind0_Geo_front23_4T, self.ind0_Geo_back41_4T))                                                                         #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                        #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                                   #h_4T initialization, shape is (63,6)
    
            h_4T[ self.ind0_Geo_left_4T,0 ]= 0                                                                                                                        #BC on left nodes
            h_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]=0
            h_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]=0
            h_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]=0
            h_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]=0
    
            h_4T[ self.ind0_Geo_right_4T,1 ]= 0                                                                                                                       #BC on right nodes
            h_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=0
            h_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=0
            h_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=0
            h_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=0
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                               #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= self.T_cooling                                                                                                                        
    
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]= self.T_cooling                                                                                                                        
            Tconv_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]= self.T_cooling                                                                                                                       
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T, self.ind0_Geo_edge1_4_7_10_13_16_19_4T, self.ind0_Geo_edge43_46_49_52_55_58_61_4T, self.ind0_Geo_edge1_22_43_4T, self.ind0_Geo_edge19_40_61_4T, self.ind0_Geo_right_4T, self.ind0_Geo_edge3_6_9_12_15_18_21_4T, self.ind0_Geo_edge45_48_51_54_57_60_63_4T, self.ind0_Geo_edge3_24_45_4T, self.ind0_Geo_edge21_42_63_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='TabTem_SurConv':  
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
            if self.status_TemBC_smoothening=='No':
                T3_4T[self.ind0_Geo_left_4T]=self.T_cooling                                                                                                          #BC on left nodes 
                T3_4T[self.ind0_Geo_right_4T]=self.T_cooling                                                                                                         #BC on right nodes 
            else:
                T3_4T[self.ind0_Geo_left_4T]=self.T_cooling_smoothened                                                                                               #BC on left nodes 
                T3_4T[self.ind0_Geo_right_4T]=self.T_cooling_smoothened                                                                                              #BC on right nodes             
            #-----------------------------get all constrained node number
            ind0_BCtem=np.concatenate((self.ind0_Geo_left_4T, self.ind0_Geo_right_4T))                                                                               #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                 #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                            #h_4T initialization, shape is (63,6)
    
            h_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= 50                                                                                                 #BC on surface nodes        
            h_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= 50
            h_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= 50
            h_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= 50               
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= self.T_cooling                                                                                              #BC on surface nodes        
            Tconv_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= self.T_cooling        
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_top2_5_8_11_14_17_20_4T, self.ind0_Geo_bottom44_47_50_53_56_59_62_4T, self.ind0_Geo_front2_23_44_4T, self.ind0_Geo_back20_41_62_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
    
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
            
            h_4T[ self.ind0_Geo_left_4T,0 ]= 30.0                                                                                                                  #BC on left nodes
            h_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= 30.0
            h_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= 30.0
            h_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= 30.0
            h_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= 30.0        
            h_4T[ self.ind0_Geo_left_Al_4T,0]=30          
                     
            h_4T[ self.ind0_Geo_right_4T,1 ]= 30.0                                                                                                                 #BC on right nodes
            h_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=30.0
            h_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=30.0
            h_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=30.0
            h_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=30.0
            h_4T[ self.ind0_Geo_right_Cu_4T,1]=30    
           
            h_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= 0                                                                                                  #BC on surface nodes        
            h_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= 0
            h_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= 0
            h_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= 0               
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling                                                                                                             #BC on left nodes
            Tconv_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= self.T_cooling        
                             
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_cooling                                                                                                            #BC on right nodes
            Tconv_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=self.T_cooling
           
            Tconv_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= self.T_cooling                                                                                             #BC on surface nodes        
            Tconv_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= self.T_cooling        
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top2_5_8_11_14_17_20_4T,self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,self.ind0_Geo_front23_4T,self.ind0_Geo_back41_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                                      #all the other nodes    
        if self.status_TabSurface_Scheme=='TabCoolAgeing':
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
    
            #-----------------------------get all constrained node number
            ind0_BCtem=[]                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int) 
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])
            
            h_4T[ self.ind0_Geo_left_4T,0 ]= 0.0                                                                                                                  #BC on left nodes
            h_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= 0.0
            h_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= 0.0
            h_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= 0.0
            h_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= 0.0        
            h_4T[ self.ind0_Geo_left_Al_4T,0]=self.Lamda_Al/11e-3 *0.05     
                     
            h_4T[ self.ind0_Geo_right_4T,1 ]= 0.0                                                                                                                 #BC on right nodes
            h_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=0.0
            h_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=0.0
            h_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=0.0
            h_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=0.0
            h_4T[ self.ind0_Geo_right_Cu_4T,1]=self.Lamda_Cu/11e-3 *0.05    
           
            h_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= 0                                                                                                  #BC on surface nodes        
            h_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= 0
            h_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= 0
            h_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= 0               
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling                                                                                                             #BC on left nodes
            Tconv_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= self.T_cooling        
                             
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_cooling                                                                                                            #BC on right nodes
            Tconv_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=self.T_cooling
           
            Tconv_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= self.T_cooling                                                                                             #BC on surface nodes        
            Tconv_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= self.T_cooling        
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top2_5_8_11_14_17_20_4T,self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,self.ind0_Geo_front23_4T,self.ind0_Geo_back41_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                                      #all the other nodes    
        if self.status_TabSurface_Scheme=='AllTem':
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                                  #T_4T initialization, shape is (63,1)        
    
            if self.status_TemBC_smoothening=='No':
                T3_4T[self.ind0_Geo_left_4T]=self.T_cooling                                                                                                                     #BC on left surface nodes 2,8,14,20
                T3_4T[self.ind0_Geo_right_4T]=self.T_cooling                                                                                                                    #BC on right surface nodes 44,50,56,62
                T3_4T[self.ind0_Geo_top2_5_8_11_14_17_20_4T]=self.T_cooling                                                                                                     #BC on front surface nodes 2,5,8,11,14,17,20
                T3_4T[self.ind0_Geo_bottom44_47_50_53_56_59_62_4T]=self.T_cooling                                                                                               #BC on back surface nodes 44,47,50,53,56,59,62
                T3_4T[self.ind0_Geo_front23_4T]=self.T_cooling                                                                                                                  #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back41_4T]=self.T_cooling                                                                                                                   #BC on back surface nodes 41
            else:
                T3_4T[self.ind0_Geo_left_4T]=self.T_cooling_smoothened                                                                                                          #BC on left surface nodes 2,8,14,20
                T3_4T[self.ind0_Geo_right_4T]=self.T_cooling_smoothened                                                                                                         #BC on right surface nodes 44,50,56,62
                T3_4T[self.ind0_Geo_top2_5_8_11_14_17_20_4T]=self.T_cooling_smoothened                                                                                          #BC on front surface nodes 2,5,8,11,14,17,20
                T3_4T[self.ind0_Geo_bottom44_47_50_53_56_59_62_4T]=self.T_cooling_smoothened                                                                                    #BC on back surface nodes 44,47,50,53,56,59,62
                T3_4T[self.ind0_Geo_front23_4T]=self.T_cooling_smoothened                                                                                                       #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back41_4T]=self.T_cooling_smoothened                                                                                                        #BC on back surface nodes 41
    
            #-----------------------------get all constrained node number
            ind0_BCtem=np.concatenate((self.ind0_Geo_left_4T, self.ind0_Geo_right_4T, self.ind0_Geo_top2_5_8_11_14_17_20_4T, self.ind0_Geo_bottom44_47_50_53_56_59_62_4T, self.ind0_Geo_front23_4T, self.ind0_Geo_back41_4T))                                                                         #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                        #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                                   #h_4T initialization, shape is (63,6)
    
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            #-----------------------------get all constrained node number                
            ind0_BCconv=[]
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
        
        if self.status_TabSurface_Scheme=='TabCooling':    
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
    
            #-----------------------------get all constrained node number
            ind0_BCtem=[]                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                 #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                            #h_4T initialization, shape is (63,6)
    
            h_4T[ self.ind0_Geo_left_4T,0 ]= 50                                                                                                                 #BC on left nodes
            h_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= 50
            h_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= 50
            h_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= 50
            h_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= 50        
                             
            h_4T[ self.ind0_Geo_right_4T,1 ]= 50                                                                                                                #BC on right nodes
            h_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=50
            h_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=50
            h_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=50
            h_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=50
           
            h_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= 0                                                                                                  #BC on surface nodes        
            h_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= 0
            h_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= 0
            h_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= 0               
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling                                                                                                      #BC on left nodes
            Tconv_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= self.T_cooling        
                             
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_cooling                                                                                                     #BC on right nodes
            Tconv_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=self.T_cooling
            Tconv_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=self.T_cooling
           
            Tconv_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= self.T_cooling                                                                                      #BC on surface nodes        
            Tconv_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= self.T_cooling        
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top2_5_8_11_14_17_20_4T,self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,self.ind0_Geo_front23_4T,self.ind0_Geo_back41_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='OneSurfaceCoolAgeing':      
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                                  #T_4T initialization, shape is (63,1)        
    
            if self.status_TemBC_smoothening=='No':
                T3_4T[self.ind0_Geo_back_4T]=self.T_cooling                                                                                                               #BC on back surface nodes 41
            else:
                T3_4T[self.ind0_Geo_back_4T]=self.T_cooling_smoothened                                                                                                    #BC on back surface nodes 41            
            #-----------------------------get all constrained node number
            ind0_BCtem=self.ind0_Geo_back_4T                                                                         #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                        #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                                   #h_4T initialization, shape is (63,6)
    
            h_4T[ self.ind0_Geo_front_4T,5 ]= 0                                                                                                                        #BC on left nodes
    
            h_4T[ self.ind0_Geo_left_4T,0 ]= 0                                                                                                                        #BC on left nodes
    
            h_4T[ self.ind0_Geo_right_4T,1 ]= 0                                                                                                                       #BC on right nodes
            
            h_4T[ self.ind0_Geo_top_4T,2 ]=0
            h_4T[ self.ind0_Geo_bottom_4T,3 ]=0
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                               #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_front_4T,5 ]= self.T_cooling                                                                                                                        
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling                                                                                                                        
    
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_cooling                                                                                                                        
    
            Tconv_4T[ self.ind0_Geo_top_4T,2 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_bottom_4T,3 ]= self.T_cooling
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_front_4T, self.ind0_Geo_left_4T, self.ind0_Geo_right_4T, self.ind0_Geo_top_4T, self.ind0_Geo_bottom_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='TwoSurfaceCoolAgeing':      
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                                  #T_4T initialization, shape is (63,1)        
    
            if self.status_TemBC_smoothening=='No':
                T3_4T[self.ind0_Geo_front_4T]=self.T_cooling                                                                                                              #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back_4T]=self.T_cooling                                                                                                               #BC on back surface nodes 41
            else:
                T3_4T[self.ind0_Geo_front_4T]=self.T_cooling_smoothened                                                                                                   #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back_4T]=self.T_cooling_smoothened                                                                                                    #BC on back surface nodes 41            
            #-----------------------------get all constrained node number
            ind0_BCtem=np.concatenate((self.ind0_Geo_front_4T,self.ind0_Geo_back_4T))                                                                         #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                        #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                                   #h_4T initialization, shape is (63,6)
    
            h_4T[ self.ind0_Geo_left_4T,0 ]= 0                                                                                                                        #BC on left nodes
    
            h_4T[ self.ind0_Geo_right_4T,1 ]= 0                                                                                                                       #BC on right nodes
            
            h_4T[ self.ind0_Geo_top_4T,2 ]=0
            h_4T[ self.ind0_Geo_bottom_4T,3 ]=0
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                               #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_cooling                                                                                                                        
    
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_cooling                                                                                                                        
    
            Tconv_4T[ self.ind0_Geo_top_4T,2 ]= self.T_cooling
            Tconv_4T[ self.ind0_Geo_bottom_4T,3 ]= self.T_cooling
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T, self.ind0_Geo_right_4T, self.ind0_Geo_top_4T, self.ind0_Geo_bottom_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='ReadBCTem':     
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                                  #T_4T initialization, shape is (63,1)        
    
            if self.status_TemBC_smoothening=='No':
                T3_4T[self.ind0_Geo_left_4T]=self.T_cooling_read                                                                                                                     #BC on left surface nodes 2,8,14,20
                T3_4T[self.ind0_Geo_right_4T]=self.T_cooling_read                                                                                                                    #BC on right surface nodes 44,50,56,62
                T3_4T[self.ind0_Geo_top2_5_8_11_14_17_20_4T]=self.T_cooling_read                                                                                                     #BC on front surface nodes 2,5,8,11,14,17,20
                T3_4T[self.ind0_Geo_bottom44_47_50_53_56_59_62_4T]=self.T_cooling_read                                                                                               #BC on back surface nodes 44,47,50,53,56,59,62
                T3_4T[self.ind0_Geo_front23_4T]=self.T_cooling_read                                                                                                                  #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back41_4T]=self.T_cooling_read                                                                                                                   #BC on back surface nodes 41
            else:
                T3_4T[self.ind0_Geo_left_4T]=self.T_cooling_smoothened                                                                                                          #BC on left surface nodes 2,8,14,20
                T3_4T[self.ind0_Geo_right_4T]=self.T_cooling_smoothened                                                                                                         #BC on right surface nodes 44,50,56,62
                T3_4T[self.ind0_Geo_top2_5_8_11_14_17_20_4T]=self.T_cooling_smoothened                                                                                          #BC on front surface nodes 2,5,8,11,14,17,20
                T3_4T[self.ind0_Geo_bottom44_47_50_53_56_59_62_4T]=self.T_cooling_smoothened                                                                                    #BC on back surface nodes 44,47,50,53,56,59,62
                T3_4T[self.ind0_Geo_front23_4T]=self.T_cooling_smoothened                                                                                                       #BC on front surface nodes 23
                T3_4T[self.ind0_Geo_back41_4T]=self.T_cooling_smoothened                                                                                                        #BC on back surface nodes 41
    
            #-----------------------------get all constrained node number
            ind0_BCtem=np.concatenate((self.ind0_Geo_left_4T, self.ind0_Geo_right_4T, self.ind0_Geo_top2_5_8_11_14_17_20_4T, self.ind0_Geo_bottom44_47_50_53_56_59_62_4T, self.ind0_Geo_front23_4T, self.ind0_Geo_back41_4T))                                                                         #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                        #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                                   #h_4T initialization, shape is (63,6)
    
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            #-----------------------------get all constrained node number                
            ind0_BCconv=[]
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
        
        if self.status_TabSurface_Scheme=='OneSurfaceCooling':
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
    
            #-----------------------------get all constrained node number
            ind0_BCtem=[]                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int) 
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])
            
            h_4T[ self.ind0_Geo_left_4T,0 ]= 60                                                                                                                  #BC on left nodes
            h_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= 70
            h_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= 70
            h_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= 70
            h_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= 70        
            h_4T[ self.ind0_Geo_left_Al_4T,0]= 1081     
                     
            h_4T[ self.ind0_Geo_right_4T,1 ]= 60                                                                                                                 #BC on right nodes
            h_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=70
            h_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=70
            h_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=70
            h_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=70
            h_4T[ self.ind0_Geo_right_Cu_4T,1]= 1809    
           
            h_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= 120                                                                                                  #BC on surface nodes        
            h_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= 120
            h_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.h_pouch
            h_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= 7               
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_initial                                                                                                             #BC on left nodes
            Tconv_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= self.T_initial
            Tconv_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= self.T_initial
            Tconv_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= self.T_initial
            Tconv_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= self.T_initial        
                             
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_initial                                                                                                            #BC on right nodes
            Tconv_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=self.T_initial
            Tconv_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=self.T_initial
            Tconv_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=self.T_initial
            Tconv_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=self.T_initial
           
            Tconv_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= self.T_initial                                                                                             #BC on surface nodes        
            Tconv_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= self.T_initial
            
            Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_cooling
            # if ip.status_P_cooling_pouch == 'Yes':
            #     Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_var_cooling_record[step]
            
            Tconv_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= self.T_initial        
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top2_5_8_11_14_17_20_4T,self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,self.ind0_Geo_front23_4T,self.ind0_Geo_back41_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                                      #all the other nodes    


        if self.status_TabSurface_Scheme=='TwoSurfaceCooling':    
    #=============================temperature BC input
            #-----------------------------input: T and its coresponding node
            T3_4T=np.nan*np.zeros([self.ntotal_4T,1])                                                                                                           #T_4T initialization, shape is (63,1)        
    
    
            #-----------------------------get all constrained node number
            ind0_BCtem=[]                                                                                                                                  #all the temperature-constrained nodes        
            ind0_BCtem_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCtem],dtype=int)                                                 #all the other nodes
    #=============================convection BC input
            #-----------------------------constrain heat convection cof: h
            h_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                            #h_4T initialization, shape is (63,6)
    
            h_4T[ self.ind0_Geo_left_4T,0 ]= 60                                                                                                                 #BC on left nodes
            h_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= 70
            h_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= 70
            h_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= 70
            h_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= 1081        
                             
            h_4T[ self.ind0_Geo_right_4T,1 ]= 60                                                                                                                #BC on right nodes
            h_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]= 70
            h_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=70
            h_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=70
            h_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=1809
           
            h_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= 120                                                                                                  #BC on surface nodes        
            h_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= 120
            h_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.h_pouch
            h_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= self.h_pouch               
            #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T=np.nan*np.zeros([self.ntotal_4T,6])                                                                                                        #Tconv_4T initialization, shape is (63,6)
    
            Tconv_4T[ self.ind0_Geo_left_4T,0 ]= self.T_initial                                                                                                      #BC on left nodes
            Tconv_4T[ self.ind0_Geo_edge1_4_7_10_13_16_19_4T,2 ]= self.T_initial
            Tconv_4T[ self.ind0_Geo_edge43_46_49_52_55_58_61_4T,3 ]= self.T_initial
            Tconv_4T[ self.ind0_Geo_edge1_22_43_4T,5 ]= self.T_initial
            Tconv_4T[ self.ind0_Geo_edge19_40_61_4T,4 ]= self.T_initial        
                             
            Tconv_4T[ self.ind0_Geo_right_4T,1 ]= self.T_initial                                                                                                     #BC on right nodes
            Tconv_4T[ self.ind0_Geo_edge3_6_9_12_15_18_21_4T,2 ]=self.T_initial
            Tconv_4T[ self.ind0_Geo_edge45_48_51_54_57_60_63_4T,3 ]=self.T_initial
            Tconv_4T[ self.ind0_Geo_edge3_24_45_4T,5 ]=self.T_initial
            Tconv_4T[ self.ind0_Geo_edge21_42_63_4T,4 ]=self.T_initial
           
            Tconv_4T[ self.ind0_Geo_top2_5_8_11_14_17_20_4T,2 ]= self.T_initial                                                                                      #BC on surface nodes        
            Tconv_4T[ self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,3]= self.T_initial
            
            Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_cooling
            # if self.status_P_cooling_pouch == 'Yes':
            #     Tconv_4T[ self.ind0_Geo_front2_23_44_4T,5 ]= self.T_var_cooling_record[step]
            
            Tconv_4T[ self.ind0_Geo_back20_41_62_4T,4 ]= self.T_cooling
            # if self.status_P_cooling_pouch == 'Yes':
            #     Tconv_4T[ self.ind0_Geo_back20_41_62_4T,5 ]= self.T_var_cooling_record[step]
                
            #-----------------------------get all constrained node number                
            ind0_BCconv=np.concatenate((self.ind0_Geo_left_4T,self.ind0_Geo_right_4T,self.ind0_Geo_top2_5_8_11_14_17_20_4T,self.ind0_Geo_bottom44_47_50_53_56_59_62_4T,self.ind0_Geo_front23_4T,self.ind0_Geo_back41_4T))     #all the convection-constrained nodes
            ind0_BCconv_others=np.array([x for x in np.arange(self.ntotal_4T) if x not in ind0_BCconv],dtype=int)                                               #all the other nodes
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
#        global ind0_Geo_top_4T, ind0_Geo_bottom_4T, ind0_Geo_left_4T, ind0_Geo_right_4T, ind0_Geo_front_4T, ind0_Geo_back_4T
#        global ind0_Geo_left_Al_4T, ind0_Geo_left_Cu_4T, ind0_Geo_left_Elb_4T, ind0_Geo_left_Elr_4T, ind0_Geo_right_Al_4T, ind0_Geo_right_Cu_4T, ind0_Geo_right_Elb_4T, ind0_Geo_right_Elr_4T 
#        global ind0_Geo_top2_5_8_11_14_17_20_4T, ind0_Geo_bottom44_47_50_53_56_59_62_4T, ind0_Geo_front23_4T, ind0_Geo_back41_4T
#        global ind0_Geo_edge1_4_7_10_13_16_19_4T, ind0_Geo_edge43_46_49_52_55_58_61_4T, ind0_Geo_edge1_22_43_4T, ind0_Geo_edge19_40_61_4T, ind0_Geo_edge3_6_9_12_15_18_21_4T, ind0_Geo_edge45_48_51_54_57_60_63_4T, ind0_Geo_edge3_24_45_4T, ind0_Geo_edge21_42_63_4T
#        global ind0_Geo_front2_23_44_4T, ind0_Geo_back20_41_62_4T
#        global ind0_Geo_front_2_4T, ind0_Geo_front_44_4T, ind0_Geo_back_20_4T, ind0_Geo_back_62_4T
#        global ind0_Geo_left_Al_OneThird_4T, ind0_Geo_left_Cu_TwoThird_4T
#        global ind0_Geo_centerlayer_10_11_12_31_32_33_52_53_54_4T, ind0_Geo_core_4T, ind0_Geo_Probe26_32_38_4T
#        global ind0_Geo_Probe42_4T
        ind0_Geo_top_4T=np.where(self.yn_4T==1)[0]                     #top nodes i.g. ind0=[0,1,2~20]
        ind0_Geo_bottom_4T=np.where(self.yn_4T==self.ny)[0]            #bottom nodes i.g. ind0=[42,43,44~62]
        ind0_Geo_left_4T=np.where(self.xn_4T==1)[0]                    #left surface nodes i.g. ind0=[0,3,6,9,12,15,18...21,24,27,30,33,36,39...42,45,48,51,54,57,60]
        ind0_Geo_right_4T=np.where(self.xn_4T==self.nx)[0]             #right surface nodes i.g. ind0=[2,5,8,11,14,17,20...23,26,29,32,35,38,41...44,47,50,53,56,59,62]
        ind0_Geo_front_4T=np.where(self.zn_4T==1)[0]                   #front surface nodes i.g. ind0=[0,1,2,21,22,23,42,43,44]
        ind0_Geo_back_4T=np.where(self.zn_4T==self.nz_4T)[0]           #front surface nodes i.g. ind0=[18,19,20,39,40,41,60,61,62]
        
        ind0_Geo_left_Al_4T=np.array([x for x in ind0_Geo_left_4T if self.mat_4T[x]==1 ],dtype=int)
        ind0_Geo_left_Cu_4T=np.array([x for x in ind0_Geo_left_4T if self.mat_4T[x]==2 ],dtype=int)
        ind0_Geo_left_Elb_4T=np.array([x for x in ind0_Geo_left_4T if self.mat_4T[x]==3 ],dtype=int)
        ind0_Geo_left_Elr_4T=np.array([x for x in ind0_Geo_left_4T if self.mat_4T[x]==4 ],dtype=int)
        ind0_Geo_right_Al_4T=np.array([x for x in ind0_Geo_right_4T if self.mat_4T[x]==1 ],dtype=int)
        ind0_Geo_right_Cu_4T=np.array([x for x in ind0_Geo_right_4T if self.mat_4T[x]==2 ],dtype=int)
        ind0_Geo_right_Elb_4T=np.array([x for x in ind0_Geo_right_4T if self.mat_4T[x]==3 ],dtype=int)
        ind0_Geo_right_Elr_4T=np.array([x for x in ind0_Geo_right_4T if self.mat_4T[x]==4 ],dtype=int)
        
        indtemp1=np.array([x for x in ind0_Geo_top_4T if self.xn_4T[x]!=1 ],dtype=int)
        ind0_Geo_top2_5_8_11_14_17_20_4T=np.array([x for x in indtemp1 if self.xn_4T[x]!=self.nx ],dtype=int)
        indtemp2=np.array([x for x in ind0_Geo_bottom_4T if self.xn_4T[x]!=1 ],dtype=int)
        ind0_Geo_bottom44_47_50_53_56_59_62_4T=np.array([x for x in indtemp2 if self.xn_4T[x]!=self.nx ],dtype=int)
    
        indtemp3=np.array([x for x in ind0_Geo_front_4T if x not in ind0_Geo_left_4T ],dtype=int)
        indtemp4=np.array([x for x in indtemp3 if x not in ind0_Geo_right_4T ],dtype=int)
        indtemp5=np.array([x for x in indtemp4 if x not in ind0_Geo_top_4T ],dtype=int)
        ind0_Geo_front23_4T=np.array([x for x in indtemp5 if x not in ind0_Geo_bottom_4T ],dtype=int)
    
        indtemp6=np.array([x for x in ind0_Geo_back_4T if x not in ind0_Geo_left_4T ],dtype=int)
        indtemp7=np.array([x for x in indtemp6 if x not in ind0_Geo_right_4T ],dtype=int)
        indtemp8=np.array([x for x in indtemp7 if x not in ind0_Geo_top_4T ],dtype=int)
        ind0_Geo_back41_4T=np.array([x for x in indtemp8 if x not in ind0_Geo_bottom_4T ],dtype=int) 
    
        ind0_Geo_edge1_4_7_10_13_16_19_4T=np.array([x for x in ind0_Geo_left_4T if self.yn_4T[x]==1 ],dtype=int)    
        ind0_Geo_edge43_46_49_52_55_58_61_4T=np.array([x for x in ind0_Geo_left_4T if self.yn_4T[x]==self.ny ],dtype=int)    
        ind0_Geo_edge1_22_43_4T=np.array([x for x in ind0_Geo_left_4T if self.zn_4T[x]==1 ],dtype=int)    
        ind0_Geo_edge19_40_61_4T=np.array([x for x in ind0_Geo_left_4T if self.zn_4T[x]==self.nz_4T ],dtype=int)    
    
        ind0_Geo_edge3_6_9_12_15_18_21_4T=np.array([x for x in ind0_Geo_right_4T if self.yn_4T[x]==1 ],dtype=int)    
        ind0_Geo_edge45_48_51_54_57_60_63_4T=np.array([x for x in ind0_Geo_right_4T if self.yn_4T[x]==self.ny ],dtype=int)    
        ind0_Geo_edge3_24_45_4T=np.array([x for x in ind0_Geo_right_4T if self.zn_4T[x]==1 ],dtype=int)    
        ind0_Geo_edge21_42_63_4T=np.array([x for x in ind0_Geo_right_4T if self.zn_4T[x]==self.nz_4T ],dtype=int)
    
        indtemp9=np.append(ind0_Geo_edge1_22_43_4T,ind0_Geo_edge3_24_45_4T)
        indtemp10=np.append(ind0_Geo_edge19_40_61_4T,ind0_Geo_edge21_42_63_4T)
        ind0_Geo_front2_23_44_4T=np.array([x for x in ind0_Geo_front_4T if x not in indtemp9 ],dtype=int)
        ind0_Geo_back20_41_62_4T=np.array([x for x in ind0_Geo_back_4T if x not in indtemp10 ],dtype=int)
    
        ind0_Geo_front_2_4T=np.array([x for x in ind0_Geo_front2_23_44_4T if self.yn_4T[x]==1 ],dtype=int)
        ind0_Geo_front_44_4T=np.array([x for x in ind0_Geo_front2_23_44_4T if self.yn_4T[x]==self.ny ],dtype=int)
        ind0_Geo_back_20_4T=np.array([x for x in ind0_Geo_back20_41_62_4T if self.yn_4T[x]==1 ],dtype=int)
        ind0_Geo_back_62_4T=np.array([x for x in ind0_Geo_back20_41_62_4T if self.yn_4T[x]==self.ny ],dtype=int)
        
        ind0_Geo_left_Al_OneThird_4T=np.array([x for x in ind0_Geo_left_Al_4T if self.yn[x]==int(self.ny/3) ],dtype=int)
        ind0_Geo_left_Cu_TwoThird_4T=np.array([x for x in ind0_Geo_left_Cu_4T if self.yn[x]==self.ny-int(self.ny/3)+1 ],dtype=int)
    
        ind0_Geo_centerlayer_10_11_12_31_32_33_52_53_54_4T=np.array([x for x in np.append(self.Elb_4T,self.Elr_4T) if self.zn_4T[x]==(self.nz_4T+1)/2 ],dtype=int)   
        ind0_Geo_core_4T=np.array([x for x in ind0_Geo_centerlayer_10_11_12_31_32_33_52_53_54_4T if self.xn_4T[x]==int((self.nx+1)/2) and self.yn_4T[x]==int((self.ny+1)/2) ],dtype=int)   
        ind0_Geo_Probe23_26_29_32_38_41_4T=np.array([x for x in np.arange(self.ntotal_4T) if self.xn_4T[x]==self.xn_4T[ind0_Geo_core_4T] and self.yn_4T[x]==self.yn_4T[ind0_Geo_core_4T] ],dtype=int)
        ind0_Geo_Probe26_32_38_4T=np.array([x for x in ind0_Geo_Probe23_26_29_32_38_41_4T if self.mat_4T[x]>=3 ],dtype=int)
    
        ind0_Geo_Probe42_4T=np.array([x for x in ind0_Geo_back_4T if (self.xn_4T[x]==self.nx and self.yn_4T[x]==int(self.ny/2)+1 ) ],dtype=int)
        ind0_Geo_Probe41_4T=np.array([x for x in ind0_Geo_back_4T if (self.xn_4T[x]==int(self.nx/2)+1 and self.yn_4T[x]==int(self.ny/2)+1 ) ],dtype=int)
        ind0_Geo_Probe40_4T=np.array([x for x in ind0_Geo_back_4T if (self.xn_4T[x]==1 and self.yn_4T[x]==int(self.ny/2)+1 ) ],dtype=int)
        ind0_Geo_Probe20_4T=np.array([x for x in ind0_Geo_back_4T if (self.xn_4T[x]==int(self.nx/2)+1 and self.yn_4T[x]==1 ) ],dtype=int)                                                                                                                                                

        (
        self.ind0_Geo_top_4T, self.ind0_Geo_bottom_4T, self.ind0_Geo_left_4T, self.ind0_Geo_right_4T, self.ind0_Geo_front_4T, self.ind0_Geo_back_4T,
        self.ind0_Geo_left_Al_4T, self.ind0_Geo_left_Cu_4T, self.ind0_Geo_left_Elb_4T, self.ind0_Geo_left_Elr_4T, self.ind0_Geo_right_Al_4T, self.ind0_Geo_right_Cu_4T, self.ind0_Geo_right_Elb_4T, self.ind0_Geo_right_Elr_4T, 
        self.ind0_Geo_top2_5_8_11_14_17_20_4T, self.ind0_Geo_bottom44_47_50_53_56_59_62_4T, self.ind0_Geo_front23_4T, self.ind0_Geo_back41_4T,
        self.ind0_Geo_edge1_4_7_10_13_16_19_4T, self.ind0_Geo_edge43_46_49_52_55_58_61_4T, self.ind0_Geo_edge1_22_43_4T, self.ind0_Geo_edge19_40_61_4T, self.ind0_Geo_edge3_6_9_12_15_18_21_4T, self.ind0_Geo_edge45_48_51_54_57_60_63_4T, self.ind0_Geo_edge3_24_45_4T, self.ind0_Geo_edge21_42_63_4T,
        self.ind0_Geo_front2_23_44_4T, self.ind0_Geo_back20_41_62_4T,
        self.ind0_Geo_front_2_4T, self.ind0_Geo_front_44_4T, self.ind0_Geo_back_20_4T, self.ind0_Geo_back_62_4T,
        self.ind0_Geo_left_Al_OneThird_4T, self.ind0_Geo_left_Cu_TwoThird_4T,
        self.ind0_Geo_centerlayer_10_11_12_31_32_33_52_53_54_4T, self.ind0_Geo_core_4T, self.ind0_Geo_Probe26_32_38_4T,
        self.ind0_Geo_Probe42_4T,
        self.ind0_Geo_Probe41_4T,
        self.ind0_Geo_Probe40_4T,
        self.ind0_Geo_Probe20_4T
        )=(
        ind0_Geo_top_4T, ind0_Geo_bottom_4T, ind0_Geo_left_4T, ind0_Geo_right_4T, ind0_Geo_front_4T, ind0_Geo_back_4T,
        ind0_Geo_left_Al_4T, ind0_Geo_left_Cu_4T, ind0_Geo_left_Elb_4T, ind0_Geo_left_Elr_4T, ind0_Geo_right_Al_4T, ind0_Geo_right_Cu_4T, ind0_Geo_right_Elb_4T, ind0_Geo_right_Elr_4T, 
        ind0_Geo_top2_5_8_11_14_17_20_4T, ind0_Geo_bottom44_47_50_53_56_59_62_4T, ind0_Geo_front23_4T, ind0_Geo_back41_4T,
        ind0_Geo_edge1_4_7_10_13_16_19_4T, ind0_Geo_edge43_46_49_52_55_58_61_4T, ind0_Geo_edge1_22_43_4T, ind0_Geo_edge19_40_61_4T, ind0_Geo_edge3_6_9_12_15_18_21_4T, ind0_Geo_edge45_48_51_54_57_60_63_4T, ind0_Geo_edge3_24_45_4T, ind0_Geo_edge21_42_63_4T,
        ind0_Geo_front2_23_44_4T, ind0_Geo_back20_41_62_4T,
        ind0_Geo_front_2_4T, ind0_Geo_front_44_4T, ind0_Geo_back_20_4T, ind0_Geo_back_62_4T,
        ind0_Geo_left_Al_OneThird_4T, ind0_Geo_left_Cu_TwoThird_4T,
        ind0_Geo_centerlayer_10_11_12_31_32_33_52_53_54_4T, ind0_Geo_core_4T, ind0_Geo_Probe26_32_38_4T,
        ind0_Geo_Probe42_4T,
        ind0_Geo_Probe41_4T,
        ind0_Geo_Probe40_4T,
        ind0_Geo_Probe20_4T
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
        Lamda_4T=np.zeros([self.ntotal_4T,6])                                                                                                        #Lamda_4T shape is (63,6). λ term in 6-node stencil
        Lamda_4T[ self.Al_4T,0 ]=self.Lamda_Al                                                                                                            #fill in in 1st column(jx1) for Al node                                                                                                           
        Lamda_4T[ self.Al_4T,1 ]=self.Lamda_Al
        Lamda_4T[ self.Al_4T,2 ]=self.Lamda_Al
        Lamda_4T[ self.Al_4T,3 ]=self.Lamda_Al
        Lamda_4T[ self.Al_4T,4 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)
        Lamda_4T[ self.Al_4T,5 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)
            
        Lamda_4T[ self.Cu_4T,0 ]=self.Lamda_Cu                                                                                                            #fill in in 1st column(jx1) for Cu node
        Lamda_4T[ self.Cu_4T,1 ]=self.Lamda_Cu
        Lamda_4T[ self.Cu_4T,2 ]=self.Lamda_Cu
        Lamda_4T[ self.Cu_4T,3 ]=self.Lamda_Cu
        Lamda_4T[ self.Cu_4T,4 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)
        Lamda_4T[ self.Cu_4T,5 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)
            
        Lamda_4T[ self.Elb_4T,0 ]=self.Lamda_El_x                                                                                                         #fill in in 1st column(jx1) for Elb node
        Lamda_4T[ self.Elb_4T,1 ]=self.Lamda_El_x
        Lamda_4T[ self.Elb_4T,2 ]=self.Lamda_El_y
        Lamda_4T[ self.Elb_4T,3 ]=self.Lamda_El_y
        Lamda_4T[ self.Elb_4T,4 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)
        Lamda_4T[ self.Elb_4T,5 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)
            
        Lamda_4T[ self.Elr_4T,0 ]=self.Lamda_El_x                                                                                                         #fill in in 1st column(jx1) for Elr node
        Lamda_4T[ self.Elr_4T,1 ]=self.Lamda_El_x
        Lamda_4T[ self.Elr_4T,2 ]=self.Lamda_El_y
        Lamda_4T[ self.Elr_4T,3 ]=self.Lamda_El_y
        Lamda_4T[ self.Elr_4T,4 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)
        Lamda_4T[ self.Elr_4T,5 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)                
        
        #modify: for nodes 1~3,22~24,43~45, λ in z2 direction is λAl
        Lamda_4T[ self.ind0_Geo_front_4T,5 ]=self.Lamda_Al      
        #modify:for nodes 19~21,40~42,61~63, λ in z1 direction is λCu
        Lamda_4T[ self.ind0_Geo_back_4T,4 ]=self.Lamda_Cu      
        #-----------------------------------fill in RouXc_4T
        RouXc_4T=np.zeros([self.ntotal_4T,1])                                                                                                        #RouXc_4T shape is (63,1). ρc term in 6-node stencil
        RouXc_4T[ self.Al_4T,0 ]=self.rou_Al*self.c_Al                                                                                                         #fill in in elements for Al node
        RouXc_4T[ self.Cu_4T,0 ]=self.rou_Cu*self.c_Cu                                                                                                         #fill in in elements for Cu node        
        RouXc_4T[ self.Elb_4T,0 ]=self.rouXc_El                                                                                                           #fill in in elements for Elb node
        RouXc_4T[ self.Elr_4T,0 ]=self.rouXc_El                                                                                                           #fill in in elements for Elr node
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
    
        MatThickness_lib=np.array([[self.delta_Al],[self.delta_Cu],[self.delta_El],[self.delta_El]])
        delta_z1_4T=MatThickness_lib[self.mat_4T-1,0]                                                                                      #get δz1 for each node
        delta_z2_4T=delta_z1_4T.copy()                                                                                                       #get δz2(=δz1) for each node
    
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
        MatrixCN[self.ind0_jx1NonNaN_4T_ALL,self.ind0_jx1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jx1NonNaN_4T,0]/self.RouXc_4T_ALL[self.ind0_jx1NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx1NonNaN_4T_ALL])/self.Delta_x1_4T_ALL[self.ind0_jx1NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jx2NaN_4T_ALL,self.ind0_jx2NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jx2NaN_4T_ALL,1]/self.RouXc_4T_ALL[self.ind0_jx2NaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx2NaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx2NaN_4T_ALL])       #jx2 components in diagonal terms
        MatrixCN[self.ind0_jx2NonNaN_4T_ALL,self.ind0_jx2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jx2NonNaN_4T,1]/self.RouXc_4T_ALL[self.ind0_jx2NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL])/self.Delta_x2_4T_ALL[self.ind0_jx2NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jy1NaN_4T_ALL,self.ind0_jy1NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.ind0_jy1NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy1NaN_4T_ALL])       #jy1 components in diagonal terms
        MatrixCN[self.ind0_jy1NonNaN_4T_ALL,self.ind0_jy1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jy1NonNaN_4T,2]/self.RouXc_4T_ALL[self.ind0_jy1NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy1NonNaN_4T_ALL])/self.Delta_y1_4T_ALL[self.ind0_jy1NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jy2NaN_4T_ALL,self.ind0_jy2NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.ind0_jy2NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy2NaN_4T_ALL])       #jy2 components in diagonal terms
        MatrixCN[self.ind0_jy2NonNaN_4T_ALL,self.ind0_jy2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jy2NonNaN_4T,3]/self.RouXc_4T_ALL[self.ind0_jy2NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL])/self.Delta_y2_4T_ALL[self.ind0_jy2NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jz1NaN_4T_ALL,self.ind0_jz1NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jz1NaN_4T_ALL,4]/self.RouXc_4T_ALL[self.ind0_jz1NaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz1NaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz1NaN_4T_ALL])       #jz1 components in diagonal terms
        MatrixCN[self.ind0_jz1NonNaN_4T_ALL,self.ind0_jz1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jz1NonNaN_4T,4]/self.RouXc_4T_ALL[self.ind0_jz1NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz1NonNaN_4T_ALL])/self.Delta_z1_4T_ALL[self.ind0_jz1NonNaN_4T_ALL]
    
        MatrixCN[self.ind0_jz2NaN_4T_ALL,self.ind0_jz2NaN_4T_ALL] += self.h_4T_ALL[self.ind0_jz2NaN_4T_ALL,5]/self.RouXc_4T_ALL[self.ind0_jz2NaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz2NaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz2NaN_4T_ALL])       #jz2 components in diagonal terms
        MatrixCN[self.ind0_jz2NonNaN_4T_ALL,self.ind0_jz2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.ind0_jz2NonNaN_4T,5]/self.RouXc_4T_ALL[self.ind0_jz2NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL])/self.Delta_z2_4T_ALL[self.ind0_jz2NonNaN_4T_ALL]
    
        MatrixCN[np.arange(self.n_4T_ALL),np.arange(self.n_4T_ALL)] += 1                                                                                               #"1" components in diagonal terms
        MatrixCN[self.ind0_BCtem_ALL,self.ind0_BCtem_ALL]=inf
        return MatrixCN

   
    #########################################################   
    ###########       function for VectorCN       ###########
    #########################################################
    def fun_VectorCN(self):                #VectorCN = VectorCN_preTp*Tp + VectorCN_conv_q;    VectorCN_preTp is very similar to MatrixCN, so form VectorCN based on MatrixCN
#        global VectorCN, VectorCN_conv_q
        VectorCN=np.zeros([self.n_4T_ALL,1])       
    #    VectorCN_preTp=np.zeros([n_4T_ALL,n_4T_ALL])
        self.VectorCN_conv_q=np.zeros([self.n_4T_ALL,1])
        #==================================================add Tp term  (this part is moved to the function fun_VectorCN_preTp(self))        
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
        self.VectorCN_conv_q += self.q_4T_ALL*self.dt/self.RouXc_4T_ALL                                                                                                                                                                                                     #heat gen components
    
        VectorCN= self.VectorCN_preTp .dot( self.T1_4T_ALL ) + self.VectorCN_conv_q
        #======================================penalty on Temperature-constrained BC nodes (apply temperature BC)
        VectorCN[self.ind0_BCtem_ALL,0]=(self.T3_4T_ALL[self.ind0_BCtem_ALL,0]*inf)
    
        return VectorCN

        
    #########################################################   
    #########     function for Visualization     ############
    #########################################################
    #plot_steps_available=np.where(~np.isnan(T_avg_record))[0]      #in cycling mode, there are NaN values in the last cycles. So here plot_step is the last non-NaN step 
    #plot_step=plot_steps_available[-1]                             #plot the last step of non-NaN steps 
    def fun_plotly_T(self, XYZ_Module, plot_step, vmin, vmax):        
        #----------------------plot T----------------------
        if self.status_Model == 'EandT' or self.status_Model == 'T':
            plot_Variable=self.T_record[:,plot_step]-273.15          #variable to be visualized
            #===================================prepare X,Y,Z,values for visualization
            ind0_plot=np.arange(self.ntotal_4T).reshape(self.ny,self.nz_4T,self.nx)
            #X=self.xi_4T[ind0_plot]; Y=self.yi_4T[ind0_plot]; Z=self.zi_4T[ind0_plot]
            X=(self.xi_4T[ind0_plot])*self.Lx_electrodes_real/((self.xi_4T[ind0_plot]).max()); Y=self.yi_4T[ind0_plot]*self.Ly_electrodes_real/((self.yi_4T[ind0_plot]).max()); Z=self.zi_4T[ind0_plot]
            values=plot_Variable[ind0_plot]
            #===================================visualization
            fig = go.Figure(data=go.Volume(   
                x=X.flatten(),    
                y=Y.flatten(),    
                z=Z.flatten(),    
                value=values.flatten(), 
                isomin=vmin,       
                isomax=vmax,       
                opacity=0.2,                      
                surface_count=20,                 
                colorbar={"title": '°C'}, 
                colorscale='RdBu_r'              
                ))
            fig.update_layout(
                    scene_aspectmode='manual',
                    scene_aspectratio=dict(x=2, y=1, z=0.25),      #change x,y,z ratio
                    title_text='T contour', title_x=0.5,
                    font=dict(family='Arial', size=16),            #font size
                    ) 
            plot(fig, auto_open=True)    
    def fun_plotly_SoC(self, XYZ_Module, plot_step, vmin, vmax):        
        #----------------------plot SoC----------------------
        if self.status_Model == 'EandT' or self.status_Model == 'E':  
            plot_Variable=self.SoC_ele_record[:,plot_step]          #variable to be visualized
            #===================================prepare X,Y,Z,values for visualization
            ind0_plot=self.List_ele2node_4T.reshape(self.ny,(2*self.nstack-1),self.nx)
            
            X=self.xi_4T[ind0_plot]; Y=self.yi_4T[ind0_plot]; Z=self.zi_4T[ind0_plot]
            ind0_ele_plot=self.List_node2ele_4T[ind0_plot,0]
            values=plot_Variable[ind0_ele_plot]
            #===================================visualization
            fig = go.Figure(data=go.Volume(   
                x=X.flatten(),    
                y=Y.flatten(),    
                z=Z.flatten(),    
                value=values.flatten(), 
                isomin=vmin,       
                isomax=vmax,       
                opacity=0.2,                      
                surface_count=20,                 
                colorscale='Greens'              
                ))
            fig.update_layout(
                    scene_aspectmode='manual',
                    scene_aspectratio=dict(x=2, y=1, z=0.25),      #change x,y,z ratio
                    title_text='SoC contour', title_x=0.5,
                    font=dict(family='Arial', size=16),            #font size
                    ) 
            plot(fig, auto_open=True)  

    def fun_mayavi_by_node(self, XYZ_Module, plot_Variable_ALL, vmin, vmax, title_string, colormap_string):        
        
#        self.plot_steps_available=np.where(~np.isnan(self.T_avg_record))[0]      #in cycling mode, there are NaN values in the last cycles. So here plot_steps_available is all the steps with non-NaN values 
#        self.plot_step=self.plot_steps_available[-1]                             #plot the last step from non-NaN steps 
        
#        self.plot_Variable_ALL=self.T_record[:,plot_step]-273.15  #variable to be visualized  
        self.xn_PlotSlice=int((self.nx+1)/2); self.zn_PlotSlice=int((self.nz_4T+1)/2)          #two cross sections for Jellyroll plotting
        #self.status_plot_Can=self.status_ThermalPatition_Can                       #plot Can or not
        
        self.X_Module = XYZ_Module[0]         #this cell's X location in Module
        self.Y_Module = XYZ_Module[1]
        self.Z_Module = XYZ_Module[2]
        #===================================prepare X,Y,Z,C for visualization
        self.yi_plot_4T_ALL=self.yi_4T_ALL.copy()
        #-----------------------plot front surface
        self.ind0_plot_front=self.ind0_Geo_front_4T.reshape(self.ny,self.nx)
        self.X1=self.zi_4T_ALL[self.ind0_plot_front]
        self.Y1=self.xi_4T_ALL[self.ind0_plot_front]
        self.Z1=self.yi_4T_ALL[self.ind0_plot_front]
        self.C1=plot_Variable_ALL[self.ind0_plot_front]
        self.X1 += self.X_Module; self.Y1 += self.Y_Module; self.Z1 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot back surface kerb outside
        self.ind0_plot_back_kerb_outside=np.array([x for x in self.ind0_Geo_back_4T if self.xn_4T[x] <= self.xn_PlotSlice ],dtype=int).reshape(self.ny,-1)        
        self.X2=self.zi_4T_ALL[self.ind0_plot_back_kerb_outside]
        self.Y2=self.xi_4T_ALL[self.ind0_plot_back_kerb_outside]
        self.Z2=self.yi_4T_ALL[self.ind0_plot_back_kerb_outside]
        self.C2=plot_Variable_ALL[self.ind0_plot_back_kerb_outside]
        self.X2 += self.X_Module; self.Y2 += self.Y_Module; self.Z2 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot back surface kerb inside        
        self.ind0_plot_back_kerb_inside=np.array([x for x in np.arange(self.ntotal_4T) if self.zn_4T[x] == 2*self.nstack and self.xn_4T[x] >= self.xn_PlotSlice ],dtype=int).reshape(self.ny,-1)           
        self.X3=self.zi_4T_ALL[self.ind0_plot_back_kerb_inside]
        self.Y3=self.xi_4T_ALL[self.ind0_plot_back_kerb_inside]
        self.Z3=self.yi_4T_ALL[self.ind0_plot_back_kerb_inside]
        self.C3=plot_Variable_ALL[self.ind0_plot_back_kerb_inside]
        self.X3 += self.X_Module; self.Y3 += self.Y_Module; self.Z3 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot right surface kerb outside
        self.ind0_plot_right_kerb_outside=np.array([x for x in self.ind0_Geo_right_4T if self.zn_4T[x] <= self.zn_PlotSlice ],dtype=int).reshape(self.ny,-1)           
        self.X4=self.zi_4T_ALL[self.ind0_plot_right_kerb_outside]
        self.Y4=self.xi_4T_ALL[self.ind0_plot_right_kerb_outside]
        self.Z4=self.yi_4T_ALL[self.ind0_plot_right_kerb_outside]
        self.C4=plot_Variable_ALL[self.ind0_plot_right_kerb_outside]
        self.X4 += self.X_Module; self.Y4 += self.Y_Module; self.Z4 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot right surface kerb inside
        self.ind0_plot_right_kerb_inside=np.array([x for x in np.arange(self.ntotal_4T) if self.xn_4T[x] == self.xn_PlotSlice and self.zn_4T[x] >= self.zn_PlotSlice ],dtype=int).reshape(self.ny,-1)          
        self.X5=self.zi_4T_ALL[self.ind0_plot_right_kerb_inside]
        self.Y5=self.xi_4T_ALL[self.ind0_plot_right_kerb_inside]
        self.Z5=self.yi_4T_ALL[self.ind0_plot_right_kerb_inside]
        self.C5=plot_Variable_ALL[self.ind0_plot_right_kerb_inside]
        self.X5 += self.X_Module; self.Y5 += self.Y_Module; self.Z5 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot left surface kerb inside
        self.ind0_plot_left=self.ind0_Geo_left_4T.reshape(self.ny,-1)          
        self.X6=self.zi_4T_ALL[self.ind0_plot_left]
        self.Y6=self.xi_4T_ALL[self.ind0_plot_left]
        self.Z6=self.yi_4T_ALL[self.ind0_plot_left]
        self.C6=plot_Variable_ALL[self.ind0_plot_left]
        self.X6 += self.X_Module; self.Y6 += self.Y_Module; self.Z6 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot bottom surface1
        self.ind0_plot_bottom1=np.array([x for x in self.ind0_Geo_bottom_4T if self.xn_4T[x] >= self.xn_PlotSlice and self.zn_4T[x] <= self.zn_PlotSlice ],dtype=int).reshape(2*self.nstack,-1)          
        self.X7=self.zi_4T_ALL[self.ind0_plot_bottom1]
        self.Y7=self.xi_4T_ALL[self.ind0_plot_bottom1]
        self.Z7=self.yi_4T_ALL[self.ind0_plot_bottom1]
        self.C7=plot_Variable_ALL[self.ind0_plot_bottom1]
        self.X7 += self.X_Module; self.Y7 += self.Y_Module; self.Z7 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot bottom surface2
        self.ind0_plot_bottom2=np.array([x for x in self.ind0_Geo_bottom_4T if self.xn_4T[x] <= self.xn_PlotSlice ],dtype=int).reshape(self.nz_4T,-1)          
        self.X8=self.zi_4T_ALL[self.ind0_plot_bottom2]
        self.Y8=self.xi_4T_ALL[self.ind0_plot_bottom2]
        self.Z8=self.yi_4T_ALL[self.ind0_plot_bottom2]
        self.C8=plot_Variable_ALL[self.ind0_plot_bottom2]
        self.X8 += self.X_Module; self.Y8 += self.Y_Module; self.Z8 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot top surface1
        self.ind0_plot_top1=np.array([x for x in self.ind0_Geo_top_4T if self.xn_4T[x] >= self.xn_PlotSlice and self.zn_4T[x] <= self.zn_PlotSlice ],dtype=int).reshape(2*self.nstack,-1)          
        self.X9=self.zi_4T_ALL[self.ind0_plot_top1]
        self.Y9=self.xi_4T_ALL[self.ind0_plot_top1]
        self.Z9=self.yi_4T_ALL[self.ind0_plot_top1]
        self.C9=plot_Variable_ALL[self.ind0_plot_top1]
        self.X9 += self.X_Module; self.Y9 += self.Y_Module; self.Z9 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot top surface2
        self.ind0_plot_top2=np.array([x for x in self.ind0_Geo_top_4T if self.xn_4T[x] <= self.xn_PlotSlice ],dtype=int).reshape(self.nz_4T,-1)          
        self.X10=self.zi_4T_ALL[self.ind0_plot_top2]
        self.Y10=self.xi_4T_ALL[self.ind0_plot_top2]
        self.Z10=self.yi_4T_ALL[self.ind0_plot_top2]
        self.C10=plot_Variable_ALL[self.ind0_plot_top2]
        self.X10 += self.X_Module; self.Y10 += self.Y_Module; self.Z10 += self.Z_Module    #Get cells' spatial locations in Module
        
        #===================================visualization
#       self.fig = mlab.figure(bgcolor=(1,1,1))    
#       self.vmin=self.plot_Variable_ALL.min();  self.vmax=self.plot_Variable_ALL.max()
        self.surf1 = mlab.mesh(self.X1, self.Y1, self.Z1, scalars=self.C1, colormap=colormap_string)
        self.surf2 = mlab.mesh(self.X2, self.Y2, self.Z2, scalars=self.C2, colormap=colormap_string)
        self.surf3 = mlab.mesh(self.X3, self.Y3, self.Z3, scalars=self.C3, colormap=colormap_string)
        self.surf4 = mlab.mesh(self.X4, self.Y4, self.Z4, scalars=self.C4, colormap=colormap_string)
        self.surf5 = mlab.mesh(self.X5, self.Y5, self.Z5, scalars=self.C5, colormap=colormap_string)
        self.surf6 = mlab.mesh(self.X6, self.Y6, self.Z6, scalars=self.C6, colormap=colormap_string)
        self.surf7 = mlab.mesh(self.X7, self.Y7, self.Z7, scalars=self.C7, colormap=colormap_string)
        self.surf8 = mlab.mesh(self.X8, self.Y8, self.Z8, scalars=self.C8, colormap=colormap_string)
        self.surf9 = mlab.mesh(self.X9, self.Y9, self.Z9, scalars=self.C9, colormap=colormap_string)
        self.surf10 = mlab.mesh(self.X10, self.Y10, self.Z10, scalars=self.C10, colormap=colormap_string)

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
        self.surf7.module_manager.scalar_lut_manager.use_default_range = False
        self.surf7.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf8.module_manager.scalar_lut_manager.use_default_range = False
        self.surf8.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf9.module_manager.scalar_lut_manager.use_default_range = False
        self.surf9.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf10.module_manager.scalar_lut_manager.use_default_range = False
        self.surf10.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.cb=mlab.colorbar(title=title_string,orientation='vertical',label_fmt='%.2f',nb_labels=5)
        self.cb.scalar_bar.unconstrained_font_size = True
        self.cb.title_text_property.font_family = 'times'; self.cb.title_text_property.bold=False; self.cb.title_text_property.italic=False; self.cb.title_text_property.color=(0,0,0); self.cb.title_text_property.font_size=20
        self.cb.label_text_property.font_family = 'times'; self.cb.label_text_property.bold=True;  self.cb.label_text_property.italic=False; self.cb.label_text_property.color=(0,0,0); self.cb.label_text_property.font_size=15
        if self.status_mayavi_show_cell_num == 'Yes':
            mlab.text3d(self.X_Module,self.Y_Module,self.Z_Module-0.02, 'cell_%s'%str(self.cell_ind0+1), scale=.005, color=(0,0,0))
    
    def fun_mayavi_by_ele(self, XYZ_Module, plot_Variable_ALL, vmin, vmax, title_string, colormap_string):        
        #===================================prepare X,Y,Z,C for visualization
        #-----------------------plot front surface
        self.ind0_plot_front=np.array([x for x in self.Elb_4T if self.zn_4T[x]==2],dtype=int).reshape(self.ny,self.nx)
        self.X1=self.zi_4T_ALL[self.ind0_plot_front]
        self.Y1=self.xi_4T_ALL[self.ind0_plot_front]
        self.Z1=self.yi_4T_ALL[self.ind0_plot_front]
        self.ind0_ele_plot_front=self.List_node2ele_4T[self.ind0_plot_front,0]
        self.C1=plot_Variable_ALL[self.ind0_ele_plot_front]
        self.X1 += self.X_Module; self.Y1 += self.Y_Module; self.Z1 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot back surface kerb outside
        self.ind0_plot_back_kerb_outside=np.array([x for x in self.Elb_4T if self.zn_4T[x]==(self.nz_4T-1) and self.xn_4T[x] <= self.xn_PlotSlice ],dtype=int).reshape(self.ny,-1)        
        self.X2=self.zi_4T_ALL[self.ind0_plot_back_kerb_outside]
        self.Y2=self.xi_4T_ALL[self.ind0_plot_back_kerb_outside]
        self.Z2=self.yi_4T_ALL[self.ind0_plot_back_kerb_outside]
        self.ind0_ele_plot_back_kerb_outside=self.List_node2ele_4T[self.ind0_plot_back_kerb_outside,0]
        self.C2=plot_Variable_ALL[self.ind0_ele_plot_back_kerb_outside]
        self.X2 += self.X_Module; self.Y2 += self.Y_Module; self.Z2 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot back surface kerb inside        
        self.ind0_plot_back_kerb_inside=np.array([x for x in np.arange(self.ntotal_4T) if self.zn_4T[x] == 2*self.nstack and self.xn_4T[x] >= self.xn_PlotSlice ],dtype=int).reshape(self.ny,-1)           
        self.X3=self.zi_4T_ALL[self.ind0_plot_back_kerb_inside]
        self.Y3=self.xi_4T_ALL[self.ind0_plot_back_kerb_inside]
        self.Z3=self.yi_4T_ALL[self.ind0_plot_back_kerb_inside]
        self.ind0_ele_plot_back_kerb_inside=self.List_node2ele_4T[self.ind0_plot_back_kerb_inside,0]
        self.C3=plot_Variable_ALL[self.ind0_ele_plot_back_kerb_inside]
        self.X3 += self.X_Module; self.Y3 += self.Y_Module; self.Z3 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot right surface kerb outside
        self.ind0_plot_right_kerb_outside=np.array([x for x in self.ind0_Geo_right_4T if x in self.El_4T and self.zn_4T[x] <= self.zn_PlotSlice ],dtype=int).reshape(self.ny,-1)           
        self.X4=self.zi_4T_ALL[self.ind0_plot_right_kerb_outside]
        self.Y4=self.xi_4T_ALL[self.ind0_plot_right_kerb_outside]
        self.Z4=self.yi_4T_ALL[self.ind0_plot_right_kerb_outside]
        self.ind0_ele_plot_right_kerb_outside=self.List_node2ele_4T[self.ind0_plot_right_kerb_outside,0]
        self.C4=plot_Variable_ALL[self.ind0_ele_plot_right_kerb_outside]
        self.X4 += self.X_Module; self.Y4 += self.Y_Module; self.Z4 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot right surface kerb inside
        self.ind0_plot_right_kerb_inside=np.array([x for x in self.El_4T if self.xn_4T[x] == self.xn_PlotSlice and self.zn_4T[x] >= self.zn_PlotSlice ],dtype=int).reshape(self.ny,-1)          
        self.X5=self.zi_4T_ALL[self.ind0_plot_right_kerb_inside]
        self.Y5=self.xi_4T_ALL[self.ind0_plot_right_kerb_inside]
        self.Z5=self.yi_4T_ALL[self.ind0_plot_right_kerb_inside]
        self.ind0_ele_plot_right_kerb_inside=self.List_node2ele_4T[self.ind0_plot_right_kerb_inside,0]
        self.C5=plot_Variable_ALL[self.ind0_ele_plot_right_kerb_inside]
        self.X5 += self.X_Module; self.Y5 += self.Y_Module; self.Z5 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot left surface kerb inside
        self.ind0_plot_left=np.array([x for x in self.El_4T if self.xn_4T[x] == 1 ],dtype=int).reshape(self.nx,-1)         
        self.X6=self.zi_4T_ALL[self.ind0_plot_left]
        self.Y6=self.xi_4T_ALL[self.ind0_plot_left]
        self.Z6=self.yi_4T_ALL[self.ind0_plot_left]
        self.ind0_ele_plot_left=self.List_node2ele_4T[self.ind0_plot_left,0]
        self.C6=plot_Variable_ALL[self.ind0_ele_plot_left]
        self.X6 += self.X_Module; self.Y6 += self.Y_Module; self.Z6 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot bottom surface1
        ind0_plot_bottom1=np.array([x for x in self.El_4T if self.yn_4T[x] == self.ny ],dtype=int)          
        self.ind0_plot_bottom1=np.array([x for x in ind0_plot_bottom1 if self.xn_4T[x] >= self.xn_PlotSlice and self.zn_4T[x] <= self.zn_PlotSlice ],dtype=int).reshape(self.nstack,-1)          
        self.X7=self.zi_4T_ALL[self.ind0_plot_bottom1]
        self.Y7=self.xi_4T_ALL[self.ind0_plot_bottom1]
        self.Z7=self.yi_4T_ALL[self.ind0_plot_bottom1]
        self.ind0_ele_plot_bottom1=self.List_node2ele_4T[self.ind0_plot_bottom1,0]
        self.C7=plot_Variable_ALL[self.ind0_ele_plot_bottom1]
        self.X7 += self.X_Module; self.Y7 += self.Y_Module; self.Z7 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot bottom surface2
        ind0_plot_bottom2=np.array([x for x in self.El_4T if self.yn_4T[x] == self.ny ],dtype=int)          
        self.ind0_plot_bottom2=np.array([x for x in ind0_plot_bottom2 if self.xn_4T[x] <= self.xn_PlotSlice ],dtype=int).reshape(2*self.nstack-1,-1)          
        self.X8=self.zi_4T_ALL[self.ind0_plot_bottom2]
        self.Y8=self.xi_4T_ALL[self.ind0_plot_bottom2]
        self.Z8=self.yi_4T_ALL[self.ind0_plot_bottom2]
        self.ind0_ele_plot_bottom2=self.List_node2ele_4T[self.ind0_plot_bottom2,0]
        self.C8=plot_Variable_ALL[self.ind0_ele_plot_bottom2]
        self.X8 += self.X_Module; self.Y8 += self.Y_Module; self.Z8 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot top surface1
        ind0_plot_top1=np.array([x for x in self.El_4T if self.yn_4T[x] == 1 ],dtype=int)          
        self.ind0_plot_top1=np.array([x for x in ind0_plot_top1 if self.xn_4T[x] >= self.xn_PlotSlice and self.zn_4T[x] <= self.zn_PlotSlice ],dtype=int).reshape(self.nstack,-1)          
        self.X9=self.zi_4T_ALL[self.ind0_plot_top1]
        self.Y9=self.xi_4T_ALL[self.ind0_plot_top1]
        self.Z9=self.yi_4T_ALL[self.ind0_plot_top1]
        self.ind0_ele_plot_top1=self.List_node2ele_4T[self.ind0_plot_top1,0]
        self.C9=plot_Variable_ALL[self.ind0_ele_plot_top1]
        self.X9 += self.X_Module; self.Y9 += self.Y_Module; self.Z9 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot top surface2
        ind0_plot_top2=np.array([x for x in self.El_4T if self.yn_4T[x] == 1 ],dtype=int)          
        self.ind0_plot_top2=np.array([x for x in ind0_plot_top2 if self.xn_4T[x] <= self.xn_PlotSlice ],dtype=int).reshape(2*self.nstack-1,-1)          
        self.X10=self.zi_4T_ALL[self.ind0_plot_top2]
        self.Y10=self.xi_4T_ALL[self.ind0_plot_top2]
        self.Z10=self.yi_4T_ALL[self.ind0_plot_top2]
        self.ind0_ele_plot_top2=self.List_node2ele_4T[self.ind0_plot_top2,0]
        self.C10=plot_Variable_ALL[self.ind0_ele_plot_top2]
        self.X10 += self.X_Module; self.Y10 += self.Y_Module; self.Z10 += self.Z_Module    #Get cells' spatial locations in Module
        
        #===================================visualization
#       self.fig = mlab.figure(bgcolor=(1,1,1))    
#       self.vmin=self.plot_Variable_ALL.min();  self.vmax=self.plot_Variable_ALL.max()
        self.surf1 = mlab.mesh(self.X1, self.Y1, self.Z1, scalars=self.C1, colormap=colormap_string)
        self.surf2 = mlab.mesh(self.X2, self.Y2, self.Z2, scalars=self.C2, colormap=colormap_string)
        self.surf3 = mlab.mesh(self.X3, self.Y3, self.Z3, scalars=self.C3, colormap=colormap_string)
        self.surf4 = mlab.mesh(self.X4, self.Y4, self.Z4, scalars=self.C4, colormap=colormap_string)
        self.surf5 = mlab.mesh(self.X5, self.Y5, self.Z5, scalars=self.C5, colormap=colormap_string)
        self.surf6 = mlab.mesh(self.X6, self.Y6, self.Z6, scalars=self.C6, colormap=colormap_string)
        self.surf7 = mlab.mesh(self.X7, self.Y7, self.Z7, scalars=self.C7, colormap=colormap_string)
        self.surf8 = mlab.mesh(self.X8, self.Y8, self.Z8, scalars=self.C8, colormap=colormap_string)
        self.surf9 = mlab.mesh(self.X9, self.Y9, self.Z9, scalars=self.C9, colormap=colormap_string)
        self.surf10 = mlab.mesh(self.X10, self.Y10, self.Z10, scalars=self.C10, colormap=colormap_string)

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
        self.surf7.module_manager.scalar_lut_manager.use_default_range = False
        self.surf7.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf8.module_manager.scalar_lut_manager.use_default_range = False
        self.surf8.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf9.module_manager.scalar_lut_manager.use_default_range = False
        self.surf9.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.surf10.module_manager.scalar_lut_manager.use_default_range = False
        self.surf10.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.cb=mlab.colorbar(title=title_string,orientation='vertical',label_fmt='%.2f',nb_labels=5)
        self.cb.scalar_bar.unconstrained_font_size = True
        self.cb.title_text_property.font_family = 'times'; self.cb.title_text_property.bold=False; self.cb.title_text_property.italic=False; self.cb.title_text_property.color=(0,0,0); self.cb.title_text_property.font_size=20
        self.cb.label_text_property.font_family = 'times'; self.cb.label_text_property.bold=True;  self.cb.label_text_property.italic=False; self.cb.label_text_property.color=(0,0,0); self.cb.label_text_property.font_size=15
        if self.status_mayavi_show_cell_num == 'Yes':
            mlab.text3d(self.X_Module,self.Y_Module,self.Z_Module-0.02, 'cell_%s'%str(self.cell_ind0+1), scale=.005, color=(0,0,0))
    
    
    
    
    
    
    
    
    
    
