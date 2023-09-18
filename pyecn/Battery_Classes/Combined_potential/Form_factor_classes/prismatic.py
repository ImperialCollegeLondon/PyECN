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

inf=1e10


class Prismatic:
          
    def fun_SpiralandStripefrom0(self,a0_local,b0_local,Lx0_pouch,   theta_local, an_local):
        L_spiral = (a0_local*theta_local+b0_local)*np.sqrt(a0_local**2+b0_local**2+2*a0_local*b0_local*theta_local+a0_local**2*theta_local**2)/2/a0_local \
             + a0_local/2*np.log(theta_local+b0_local/a0_local+np.sqrt((theta_local+b0_local/a0_local)**2+1)) \
             -np.sqrt(a0_local**2+b0_local**2)*b0_local/2/a0_local- a0_local/2*np.log(b0_local/a0_local+np.sqrt((b0_local/a0_local)**2+1))
        
    #    OneFullStripe_penalty = np.heaviside(an_local-(ind0_SpiralandStripe_boundary1+nx_pouch+1),1)     #line1_penalty is vector in 1,2...ntotal order. 0 means that node has full length in stripe1
    #    TwoFullStripe_penalty = np.heaviside(an_local-(ind0_SpiralandStripe_boundary2+nx_pouch+1),1)     #line1_penalty is vector in 1,2...ntotal order. 0 means that node has full length in two stripes: stripe2 & stripe2
    #
    #    node8to10_penalty = OneFullStripe_penalty * np.heaviside(an_local-(ind0_SpiralandStripe_boundary2+1+1),1)
    #    node3to5_penalty = OneFullStripe_penalty * np.heaviside(an_local-(ind0_SpiralandStripe_boundary2+1+1),1)
    
        node1to2_penalty = np.heaviside((self.ind0_SpiralandStripe_boundary1+1)-an_local,1)
        node3to5_penalty = np.heaviside(an_local-(self.ind0_SpiralandStripe_boundary1+1+1),1) * np.heaviside((self.ind0_SpiralandStripe_boundary1+1+self.nx_pouch)-an_local,1)
        node6to7_penalty = np.heaviside(an_local-(self.ind0_SpiralandStripe_boundary1+1+self.nx_pouch+1),1) * np.heaviside((self.ind0_SpiralandStripe_boundary2+self.nx_pouch+1)-an_local,1)
        node8to10_penalty = np.heaviside(an_local-(self.ind0_SpiralandStripe_boundary2+self.nx_pouch+1+1),1) * np.heaviside((self.ind0_SpiralandStripe_boundary2+self.nx_pouch+1+self.nx_pouch)-an_local,1)
        node_lastsection_penalty = np.heaviside(an_local-(self.ind0_SpiralandStripe_boundary2+2*self.nx_pouch+1+1),1)   
        
        L_pouch = (   node1to2_penalty * 0                                                                     
                    + node3to5_penalty * ((an_local-1)%self.nx-self.ind0_SpiralandStripe_boundary1)                      
                    + node6to7_penalty * self.nx_pouch                                                             
                    + node8to10_penalty * ((an_local-1)%self.nx-self.ind0_SpiralandStripe_boundary2)                    
                    + node_lastsection_penalty * 2*self.nx_pouch                                                    
                    + np.floor(theta_local/2/np.pi) * 2*self.nx_pouch   ) * Lx0_pouch/self.nx_pouch                    
        
        L = L_spiral + L_pouch
            
        return L            
    def fun_prismatic_mayavi(self,ind0_plot):
        node1to2_penalty = np.heaviside((self.ind0_SpiralandStripe_boundary1+1)-self.an_4T[ind0_plot],1)
        node3to5_penalty = np.heaviside(self.an_4T[ind0_plot]-(self.ind0_SpiralandStripe_boundary1+1+1),1) * np.heaviside((self.ind0_SpiralandStripe_boundary1+1+self.nx_pouch)-self.an_4T[ind0_plot],1)
        node6to7_penalty = np.heaviside(self.an_4T[ind0_plot]-(self.ind0_SpiralandStripe_boundary1+1+self.nx_pouch+1),1) * np.heaviside((self.ind0_SpiralandStripe_boundary2+self.nx_pouch+1)-self.an_4T[ind0_plot],1)
        node8to10_penalty = np.heaviside(self.an_4T[ind0_plot]-(self.ind0_SpiralandStripe_boundary2+self.nx_pouch+1+1),1) * np.heaviside((self.ind0_SpiralandStripe_boundary2+self.nx_pouch+1+self.nx_pouch)-self.an_4T[ind0_plot],1)
        node_lastsection_penalty = np.heaviside(self.an_4T[ind0_plot]-(self.ind0_SpiralandStripe_boundary2+2*self.nx_pouch+1+1),1)   
        
        L_pouch = (   node1to2_penalty * 0                                                                            
                    + node3to5_penalty * ((self.an_4T[ind0_plot]-1)%self.nx-self.ind0_SpiralandStripe_boundary1) 
                    + node6to7_penalty * self.nx_pouch                                                
                    + node8to10_penalty * (self.ind0_SpiralandStripe_boundary2 + 2*self.nx_pouch - (self.an_4T[ind0_plot]-1)%self.nx)  
                    + node_lastsection_penalty * 0  ) * self.Lx_pouch/self.nx_pouch                                     
        return L_pouch
    #########################################################   
    ################## function for matrix1 #################
    #########################################################
    def fun_matrix1(self):
        node=np.arange(1, self.ntotal+1)
        an_unit=np.linspace(1,self.nx,self.nx,dtype=int)     #repeating node number for an
        ra_unit=np.linspace(1,self.nz,self.nz,dtype=int)   #repeating node number for ra
        ax_unit=np.linspace(1,self.ny,self.ny,dtype=int)     #repeating node number for ax
        an=np.tile(an_unit,self.nz*self.ny) 
        ra=np.repeat(ra_unit,self.nx); ra=np.tile(ra,self.ny)
        ax=np.repeat(ax_unit,self.nx*self.nz)        
    
        lap=(ra-1)//((self.ne+1)*2)+1

        theta_unit = np.concatenate(( self.theta_unit0[:self.ind0_SpiralandStripe_boundary1+1], np.repeat(np.pi/2,self.nx_pouch), self.theta_unit0[self.ind0_SpiralandStripe_boundary1+1:self.ind0_SpiralandStripe_boundary2+1], np.repeat(3*np.pi/2,self.nx_pouch), self.theta_unit0[self.ind0_SpiralandStripe_boundary2+1:] )) 
        theta = 2*np.pi*(lap-1) + np.tile(theta_unit,self.nz*self.ny)

        mat_unit=np.array([1,3,2,4])                      #repeating node number for mat
        mat_tmp1=np.repeat(mat_unit,[self.nx,self.ne*self.nx,self.nx,self.ne*self.nx])  #within one lap Al,El_b,Cu,El_r [1,3,2,4]  ax=1
        mat_tmp2=np.tile(mat_tmp1,self.nstack)                   #within two laps Al,El_b,Cu,El_r [1,3,2,4,1,3,2,4]  ax=1
        mat_tmp3=mat_tmp2[:(self.nz*self.nx)]                      #within two laps after trimming Al,El_b,Cu,El_r [1,3,2,4,1,3,2]  ax=1
        mat=np.tile(mat_tmp3,self.ny)
    
        Al=np.where(mat==1)[0]; Cu=np.where(mat==2)[0]; Elb=np.where(mat==3)[0]; Elr=np.where(mat==4)[0]   #for case of 192 nodes, ind from 0 to 191
    
        xi=np.zeros(self.ntotal)
        xi[Al]=self.fun_SpiralandStripefrom0(self.a0,self.b01,self.Lx_pouch, theta[Al],an[Al])    
        xi[Cu]=self.fun_SpiralandStripefrom0(self.a0,self.b02,self.Lx_pouch, theta[Cu],an[Cu])
        xi[Elb]=self.fun_SpiralandStripefrom0(self.a0,self.b03,self.Lx_pouch, theta[Elb],an[Elb])
        xi[Elr]=self.fun_SpiralandStripefrom0(self.a0,self.b04,self.Lx_pouch, theta[Elr],an[Elr])
     
        zi=np.zeros(self.ntotal)
        zi[Al]=self.a0*theta[Al]+self.b01
        zi[Cu]=self.a0*theta[Cu]+self.b02
        zi[Elb]=self.a0*theta[Elb]+self.b03
        zi[Elr]=self.a0*theta[Elr]+self.b04
    
        if self.ny==1:
            yi=np.zeros(self.ntotal)
        else:
            yi=(ax-1)*self.LG/(self.ny-1)
    
    #    node1=node.astype(float)
    #    for i in (node-1):                                     #for later use: for node neighbors in thermal model 
    #        if (ra[i]%(ne+1)!=1) and (ra[i]%(ne+1)!=2):
    #            node1[i]=None
        #---------------------------- neighbor node number ------------------------------
    
        jx1=np.zeros(self.ntotal,dtype=int)            #initialize left-neighbor node number in x direction
        jx2=np.zeros(self.ntotal,dtype=int)            #initialize right-neighbor node number in x direction
        jy1=np.zeros(self.ntotal,dtype=int)            #initialize up-neighbor node number in y direction
        jy2=np.zeros(self.ntotal,dtype=int)            #initialize down-neighbor node number in y direction
        jz1=np.zeros(self.ntotal,dtype=int)            #initialize inner-neighbor node number in z direction
        jz2=np.zeros(self.ntotal,dtype=int)            #initialize outer-neighbor node number in z direction
        
        for i in (node-1):
            if self.nx==1:                                     #for lumped model(nx=1), all nodes in line, no left neighbor
                jx1[i]=np.array([-9999])        
            elif an[i]==1:
                if ra[i] <= self.ne*2+2:                               
                    jx1[i]=np.array([-9999])                           #for node[i] with an==1, starting nodes, no left-neighbor number jx1[i]
                else:
                    jx1[i]=node[i] - ((self.ne*2+2)*self.nx-self.nx+1)  
            else:
                jx1[i]=node[i]-1                          #for node[i], left-neighbor number jx1[i] is node[i]-1
    
        for i in (node-1):
            if self.nx==1:                                     #for lumped model(nx=1), all nodes in line, no right neighbor
                jx2[i]=np.array([-9999])
            elif an[i]==self.nx:
                if ra[i] > self.nz-2*self.ne-2: #for node[i] with an==4, ending nodes, no right-neighbor number jx2[i]
                    jx2[i]=np.array([-9999])
                else:
                    jx2[i]=node[i] + ((self.ne*2+2)*self.nx-self.nx+1)            
            else:
                jx2[i]=node[i]+1                          #for node[i], right-neighbor number jx2[i] is node[i]+1
    
        for i in (node-1):
            if ax[i]==1:
                jy1[i]=np.array([-9999])                               #for node[i] in the top layer, no up-neighbor number jy1[i] 
            else:
                jy1[i]=node[i]-self.nx*self.nz                    
        for i in (node-1):
            if ax[i]==self.ny:
                jy2[i]=np.array([-9999])                               #for node[i] in the bottom layer, no down-neighbor number jy2[i]
            else:
                jy2[i]=node[i]+self.nx*self.nz
                
        for i in (node-1):
            if ra[i]==1:
                jz1[i]=np.array([-9999])                               #for node[i] in the innermost lap, no inner-neighbor number jz1[i]
            else:
                jz1[i]=node[i]-self.nx
        for i in (node-1):
            if ra[i]==self.nz:
                jz2[i]=np.array([-9999])
            else:
                jz2[i]=node[i]+self.nx                        #for node[i] in the outermost lap, no outer-neighbor number jz2[i]
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
        Lx_ele_lib=np.zeros([2*self.nstack-1,self.nx])#elementary spiral length library, for example nx=4,ny=3,nstack=2 there are 10 kinds of volume, which is a library
        V_ele_lib=np.zeros([2*self.nstack-1,self.nx]) #volume library, for example nx=4,ny=3,nstack=2 there are 10 kinds of volume, which is a library
        for i0 in np.arange(2*self.nstack-1):    #i0: 0index of rows in library array, for example, 0,1,2
            for j0 in np.arange(self.nx):      #j0: 0index 0f column in library array, for example, 0,1,2,3
                node_i0=i0*(self.nx*(self.ne+1))+j0  #node_i0: 0index of node, described in i0, j0
                if not( (i0==2*self.nstack-3 and j0==self.nx-1) or (i0==2*self.nstack-2 and j0==self.nx-1) ):  #There shouldn't be the two ending volumes; if not the two volumes case, do the following                       
                    if i0%2==0 and an[node_i0]!=self.nx:            #if ELb case, use b03; if other nodes, for example node 1,2,3
                        Lx_ele_lib[i0,j0]= self.fun_SpiralandStripefrom0(self.a0,self.b03,self.Lx_pouch, theta[node_i0+1],an[node_i0+1])-self.fun_SpiralandStripefrom0(self.a0,self.b03,self.Lx_pouch, theta[node_i0],an[node_i0]) 
                        V_ele_lib[i0,j0]=Lx_ele_lib[i0,j0] * (self.LG/(self.ny-1)) * self.delta_El
                    elif i0%2==0 and an[node_i0]==self.nx:          #if ELb case, use b03; if ending node in angular direction, for example in nx=4,ny=3,nstack=2 case, node 4 connects node 41
                        Lx_ele_lib[i0,j0]= self.fun_SpiralandStripefrom0(self.a0,self.b03,self.Lx_pouch, theta[node_i0+self.nx*2*(self.ne+1)-self.nx+1],an[node_i0+self.nx*2*(self.ne+1)-self.nx+1])-self.fun_SpiralandStripefrom0(self.a0,self.b03,self.Lx_pouch, theta[node_i0],an[node_i0]) 
                        V_ele_lib[i0,j0]=Lx_ele_lib[i0,j0] * (self.LG/(self.ny-1)) * self.delta_El
                    elif i0%2==1 and an[node_i0]!=self.nx:          #if ELr case, use b04
                        Lx_ele_lib[i0,j0]= self.fun_SpiralandStripefrom0(self.a0,self.b04,self.Lx_pouch, theta[node_i0+1],an[node_i0+1])-self.fun_SpiralandStripefrom0(self.a0,self.b04,self.Lx_pouch, theta[node_i0],an[node_i0]) 
                        V_ele_lib[i0,j0]=Lx_ele_lib[i0,j0] * (self.LG/(self.ny-1)) * self.delta_El
                    else:
                        Lx_ele_lib[i0,j0]= self.fun_SpiralandStripefrom0(self.a0,self.b04,self.Lx_pouch, theta[node_i0+self.nx*2*(self.ne+1)-self.nx+1],an[node_i0+self.nx*2*(self.ne+1)-self.nx+1])-self.fun_SpiralandStripefrom0(self.a0,self.b04,self.Lx_pouch, theta[node_i0],an[node_i0]) 
                        V_ele_lib[i0,j0]=Lx_ele_lib[i0,j0] * (self.LG/(self.ny-1)) * self.delta_El
                else:    #case of the two ending volumes
                    if i0%2==0:
                        Lx_ele_lib[i0,j0]= self.fun_SpiralandStripefrom0(self.a0,self.b03,self.Lx_pouch, (theta[node_i0]+2*np.pi/self.nx_cylindrical),1)-self.fun_SpiralandStripefrom0(self.a0,self.b03,self.Lx_pouch, theta[node_i0],an[node_i0]) 
                        V_ele_lib[i0,j0]=Lx_ele_lib[i0,j0] * (self.LG/(self.ny-1)) * self.delta_El
                    else:
                        Lx_ele_lib[i0,j0]= self.fun_SpiralandStripefrom0(self.a0,self.b04,self.Lx_pouch, (theta[node_i0]+2*np.pi/self.nx_cylindrical),1)-self.fun_SpiralandStripefrom0(self.a0,self.b04,self.Lx_pouch, theta[node_i0],an[node_i0]) 
                        V_ele_lib[i0,j0]=Lx_ele_lib[i0,j0] * (self.LG/(self.ny-1)) * self.delta_El
                        
        #the above gets the volume library
        V=np.zeros([self.ntotal,1])  #V is volume of each node, in the form of 1,2...ntotal. When two nodes belong to the same element, they have the same volume i.e. the elementary volume
        for i0 in node-1:   
            if mat[i0]>=3:                #for El nodes  
                i00=(ra[i0]-1)//(self.ne+1); j00=an[i0]-1    #i00 is the row 0index in V_ele_lib, for example, 0,1,2; j00 is the column 0index in V_ele_lib, for example, 0,1,2,3
                V[i0]=V_ele_lib[i00,j00]  
            elif mat[i0]==1:              #for Al nodes 
                if an[i0]!=self.nx:
                    A_Al=self.delta_Al*(xi[i0+1]-xi[i0])
                else:
                    A_Al=self.delta_Al*(self.fun_SpiralandStripefrom0(self.a0,self.b01,self.Lx_pouch,  lap[i0]*2*np.pi,1) - self.fun_SpiralandStripefrom0(self.a0,self.b01,self.Lx_pouch,  theta[i0],an[i0]))
                V[i0]=A_Al*self.LG/(self.ny-1)
            else:                         #for Cu nodes
                if an[i0]!=self.nx:
                    A_Cu=self.delta_Cu*(xi[i0+1]-xi[i0])
                else:
                    A_Cu=self.delta_Cu*(self.fun_SpiralandStripefrom0(self.a0,self.b02,self.Lx_pouch,  lap[i0]*2*np.pi,1) - self.fun_SpiralandStripefrom0(self.a0,self.b02,self.Lx_pouch,  theta[i0],an[i0]))
                V[i0]=A_Cu*self.LG/(self.ny-1)                    
        Lx_ele=np.zeros([self.nECN,1])   #Lx_ele is spiral length of each ECN element, in the form of 1,2...nECN
        V_ele=np.zeros([self.nECN,1])    #V_ele is volume of each ECN element, in the form of 1,2...nECN
        for i0 in np.arange(self.nECN):
            nECN_1layer = self.nECN//self.ny
            ind_temp=i0%nECN_1layer    #element repeating 0index in each layer (e.g. 0~3 when nx=4,ny=3,nstack=2)
            i00=ind_temp//self.nx; j00=ind_temp%self.nx   #i00 is the row 0index in V_ele_lib, for example, 0,1,2; j00 is the column 0index in V_ele_lib, for example, 0,1,2,3
            Lx_ele[i0]=Lx_ele_lib[i00,j00]
            V_ele[i0]=V_ele_lib[i00,j00]   
        Axy_ele=Lx_ele*self.LG/(self.ny-1)     #Axy_ele is Electrodes cross section area of each ECN element, in the form of 1,2...nECN
        #---------------------------- nodes in each element ------------------------------
        El_in_nodeE=np.zeros([self.nECN*self.ne,1],dtype=int); counter0=0   #El_in_nodeE is all the nodes in electrode
        for i0 in node-1:
            if mat[i0]>2:
                El_in_nodeE[counter0]=node[i0]
                counter0=counter0+1
        #the above gets all the nodes in electrode: El_in_nodeE
        
        n_before=self.nx*(2*self.nstack-1)*(ax[El_in_nodeE-1]-1) + self.nx*((ra[El_in_nodeE-1]-1)//(self.ne+1)) + (an[El_in_nodeE-1]-1)   #copied
        indele=n_before   #copied
        #the two lines above are copied from fun_node2ele;  Here we want to use this fun by "indele=fun_node2ele(El_in_nodeE)" but can not. Because the fun needs ax,ra,an but they are not generated until finishing fun_matrix1
        #the above transforms node: El_in_nodeE into element: indele
        
        node_ele=np.zeros([self.nECN,self.ne],dtype=int)  #node_ele is El nodes for each element
        for i0 in np.arange(self.nECN):
            node_ele[i0] = El_in_nodeE[np.where(indele==i0)]
        #the above gets node_ele   e.g. when nx=2,ny=2,nstack=1,nRC=1, there are 4 ECN elements, node_ele is array[[3,5],[4,6],[11,13],[12,14]]
                     
        return node, ax, ra, an, lap, theta, mat, xi, yi, zi, V, V_ele, Lx_ele, Axy_ele, node_ele, Al, Cu, Elb, Elr,                                                                                          \
               jx1, jx2, jy1, jy2, jz1, jz2, ind0_jx1, ind0_jx2, ind0_jy1, ind0_jy2, ind0_jz1, ind0_jz2,                                                                                             \
               ind0_jx1NaN, ind0_jx2NaN, ind0_jy1NaN, ind0_jy2NaN, ind0_jz1NaN, ind0_jz2NaN, ind0_jx1NonNaN, ind0_jx2NonNaN, ind0_jy1NonNaN, ind0_jy2NonNaN, ind0_jz1NonNaN, ind0_jz2NonNaN            
    #########################################################   
    ######## function for ρ_bulk, ρc_bulk and c_bulk ########
    ########      λx_bulk, λy_bulk and λz_bulk       ########
    #########################################################
    def fun_bulk_4T(self):
#        global rou_bulk, c_bulk, rouXc_bulk, V_sum
        if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
            self.V_Sep_4T=self.V_stencil_4T_ALL[self.ind0_Geo_core_AddSep_4T_4SepFill].reshape(-1,1)
            self.V_sum=np.sum(self.V_stencil_4T_ALL[:(self.ntotal_4T+self.nAddCore_4T)])
            self.rou_bulk=np.sum(self.rou_Al*(self.V_stencil_4T_ALL[self.Al_4T]/self.V_sum)) + np.sum(self.rou_Cu*(self.V_stencil_4T_ALL[self.Cu_4T]/self.V_sum)) + np.sum(self.rou_El*(self.V_stencil_4T_ALL[self.Elb_4T]/self.V_sum)) + np.sum(self.rou_El*(self.V_stencil_4T_ALL[self.Elr_4T]/self.V_sum)) + np.sum(self.rou_Sep*(self.V_Sep_4T/self.V_sum))
            self.rouXc_bulk=np.sum(self.rou_Al*self.c_Al*(self.V_stencil_4T_ALL[self.Al_4T]/self.V_sum)) + np.sum(self.rou_Cu*self.c_Cu*(self.V_stencil_4T_ALL[self.Cu_4T]/self.V_sum)) + np.sum(self.rouXc_El*(self.V_stencil_4T_ALL[self.Elb_4T]/self.V_sum)) + np.sum(self.rouXc_El*(self.V_stencil_4T_ALL[self.Elr_4T]/self.V_sum)) + np.sum(self.rou_Sep*self.c_Sep*(self.V_Sep_4T/self.V_sum))
            self.c_bulk=self.rouXc_bulk/self.rou_bulk
        else:
            self.V_sum=np.sum(self.V_stencil_4T_ALL[:self.ntotal_4T])
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
#        global weight_core, weight_Jellyroll, weight_Membrane, weight_Can, weight_Total, EnergyDensity     
        SpiralLength_Al = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real/2+self.delta_core_real, self.nstack_real*2*np.pi )
        SpiralLength_Cu = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real+self.delta_El_real+self.delta_Cu_real/2+self.delta_core_real, self.nstack_real*2*np.pi )
        SpiralLength_Elb_Ca = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real+self.delta_Ca_real/2+self.delta_core_real, self.nstack_real*2*np.pi )
        SpiralLength_Elb_Sep = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real+self.delta_Ca_real+self.delta_Sep_real/2+self.delta_core_real, self.nstack_real*2*np.pi )
        SpiralLength_Elb_An = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real+self.delta_Ca_real+self.delta_Sep_real+self.delta_An_real/2+self.delta_core_real, self.nstack_real*2*np.pi )
        SpiralLength_Elr_An = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real+self.delta_El_real+self.delta_Cu_real+self.delta_An_real/2+self.delta_core_real, (self.nstack_real-1)*2*np.pi )
        SpiralLength_Elr_Sep = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real+self.delta_El_real+self.delta_Cu_real+self.delta_An_real+self.delta_Sep_real/2+self.delta_core_real, (self.nstack_real-1)*2*np.pi )
        SpiralLength_Elr_Ca = self.fun_spiralfrom0( (2*self.delta_El_real+self.delta_Cu_real+self.delta_Al_real)/2/np.pi, self.delta_Al_real+self.delta_El_real+self.delta_Cu_real+self.delta_An_real+self.delta_Sep_real+self.delta_Ca_real/2+self.delta_core_real, (self.nstack_real-1)*2*np.pi )
    
        weight_Al = SpiralLength_Al*self.Ly_electrodes_real*self.delta_Al_real*self.rou_Al 
        weight_Cu = SpiralLength_Cu*self.Ly_electrodes_real*self.delta_Cu_real*self.rou_Cu 
        weight_Elb = (SpiralLength_Elb_Ca*self.delta_Ca_real*self.rou_Ca + SpiralLength_Elb_Sep*self.delta_Sep_real*self.rou_Sep + SpiralLength_Elb_An*self.delta_An_real*self.rou_An)*self.Ly_electrodes_real
        weight_Elr = (SpiralLength_Elr_Ca*self.delta_Ca_real*self.rou_Ca + SpiralLength_Elr_Sep*self.delta_Sep_real*self.rou_Sep + SpiralLength_Elr_An*self.delta_An_real*self.rou_An)*self.Ly_electrodes_real
    
        self.weight_core = self.S_SpiralandLine * self.Ly_electrodes_real * self.rou_Sep
        
        self.weight_Jellyroll = weight_Al + weight_Cu + weight_Elb + weight_Elr + self.weight_core
    
        self.weight_Membrane = np.pi * ( (self.delta_cell + self.delta_Membrane)**2 - self.delta_cell**2 ) * self.Ly_electrodes_real * self.rou_Sep
    
        Can_volume = np.pi * ( (self.delta_cell + self.delta_Membrane + self.delta_Can_real)**2 - (self.delta_cell + self.delta_Membrane)**2 ) * (self.LG_Can-self.delta_Can_real*2)  +  np.pi*(self.delta_cell + self.delta_Membrane + self.delta_Can_real)**2*self.delta_Can_real*2
        self.weight_Can = Can_volume * self.rou_Can
            
        self.weight_Total = self.weight_core + self.weight_Jellyroll + self.weight_Membrane + self.weight_Can
        
        self.EnergyDensity = self.SpecSheet_Energy/self.weight_Total
    #########################################################   
    #  functions for weighted avg & std of nodes temperature#
    #########################################################
    def fun_weighted_avg_and_std(self, T, rou_c_V_weights):
        if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
            weight_avg=np.average(self.T3_4T_ALL[:(self.ntotal_4T+self.nAddCore_4T)],weights=self.rou_c_V_weights)
            weight_std=np.sqrt( np.average( (self.T3_4T_ALL[:(self.ntotal_4T+self.nAddCore_4T)]-weight_avg)**2,weights=self.rou_c_V_weights ) )
            T_Delta=np.max(self.T3_4T_ALL[:(self.ntotal_4T+self.nAddCore_4T)])-np.min(self.T3_4T_ALL[:(self.ntotal_4T+self.nAddCore_4T)])
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
                            b=self.b01; delta=self.delta_Al; Conductivity=self.Conductivity_Al
                        if self.mat[i0]==2:
                            b=self.b02; delta=self.delta_Cu; Conductivity=self.Conductivity_Cu
                        
                        if self.yi[i0]==self.yi[j0]:                     #1. horizontal resistance     yi[j0]>yi[i0] because j0 is larger than i0
                            L=self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  self.theta[j0],self.an[j0]) - self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  self.theta[i0],self.an[i0])    
                            A=delta*self.LG/(self.ny-1)
                            R=L/A/Conductivity *(self.scalefactor_z**2)    # ※ difference between Cylindrical code version and Pouch code version, refer to p217 ECM52 and p125                 
                        else:                                  #2. vertical resistance     yi[j0]>yi[i0] because j0 is larger than i0
                            L=self.LG/(self.ny-1)
                            if self.an[i0] != self.nx:
                                A=delta*(self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  self.theta[i0+1],self.an[i0+1]) - self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  self.theta[i0],self.an[i0]))
                            else:
                                if self.ra[i0] <= self.nz-2*(self.ne+1):    #for vertical resistance of node4,68 and node24,88,
                                    A=delta*(self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  self.theta[i0+self.nx*2*(self.ne+1)-self.nx+1],self.an[i0+self.nx*2*(self.ne+1)-self.nx+1]) - self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  self.theta[i0],self.an[i0]))
                                else:                          #for vertical resistance of node44,108 and node64,128
                                    A=delta*(self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  (self.theta[i0]+2*np.pi/self.nx_cylindrical),1) - self.fun_SpiralandStripefrom0(self.a0,b,self.Lx_pouch,  self.theta[i0],1))
                            R=L/A/Conductivity                   
                        if self.mat[i0]==1:
                            self.RAl_pair[counter0,0]=i0; self.RAl_pair[counter0,1]=j0; self.RAl_pair[counter0,2]=R; counter0=counter0+1
                        if self.mat[i0]==2:
                            self.RCu_pair[counter1,0]=i0; self.RCu_pair[counter1,1]=j0; self.RCu_pair[counter1,2]=R; counter1=counter1+1
                        
                    elif self.an[i0]==self.an[j0] and self.ax[i0]==self.ax[j0]:                                                  #ii. Elb and Elr case
                        temp4=(self.ra[i0]-1)%(self.ne+1); temp5=(self.ra[j0]-1)%(self.ne+1)            #no. in lumped RCs. For example, temp3 and temp4 can be 0,1,2,3,4                                 
                        indRC=np.where((ECN_pair_lib==[temp4,temp5]).all(1))[0]             #find the row index of link case [temp1,temp2] in ECN_pair_lib    np.where((ECN_pair_lib==[temp4,temp5]).all(1)) returns tuple, so add [0] in the end
                        if indRC==0:        #1. case of R0
                            self.R0_pair[counter2,0]=i0; self.R0_pair[counter2,1]=j0; counter2=counter2+1
                        elif indRC==self.nRC+1:  #2. case of Ei
                            self.Ei_pair[counter4,0]=i0; self.Ei_pair[counter4,1]=j0; counter4=counter4+1          #for later use in computing RectangleC2                        
                        else:               #3. case of R123 and C123                    
                            self.RC_pair[counter3,0]=i0; self.RC_pair[counter3,1]=j0; self.RC_pair[counter3,4]=indRC; counter3=counter3+1
    
#        global ind0_EleIsElb, ind0_EleIsElr, ind0_AlOfElb, ind0_CuOfElb, ind0_AlOfElr, ind0_CuOfElr, ind0_ele_RC_pair, ind0_ele_RC_pair_4T
        self.ind0_EleIsElb = np.where(self.mat[self.Ei_pair[:,0].astype(int)]==3)[0];   self.ind0_EleIsElr = np.where(self.mat[self.Ei_pair[:,0].astype(int)]==4)[0]
        self.ind0_AlOfElb = self.jz1[ self.List_ele2node[self.ind0_EleIsElb][:,0] ]-1                    #for all Elb elements (in the increasing order), get ind0 of Al node 
        self.ind0_CuOfElb = self.jz2[ self.List_ele2node[self.ind0_EleIsElb][:,-1] ]-1                   #for all Elb elements (in the increasing order), get ind0 of Cu node 
        self.ind0_AlOfElr = self.jz2[ self.List_ele2node[self.ind0_EleIsElr][:,-1] ]-1                   #for all Elr elements (in the increasing order), get ind0 of Al node 
        self.ind0_CuOfElr = self.jz1[ self.List_ele2node[self.ind0_EleIsElr][:,0] ]-1                    #for all Elr elements (in the increasing order), get ind0 of Cu node 
        self.ind0_ele_RC_pair = np.concatenate(np.split(np.arange(self.nECN*self.nRC).reshape(-1,self.nx).T,self.nECN/self.nx,1))                                         #prepare this for MatrixC_neo and I_neo. RC_pair is not in the sequence of elements, ind0_ele_RC_pair gives the ind0 in element sequence    notebook p62                     
        if self.nRC != 0:
            self.ind0_ele_RC_pair_4T = self.ind0_ele_RC_pair[:,0]                  

    #########################################################   
    ###########      function for Thermal BC      ###########
    #########################################################
    def fun_BC_4T_ALL(self):         # output vector T_4T (after constraining temperature on BC nodes)
    #    if status_ThermalBC_Core=='BCFix':
    #        T_4T=np.zeros([ntotal_4T,1])             
    #        T_4T[ind0_Geo_top_4T_4BCFix]=T_cooling                   #constrain temperature on base top nodes 
    #        T_4T[ind0_Geo_bottom_4T_4BCFix]=T_cooling                #constrain temperature on base bottom nodes
    #        T_4T[ind0_Geo_surface_4T_4BCFix]=T_cooling               #constrain temperature on jelly roll surface nodes
    #        T_4T[ind0_Geo_finalcrosssection_4T_4BCFix]=T_cooling     #constrain temperature on final cross section nodes
    #        T_4T[ind0_Geo_core_4T_4BCFix]=T_cooling                  #constrain temperature on core nodes    
#        global T3_4T_ALL, ind0_BCtem_ALL, ind0_BCtem_others_ALL, h_4T_ALL, Tconv_4T_ALL, ind0_BCconv_ALL, ind0_BCconv_others_ALL
        if self.status_TabSurface_Scheme=='TabConv_SurTem':     
            #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
            T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        

            if self.status_TemBC_smoothening=='No':
                T3_4T_4SepFill[self.ind0_Geo_surface_4T_4SepFill]=self.T_cooling                                                                                                          #BC on surface nodes
                T3_4T_4SepFill[self.ind0_Geo_finalcrosssection_4T_4SepFill]=self.T_cooling                                                                                                #BC on final cross section nodes
            else:
                T3_4T_4SepFill[self.ind0_Geo_surface_4T_4SepFill]=self.T_cooling_smoothened                                                                                               #BC on surface nodes
                T3_4T_4SepFill[self.ind0_Geo_finalcrosssection_4T_4SepFill]=self.T_cooling_smoothened                                                                                     #BC on final cross section nodes                
                #-----------------------------get all constrained node number
            ind0_BCtem_4SepFill=np.concatenate((self.ind0_Geo_surface_4T_4SepFill,self.ind0_Geo_finalcrosssection_4T_4SepFill))      #all the temperature-constrained nodes        
            ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
            h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

            h_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= 50                                                                                                                     #BC on top nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= 50
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= 50         
        
            h_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= 50                                                                                                                  #BC on bottom nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= 50         
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= 50        
                #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

            Tconv_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= self.T_cooling        
                #-----------------------------get all constrained node number                
            ind0_BCconv_4SepFill=np.concatenate((self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill))     #all the convection-constrained nodes
            ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='TabTem_SurConv':     
            #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
            T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        

            if self.status_TemBC_smoothening=='No':
                T3_4T_4SepFill[self.ind0_Geo_top_4T_4SepFill]=self.T_cooling                                                                                                              #BC on top nodes
                T3_4T_4SepFill[self.ind0_Geo_bottom_4T_4SepFill]=self.T_cooling                                                                                                           #BC on bottom nodes
            else:
                T3_4T_4SepFill[self.ind0_Geo_top_4T_4SepFill]=self.T_cooling_smoothened                                                                                                   #BC on top nodes
                T3_4T_4SepFill[self.ind0_Geo_bottom_4T_4SepFill]=self.T_cooling_smoothened                                                                                                #BC on bottom nodes                
                #-----------------------------get all constrained node number
            ind0_BCtem_4SepFill=np.concatenate((self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill))      #all the temperature-constrained nodes        
            ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
            h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

            h_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= 50                                                                                                                 #BC on surface nodes        
            h_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= 50
        
            h_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= 50                                                                                                       #BC on finalcrosssection nodes
                #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

            Tconv_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= self.T_cooling                                                                                                      
            Tconv_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= self.T_cooling        
                #-----------------------------get all constrained node number                
            ind0_BCconv_4SepFill=np.concatenate((self.ind0_Geo_surface_4T_4SepFill,self.ind0_Geo_finalcrosssection_4T_4SepFill))                                                          #all the convection-constrained nodes
            ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='AllConv':     
            #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
            T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        


                #-----------------------------get all constrained node number
            ind0_BCtem_4SepFill=np.array([],dtype=int)                                                                                                                                              #all the temperature-constrained nodes        
            ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
            h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

            h_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= 0.0                                                                                                                     #BC on top nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= 0.0
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= 0.0         
        
            h_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= 0.0                                                                                                                  #BC on bottom nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= 0.0         
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= 0.0
        
            h_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= 0.0                                                                                                                 #BC on surface nodes        
            h_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= 0.0
        
            h_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= 0.0                                                                                                       #BC on finalcrosssection nodes        
                #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

            Tconv_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= self.T_cooling

            Tconv_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= self.T_cooling
            Tconv_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= self.T_cooling                
                #-----------------------------get all constrained node number                
            ind0_BCconv_4SepFill=np.concatenate((self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill,self.ind0_Geo_surface_4T_4SepFill,self.ind0_Geo_finalcrosssection_4T_4SepFill))     #all the convection-constrained nodes
            ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='AllTem':     
            #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
            T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        
            if self.status_TemBC_smoothening=='No':
                T3_4T_4SepFill[self.ind0_Geo_surface_4T_4SepFill]=self.T_cooling                                                                                                          #constrain temperature on surface nodes
                T3_4T_4SepFill[self.ind0_Geo_finalcrosssection_4T_4SepFill]=self.T_cooling                                                                                                #constrain temperature on final cross section nodes
                T3_4T_4SepFill[self.ind0_Geo_top_4T_4SepFill]=self.T_cooling                                                                                                              #constrain temperature on base top nodes
                T3_4T_4SepFill[self.ind0_Geo_bottom_4T_4SepFill]=self.T_cooling                                                                                                           #constrain temperature on base bottom nodes
            else:
                T3_4T_4SepFill[self.ind0_Geo_surface_4T_4SepFill]=self.T_cooling_smoothened                                                                                               #constrain temperature on surface nodes
                T3_4T_4SepFill[self.ind0_Geo_finalcrosssection_4T_4SepFill]=self.T_cooling_smoothened                                                                                     #constrain temperature on final cross section nodes
                T3_4T_4SepFill[self.ind0_Geo_top_4T_4SepFill]=self.T_cooling_smoothened                                                                                                   #constrain temperature on base top nodes
                T3_4T_4SepFill[self.ind0_Geo_bottom_4T_4SepFill]=self.T_cooling_smoothened                                                                                                #constrain temperature on base bottom nodes
                
                #-----------------------------get all constrained node number
            ind0_BCtem_4SepFill=np.concatenate((self.ind0_Geo_surface_4T_4SepFill,self.ind0_Geo_finalcrosssection_4T_4SepFill,self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill))      #all the temperature-constrained nodes        
            ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
            h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

#            h_4T_4SepFill[ ind0_Geo_top_4T_4SepFill,2 ]= 50                                                                                                                     #BC on top nodes
#            h_4T_4SepFill[ np.concatenate((ind0_Geo_edge25to27_4T_4SepFill,ind0_Geo_edge28_4T_4SepFill)),5 ]= 50
#            h_4T_4SepFill[ np.concatenate((ind0_Geo_top16_4T_4SepFill, ind0_Geo_top20_4T_4SepFill, ind0_Geo_top24_4T_4SepFill, ind0_Geo_edge28_4T_4SepFill)),1 ]= 50         
#        
#            h_4T_4SepFill[ ind0_Geo_bottom_4T_4SepFill,3 ]= 50                                                                                                                  #BC on bottom nodes
#            h_4T_4SepFill[ np.concatenate((ind0_Geo_edge81to83_4T_4SepFill,ind0_Geo_edge84_4T_4SepFill)),5 ]= 50         
#            h_4T_4SepFill[ np.concatenate((ind0_Geo_bottom72_4T_4SepFill, ind0_Geo_bottom76_4T_4SepFill, ind0_Geo_bottom80_4T_4SepFill, ind0_Geo_edge84_4T_4SepFill)),1 ]= 50
        
        
                #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

#            Tconv_4T_4SepFill[ ind0_Geo_top_4T_4SepFill,2 ]= T_cooling
#            Tconv_4T_4SepFill[ np.concatenate((ind0_Geo_edge25to27_4T_4SepFill,ind0_Geo_edge28_4T_4SepFill)),5 ]= T_cooling
#            Tconv_4T_4SepFill[ np.concatenate((ind0_Geo_top16_4T_4SepFill, ind0_Geo_top20_4T_4SepFill, ind0_Geo_top24_4T_4SepFill, ind0_Geo_edge28_4T_4SepFill)),1 ]= T_cooling
#        
#            Tconv_4T_4SepFill[ ind0_Geo_bottom_4T_4SepFill,3 ]= T_cooling
#            Tconv_4T_4SepFill[ np.concatenate((ind0_Geo_edge81to83_4T_4SepFill,ind0_Geo_edge84_4T_4SepFill)),5 ]= T_cooling
#            Tconv_4T_4SepFill[ np.concatenate((ind0_Geo_bottom72_4T_4SepFill, ind0_Geo_bottom76_4T_4SepFill, ind0_Geo_bottom80_4T_4SepFill, ind0_Geo_edge84_4T_4SepFill)),1 ]= T_cooling
        
                #-----------------------------get all constrained node number                
            ind0_BCconv_4SepFill=np.array([],dtype=int)     #all the convection-constrained nodes
            ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='SurfaceCooling':     
            #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
            T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        


                #-----------------------------get all constrained node number
            ind0_BCtem_4SepFill=np.array([],dtype=int)                                                                                                                                              #all the temperature-constrained nodes        
            ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
            h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

            h_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= 0                                                                                                                     #BC on top nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= 0
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= 0         
        
            h_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= 0                                                                                                                  #BC on bottom nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= 0         
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= 0
        
            h_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= 50                                                                                                                 #BC on surface nodes        
            h_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= 50
        
            h_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= 50                                                                                                       #BC on finalcrosssection nodes        
                #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

            Tconv_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= self.T_cooling

            Tconv_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= self.T_cooling
            Tconv_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= self.T_cooling                
                #-----------------------------get all constrained node number                
            ind0_BCconv_4SepFill=np.concatenate((self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill,))     #all the convection-constrained nodes
            ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='TabCooling':     
            #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
            T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        


                #-----------------------------get all constrained node number
            ind0_BCtem_4SepFill=np.array([],dtype=int)                                                                                                                                              #all the temperature-constrained nodes        
            ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
            h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

            h_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= 50                                                                                                                     #BC on top nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= 50
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= 50         
        
            h_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= 50                                                                                                                  #BC on bottom nodes
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= 50         
            h_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= 50
        
            h_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= 0                                                                                                                 #BC on surface nodes        
            h_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= 0
        
            h_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= 0                                                                                                       #BC on finalcrosssection nodes        
                #-----------------------------constrain heat convection temperature: Tconv
            Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

            Tconv_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= self.T_cooling
            Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= self.T_cooling

            Tconv_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= self.T_cooling
            Tconv_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= self.T_cooling
        
            Tconv_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= self.T_cooling                
                #-----------------------------get all constrained node number                
            ind0_BCconv_4SepFill=np.concatenate((self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill,))     #all the convection-constrained nodes
            ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
        if self.status_TabSurface_Scheme=='UserDefine':      #first half: TabCooling, second half: SurfaceCooling  
            if step <= self.nt/2:
                #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
                T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        

                #-----------------------------get all constrained node number
                ind0_BCtem_4SepFill=np.array([],dtype=int)                                                                                                                                              #all the temperature-constrained nodes        
                ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
                h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

                h_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= 50                                                                                                                     #BC on top nodes
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= 50
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= 50         
        
                h_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= 50                                                                                                                  #BC on bottom nodes
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= 50         
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= 50
        
                h_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= 0                                                                                                                 #BC on surface nodes        
                h_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= 0
        
                h_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= 0                                                                                                       #BC on finalcrosssection nodes        
                #-----------------------------constrain heat convection temperature: Tconv
                Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

                Tconv_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= self.T_cooling
        
                Tconv_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= self.T_cooling

                Tconv_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= self.T_cooling
                Tconv_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= self.T_cooling
        
                Tconv_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= self.T_cooling                
                #-----------------------------get all constrained node number                
                ind0_BCconv_4SepFill=np.concatenate((self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill))     #all the convection-constrained nodes
                ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
            else:
            #=============================temperature BC input
                #-----------------------------input: T and its coresponding node
                T3_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                                           #T_4T initialization, shape is (87,1)        

                #-----------------------------get all constrained node number
                ind0_BCtem_4SepFill=np.array([],dtype=int)                                                                                                                                              #all the temperature-constrained nodes        
                ind0_BCtem_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCtem_4SepFill],dtype=int)                                                 #all the other nodes
            #=============================convection BC input
                #-----------------------------constrain heat convection cof: h
                h_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                            #h_4T initialization, shape is (87,6)

                h_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= 0                                                                                                                     #BC on top nodes
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= 0
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= 0         
        
                h_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= 0                                                                                                                  #BC on bottom nodes
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= 0         
                h_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= 0
        
                h_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= 50                                                                                                                 #BC on surface nodes        
                h_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= 50
        
                h_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= 50                                                                                                       #BC on finalcrosssection nodes        
                #-----------------------------constrain heat convection temperature: Tconv
                Tconv_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                                        #Tconv_4T initialization, shape is (87,6)

                Tconv_4T_4SepFill[ self.ind0_Geo_top_4T_4SepFill,2 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge25to27_4T_4SepFill,self.ind0_Geo_edge28_4T_4SepFill)),5 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill)),1 ]= self.T_cooling
        
                Tconv_4T_4SepFill[ self.ind0_Geo_bottom_4T_4SepFill,3 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_edge81to83_4T_4SepFill,self.ind0_Geo_edge84_4T_4SepFill)),5 ]= self.T_cooling
                Tconv_4T_4SepFill[ np.concatenate((self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill)),1 ]= self.T_cooling

                Tconv_4T_4SepFill[ self.ind0_Geo_surface_4T_4SepFill,5 ]= self.T_cooling
                Tconv_4T_4SepFill[self.ind0_Geo_surface56_4T_4SepFill,1]= self.T_cooling
                
                Tconv_4T_4SepFill[ self.ind0_Geo_finalcrosssection_4T_4SepFill,1 ]= self.T_cooling                
                #-----------------------------get all constrained node number                
                ind0_BCconv_4SepFill=np.concatenate((self.ind0_Geo_top_4T_4SepFill,self.ind0_Geo_bottom_4T_4SepFill))     #all the convection-constrained nodes
                ind0_BCconv_others_4SepFill=np.array([x for x in np.arange(self.ntotal_4T+self.nAddCore_4T) if x not in ind0_BCconv_4SepFill],dtype=int)                                               #all the other nodes
        #-----------------------------exposed boundary nodes for Roll first_cross_section and edge. For core area due to SepFill case (Usually unchanged as long as it is cylindrical cell)
        h_4T_4SepFill[ self.ind0_Geo_firstcrosssectionWithEnds_4T_4SepFill,0 ]=0                                                                                                  #for these nodes, h=0
        Tconv_4T_4SepFill[ self.ind0_Geo_firstcrosssectionWithEnds_4T_4SepFill,0 ]=-9999

        self.T3_4T_ALL=T3_4T_4SepFill.copy()
        self.ind0_BCtem_ALL=ind0_BCtem_4SepFill.copy()
        self.ind0_BCtem_others_ALL=ind0_BCtem_others_4SepFill.copy()
        self.h_4T_ALL=h_4T_4SepFill.copy()
        self.Tconv_4T_ALL=Tconv_4T_4SepFill.copy()
        self.ind0_BCconv_ALL=ind0_BCconv_4SepFill.copy()
        self.ind0_BCconv_others_ALL=ind0_BCconv_others_4SepFill.copy()

    #########################################################   
    ######       function for thermal Geometry        #######
    ######################################################### 
    def fun_get_Geo_4T(self):
        if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
#            global ind0_Geo_top_4T_4SepFill, ind0_Geo_bottom_4T_4SepFill, ind0_Geo_surface_4T_4SepFill, ind0_Geo_core_4T_4SepFill, ind0_Geo_firstcrosssection_4T_4SepFill, ind0_Geo_finalcrosssection_4T_4SepFill      #0ind of geometry boundary nodes         
#            global ind0_Geo_top_edge_inner_4T_4SepFill, ind0_Geo_top_edge_outer_4T_4SepFill, ind0_Geo_bottom_edge_inner_4T_4SepFill, ind0_Geo_bottom_edge_outer_4T_4SepFill 
#            global ind0_Geo_coreWithEnds_4T_4SepFill, ind0_Geo_surfaceWithEnds_4T_4SepFill
#            global ind0_Geo_topNoSep_4T_4SepFill, ind0_Geo_bottomNoSep_4T_4SepFill
#            global ind0_Geo_top16_4T_4SepFill, ind0_Geo_top20_4T_4SepFill, ind0_Geo_top24_4T_4SepFill, ind0_Geo_top5_4T_4SepFill, ind0_Geo_top9_4T_4SepFill, ind0_Geo_top13_4T_4SepFill 
#            global ind0_Geo_bottom72_4T_4SepFill, ind0_Geo_bottom76_4T_4SepFill, ind0_Geo_bottom80_4T_4SepFill, ind0_Geo_bottom61_4T_4SepFill, ind0_Geo_bottom65_4T_4SepFill, ind0_Geo_bottom69_4T_4SepFill
#            global ind0_Geo_surface53to55_4T_4SepFill, ind0_Geo_surface56_4T_4SepFill
#            global ind0_Geo_finalcrosssection44_4T_4SepFill, ind0_Geo_finalcrosssection48_4T_4SepFill,ind0_Geo_finalcrosssection52_4T_4SepFill
#            global ind0_Geo_core29_4T_4SepFill, ind0_Geo_core33_4T_4SepFill, ind0_Geo_core37_4T_4SepFill, ind0_Geo_core41_4T_4SepFill, ind0_Geo_core30to32_4T_4SepFill
#            global ind0_Geo_edge25to27_4T_4SepFill, ind0_Geo_edge28_4T_4SepFill, ind0_Geo_edge2to4_4T_4SepFill, ind0_Geo_edge1_4T_4SepFill, ind0_Geo_edge81to83_4T_4SepFill, ind0_Geo_edge84_4T_4SepFill, ind0_Geo_edge58to60_4T_4SepFill, ind0_Geo_edge57_4T_4SepFill        
#            global ind0_Geo_Sep85_4T_4SepFill, ind0_Geo_Sep87_4T_4SepFill
#            global ind0_Geo_firstcrosssectionWithEnds_4T_4SepFill, ind0_Geo_finalcrosssectionWithEnds_4T_4SepFill
#            global ind0_Geo_node1_29_57_4T_4SepFill, ind0_Geo_node4_32_60_4T_4SepFill, ind0_Geo_node2_3_30_31_58_59_4T_4SepFill
#            global ind0_Geo_core_AddSep_4T_4SepFill, ind0_Geo_core_Al_4T_4SepFill
#            global ind0_Geo_Probe36_44_52_4T
            #---------------------------------------first level nodes                                                                                                                      #for status_ThermalBC_Core=='SepFill', these should be modified                 
                #---------------------------------------nodes in top/bottom base
            ind0_Geo_topNoSep_4T_4SepFill=np.where(self.ax_4T==1)[0]                             #i.g. ind0=[0,1,2~27]
            ind0_Geo_bottomNoSep_4T_4SepFill=np.where(self.ax_4T==self.ny)[0]                         #i.g. ind0=[56,57,58~83]
            ind0_Geo_top_4T=np.where(self.ax_4T==1)[0]
            ind0_Geo_top_4T_4SepFill=np.append(ind0_Geo_top_4T,self.ntotal_4T)
            ind0_Geo_bottom_4T=np.where(self.ax_4T==self.ny)[0]
            ind0_Geo_bottom_4T_4SepFill=np.append(ind0_Geo_bottom_4T,self.ntotal_4T+self.nAddCore_4T-1)
                #---------------------------------------nodes in jelly roll surface
            indtemp1=np.where(self.ra_4T==self.nz_4T)[0]     #0ind of all surface nodes   
            indtemp2=np.append(ind0_Geo_top_4T_4SepFill,ind0_Geo_bottom_4T_4SepFill)   #0ind of top and bottom nodes; prep for next line
            ind0_Geo_surface_4T_4SepFill=np.array([x for x in indtemp1 if x not in indtemp2],dtype=int)  #0ind of surface nodes; nodes double belonging to surface and top/bottom are excluded
                #---------------------------------------nodes in jelly roll core
            indtemp3_1=np.where(self.ra_4T==1)[0]   #indtemp3_1: 0ind of nodes on inner layer           
            indtemp3_2=np.argwhere(np.isnan(self.jx1_4T)).reshape(-1); indtemp3_3=np.array([x for x in indtemp3_2 if x not in indtemp3_1],dtype=int) #indtemp3_2: 0ind of nodes on starting cross section
            indtemp3=np.append(indtemp3_1,indtemp3_3)      #0ind of all core nodes: inner layer and starting cross section
            ind0_Geo_core_4T_4SepFill=np.array([x for x in indtemp3 if x not in indtemp2],dtype=int)    #0ind of core nodes
                #---------------------------------------nodes in final cross section
            indtemp4=np.where(self.an_4T==self.nx)[0]            #0ind of all nodes in final cross section
            indtemp5=np.where(self.ra_4T > (self.nz_4T-4) )[0]  #0ind of nodes in outer jelly roll; prep for next line
            indtemp6=np.array([x for x in indtemp4 if x in indtemp5])   #exclude outer jelly roll nodes from final cross section nodes
            ind0_Geo_finalcrosssection_4T_4SepFill=np.array([x for x in indtemp6 if (self.ax_4T[x]!=1 and self.ax_4T[x]!=self.ny and self.ra_4T[x]!=self.nz_4T)],dtype=int)  #0ind of final cross section; nodes double belonging to surface and top/bottom are excluded
            indtemp4_1=np.where(self.an_4T==1)[0]            #0ind of all nodes in first cross section
            indtemp5_1=np.where(self.ra_4T <= 4 )[0]         #0ind of nodes in inner jelly roll; prep for next line
            indtemp6_1=np.array([x for x in indtemp4_1 if x in indtemp5_1])   #exclude inner jelly roll nodes from first cross section nodes
            ind0_Geo_firstcrosssection_4T_4SepFill=np.array([x for x in indtemp6_1 if (self.ax_4T[x]!=1 and self.ax_4T[x]!=self.ny and self.ra_4T[x]!=1)],dtype=int)  #0ind of first cross section; nodes double belonging to surface and top/bottom are excluded
            #---------------------------------------second level nodes
            ind0_Geo_top_edge_inner_4T_4SepFill=np.array([x for x in ind0_Geo_topNoSep_4T_4SepFill if self.ra_4T[x]==1 ],dtype=int)            #i.g. ind0=[0,1,2,3]
            ind0_Geo_bottom_edge_inner_4T_4SepFill=np.array([x for x in ind0_Geo_bottomNoSep_4T_4SepFill if self.ra_4T[x]==1 ],dtype=int)      #i.g. ind0=[56,57,58,59]
            
            ind0_Geo_top_edge_outer_4T_4SepFill=np.array([x for x in ind0_Geo_topNoSep_4T_4SepFill if self.ra_4T[x]==self.nz_4T ],dtype=int)       #i.g. ind0=[24,25,26,27]
            ind0_Geo_bottom_edge_outer_4T_4SepFill=np.array([x for x in ind0_Geo_bottomNoSep_4T_4SepFill if self.ra_4T[x]==self.nz_4T ],dtype=int) #i.g. ind0=[80,81,82,83]
            ind0_Geo_coreWithEnds_4T_4SepFill=np.concatenate((ind0_Geo_top_edge_inner_4T_4SepFill,ind0_Geo_core_4T_4SepFill,ind0_Geo_bottom_edge_inner_4T_4SepFill))
            ind0_Geo_surfaceWithEnds_4T_4SepFill=np.concatenate((ind0_Geo_top_edge_outer_4T_4SepFill,ind0_Geo_surface_4T_4SepFill,ind0_Geo_bottom_edge_outer_4T_4SepFill))
            #---------------------------------------third level nodes
            ind0_Geo_top5_4T_4SepFill=np.array([x for x in ind0_Geo_top_4T if (self.mat_4T[x]==3 and self.ra_4T[x]==2 and self.an_4T[x]==1) ],dtype=int)
            ind0_Geo_top9_4T_4SepFill=np.array([x for x in ind0_Geo_top_4T if (self.mat_4T[x]==2 and self.ra_4T[x]==3 and self.an_4T[x]==1) ],dtype=int)
            ind0_Geo_top13_4T_4SepFill=np.array([x for x in ind0_Geo_top_4T if (self.mat_4T[x]==4 and self.ra_4T[x]==4 and self.an_4T[x]==1) ],dtype=int)
            ind0_Geo_top16_4T_4SepFill=np.array([x for x in ind0_Geo_top_4T if (self.mat_4T[x]==4 and self.ra_4T[x]==self.nz_4T-3 and self.an_4T[x]==self.nx) ],dtype=int)
            ind0_Geo_top20_4T_4SepFill=np.array([x for x in ind0_Geo_top_4T if (self.mat_4T[x]==1 and self.ra_4T[x]==self.nz_4T-2 and self.an_4T[x]==self.nx) ],dtype=int)
            ind0_Geo_top24_4T_4SepFill=np.array([x for x in ind0_Geo_top_4T if (self.mat_4T[x]==3 and self.ra_4T[x]==self.nz_4T-1 and self.an_4T[x]==self.nx) ],dtype=int)
            ind0_Geo_bottom61_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_4T if (self.mat_4T[x]==3 and self.ra_4T[x]==2 and self.an_4T[x]==1) ],dtype=int)
            ind0_Geo_bottom65_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_4T if (self.mat_4T[x]==2 and self.ra_4T[x]==3 and self.an_4T[x]==1) ],dtype=int)
            ind0_Geo_bottom69_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_4T if (self.mat_4T[x]==4 and self.ra_4T[x]==4 and self.an_4T[x]==1) ],dtype=int)
            ind0_Geo_bottom72_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_4T if (self.mat_4T[x]==4 and self.ra_4T[x]==self.nz_4T-3 and self.an_4T[x]==self.nx) ],dtype=int)
            ind0_Geo_bottom76_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_4T if (self.mat_4T[x]==1 and self.ra_4T[x]==self.nz_4T-2 and self.an_4T[x]==self.nx) ],dtype=int)
            ind0_Geo_bottom80_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_4T if (self.mat_4T[x]==3 and self.ra_4T[x]==self.nz_4T-1 and self.an_4T[x]==self.nx) ],dtype=int)
            ind0_Geo_surface53to55_4T_4SepFill=np.array([x for x in ind0_Geo_surface_4T_4SepFill if self.an_4T[x]!=self.nx ],dtype=int)
            ind0_Geo_surface56_4T_4SepFill=np.array([x for x in ind0_Geo_surface_4T_4SepFill if self.an_4T[x]==self.nx ],dtype=int)
            ind0_Geo_finalcrosssection44_4T_4SepFill=np.array([x for x in ind0_Geo_finalcrosssection_4T_4SepFill if self.mat_4T[x]==4 ],dtype=int)
            ind0_Geo_finalcrosssection48_4T_4SepFill=np.array([x for x in ind0_Geo_finalcrosssection_4T_4SepFill if self.mat_4T[x]==1 ],dtype=int)
            ind0_Geo_finalcrosssection52_4T_4SepFill=np.array([x for x in ind0_Geo_finalcrosssection_4T_4SepFill if self.mat_4T[x]==3 ],dtype=int)
            indtemp20=np.array([x for x in ind0_Geo_core_4T_4SepFill if self.an_4T[x]==1 ],dtype=int)
            ind0_Geo_core29_4T_4SepFill=np.array([x for x in indtemp20 if self.mat_4T[x]==1 ],dtype=int)
            ind0_Geo_core33_4T_4SepFill=np.array([x for x in indtemp20 if self.mat_4T[x]==3 ],dtype=int)
            ind0_Geo_core37_4T_4SepFill=np.array([x for x in indtemp20 if self.mat_4T[x]==2 ],dtype=int)
            ind0_Geo_core41_4T_4SepFill=np.array([x for x in indtemp20 if self.mat_4T[x]==4 ],dtype=int)
            ind0_Geo_core30to32_4T_4SepFill=np.array([x for x in ind0_Geo_core_4T_4SepFill if x not in np.concatenate((ind0_Geo_core29_4T_4SepFill,ind0_Geo_core33_4T_4SepFill,ind0_Geo_core37_4T_4SepFill,ind0_Geo_core41_4T_4SepFill)) ],dtype=int)
            ind0_Geo_edge25to27_4T_4SepFill=np.array([x for x in ind0_Geo_top_edge_outer_4T_4SepFill if self.an_4T[x]!=self.nx ],dtype=int)
            ind0_Geo_edge28_4T_4SepFill=np.array([x for x in ind0_Geo_top_edge_outer_4T_4SepFill if self.an_4T[x]==self.nx ],dtype=int)
            ind0_Geo_edge2to4_4T_4SepFill=np.array([x for x in ind0_Geo_top_edge_inner_4T_4SepFill if self.an_4T[x]!=1 ],dtype=int)
            ind0_Geo_edge1_4T_4SepFill=np.array([x for x in ind0_Geo_top_edge_inner_4T_4SepFill if self.an_4T[x]==1 ],dtype=int)
            ind0_Geo_edge81to83_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_edge_outer_4T_4SepFill if self.an_4T[x]!=self.nx ],dtype=int)
            ind0_Geo_edge84_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_edge_outer_4T_4SepFill if self.an_4T[x]==self.nx ],dtype=int)
            ind0_Geo_edge58to60_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_edge_inner_4T_4SepFill if self.an_4T[x]!=1 ],dtype=int)
            ind0_Geo_edge57_4T_4SepFill=np.array([x for x in ind0_Geo_bottom_edge_inner_4T_4SepFill if self.an_4T[x]==1 ],dtype=int)                     
            ind0_Geo_Sep85_4T_4SepFill=np.array([self.ntotal_4T])
            ind0_Geo_Sep87_4T_4SepFill=np.array([self.ntotal_4T+self.nAddCore_4T-1])
            ind0_Geo_firstcrosssectionWithEnds_4T_4SepFill=np.concatenate((ind0_Geo_edge1_4T_4SepFill, ind0_Geo_top5_4T_4SepFill, ind0_Geo_top9_4T_4SepFill, ind0_Geo_top13_4T_4SepFill,                       \
                                                                           ind0_Geo_core29_4T_4SepFill, ind0_Geo_firstcrosssection_4T_4SepFill,                                                                \
                                                                           ind0_Geo_edge57_4T_4SepFill, ind0_Geo_bottom61_4T_4SepFill, ind0_Geo_bottom65_4T_4SepFill, ind0_Geo_bottom69_4T_4SepFill))
            ind0_Geo_finalcrosssectionWithEnds_4T_4SepFill=np.concatenate((ind0_Geo_top16_4T_4SepFill, ind0_Geo_top20_4T_4SepFill, ind0_Geo_top24_4T_4SepFill, ind0_Geo_edge28_4T_4SepFill,                    \
                                                                           ind0_Geo_finalcrosssection_4T_4SepFill, ind0_Geo_surface56_4T_4SepFill,                                                             \
                                                                           ind0_Geo_bottom72_4T_4SepFill, ind0_Geo_bottom76_4T_4SepFill, ind0_Geo_bottom80_4T_4SepFill, ind0_Geo_edge84_4T_4SepFill))
            ind0_Geo_node1_29_57_4T_4SepFill        =np.array([x for x in ind0_Geo_coreWithEnds_4T_4SepFill if self.an_4T[x]==1 ],dtype=int)
            ind0_Geo_node4_32_60_4T_4SepFill        =np.array([x for x in ind0_Geo_coreWithEnds_4T_4SepFill if self.an_4T[x]==self.nx ],dtype=int)        
            ind0_Geo_node2_3_30_31_58_59_4T_4SepFill=np.array([x for x in ind0_Geo_coreWithEnds_4T_4SepFill if (self.an_4T[x]>1 and self.an_4T[x]<self.nx) ],dtype=int)
            ind0_Geo_core_AddSep_4T_4SepFill=np.arange(self.node_4T[-1],self.node_4T[-1]+self.nAddCore_4T)                                                                                    #node number of added separator in the core; e.g. when nx=4,ny=3,nstack=2, nRC=0, ind0_Geo_core_AddSep_4T_4SepFill is [85,86,87]
            ind0_Geo_core_Al_4T_4SepFill=np.where(self.ra_4T==1)[0]                                                                                                                 #node number for all the nodes around the added separator nodes;  e.g. when nx=4,ny=3,nstack=2, nRC=0, ind0_Geo_core_Al_4T_4SepFill is [1,2,3,4,29,30,31,32,57,58,59,60]            
    
            ind0_Geo_Probe32_36_40_44_48_52_56_4T=np.array([x for x in np.arange(self.ntotal_4T) if self.an_4T[x]==self.nx and self.ax_4T[x]==int((self.ny+1)/2) ],dtype=int)
            ind0_Geo_Probe36_44_52_4T=np.array([x for x in ind0_Geo_Probe32_36_40_44_48_52_56_4T if self.mat_4T[x]>=3 ],dtype=int)

            ind0_Geo_left_4T_4SepFill = np.array([x for x in ind0_Geo_surfaceWithEnds_4T_4SepFill if self.an_4T[x] >= self.ind0_SpiralandStripe_boundary1+1 and self.an_4T[x] <= self.ind0_SpiralandStripe_boundary1+self.nx_pouch+1 ],dtype=int)
            ind0_Geo_right_4T_4SepFill = np.array([x for x in ind0_Geo_surfaceWithEnds_4T_4SepFill if self.an_4T[x] >= self.ind0_SpiralandStripe_boundary2+self.nx_pouch+1 and self.an_4T[x] <= self.ind0_SpiralandStripe_boundary2+2*self.nx_pouch+1 ],dtype=int)
            
            ind0temp7 = int(np.size(ind0_Geo_right_4T_4SepFill)/2)
            ind0_Geo_Probe138_4T = ind0_Geo_right_4T_4SepFill[ind0temp7]
            (
            self.ind0_Geo_top_4T_4SepFill, self.ind0_Geo_bottom_4T_4SepFill, self.ind0_Geo_surface_4T_4SepFill, self.ind0_Geo_core_4T_4SepFill, self.ind0_Geo_firstcrosssection_4T_4SepFill, self.ind0_Geo_finalcrosssection_4T_4SepFill,
            self.ind0_Geo_top_edge_inner_4T_4SepFill, self.ind0_Geo_top_edge_outer_4T_4SepFill, self.ind0_Geo_bottom_edge_inner_4T_4SepFill, self.ind0_Geo_bottom_edge_outer_4T_4SepFill, 
            self.ind0_Geo_coreWithEnds_4T_4SepFill, self.ind0_Geo_surfaceWithEnds_4T_4SepFill,
            self.ind0_Geo_topNoSep_4T_4SepFill, self.ind0_Geo_bottomNoSep_4T_4SepFill,
            self.ind0_Geo_top16_4T_4SepFill, self.ind0_Geo_top20_4T_4SepFill, self.ind0_Geo_top24_4T_4SepFill, self.ind0_Geo_top5_4T_4SepFill, self.ind0_Geo_top9_4T_4SepFill, self.ind0_Geo_top13_4T_4SepFill ,
            self.ind0_Geo_bottom72_4T_4SepFill, self.ind0_Geo_bottom76_4T_4SepFill, self.ind0_Geo_bottom80_4T_4SepFill, self.ind0_Geo_bottom61_4T_4SepFill, self.ind0_Geo_bottom65_4T_4SepFill, self.ind0_Geo_bottom69_4T_4SepFill,
            self.ind0_Geo_surface53to55_4T_4SepFill, self.ind0_Geo_surface56_4T_4SepFill,
            self.ind0_Geo_finalcrosssection44_4T_4SepFill, self.ind0_Geo_finalcrosssection48_4T_4SepFill,self.ind0_Geo_finalcrosssection52_4T_4SepFill,
            self.ind0_Geo_core29_4T_4SepFill, self.ind0_Geo_core33_4T_4SepFill, self.ind0_Geo_core37_4T_4SepFill, self.ind0_Geo_core41_4T_4SepFill, self.ind0_Geo_core30to32_4T_4SepFill,
            self.ind0_Geo_edge25to27_4T_4SepFill, self.ind0_Geo_edge28_4T_4SepFill, self.ind0_Geo_edge2to4_4T_4SepFill, self.ind0_Geo_edge1_4T_4SepFill, self.ind0_Geo_edge81to83_4T_4SepFill, self.ind0_Geo_edge84_4T_4SepFill, self.ind0_Geo_edge58to60_4T_4SepFill, self.ind0_Geo_edge57_4T_4SepFill,
            self.ind0_Geo_Sep85_4T_4SepFill, self.ind0_Geo_Sep87_4T_4SepFill,
            self.ind0_Geo_firstcrosssectionWithEnds_4T_4SepFill, self.ind0_Geo_finalcrosssectionWithEnds_4T_4SepFill,
            self.ind0_Geo_node1_29_57_4T_4SepFill, self.ind0_Geo_node4_32_60_4T_4SepFill, self.ind0_Geo_node2_3_30_31_58_59_4T_4SepFill,
            self.ind0_Geo_core_AddSep_4T_4SepFill, self.ind0_Geo_core_Al_4T_4SepFill,
            self.ind0_Geo_Probe36_44_52_4T,
            self.ind0_Geo_left_4T_4SepFill,
            self.ind0_Geo_right_4T_4SepFill,
            self.ind0_Geo_Probe138_4T
            ) = (
            ind0_Geo_top_4T_4SepFill, ind0_Geo_bottom_4T_4SepFill, ind0_Geo_surface_4T_4SepFill, ind0_Geo_core_4T_4SepFill, ind0_Geo_firstcrosssection_4T_4SepFill, ind0_Geo_finalcrosssection_4T_4SepFill,      #0ind of geometry boundary nodes         
            ind0_Geo_top_edge_inner_4T_4SepFill, ind0_Geo_top_edge_outer_4T_4SepFill, ind0_Geo_bottom_edge_inner_4T_4SepFill, ind0_Geo_bottom_edge_outer_4T_4SepFill, 
            ind0_Geo_coreWithEnds_4T_4SepFill, ind0_Geo_surfaceWithEnds_4T_4SepFill,
            ind0_Geo_topNoSep_4T_4SepFill, ind0_Geo_bottomNoSep_4T_4SepFill,
            ind0_Geo_top16_4T_4SepFill, ind0_Geo_top20_4T_4SepFill, ind0_Geo_top24_4T_4SepFill, ind0_Geo_top5_4T_4SepFill, ind0_Geo_top9_4T_4SepFill, ind0_Geo_top13_4T_4SepFill ,
            ind0_Geo_bottom72_4T_4SepFill, ind0_Geo_bottom76_4T_4SepFill, ind0_Geo_bottom80_4T_4SepFill, ind0_Geo_bottom61_4T_4SepFill, ind0_Geo_bottom65_4T_4SepFill, ind0_Geo_bottom69_4T_4SepFill,
            ind0_Geo_surface53to55_4T_4SepFill, ind0_Geo_surface56_4T_4SepFill,
            ind0_Geo_finalcrosssection44_4T_4SepFill, ind0_Geo_finalcrosssection48_4T_4SepFill,ind0_Geo_finalcrosssection52_4T_4SepFill,
            ind0_Geo_core29_4T_4SepFill, ind0_Geo_core33_4T_4SepFill, ind0_Geo_core37_4T_4SepFill, ind0_Geo_core41_4T_4SepFill, ind0_Geo_core30to32_4T_4SepFill,
            ind0_Geo_edge25to27_4T_4SepFill, ind0_Geo_edge28_4T_4SepFill, ind0_Geo_edge2to4_4T_4SepFill, ind0_Geo_edge1_4T_4SepFill, ind0_Geo_edge81to83_4T_4SepFill, ind0_Geo_edge84_4T_4SepFill, ind0_Geo_edge58to60_4T_4SepFill, ind0_Geo_edge57_4T_4SepFill,
            ind0_Geo_Sep85_4T_4SepFill, ind0_Geo_Sep87_4T_4SepFill,
            ind0_Geo_firstcrosssectionWithEnds_4T_4SepFill, ind0_Geo_finalcrosssectionWithEnds_4T_4SepFill,
            ind0_Geo_node1_29_57_4T_4SepFill, ind0_Geo_node4_32_60_4T_4SepFill, ind0_Geo_node2_3_30_31_58_59_4T_4SepFill,
            ind0_Geo_core_AddSep_4T_4SepFill, ind0_Geo_core_Al_4T_4SepFill,
            ind0_Geo_Probe36_44_52_4T,
            ind0_Geo_left_4T_4SepFill,
            ind0_Geo_right_4T_4SepFill,
            ind0_Geo_Probe138_4T
            )
        
    #########################################################   
    ### function for reading LUT_SoC,LUT_T,LUT_Ri_PerA,LUT_Ci_PerA ####
    #########################################################
    def fun_pre_Thermal(self):
        if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
        #=======================================for SepFill case (prep for BOTH Regular stencil and Irregular stencil), Sep nodes are added, so modify node info at first
#            global xi_4T_ALL, yi_4T_ALL, zi_4T_ALL, theta_4T_ALL
#            global mat_4T_ALL
#            global jx1_4T_ALL, jx2_4T_ALL, jy1_4T_ALL, jy2_4T_ALL, jz1_4T_ALL, jz2_4T_ALL, jxz_4T_ALL
#            global Reg_ind0_jx1NaN_4T_ALL, Reg_ind0_jx2NaN_4T_ALL, Reg_ind0_jy1NaN_4T_ALL, Reg_ind0_jy2NaN_4T_ALL, Reg_ind0_jz1NaN_4T_ALL, Reg_ind0_jz2NaN_4T_ALL, Reg_ind0_jxzNaN_4T_ALL, Reg_ind0_4T_ALL
#            global Reg_ind0_jx1NonNaN_4T_ALL, Reg_ind0_jx2NonNaN_4T_ALL, Reg_ind0_jy1NonNaN_4T_ALL, Reg_ind0_jy2NonNaN_4T_ALL, Reg_ind0_jz1NonNaN_4T_ALL, Reg_ind0_jz2NonNaN_4T_ALL, Reg_ind0_jxzNaN_4T_ALL, Irreg_ind0_4T_ALL
            #-----------------------------------add the separator node in the cell core and modify the xi_4T, yi_4T and zi_4T                                                                                                               
            xi_4T_4SepFill=np.concatenate(( self.xi_4T,np.zeros([self.nAddCore_4T]) ))                                                                                 #e.g. when nx=4,ny=3,nstack=2, nRC=0, xi_4T shape is (87,1) containing the 3 Sep nodes in the end. xi_4T shape was (84,1) before
            yi_4T_4SepFill=np.concatenate(( self.yi_4T,self.LG/(self.ny-1)*np.arange(self.nAddCore_4T) )) 
            zi_4T_4SepFill=np.concatenate(( self.zi_4T,np.zeros([self.nAddCore_4T]) ))
            
            theta_4T_4SepFill=np.concatenate(( self.theta_4T,np.zeros([self.nAddCore_4T]) ))
            #-----------------------------------add the separator node in the cell core and modify mat_4T
            mat_4T_4SepFill=np.concatenate(( self.mat_4T,np.tile(np.array([5]),self.nAddCore_4T) ))                                                                   #material discrimination: 1:Al, 2:Cu, 3:Elb, 4:Elr, 5:Sep
            #-----------------------------------modify neighboring node of Al node in the cell core
            jx1_4T_4SepFill=np.concatenate((self.jx1_4T,-9999*np.ones([self.nAddCore_4T],dtype=int) ))                                                                          #modify the jx1_4T. node [85,86,87] jx1_4T is [nan,nan,nan]. jx1_4T shape is (87,1)
            jx2_4T_4SepFill=np.concatenate((self.jx2_4T,-9999*np.ones([self.nAddCore_4T],dtype=int) ))                                                                          #modify the jx2_4T. node [85,86,87] jx2_4T is [nan,nan,nan]. jx2_4T shape is (87,1) 
            jy1_4T_temp=self.ind0_Geo_core_AddSep_4T_4SepFill-1+1; jy1_4T_temp[0]=np.array([-9999])
            jy1_4T_4SepFill=np.concatenate((self.jy1_4T,jy1_4T_temp ))                                                                                            #modify the jy1_4T: upper jy1 node. node [85,86,87] jy1 node is [nan,85,86]. jy1_4T shape is (87,1)
            jy2_4T_temp=self.ind0_Geo_core_AddSep_4T_4SepFill+1+1; jy2_4T_temp[-1]=np.array([-9999])
            jy2_4T_4SepFill=np.concatenate((self.jy2_4T,jy2_4T_temp ))                                                                                            #modify the jy2_4T: lower jy2 node. node [85,86,87] jy2 node is [86,87,nan]. jy2_4T shape is (87,1)            
            jz1_4T_temp=self.jz1_4T.copy(); jz1_4T_temp[self.ind0_Geo_core_Al_4T_4SepFill]=self.ind0_Geo_core_AddSep_4T_4SepFill.repeat(self.nx)+1; 
            jz1_4T_4SepFill=np.concatenate(( jz1_4T_temp,-9999*np.ones([self.nAddCore_4T],dtype=int) ))                                                                    #modify the jz1_4T. For node [1,2,3,4,29,30,31,32,57,58,59,60], jz1 was Nan before, now should be [85,85,85,85,86,86,86,86,87,87,87,87]. jz1 shape is (87,1)
            jz2_4T_4SepFill=np.concatenate((self.jz2_4T,-9999*np.ones([self.nAddCore_4T],dtype=int) ))                                                                          #modify the jz2_4T. node [85,86,87] jz2 is [nan,nan,nan]. jz2 shape is (87,1)
            jxz_4T_4SepFill=-9999*np.ones([self.ntotal_4T+self.nAddCore_4T,self.nx],dtype=int)
            jxz_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=self.ind0_Geo_core_Al_4T_4SepFill.reshape(self.ny,self.nx)+1                                                                    #add the jxz_4T: neighbor nodes in xz plane. only node [85,86,87] have xz plane neighbors [[1,2,3,4],[29,30,31,32],[57,58,59,60]]. jxz shape is (87,4)
            #-----------------------------------in Regular stencil, prep (e.g. node 1~84) jx1, jx2 NaN and NonNaN element 0-index to be used in fun_Thermal
            Reg_ind0_jx1NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jx1_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)           #nodes used in Regular stencil.these nodes Do NOT have jx1_4T. e.g. node1
            Reg_ind0_jx2NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jx2_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jy1NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy1_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jy2NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy2_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jz1NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jz1_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jz2NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jz2_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jx1NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jx1_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)        #nodes used in Regular stencil.these nodes HAVE jx1_4T. e.g. node2
            Reg_ind0_jx2NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jx2_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jy1NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy1_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jy2NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy2_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jz1NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jz1_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jz2NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jz2_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)
            Reg_ind0_jxzNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jxz_4T_4SepFill[x,0]==-9999 and zi_4T_4SepFill[x]!=0) ],dtype=int)         #nodes used in Regular stencil.these nodes Do NOT have jxz_4T. e.g. node1
            Reg_ind0_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if zi_4T_4SepFill[x]!=0 ],dtype=int)                                                  #all nodes used in Regular stencil.
            #-----------------------------------in Irregular stencil, prep (e.g. node 85~87) jy1, jy2 NaN and NonNaN element 0-index to be used in fun_Thermal
#            global Irreg_ind0_jy1NaN_4T_ALL, Irreg_ind0_jy2NaN_4T_ALL
#            global Irreg_ind0_jy1NonNaN_4T_ALL, Irreg_ind0_jy2NonNaN_4T_ALL
#            global Irreg_ind0_jxzNonNaN_4T_ALL
            Irreg_ind0_jy1NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy1_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]==0) ],dtype=int)         #nodes used in Irregular stencil.these nodes Do NOT have jy1_4T. e.g. node85
            Irreg_ind0_jy1NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy1_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]==0) ],dtype=int)      #nodes used in Regular stencil.these nodes HAVE jy1_4T. e.g. node86
            Irreg_ind0_jy2NaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy2_4T_4SepFill[x]==-9999 and zi_4T_4SepFill[x]==0) ],dtype=int)
            Irreg_ind0_jy2NonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jy2_4T_4SepFill[x]!=-9999 and zi_4T_4SepFill[x]==0) ],dtype=int)
            Irreg_ind0_jxzNonNaN_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if (jxz_4T_4SepFill[x,0]!=-9999 and zi_4T_4SepFill[x]==0) ],dtype=int)    #nodes used in Irregular stencil.these nodes HAVE jxz_4T. e.g. node85
            Irreg_ind0_4T_4SepFill=np.array([x for x in self.node_4T_4SepFill-1 if zi_4T_4SepFill[x]==0 ],dtype=int)                                                #all nodes used in Irregular stencil.
        #=======================================prep for Regular stencil of node 1~84    "_4SepFill" suffix is for node 1~87
#            global Lamda_4T_ALL, RouXc_4T_ALL                                                                                                                                                       
#            global Delta_x1_4T_ALL, Delta_x2_4T_ALL, Delta_y1_4T_ALL, Delta_y2_4T_ALL, Delta_z1_4T_ALL, Delta_z2_4T_ALL                                                                            
#            global delta_x1_4T_ALL, delta_x2_4T_ALL, delta_y1_4T_ALL, delta_y2_4T_ALL, delta_z1_4T_ALL, delta_z2_4T_ALL, delta_xyz_4T_ALL, V_stencil_4T_ALL
#            #-----------------------------------fill in Lamda_4T_4SepFill for node 1~84
            Lamda_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                                                            #Lamda_4T_4SepFill shape is (87,6). λ term in 6-node stencil
            Lamda_4T_4SepFill[ self.Al_4T,0 ]=self.Lamda_Al                                                                                                            #fill in in 1st column(jx1) for Al node                                                                                                           
            Lamda_4T_4SepFill[ self.Al_4T,1 ]=self.Lamda_Al
            Lamda_4T_4SepFill[ self.Al_4T,2 ]=self.Lamda_Al
            Lamda_4T_4SepFill[ self.Al_4T,3 ]=self.Lamda_Al
            Lamda_4T_4SepFill[ self.Al_4T,4 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)
            Lamda_4T_4SepFill[ self.Al_4T,5 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)
            
            Lamda_4T_4SepFill[ self.Cu_4T,0 ]=self.Lamda_Cu                                                                                                            #fill in in 1st column(jx1) for Cu node
            Lamda_4T_4SepFill[ self.Cu_4T,1 ]=self.Lamda_Cu
            Lamda_4T_4SepFill[ self.Cu_4T,2 ]=self.Lamda_Cu
            Lamda_4T_4SepFill[ self.Cu_4T,3 ]=self.Lamda_Cu
            Lamda_4T_4SepFill[ self.Cu_4T,4 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)
            Lamda_4T_4SepFill[ self.Cu_4T,5 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)
            
            Lamda_4T_4SepFill[ self.Elb_4T,0 ]=self.Lamda_El_x                                                                                                         #fill in in 1st column(jx1) for Elb node
            Lamda_4T_4SepFill[ self.Elb_4T,1 ]=self.Lamda_El_x
            Lamda_4T_4SepFill[ self.Elb_4T,2 ]=self.Lamda_El_y
            Lamda_4T_4SepFill[ self.Elb_4T,3 ]=self.Lamda_El_y
            Lamda_4T_4SepFill[ self.Elb_4T,4 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)
            Lamda_4T_4SepFill[ self.Elb_4T,5 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)
            
            Lamda_4T_4SepFill[ self.Elr_4T,0 ]=self.Lamda_El_x                                                                                                         #fill in in 1st column(jx1) for Elr node
            Lamda_4T_4SepFill[ self.Elr_4T,1 ]=self.Lamda_El_x
            Lamda_4T_4SepFill[ self.Elr_4T,2 ]=self.Lamda_El_y
            Lamda_4T_4SepFill[ self.Elr_4T,3 ]=self.Lamda_El_y
            Lamda_4T_4SepFill[ self.Elr_4T,4 ]=0.5*(self.delta_El+self.delta_Cu) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Cu/self.Lamda_Cu)
            Lamda_4T_4SepFill[ self.Elr_4T,5 ]=0.5*(self.delta_El+self.delta_Al) / (0.5*self.delta_El/self.Lamda_El_z + 0.5*self.delta_Al/self.Lamda_Al)        
    
            #modify: for nodes 25~28,53~56,81~84, λ in z2 direction is λCu
            Lamda_4T_4SepFill[ self.ind0_Geo_surfaceWithEnds_4T_4SepFill,5 ]=self.Lamda_Cu
            #modify: for heat transfer between node1~4,29~32,57~60 and node85,86,87. refer to p204 ECM47.py        
            if self.status_ThermalBC_Core=='SepAir':                                                                
                Lamda_4T_4SepFill[ self.ind0_Geo_coreWithEnds_4T_4SepFill,4 ]=0.5*(self.b0_Sep*(1-self.n_Air)+self.delta_Al) / ((0.5*(self.b0_Sep*(1-self.n_Air)+self.delta_Al)-0.5*self.delta_Al)/self.Lamda_Sep + 0.5*self.delta_Al/self.Lamda_Al)        #i.g. for node1~4,29~32,57~60, the 5th column of Lamda_4T_SepFill need to be changed. 
            if self.status_ThermalBC_Core=='InsuFill':
                Lamda_4T_4SepFill[ self.ind0_Geo_coreWithEnds_4T_4SepFill,4 ]=0.0                                                                                                                     
            if self.status_ThermalBC_Core=='SepFill':
                Lamda_4T_4SepFill[ self.ind0_Geo_coreWithEnds_4T_4SepFill,4 ]=(self.a0*self.theta_4T[ self.ind0_Geo_coreWithEnds_4T_4SepFill ]+self.b0_Sep+ 0.5*self.delta_Al) / ((self.a0*self.theta_4T[ self.ind0_Geo_coreWithEnds_4T_4SepFill ]+self.b0_Sep)/self.Lamda_Sep + 0.5*self.delta_Al/self.Lamda_Al)                                                                           
            #-----------------------------------fill in RouXc_4T_4SepFill for node 1~84
            RouXc_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])                                                                                            #RouXc_4T_4SepFill shape is (87,1). ρc term in 6-node stencil
            RouXc_4T_4SepFill[ self.Al_4T,0 ]=self.rou_Al*self.c_Al                                                                                                         #fill in in elements for Al node
            RouXc_4T_4SepFill[ self.Cu_4T,0 ]=self.rou_Cu*self.c_Cu                                                                                                         #fill in in elements for Cu node        
            RouXc_4T_4SepFill[ self.Elb_4T,0 ]=self.rouXc_El                                                                                                           #fill in in elements for Elb node
            RouXc_4T_4SepFill[ self.Elr_4T,0 ]=self.rouXc_El                                                                                                           #fill in in elements for Elr node
            RouXc_4T_4SepFill[ self.ind0_Geo_core_AddSep_4T_4SepFill,0 ]=self.rou_Sep*self.c_Sep                                                                                                 #fill in in elements for Sep node
            #-----------------------------------fill in Delta_x1_4T_4SepFill for node 1~84
            Delta_x1_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])                                                                  #get Δx1 for each node
            Delta_x1_4T_4SepFill[Reg_ind0_jx1NonNaN_4T_4SepFill]=xi_4T_4SepFill[Reg_ind0_jx1NonNaN_4T_4SepFill] - xi_4T_4SepFill[jx1_4T_4SepFill[Reg_ind0_jx1NonNaN_4T_4SepFill]-1]                                              
    
            Delta_x2_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])                                                                  #get Δx2 for each node
            Delta_x2_4T_4SepFill[Reg_ind0_jx2NonNaN_4T_4SepFill]=xi_4T_4SepFill[jx2_4T_4SepFill[Reg_ind0_jx2NonNaN_4T_4SepFill]-1] - xi_4T_4SepFill[Reg_ind0_jx2NonNaN_4T_4SepFill]
    
            Delta_y1_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])                                                                  #get Δy1 for each node
            Delta_y1_4T_4SepFill[Reg_ind0_jy1NonNaN_4T_4SepFill]=yi_4T_4SepFill[Reg_ind0_jy1NonNaN_4T_4SepFill] - yi_4T_4SepFill[jy1_4T_4SepFill[Reg_ind0_jy1NonNaN_4T_4SepFill]-1]
    
            Delta_y2_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])                                                                  #get Δy2 for each node
            Delta_y2_4T_4SepFill[Reg_ind0_jy2NonNaN_4T_4SepFill]=yi_4T_4SepFill[jy2_4T_4SepFill[Reg_ind0_jy2NonNaN_4T_4SepFill]-1] - yi_4T_4SepFill[Reg_ind0_jy2NonNaN_4T_4SepFill]
    
            Delta_z1_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])                                                                  #get Δz1 for each node
            Delta_z1_4T_4SepFill[Reg_ind0_jz1NonNaN_4T_4SepFill]=zi_4T_4SepFill[Reg_ind0_jz1NonNaN_4T_4SepFill] - zi_4T_4SepFill[jz1_4T_4SepFill[Reg_ind0_jz1NonNaN_4T_4SepFill]-1]
    
            Delta_z2_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])                                                                  #get Δz2 for each node
            Delta_z2_4T_4SepFill[Reg_ind0_jz2NonNaN_4T_4SepFill]=zi_4T_4SepFill[jz2_4T_4SepFill[Reg_ind0_jz2NonNaN_4T_4SepFill]-1] - zi_4T_4SepFill[Reg_ind0_jz2NonNaN_4T_4SepFill]
    
                    #modify for heat transfer between node1~4,29~32,57~60 and node85,86,87. refer to p204 ECM47.py        
            if self.status_ThermalBC_Core=='SepAir':
                Delta_z1_4T_4SepFill[self.ind0_Geo_coreWithEnds_4T_4SepFill]=(self.b0_Sep*(1-self.n_Air)+self.delta_Al)/2         #refer to ppt p204 ECM47.py  i.g. for node1~4,29~32,57~60
            if self.status_ThermalBC_Core=='InsuFill':
                Delta_z1_4T_4SepFill[self.ind0_Geo_coreWithEnds_4T_4SepFill]=self.a0*self.theta_4T[ self.ind0_Geo_coreWithEnds_4T_4SepFill ]+self.b0_Sep +0.5*self.delta_Al                   #refer to ppt p204 ECM47.py  i.g. for node1~4,29~32,57~60
            if self.status_ThermalBC_Core=='SepFill':
                Delta_z1_4T_4SepFill[self.ind0_Geo_coreWithEnds_4T_4SepFill]=self.a0*self.theta_4T[ self.ind0_Geo_coreWithEnds_4T_4SepFill ]+self.b0_Sep+0.5*self.delta_Al                   #refer to ppt p204 ECM47.py  i.g. for node1~4,29~32,57~60       
            #-----------------------------------fill in delta_x1_4T_4SepFill for node 1~84
            delta_x1_4T_4SepFill=Delta_x1_4T_4SepFill.copy()                                                                                      #get δx1 for each node
            delta_x1_4T_4SepFill[Reg_ind0_jx1NaN_4T_4SepFill]=xi_4T_4SepFill[jx2_4T_4SepFill[Reg_ind0_jx1NaN_4T_4SepFill]-1]-xi_4T_4SepFill[Reg_ind0_jx1NaN_4T_4SepFill]
    
            delta_x2_4T_4SepFill=Delta_x2_4T_4SepFill.copy()                                                                                      #get δx2 for each node
            delta_x2_4T_4SepFill[Reg_ind0_jx2NaN_4T_4SepFill]=xi_4T_4SepFill[Reg_ind0_jx2NaN_4T_4SepFill]-xi_4T_4SepFill[jx1_4T_4SepFill[Reg_ind0_jx2NaN_4T_4SepFill]-1]
            
            delta_y1_4T_4SepFill=Delta_y1_4T_4SepFill.copy()                                                                                      #get δy1 for each node
            delta_y1_4T_4SepFill[Reg_ind0_jy1NaN_4T_4SepFill]=yi_4T_4SepFill[jy2_4T_4SepFill[Reg_ind0_jy1NaN_4T_4SepFill]-1]-yi_4T_4SepFill[Reg_ind0_jy1NaN_4T_4SepFill]
            
            delta_y2_4T_4SepFill=Delta_y2_4T_4SepFill.copy()                                                                                      #get δy2 for each node
            delta_y2_4T_4SepFill[Reg_ind0_jy2NaN_4T_4SepFill]=yi_4T_4SepFill[Reg_ind0_jy2NaN_4T_4SepFill]-yi_4T_4SepFill[jy1_4T_4SepFill[Reg_ind0_jy2NaN_4T_4SepFill]-1]
            
            MatThickness_lib=np.array([[self.delta_Al],[self.delta_Cu],[self.delta_El],[self.delta_El]])
            delta_z1_4T=MatThickness_lib[self.mat_4T-1,0]                                                                                       #get δz1 for each node
            delta_z1_4T_4SepFill=np.concatenate(( delta_z1_4T,np.nan*np.zeros([self.nAddCore_4T]) ))
            delta_z2_4T_4SepFill=delta_z1_4T_4SepFill.copy()                                                                                      #get δz2(=δz1) for each node
    
            delta_xyz_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,6])                                                               #(δx1+δx2,δx1+δx2,δy1+δy2,δy1+δy2,2δz,2δz), shape is (87,6)
            delta_xyz_4T_4SepFill[Reg_ind0_4T_4SepFill,0:2]=(delta_x1_4T_4SepFill[Reg_ind0_4T_4SepFill]+delta_x2_4T_4SepFill[Reg_ind0_4T_4SepFill]).reshape(-1,1) 
            delta_xyz_4T_4SepFill[Reg_ind0_4T_4SepFill,2:4]=(delta_y1_4T_4SepFill[Reg_ind0_4T_4SepFill]+delta_y2_4T_4SepFill[Reg_ind0_4T_4SepFill]).reshape(-1,1)
            delta_xyz_4T_4SepFill[Reg_ind0_4T_4SepFill,4:6]=(delta_z1_4T_4SepFill[Reg_ind0_4T_4SepFill]+delta_z2_4T_4SepFill[Reg_ind0_4T_4SepFill]).reshape(-1,1)
    
            V_stencil_4T_4SepFill=(delta_xyz_4T_4SepFill[:,0]/2) * (delta_xyz_4T_4SepFill[:,2]/2) * (delta_xyz_4T_4SepFill[:,4]/2)  #node volume used in thermal stencil delta_xyz_4T_ALL: (δx1+δx2,δx1+δx2,δy1+δy2,δy1+δy2,2δz,2δz), shape is (87,1)
        #=======================================prep for Irregular stencil of node 85~87    "_4SepFill" suffix is for node 1~87
            #-----------------------------------fill in Lamda_4T_4SepFill for node 85,86,87;  for heat transfer between node85~87        
            if self.status_ThermalBC_Core=='SepAir':                                                                
                Lamda_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill,2]=self.Lamda_Sep        #i.g. for node85~87, the 3rd column of Lamda_4T_SepFill need to be changed. 
                Lamda_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill,3]=self.Lamda_Sep        #i.g. for node85~87, the 4th column of Lamda_4T_SepFill need to be changed. 
            if self.status_ThermalBC_Core=='InsuFill':
                Lamda_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill,2]=0.0              
                Lamda_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill,3]=0.0              
            if self.status_ThermalBC_Core=='SepFill':
                Lamda_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill,2]=self.Lamda_Sep                                                                             
                Lamda_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill,3]=self.Lamda_Sep                                                                             
            #-----------------------------------fill in RouXc_4T_4SepFill for node 85,86,87
            RouXc_4T_4SepFill[ self.ind0_Geo_core_AddSep_4T_4SepFill,0 ]=self.rou_Sep*self.c_Sep                                                                                                 #fill in in elements for Sep node
            #-----------------------------------fill in Delta_y1_4T_4SepFill for node 85,86,87
            Delta_y1_4T_4SepFill[Irreg_ind0_jy1NonNaN_4T_4SepFill]=yi_4T_4SepFill[Irreg_ind0_jy1NonNaN_4T_4SepFill] - yi_4T_4SepFill[jy1_4T_4SepFill[Irreg_ind0_jy1NonNaN_4T_4SepFill]-1]
    
            Delta_y2_4T_4SepFill[Irreg_ind0_jy2NonNaN_4T_4SepFill]=yi_4T_4SepFill[jy2_4T_4SepFill[Irreg_ind0_jy2NonNaN_4T_4SepFill]-1] - yi_4T_4SepFill[Irreg_ind0_jy2NonNaN_4T_4SepFill]
            #-----------------------------------fill in delta_y1_4T_4SepFill for node 85,86,87
            delta_y1_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=Delta_y1_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]
            delta_y2_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=Delta_y2_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]
    
            delta_y1_4T_4SepFill[Irreg_ind0_jy1NaN_4T_4SepFill]=yi_4T_4SepFill[jy2_4T_4SepFill[Irreg_ind0_jy1NaN_4T_4SepFill]-1]-yi_4T_4SepFill[Irreg_ind0_jy1NaN_4T_4SepFill]
    
            delta_y2_4T_4SepFill[Irreg_ind0_jy2NaN_4T_4SepFill]=yi_4T_4SepFill[Irreg_ind0_jy2NaN_4T_4SepFill]-yi_4T_4SepFill[jy1_4T_4SepFill[Irreg_ind0_jy2NaN_4T_4SepFill]-1]
    
            delta_xyz_4T_4SepFill[Irreg_ind0_4T_4SepFill,0:2]=(delta_x1_4T_4SepFill[Irreg_ind0_4T_4SepFill]+delta_x2_4T_4SepFill[Irreg_ind0_4T_4SepFill]).reshape(-1,1)                        
            delta_xyz_4T_4SepFill[Irreg_ind0_4T_4SepFill,2:4]=(delta_y1_4T_4SepFill[Irreg_ind0_4T_4SepFill]+delta_y2_4T_4SepFill[Irreg_ind0_4T_4SepFill]).reshape(-1,1)
            delta_xyz_4T_4SepFill[Irreg_ind0_4T_4SepFill,4:6]=(delta_z1_4T_4SepFill[Irreg_ind0_4T_4SepFill]+delta_z2_4T_4SepFill[Irreg_ind0_4T_4SepFill]).reshape(-1,1)
            #-----------------------------------prep α and S for node 85,86,87
#            global Alpha_Irreg_ALL, S_Irreg_ALL
            Alpha_Irreg_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,self.nx])
            if self.status_ThermalBC_Core=='SepAir':                                                                
                Lamda_temp=0.5*(self.b0_Sep*(1-self.n_Air)+self.delta_Al) / ((0.5*(self.b0_Sep*(1-self.n_Air)+self.delta_Al)-0.5*self.delta_Al)/self.Lamda_Sep + 0.5*self.delta_Al/self.Lamda_Al)                    
                Alpha_Irreg_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=Lamda_temp/self.rou_Sep/self.c_Sep
            if self.status_ThermalBC_Core=='InsuFill':
                Alpha_Irreg_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=0.0
            if self.status_ThermalBC_Core=='SepFill':
                Lamda_temp=(self.a0*self.theta_4T[ self.ind0_Geo_coreWithEnds_4T_4SepFill ]+self.b0_Sep+ 0.5*self.delta_Al) / ((self.a0*self.theta_4T[ self.ind0_Geo_coreWithEnds_4T_4SepFill ]+self.b0_Sep)/self.Lamda_Sep + 0.5*self.delta_Al/self.Lamda_Al)
                Alpha_Irreg_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=Lamda_temp.reshape(self.ny,-1)/self.rou_Sep/self.c_Sep                                                                             
            #-----------------------------------prep delta_x1_Sub4Irreg_4T_4SepFill for node 85,86,87
#            global delta_x1_Sub4Irreg_4T_ALL, delta_x2_Sub4Irreg_4T_ALL
            S_Irreg_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T,1])
            S_Irreg_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=self.S_SpiralandLine
    
            delta_x1_Sub4Irreg_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])  
            delta_x1_Sub4Irreg_4T_4SepFill[self.ind0_Geo_coreWithEnds_4T_4SepFill]=delta_x1_4T_4SepFill[self.ind0_Geo_coreWithEnds_4T_4SepFill]
    
            delta_x2_Sub4Irreg_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])  
            delta_x2_Sub4Irreg_4T_4SepFill[self.ind0_Geo_coreWithEnds_4T_4SepFill]=delta_x2_4T_4SepFill[self.ind0_Geo_coreWithEnds_4T_4SepFill]
    
            delta_y1_Irreg_4T_4SepFill=np.nan*np.zeros([self.ntotal_4T+self.nAddCore_4T])  
            delta_y1_Irreg_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]=delta_y1_4T_4SepFill[self.ind0_Geo_core_AddSep_4T_4SepFill]  
    
            V_stencil_4T_4SepFill[Irreg_ind0_4T_4SepFill] = 0.5*(delta_y1_4T_4SepFill[Irreg_ind0_4T_4SepFill] + delta_y2_4T_4SepFill[Irreg_ind0_4T_4SepFill])*S_Irreg_4SepFill[Irreg_ind0_4T_4SepFill,0]     #node volume used in thermal stencil delta_xyz_4T_ALL: (δx1+δx2,δx1+δx2,δy1+δy2,δy1+δy2,2δz,2δz), shape is (87,1)
        #=======================================update variables to be used in stencil (fun_Thermal()); for BOTH Regular stencil and Irregular stencil
            self.xi_4T_ALL=xi_4T_4SepFill.copy();  self.yi_4T_ALL=yi_4T_4SepFill.copy();  self.zi_4T_ALL=zi_4T_4SepFill.copy();  self.theta_4T_ALL=theta_4T_4SepFill.copy()
            self.jx1_4T_ALL=jx1_4T_4SepFill.copy();  self.jx2_4T_ALL=jx2_4T_4SepFill.copy();  self.jy1_4T_ALL=jy1_4T_4SepFill.copy();  self.jy2_4T_ALL=jy2_4T_4SepFill.copy();  self.jz1_4T_ALL=jz1_4T_4SepFill.copy();  self.jz2_4T_ALL=jz2_4T_4SepFill.copy();  self.jxz_4T_ALL=jxz_4T_4SepFill.copy()
            self.Reg_ind0_jx1NaN_4T_ALL=Reg_ind0_jx1NaN_4T_4SepFill.copy();        self.Reg_ind0_jx2NaN_4T_ALL=Reg_ind0_jx2NaN_4T_4SepFill.copy();        self.Reg_ind0_jy1NaN_4T_ALL=Reg_ind0_jy1NaN_4T_4SepFill.copy();        self.Reg_ind0_jy2NaN_4T_ALL=Reg_ind0_jy2NaN_4T_4SepFill.copy();        self.Reg_ind0_jz1NaN_4T_ALL=Reg_ind0_jz1NaN_4T_4SepFill.copy();        self.Reg_ind0_jz2NaN_4T_ALL=Reg_ind0_jz2NaN_4T_4SepFill.copy();                self.Reg_ind0_4T_ALL=Reg_ind0_4T_4SepFill.copy()
            self.Reg_ind0_jx1NonNaN_4T_ALL=Reg_ind0_jx1NonNaN_4T_4SepFill.copy();  self.Reg_ind0_jx2NonNaN_4T_ALL=Reg_ind0_jx2NonNaN_4T_4SepFill.copy();  self.Reg_ind0_jy1NonNaN_4T_ALL=Reg_ind0_jy1NonNaN_4T_4SepFill.copy();  self.Reg_ind0_jy2NonNaN_4T_ALL=Reg_ind0_jy2NonNaN_4T_4SepFill.copy();  self.Reg_ind0_jz1NonNaN_4T_ALL=Reg_ind0_jz1NonNaN_4T_4SepFill.copy();  self.Reg_ind0_jz2NonNaN_4T_ALL=Reg_ind0_jz2NonNaN_4T_4SepFill.copy();  self.Reg_ind0_jxzNaN_4T_ALL=Reg_ind0_jxzNaN_4T_4SepFill.copy();   self.Irreg_ind0_4T_ALL=Irreg_ind0_4T_4SepFill.copy()
            self.mat_4T_ALL=mat_4T_4SepFill.copy()
            self.Lamda_4T_ALL=Lamda_4T_4SepFill.copy(); self.RouXc_4T_ALL=RouXc_4T_4SepFill.copy()                                                                                                                                                                                                  
            self.Delta_x1_4T_ALL=Delta_x1_4T_4SepFill.copy(); self.Delta_x2_4T_ALL=Delta_x2_4T_4SepFill.copy(); self.Delta_y1_4T_ALL=Delta_y1_4T_4SepFill.copy(); self.Delta_y2_4T_ALL=Delta_y2_4T_4SepFill.copy(); self.Delta_z1_4T_ALL=Delta_z1_4T_4SepFill.copy(); self.Delta_z2_4T_ALL=Delta_z2_4T_4SepFill.copy()                                                                                                  
            self.delta_x1_4T_ALL=delta_x1_4T_4SepFill.copy(); self.delta_x2_4T_ALL=delta_x2_4T_4SepFill.copy(); self.delta_y1_4T_ALL=delta_y1_4T_4SepFill.copy(); self.delta_y2_4T_ALL=delta_y2_4T_4SepFill.copy(); self.delta_z1_4T_ALL=delta_z1_4T_4SepFill.copy(); self.delta_z2_4T_ALL=delta_z2_4T_4SepFill.copy()
            self.delta_xyz_4T_ALL=delta_xyz_4T_4SepFill.copy()           
            self.V_stencil_4T_ALL=V_stencil_4T_4SepFill.copy()  #node volume used in thermal stencil delta_xyz_4T_ALL: (δx1+δx2,δx1+δx2,δy1+δy2,δy1+δy2,2δz,2δz), shape is (87,1)
            self.Alpha_Irreg_ALL=Alpha_Irreg_4SepFill.copy(); self.S_Irreg_ALL=S_Irreg_4SepFill.copy()
            self.delta_x1_Sub4Irreg_4T_ALL=delta_x1_Sub4Irreg_4T_4SepFill.copy(); self.delta_x2_Sub4Irreg_4T_ALL=delta_x2_Sub4Irreg_4T_4SepFill.copy()
            self.Irreg_ind0_jy1NaN_4T_ALL=Irreg_ind0_jy1NaN_4T_4SepFill.copy();       self.Irreg_ind0_jy2NaN_4T_ALL=Irreg_ind0_jy2NaN_4T_4SepFill.copy()
            self.Irreg_ind0_jy1NonNaN_4T_ALL=Irreg_ind0_jy1NonNaN_4T_4SepFill.copy(); self.Irreg_ind0_jy2NonNaN_4T_ALL=Irreg_ind0_jy2NonNaN_4T_4SepFill.copy()
            self.Irreg_ind0_jxzNonNaN_4T_ALL=Irreg_ind0_jxzNonNaN_4T_4SepFill.copy()

    #########################################################   
    ###########     function for Thermal model    ###########
    #########################################################
    def fun_Thermal(self, T1_4T_ALL,T3_4T_ALL, ind0_BCtem_ALL, ind0_BCtem_others_ALL, h_4T_ALL, Tconv_4T_ALL, ind0_BCconv_ALL, ind0_BCconv_others_ALL):     #return node temperature vector T_4T (in Thermal node framework)        
        #Core_AllForm.fun_Thermal(self, T1_4T_ALL,T3_4T_ALL, ind0_BCtem_ALL, ind0_BCtem_others_ALL, h_4T_ALL, Tconv_4T_ALL, ind0_BCconv_ALL, ind0_BCconv_others_ALL)
#     #================================================================explicit solver
    #================================================================explicit solver
        if self.status_Thermal_solver == 'Explicit':        
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
                #======================================calculate nodes suitable for Irregular stencil; e.g. Sep nodes (i.e. 85~87)            
                StencilSum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]=0        #initialize the Irregular stencil, e.g. the last 3 rows in StencilSum_4T_ALL
    
                StencilSum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] += ( self.Alpha_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]*self.dt/self.S_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]/2* ( (self.delta_x1_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1]+self.delta_x2_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1])                                                                                       \
                                                                               * (T1_4T_ALL[:,0][self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1]-T1_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] )/self.Delta_z1_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1] ) )  .dot( np.ones([self.nx,1]) )            
                
                StencilSum_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0] += h_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]) * (Tconv_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]-T1_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0])
                StencilSum_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]) * (T1_4T_ALL[self.jy1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0]) /self.Delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]
    
                StencilSum_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0] += h_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]) * (Tconv_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]-T1_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0])
                StencilSum_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0] += self.Lamda_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]) * (T1_4T_ALL[self.jy2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]-1,0]-T1_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0]) /self.Delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]
    
                T3_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]=T1_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] + StencilSum_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] + self.q_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]*self.dt/self.RouXc_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]
                #======================================apply temperature BC
                T3_4T_ALL[ind0_BCtem_ALL]=T_copy[ind0_BCtem_ALL]                                                            #replace temperature-constrained nodes in T3_4T (i.e. NaN) with temperature BC stored before (i.e. T_copy)                     
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
        return T3_4T_ALL                                                                                          #for these nodes, Tconv=NaN               

        
    #########################################################   
    ###########       function for MatrixCN       ###########
    #########################################################
    
    def fun_MatrixCN(self):
        if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir':
            MatrixCN=np.zeros([self.n_4T_ALL,self.n_4T_ALL])        
            #======================================calculate nodes suitable for Regular stencil; e.g. nodes except Sep nodes (i.e. 1~84)
                #--------------------------------------fill in jx1, jx2, jy1, jy2, jz1, jz2 terms
            MatrixCN[self.Reg_ind0_jx1NonNaN_4T_ALL,self.jx1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0]/self.RouXc_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL])/self.Delta_x1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL]    #if ind0_jx1NonNaN_4T_4SepFill nodes case: fill elements of the MatrixCN: row of ind0_jx1NonNaN_4T_4SepFill node , column of left neighbor node(jx1); if ind0_jx1NaN_4T_4SepFill nodes case: elements are zero as initiated 
            MatrixCN[self.Reg_ind0_jx2NonNaN_4T_ALL,self.jx2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,1]/self.RouXc_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL])/self.Delta_x2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL]    #if ind0_jx2NonNaN_4T_4SepFill nodes case: fill elements of the MatrixCN: row of ind0_jx2NonNaN_4T_4SepFill node , column of right neighbor node(jx2); if ind0_jx2NaN_4T_4SepFill nodes case: elements are zero as initiated 
            MatrixCN[self.Reg_ind0_jy1NonNaN_4T_ALL,self.jy1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL])/self.Delta_y1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL]    #if ind0_jy1NonNaN_4T_4SepFill nodes case: fill elements of the MatrixCN: row of ind0_jy1NonNaN_4T_4SepFill node , column of up neighbor node(jy1); if ind0_jy1NaN_4T_4SepFill nodes case: elements are zero as initiated 
            MatrixCN[self.Reg_ind0_jy2NonNaN_4T_ALL,self.jy2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL])/self.Delta_y2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL]    #if ind0_jy2NonNaN_4T_4SepFill nodes case: fill elements of the MatrixCN: row of ind0_jy2NonNaN_4T_4SepFill node , column of down neighbor node(jy2); if ind0_jy2NaN_4T_4SepFill nodes case: elements are zero as initiated 
            MatrixCN[self.Reg_ind0_jz1NonNaN_4T_ALL,self.jz1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,4]/self.RouXc_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL])/self.Delta_z1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL]    #if ind0_jz1NonNaN_4T_4SepFill nodes case: fill elements of the MatrixCN: row of ind0_jz1NonNaN_4T_4SepFill node , column of inner neighbor node(jz1); if ind0_jz1NaN_4T_4SepFill nodes case: elements are zero as initiated 
            MatrixCN[self.Reg_ind0_jz2NonNaN_4T_ALL,self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,5]/self.RouXc_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL])/self.Delta_z2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL] * (self.delta_x1_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1]+self.delta_x2_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1])/(self.delta_x1_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL])    #if ind0_jz2NonNaN_4T_4SepFill nodes case: fill elements of the MatrixCN: row of ind0_jz2NonNaN_4T_4SepFill node , column of outer neighbor node(jz2); if ind0_jz2NaN_4T_4SepFill nodes case: elements are zero as initiated 
                #--------------------------------------fill in diagonal terms
            MatrixCN[self.Reg_ind0_jx1NaN_4T_ALL,self.Reg_ind0_jx1NaN_4T_ALL] += self.h_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0]/self.RouXc_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL])       #jx1 components in diagonal terms
            MatrixCN[self.Reg_ind0_jx1NonNaN_4T_ALL,self.Reg_ind0_jx1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0]/self.RouXc_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL])/self.Delta_x1_4T_ALL[self.Reg_ind0_jx1NonNaN_4T_ALL]
    
            MatrixCN[self.Reg_ind0_jx2NaN_4T_ALL,self.Reg_ind0_jx2NaN_4T_ALL] += self.h_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1]/self.RouXc_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL])       #jx2 components in diagonal terms
            MatrixCN[self.Reg_ind0_jx2NonNaN_4T_ALL,self.Reg_ind0_jx2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,1]/self.RouXc_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL,0]*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL])/self.Delta_x2_4T_ALL[self.Reg_ind0_jx2NonNaN_4T_ALL]
    
            MatrixCN[self.Reg_ind0_jy1NaN_4T_ALL,self.Reg_ind0_jy1NaN_4T_ALL] += self.h_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL])       #jy1 components in diagonal terms
            MatrixCN[self.Reg_ind0_jy1NonNaN_4T_ALL,self.Reg_ind0_jy1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL])/self.Delta_y1_4T_ALL[self.Reg_ind0_jy1NonNaN_4T_ALL]
    
            MatrixCN[self.Reg_ind0_jy2NaN_4T_ALL,self.Reg_ind0_jy2NaN_4T_ALL] += self.h_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL])       #jy2 components in diagonal terms
            MatrixCN[self.Reg_ind0_jy2NonNaN_4T_ALL,self.Reg_ind0_jy2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL])/self.Delta_y2_4T_ALL[self.Reg_ind0_jy2NonNaN_4T_ALL]
    
            MatrixCN[self.Reg_ind0_jz1NaN_4T_ALL,self.Reg_ind0_jz1NaN_4T_ALL] += self.h_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4]/self.RouXc_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL])       #jz1 components in diagonal terms
            MatrixCN[self.Reg_ind0_jz1NonNaN_4T_ALL,self.Reg_ind0_jz1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,4]/self.RouXc_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL])/self.Delta_z1_4T_ALL[self.Reg_ind0_jz1NonNaN_4T_ALL]
    
            MatrixCN[self.Reg_ind0_jz2NaN_4T_ALL,self.Reg_ind0_jz2NaN_4T_ALL] += self.h_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5]/self.RouXc_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL])       #jz2 components in diagonal terms
            MatrixCN[self.Reg_ind0_jz2NonNaN_4T_ALL,self.Reg_ind0_jz2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,5]/self.RouXc_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL,0]*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL])/self.Delta_z2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL] * (self.delta_x1_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1]+self.delta_x2_4T_ALL[self.jz2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]-1])/(self.delta_x1_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jz2NonNaN_4T_ALL])
    
            MatrixCN[self.Reg_ind0_4T_ALL,self.Reg_ind0_4T_ALL] += 1                                                                                               #"1" components in diagonal terms
            #======================================calculate nodes suitable for Irregular stencil; e.g. Sep nodes (i.e. 85~87)
            MatrixCN[self.Irreg_ind0_4T_ALL,:]=0        #initialize the Irregular stencil, e.g. the last 3 rows in MatrixCN
                #--------------------------------------fill in jxz, jy1, jy2 terms
            Irreg_ind0_jxzNonNaN_vector=self.Irreg_ind0_jxzNonNaN_4T_ALL.reshape(-1,1)
            Irreg_ind0_jxzNonNaN_matrix=self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1
            MatrixCN[Irreg_ind0_jxzNonNaN_vector,Irreg_ind0_jxzNonNaN_matrix] += -self.Alpha_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]*self.dt/4/self.S_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]*(self.delta_x1_Sub4Irreg_4T_ALL[Irreg_ind0_jxzNonNaN_matrix]+self.delta_x2_Sub4Irreg_4T_ALL[Irreg_ind0_jxzNonNaN_matrix])/self.Delta_z1_4T_ALL[Irreg_ind0_jxzNonNaN_matrix]
                
            MatrixCN[self.Irreg_ind0_jy1NonNaN_4T_ALL,self.jy1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL])/self.Delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]    #if Irreg_ind0_jy1NonNaN nodes case: fill elements of the MatrixCN: row of Irreg_ind0_jy1NonNaN node , column of up neighbor node(jy1); if ind0_jy1NaN_4T_4SepFill nodes case: elements are zero as initiated 
            MatrixCN[self.Irreg_ind0_jy2NonNaN_4T_ALL,self.jy2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]-1] += -self.Lamda_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL])/self.Delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]    #if Irreg_ind0_jy2NonNaN nodes case: fill elements of the MatrixCN: row of Irreg_ind0_jy2NonNaN node , column of down neighbor node(jy2); if ind0_jy2NaN_4T_4SepFill nodes case: elements are zero as initiated 
                #--------------------------------------fill in diagonal terms
            MatrixCN[self.Irreg_ind0_jxzNonNaN_4T_ALL,self.Irreg_ind0_jxzNonNaN_4T_ALL] += (( self.Alpha_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]*self.dt/4/self.S_Irreg_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL] * (self.delta_x1_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1]+self.delta_x2_Sub4Irreg_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1])       \
                                                                                  /self.Delta_z1_4T_ALL[self.jxz_4T_ALL[self.Irreg_ind0_jxzNonNaN_4T_ALL]-1] )  .dot( np.ones([self.nx,1]) )).reshape(-1)                                                                                                             #jxz components in diagonal terms        
    
            MatrixCN[self.Irreg_ind0_jy1NaN_4T_ALL,self.Irreg_ind0_jy1NaN_4T_ALL] += self.h_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL])                                                       #jy1 components in diagonal terms
            MatrixCN[self.Irreg_ind0_jy1NonNaN_4T_ALL,self.Irreg_ind0_jy1NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL])/self.Delta_y1_4T_ALL[self.Irreg_ind0_jy1NonNaN_4T_ALL]
    
            MatrixCN[self.Irreg_ind0_jy2NaN_4T_ALL,self.Irreg_ind0_jy2NaN_4T_ALL] += self.h_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL])                                                       #jy2 components in diagonal terms
            MatrixCN[self.Irreg_ind0_jy2NonNaN_4T_ALL,self.Irreg_ind0_jy2NonNaN_4T_ALL] += self.Lamda_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL,0]*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL])/self.Delta_y2_4T_ALL[self.Irreg_ind0_jy2NonNaN_4T_ALL]
            
            MatrixCN[self.Irreg_ind0_4T_ALL,self.Irreg_ind0_4T_ALL] += 1                                                                                                                                                                                                   #"1" components in diagonal terms        
            #======================================penalty on Temperature-constrained BC nodes (apply temperature BC)
            MatrixCN[self.ind0_BCtem_ALL,self.ind0_BCtem_ALL]=inf
        return MatrixCN
 
    #########################################################   
    ###########       function for VectorCN       ###########
    #########################################################
    def fun_VectorCN(self):                #VectorCN = VectorCN_preTp*Tp + VectorCN_conv_q;    VectorCN_preTp is very similar to MatrixCN, so form VectorCN based on MatrixCN
#        global VectorCN, VectorCN_conv_q
        if self.status_ThermalBC_Core=='SepFill' or self.status_ThermalBC_Core=='SepAir': 
            VectorCN=np.zeros([self.n_4T_ALL,1])       
    #        VectorCN_preTp=np.zeros([n_4T_ALL,n_4T_ALL])
            self.VectorCN_conv_q=np.zeros([self.n_4T_ALL,1])
            #==================================================add Tp term  (this part is moved to the function above)        
    #        VectorCN_preTp=MatrixCN.copy()
    #        VectorCN_preTp[np.arange(n_4T_ALL),np.arange(n_4T_ALL)] -= 1
    #        VectorCN_preTp = -VectorCN_preTp
    #        VectorCN_preTp[np.arange(n_4T_ALL),np.arange(n_4T_ALL)] += 1
            #==================================================add non Tp term - conv         
                #--------------------------------------for Regular stencil
            self.VectorCN_conv_q[self.Reg_ind0_jx1NaN_4T_ALL,0] += self.h_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0]/self.RouXc_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0]*2*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Reg_ind0_jx1NaN_4T_ALL,0]                                                                                               #if ind0_jx1NaN_4T_4SepFill nodes case: fill elements of the jx1 terms
            self.VectorCN_conv_q[self.Reg_ind0_jx2NaN_4T_ALL,0] += self.h_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1]/self.RouXc_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,0]*2*self.dt/(self.delta_x1_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL]+self.delta_x2_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Reg_ind0_jx2NaN_4T_ALL,1]                                                                                               #if ind0_jx2NaN_4T_4SepFill nodes case: fill elements of the jx2 terms
            self.VectorCN_conv_q[self.Reg_ind0_jy1NaN_4T_ALL,0] += self.h_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Reg_ind0_jy1NaN_4T_ALL,2]                                                                                               #if ind0_jy1NaN_4T_4SepFill nodes case: fill elements of the jy1 terms
            self.VectorCN_conv_q[self.Reg_ind0_jy2NaN_4T_ALL,0] += self.h_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Reg_ind0_jy2NaN_4T_ALL,3]                                                                                               #if ind0_jy2NaN_4T_4SepFill nodes case: fill elements of the jy2 terms
            self.VectorCN_conv_q[self.Reg_ind0_jz1NaN_4T_ALL,0] += self.h_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4]/self.RouXc_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,0]*2*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Reg_ind0_jz1NaN_4T_ALL,4]                                                                                               #if ind0_jz1NaN_4T_4SepFill nodes case: fill elements of the jz1 terms      
            self.VectorCN_conv_q[self.Reg_ind0_jz2NaN_4T_ALL,0] += self.h_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5]/self.RouXc_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,0]*2*self.dt/(self.delta_z1_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL]+self.delta_z2_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Reg_ind0_jz2NaN_4T_ALL,5]                                                                                               #if ind0_jz2NaN_4T_4SepFill nodes case: fill elements of the jz2 terms
                #--------------------------------------for Irregular stencil
            self.VectorCN_conv_q[self.Irreg_ind0_jy1NaN_4T_ALL,0] += self.h_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]/self.RouXc_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Irreg_ind0_jy1NaN_4T_ALL,2]
            self.VectorCN_conv_q[self.Irreg_ind0_jy2NaN_4T_ALL,0] += self.h_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]/self.RouXc_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,0]*2*self.dt/(self.delta_y1_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]+self.delta_y2_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL]) * self.Tconv_4T_ALL[self.Irreg_ind0_jy2NaN_4T_ALL,3]
            #==================================================add non Tp term - heat gen q         
            self.VectorCN_conv_q += self.q_4T_ALL*self.dt/self.RouXc_4T_ALL                                                                                                                                                                                                     #heat gen components
    
            VectorCN= self.VectorCN_preTp .dot( self.T1_4T_ALL ) + self.VectorCN_conv_q
            #======================================penalty on Temperature-constrained BC nodes (apply temperature BC)
            VectorCN[self.ind0_BCtem_ALL,0]=(self.T3_4T_ALL[self.ind0_BCtem_ALL,0]*inf)
                
        return VectorCN
  
    
    
    
    #########################################################   
    #########     function for Visualization     ############
    #########################################################
    def fun_mayavi_by_node(self, XYZ_Module, plot_Variable_ALL, vmin, vmax, title_string, colormap_string):        
        
#        self.plot_steps_available=np.where(~np.isnan(self.T_avg_record))[0]      #in cycling mode, there are NaN values in the last cycles. So here plot_steps_available is all the steps with non-NaN values 
#        self.plot_step=self.plot_steps_available[-1]                             #plot the last step from non-NaN steps 
        
#        self.plot_Variable_ALL=self.T_record[:,plot_step]-273.15  #variable to be visualized  
        self.an_PlotRoll_1=1; self.an_PlotRoll_2=self.nx-4           #two cross sections for Jellyroll plotting
        self.an_PlotCan_1=1; self.an_PlotCan_2=self.nx-2           #two cross sections for Can plotting
        self.status_plot_CoreSep_T='No'                 #plot core separator T or not
        
        self.X_Module = XYZ_Module[0]         #this cell's X location in Module
        self.Y_Module = XYZ_Module[1]
        self.Z_Module = XYZ_Module[2]
        #===================================prepare X,Y,Z,C for visualization
        self.yi_plot_4T_ALL=self.yi_4T_ALL.copy()
        #-----------------------plot jellyroll surface
        self.ind0_plot_surface=self.ind0_Geo_surfaceWithEnds_4T_4SepFill.reshape(self.ny,self.nx)
        self.ind0_plot_surface=self.ind0_plot_surface[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(self.ny,-1)
        self.X1=-self.zi_4T_ALL[self.ind0_plot_surface]*np.sin(self.theta_4T_ALL[self.ind0_plot_surface])
        self.Y1=self.zi_4T_ALL[self.ind0_plot_surface]*np.cos(self.theta_4T_ALL[self.ind0_plot_surface]) 
        self.Y1 -= self.fun_prismatic_mayavi(self.ind0_plot_surface)
        self.Z1=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_surface]
        self.C1=plot_Variable_ALL[self.ind0_plot_surface]
        self.X1 += self.X_Module; self.Y1 += self.Y_Module; self.Z1 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll top
        self.ind0_plot_top=np.where(self.ax_4T==1)[0].reshape(-1,self.nx)
        if self.status_plot_CoreSep_T=='Yes':
            temp1=np.repeat(self.ind0_Geo_core_AddSep_4T_4SepFill[0],self.nx).reshape(1,-1)
            self.ind0_plot_top=np.append(temp1,self.ind0_plot_top,axis=0)
        self.ind0_plot_top=self.ind0_plot_top[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(-1,self.an_PlotRoll_2-self.an_PlotRoll_1+1)
        self.X2=-self.zi_4T_ALL[self.ind0_plot_top]*np.sin(self.theta_4T_ALL[self.ind0_plot_top])
        self.Y2=self.zi_4T_ALL[self.ind0_plot_top]*np.cos(self.theta_4T_ALL[self.ind0_plot_top]) 
        self.Y2 -= self.fun_prismatic_mayavi(self.ind0_plot_top)
        self.Z2=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_top]
        self.C2=plot_Variable_ALL[self.ind0_plot_top]
        self.X2 += self.X_Module; self.Y2 += self.Y_Module; self.Z2 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll bottom
        self.ind0_plot_bottom=np.where(self.ax_4T==self.ny)[0].reshape(-1,self.nx)
        if self.status_plot_CoreSep_T=='Yes':
            temp1=np.repeat(self.ind0_Geo_core_AddSep_4T_4SepFill[-1],self.nx).reshape(1,-1)
            self.ind0_plot_bottom=np.append(temp1,self.ind0_plot_bottom,axis=0)
        self.ind0_plot_bottom=self.ind0_plot_bottom[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(-1,self.an_PlotRoll_2-self.an_PlotRoll_1+1)
        self.X3=-self.zi_4T_ALL[self.ind0_plot_bottom]*np.sin(self.theta_4T_ALL[self.ind0_plot_bottom])
        self.Y3=self.zi_4T_ALL[self.ind0_plot_bottom]*np.cos(self.theta_4T_ALL[self.ind0_plot_bottom]) 
        self.Y3 -= self.fun_prismatic_mayavi(self.ind0_plot_bottom)
        self.Z3=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_bottom]
        self.C3=plot_Variable_ALL[self.ind0_plot_bottom]
        self.X3 += self.X_Module; self.Y3 += self.Y_Module; self.Z3 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll cross section1
        self.ind0_plot_crosssection1=np.where(self.an_4T==self.an_PlotRoll_1)[0].reshape(-1,self.nz_4T)
        if self.status_plot_CoreSep_T=='Yes':
            self.ind0_plot_crosssection1=np.append(self.ind0_Geo_core_AddSep_4T_4SepFill.reshape(-1,1),self.ind0_plot_crosssection1,axis=1)
        self.X4=-self.zi_4T_ALL[self.ind0_plot_crosssection1]*np.sin(self.theta_4T_ALL[self.ind0_plot_crosssection1])
        self.Y4=self.zi_4T_ALL[self.ind0_plot_crosssection1]*np.cos(self.theta_4T_ALL[self.ind0_plot_crosssection1]) 
        self.Y4 -= self.fun_prismatic_mayavi(self.ind0_plot_crosssection1)
        self.Z4=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_crosssection1]
        self.C4=plot_Variable_ALL[self.ind0_plot_crosssection1]
        self.X4 += self.X_Module; self.Y4 += self.Y_Module; self.Z4 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll cross section2
        self.ind0_plot_crosssection2=np.where(self.an_4T==self.an_PlotRoll_2)[0].reshape(-1,self.nz_4T)
        if self.status_plot_CoreSep_T=='Yes':
            self.ind0_plot_crosssection2=np.append(self.ind0_Geo_core_AddSep_4T_4SepFill.reshape(-1,1),self.ind0_plot_crosssection2,axis=1)
        self.X5=-self.zi_4T_ALL[self.ind0_plot_crosssection2]*np.sin(self.theta_4T_ALL[self.ind0_plot_crosssection2])
        self.Y5=self.zi_4T_ALL[self.ind0_plot_crosssection2]*np.cos(self.theta_4T_ALL[self.ind0_plot_crosssection2]) 
        self.Y5 -= self.fun_prismatic_mayavi(self.ind0_plot_crosssection2)
        self.Z5=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_crosssection2]
        self.C5=plot_Variable_ALL[self.ind0_plot_crosssection2]
        self.X5 += self.X_Module; self.Y5 += self.Y_Module; self.Z5 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll surface(inner)
        self.ind0_plot_core=self.ind0_Geo_coreWithEnds_4T_4SepFill.reshape(self.ny,self.nx)
        self.ind0_plot_core=self.ind0_plot_core[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(self.ny,-1)
        self.X9=-self.zi_4T_ALL[self.ind0_plot_core]*np.sin(self.theta_4T_ALL[self.ind0_plot_core])
        self.Y9=self.zi_4T_ALL[self.ind0_plot_core]*np.cos(self.theta_4T_ALL[self.ind0_plot_core]) 
        self.Y9 -= self.fun_prismatic_mayavi(self.ind0_plot_core)
        self.Z9=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_core]
        self.C9=plot_Variable_ALL[self.ind0_plot_core]
        self.X9 += self.X_Module; self.Y9 += self.Y_Module; self.Z9 += self.Z_Module    #Get cells' spatial locations in Module
        
        #===================================visualization
#       self.fig = mlab.figure(bgcolor=(1,1,1))    
#       self.vmin=self.plot_Variable_ALL.min();  self.vmax=self.plot_Variable_ALL.max()
        self.surf1 = mlab.mesh(self.X1, self.Y1, self.Z1, scalars=self.C1, colormap=colormap_string)
        self.surf2 = mlab.mesh(self.X2, self.Y2, self.Z2, scalars=self.C2, colormap=colormap_string)
        self.surf3 = mlab.mesh(self.X3, self.Y3, self.Z3, scalars=self.C3, colormap=colormap_string)
        self.surf4 = mlab.mesh(self.X4, self.Y4, self.Z4, scalars=self.C4, colormap=colormap_string)
        self.surf5 = mlab.mesh(self.X5, self.Y5, self.Z5, scalars=self.C5, colormap=colormap_string)
        if self.status_plot_CoreSep_T != 'Yes':
            self.surf9 = mlab.mesh(self.X9, self.Y9, self.Z9, scalars=self.C9, colormap=colormap_string)
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
        if self.status_plot_CoreSep_T != 'Yes':
            self.surf9.module_manager.scalar_lut_manager.use_default_range = False
            self.surf9.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.cb=mlab.colorbar(title=title_string,orientation='vertical',label_fmt='%.2f',nb_labels=5)
        self.cb.scalar_bar.unconstrained_font_size = True
        self.cb.title_text_property.font_family = 'times'; self.cb.title_text_property.bold=False; self.cb.title_text_property.italic=False; self.cb.title_text_property.color=(0,0,0); self.cb.title_text_property.font_size=20
        self.cb.label_text_property.font_family = 'times'; self.cb.label_text_property.bold=True;  self.cb.label_text_property.italic=False; self.cb.label_text_property.color=(0,0,0); self.cb.label_text_property.font_size=15
        if self.status_mayavi_show_cell_num == 'Yes':
            mlab.text3d(self.X_Module,self.Y_Module,self.Z_Module-0.02, 'cell_%s'%str(self.cell_ind0+1), scale=.005, color=(0,0,0))
         
        
    def fun_mayavi_by_ele(self, XYZ_Module, plot_Variable_ALL, vmin, vmax, title_string, colormap_string):        
        #===================================prepare X,Y,Z,C for visualization
        #-----------------------plot jellyroll surface
        #self.ind0_plot_surface=ind0_Geo_surfaceWithEnds_4T_4SepFill.reshape(ny,nx)
        self.ind0_plot_surface=np.array([x for x in self.Elb_4T if self.ra_4T[x]==(self.nz_4T-1)],dtype=int).reshape(self.ny,self.nx)
        self.ind0_plot_surface=self.ind0_plot_surface[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(self.ny,-1)
        self.X1=-self.zi_4T_ALL[self.ind0_plot_surface]*np.sin(self.theta_4T_ALL[self.ind0_plot_surface])
        self.Y1=self.zi_4T_ALL[self.ind0_plot_surface]*np.cos(self.theta_4T_ALL[self.ind0_plot_surface]) 
        self.Y1 -= self.fun_prismatic_mayavi(self.ind0_plot_surface)
        self.Z1=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_surface]
        self.ind0_ele_plot_surface=self.List_node2ele_4T[self.ind0_plot_surface,0]
        self.C1=plot_Variable_ALL[self.ind0_ele_plot_surface]
        self.X1 += self.X_Module; self.Y1 += self.Y_Module; self.Z1 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll top
        self.ind0_plot_top=np.where(self.ax_4T==1)[0]
        self.ind0_plot_top=np.array([x for x in self.ind0_plot_top if (self.mat_4T[x]==3 or self.mat_4T[x]==4)],dtype=int).reshape(-1,self.nx)
        #temp1=np.repeat(ind0_Geo_core_AddSep_4T_4SepFill[0],nx).reshape(1,-1)
        #self.ind0_plot_top=np.append(temp1,ind0_plot_top,axis=0)
        self.ind0_plot_top=self.ind0_plot_top[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(-1,self.an_PlotRoll_2-self.an_PlotRoll_1+1)
        self.X2=-self.zi_4T_ALL[self.ind0_plot_top]*np.sin(self.theta_4T_ALL[self.ind0_plot_top])
        self.Y2=self.zi_4T_ALL[self.ind0_plot_top]*np.cos(self.theta_4T_ALL[self.ind0_plot_top]) 
        self.Y2 -= self.fun_prismatic_mayavi(self.ind0_plot_top)
        self.Z2=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_top]
        self.ind0_ele_plot_top=self.List_node2ele_4T[self.ind0_plot_top,0]
        self.C2=plot_Variable_ALL[self.ind0_ele_plot_top]
        self.X2 += self.X_Module; self.Y2 += self.Y_Module; self.Z2 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll bottom
        self.ind0_plot_bottom=np.where(self.ax_4T==self.ny)[0]
        self.ind0_plot_bottom=np.array([x for x in self.ind0_plot_bottom if (self.mat_4T[x]==3 or self.mat_4T[x]==4)],dtype=int).reshape(-1,self.nx)
        #temp1=np.repeat(ind0_Geo_core_AddSep_4T_4SepFill[-1],nx).reshape(1,-1)
        #self.ind0_plot_bottom=np.append(temp1,ind0_plot_bottom,axis=0)
        self.ind0_plot_bottom=self.ind0_plot_bottom[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(-1,self.an_PlotRoll_2-self.an_PlotRoll_1+1)
        self.X3=-self.zi_4T_ALL[self.ind0_plot_bottom]*np.sin(self.theta_4T_ALL[self.ind0_plot_bottom])
        self.Y3=self.zi_4T_ALL[self.ind0_plot_bottom]*np.cos(self.theta_4T_ALL[self.ind0_plot_bottom]) 
        self.Y3 -= self.fun_prismatic_mayavi(self.ind0_plot_bottom)
        self.Z3=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_bottom]
        self.ind0_ele_plot_bottom=self.List_node2ele_4T[self.ind0_plot_bottom,0]
        self.C3=plot_Variable_ALL[self.ind0_ele_plot_bottom]
        self.X3 += self.X_Module; self.Y3 += self.Y_Module; self.Z3 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll cross section1
        self.ind0_plot_crosssection1=np.where(self.an_4T==self.an_PlotRoll_1)[0]
        self.ind0_plot_crosssection1=np.array([x for x in self.ind0_plot_crosssection1 if (self.mat_4T[x]==3 or self.mat_4T[x]==4)],dtype=int).reshape(self.ny,-1)
        #self.ind0_plot_crosssection1=np.append(ind0_Geo_core_AddSep_4T_4SepFill.reshape(-1,1),ind0_plot_crosssection1,axis=1)
        self.X4=-self.zi_4T_ALL[self.ind0_plot_crosssection1]*np.sin(self.theta_4T_ALL[self.ind0_plot_crosssection1])
        self.Y4=self.zi_4T_ALL[self.ind0_plot_crosssection1]*np.cos(self.theta_4T_ALL[self.ind0_plot_crosssection1]) 
        self.Y4 -= self.fun_prismatic_mayavi(self.ind0_plot_crosssection1)
        self.Z4=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_crosssection1]
        self.ind0_ele_plot_crosssection1=self.List_node2ele_4T[self.ind0_plot_crosssection1,0]
        self.C4=plot_Variable_ALL[self.ind0_ele_plot_crosssection1]
        self.X4 += self.X_Module; self.Y4 += self.Y_Module; self.Z4 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll cross section2
        self.ind0_plot_crosssection2=np.where(self.an_4T==self.an_PlotRoll_2)[0]
        self.ind0_plot_crosssection2=np.array([x for x in self.ind0_plot_crosssection2 if (self.mat_4T[x]==3 or self.mat_4T[x]==4)],dtype=int).reshape(self.ny,-1)
        #ind0_plot_crosssection2=np.append(ind0_Geo_core_AddSep_4T_4SepFill.reshape(-1,1),ind0_plot_crosssection2,axis=1)
        self.X5=-self.zi_4T_ALL[self.ind0_plot_crosssection2]*np.sin(self.theta_4T_ALL[self.ind0_plot_crosssection2])
        self.Y5=self.zi_4T_ALL[self.ind0_plot_crosssection2]*np.cos(self.theta_4T_ALL[self.ind0_plot_crosssection2]) 
        self.Y5 -= self.fun_prismatic_mayavi(self.ind0_plot_crosssection2)
        self.Z5=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_crosssection2]
        self.ind0_ele_plot_crosssection2=self.List_node2ele_4T[self.ind0_plot_crosssection2,0]
        self.C5=plot_Variable_ALL[self.ind0_ele_plot_crosssection2]
        self.X5 += self.X_Module; self.Y5 += self.Y_Module; self.Z5 += self.Z_Module    #Get cells' spatial locations in Module
        #-----------------------plot jellyroll surface(inner)
        #self.ind0_plot_surface=ind0_Geo_surfaceWithEnds_4T_4SepFill.reshape(ny,nx)
        self.ind0_plot_surface=np.array([x for x in self.Elb_4T if self.ra_4T[x]==2],dtype=int).reshape(self.ny,self.nx)
        self.ind0_plot_surface=self.ind0_plot_surface[:,self.an_PlotRoll_1-1:self.an_PlotRoll_2].reshape(self.ny,-1)
        self.X9=-self.zi_4T_ALL[self.ind0_plot_surface]*np.sin(self.theta_4T_ALL[self.ind0_plot_surface])
        self.Y9=self.zi_4T_ALL[self.ind0_plot_surface]*np.cos(self.theta_4T_ALL[self.ind0_plot_surface]) 
        self.Y9 -= self.fun_prismatic_mayavi(self.ind0_plot_surface)
        self.Z9=self.LG_Jellyroll*(1-1/2/self.ny)-self.yi_plot_4T_ALL[self.ind0_plot_surface]
        self.ind0_ele_plot_surface=self.List_node2ele_4T[self.ind0_plot_surface,0]
        self.C9=plot_Variable_ALL[self.ind0_ele_plot_surface]
        self.X9 += self.X_Module; self.Y9 += self.Y_Module; self.Z9 += self.Z_Module    #Get cells' spatial locations in Module
        
        #===================================visualization
#       self.fig = mlab.figure(bgcolor=(1,1,1))    
#       self.vmin=self.plot_Variable_ALL.min();  self.vmax=self.plot_Variable_ALL.max()
        self.surf1 = mlab.mesh(self.X1, self.Y1, self.Z1, scalars=self.C1, colormap=colormap_string)
        self.surf2 = mlab.mesh(self.X2, self.Y2, self.Z2, scalars=self.C2, colormap=colormap_string)
        self.surf3 = mlab.mesh(self.X3, self.Y3, self.Z3, scalars=self.C3, colormap=colormap_string)
        self.surf4 = mlab.mesh(self.X4, self.Y4, self.Z4, scalars=self.C4, colormap=colormap_string)
        self.surf5 = mlab.mesh(self.X5, self.Y5, self.Z5, scalars=self.C5, colormap=colormap_string)
        self.surf9 = mlab.mesh(self.X9, self.Y9, self.Z9, scalars=self.C9, colormap=colormap_string)
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
        self.surf9.module_manager.scalar_lut_manager.use_default_range = False
        self.surf9.module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])    
        self.cb=mlab.colorbar(title=title_string,orientation='vertical',label_fmt='%.2f',nb_labels=5)
        self.cb.scalar_bar.unconstrained_font_size = True
        self.cb.title_text_property.font_family = 'times'; self.cb.title_text_property.bold=False; self.cb.title_text_property.italic=False; self.cb.title_text_property.color=(0,0,0); self.cb.title_text_property.font_size=20
        self.cb.label_text_property.font_family = 'times'; self.cb.label_text_property.bold=True;  self.cb.label_text_property.italic=False; self.cb.label_text_property.color=(0,0,0); self.cb.label_text_property.font_size=15
    
    
    
    
    
    
    
    
    
    
    
