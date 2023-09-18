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
import os, sys    #for rerun the script for PID FPODT calibration
import scipy.io as sio 
import os.path
from pyecn.read_LUT import read_LUT

import pyecn.parse_inputs as ip

inf=1e10

# from Form_factor.ECN_plain.core_Allform_excld_LUT_combine_Init import Core_AllForm

class Read_LUTs:

    #########################################################   
    ########## function for reading LUT_SoC,LUT_T #########
    #########################################################                            
    def fun_read_SoCandT(self):        #this function is to generate LUT_SoC and LUT_T
#        global LUT_SoC, LUT_T, nSoC, nT
#        global LUT_SoC_min, LUT_SoC_max, LUT_T_min, LUT_T_max
       # file_name='\SoC.csv'
        # LUT_Address1=f'LUT-{ip.status_Eparam}'
        # LUT_Address2=f'LUT_Address1\{file_name}'
        LUT_SoC=read_LUT(ip.FileAddress_SOC_read).reshape(-1,1)
        LUT_T=read_LUT(ip.FileAddress_T_read).reshape(-1,1); LUT_T=LUT_T+273.15
        
        LUT_SoC_min=LUT_SoC.min(); LUT_SoC_max=LUT_SoC.max()
        LUT_T_min=LUT_T.min(); LUT_T_max=LUT_T.max()
        nSoC=np.size(LUT_SoC); nT=np.size(LUT_T)
        #-----------------------------------------------------
        LUT_SoC=np.concatenate( (np.array([[inf]]),LUT_SoC,np.array([[-inf]])) )   # this is for cut-off extrapolating; add +inf to be the first element and -inf to be the last element
        LUT_T=np.concatenate( (np.array([[-inf]]),LUT_T,np.array([[inf]])) )        # this is for cut-off extrapolating; add +inf to be the first element and -inf to be the last element

        (
        self.LUT_SoC, self.LUT_T, self.nSoC, self.nT,                                                  
        self.LUT_SoC_min, self.LUT_SoC_max, self.LUT_T_min, self.LUT_T_max,                            
        )=(                                                                                             
        LUT_SoC, LUT_T, nSoC, nT,                                                                      
        LUT_SoC_min, LUT_SoC_max, LUT_T_min, LUT_T_max
        )                        
    def fun_read_OCV(self):        #this function is to generate LUT_OCV_PerA
#        global LUT_OCV_PerA
#        global LUT_OCV_Cell_min, LUT_OCV_Cell_max 
        LUT_OCV_Cell=np.zeros([self.nT,self.nSoC,1])
        for i0_temp in range(int(ip.Temperature_Value_LUTs)):
            path = ip.OCV_names[i0_temp]
            check_file = os.path.isfile(path)
            if check_file==True:
                LUT_OCV_Cell[i0_temp,:,:]=read_LUT(ip.OCV_names[i0_temp])[:,1].reshape(-1,1)
            else:
                LUT_OCV_Cell[i0_temp,:,:]=read_LUT(ip.FileAddress_OCV_read)[:,1].reshape(-1,1)
        del i0_temp
        
        LUT_OCV_Cell_min=LUT_OCV_Cell[:,:,0].min(); LUT_OCV_Cell_max=LUT_OCV_Cell[:,:,0].max()
        LUT_OCV_PerA=LUT_OCV_Cell
        #-----------------------------------------------------
        LUT_OCV_PerA=np.concatenate( (LUT_OCV_PerA[:,0,0].reshape(self.nT,1,1),LUT_OCV_PerA,LUT_OCV_PerA[:,-1,0].reshape(self.nT,1,1)),axis=1 )               # for cut-off extrapolating; duplicate the first and last layer, along SoC axis
        LUT_OCV_PerA=np.concatenate( (LUT_OCV_PerA[0,:,:].reshape(1,self.nSoC+2,1), LUT_OCV_PerA, LUT_OCV_PerA[-1,:,:].reshape(1,self.nSoC+2,1)),axis=0 )     # duplicate the first and last layer, along the T axis

        (
        self.LUT_OCV_PerA,                                                          
        self.LUT_OCV_Cell_min, self.LUT_OCV_Cell_max                                
        )=(                                                                           
        LUT_OCV_PerA,                                                               
        LUT_OCV_Cell_min, LUT_OCV_Cell_max
        )
    def fun_read_dVdT(self):        #this function is to generate LUT_dVdT_PerA, i.e. dVdT for per volume
#        global LUT_SoC_entropy
#        global LUT_dVdT_PerA
#        global LUT_dVdT_Cell_min, LUT_dVdT_Cell_max
        
        
        LUT_SoC_entropy=read_LUT(ip.FileAddress_dVdT_read)[:,0].reshape(-1,1)
        LUT_SoC_entropy=np.concatenate( (np.array([[inf]]),LUT_SoC_entropy,np.array([[-inf]])) )   # this is for cut-off extrapolating; add +inf to be the first element and -inf to be the last element
    
        LUT_dVdT_Cell=read_LUT(ip.FileAddress_dVdT_read)[:,1].reshape(-1,1)
        
        LUT_dVdT_Cell_min=LUT_dVdT_Cell.min(); LUT_dVdT_Cell_max=LUT_dVdT_Cell.max()
        LUT_dVdT_PerA=LUT_dVdT_Cell
        #-----------------------------------------------------
        LUT_dVdT_PerA=np.concatenate( (LUT_dVdT_PerA[0].reshape(-1,1),LUT_dVdT_PerA,LUT_dVdT_PerA[-1].reshape(-1,1)) )  # this is for cut-off extrapolating; duplicate the first element and duplicate the last element    

        (
        self.LUT_SoC_entropy,                                                   
        self.LUT_dVdT_PerA,                                                     
        self.LUT_dVdT_Cell_min, self.LUT_dVdT_Cell_max                          
        )=(                                                                      
        LUT_SoC_entropy,                                                        
        LUT_dVdT_PerA,                                                          
        LUT_dVdT_Cell_min, LUT_dVdT_Cell_max
        )    
    def fun_read_RsCs(self):       #this function is to generate LUT_R0_PerA,LUT_Ri_PerA and LUT_Ci_PerA
#        global LUT_R0_PerA, LUT_Ri_PerA, LUT_Ci_PerA    
#        global LUT_R0_Cell_min, LUT_R0_Cell_max
#        global LUT_R1_Cell_min, LUT_R1_Cell_max, LUT_R2_Cell_min, LUT_R2_Cell_max, LUT_R3_Cell_min, LUT_R3_Cell_max, LUT_Ri_Cell_min, LUT_Ri_Cell_max  
#        global LUT_C1_Cell_min, LUT_C1_Cell_max, LUT_C2_Cell_min, LUT_C2_Cell_max, LUT_C3_Cell_min, LUT_C3_Cell_max, LUT_Ci_Cell_min, LUT_Ci_Cell_max  
#        global LUT_R0_PerA_min, LUT_R0_PerA_max
#        global LUT_Ri_PerA_min, LUT_Ri_PerA_max
#        global LUT_Ci_PerA_min, LUT_Ci_PerA_max
        nRC_local=ip.nRC
        LUT_R0_Cell=np.zeros([self.nT,self.nSoC,1]); LUT_Ri_Cell=np.zeros([self.nT,self.nSoC,nRC_local]); LUT_Ci_Cell=np.zeros([self.nT,self.nSoC,nRC_local]);

        for i0_temp in range(int(ip.Temperature_Value_LUTs)):
            LUT_R0_Cell[i0_temp,:,:]=read_LUT(ip.R0_names[i0_temp])[:,1].reshape(-1,1)
            LUT_Ri_Cell[i0_temp,:,:]=read_LUT(ip.Ri_names[i0_temp])[:,1:nRC_local+1]  #read R1-3 under T10 to first dimension (page) of LUT_Ri_PerA
            LUT_Ci_Cell[i0_temp,:,:]=read_LUT(ip.Ci_names[i0_temp])[:,1:nRC_local+1]  #read C1-3 under T10 to first dimension (page) of LUT_Ci_PerA
        del i0_temp


        if ip.status_Module == 'Yes':           
            #-----New Lines For splitting of Rs and Cs in case of MultipleJelly Roles
            LUT_R0_Cell=LUT_R0_Cell*(ip.LUT_Scale_Factor_Rs)
            LUT_Ri_Cell=LUT_Ri_Cell*(ip.LUT_Scale_Factor_Rs)
            LUT_Ci_Cell=LUT_Ci_Cell*(ip.LUT_Scale_Factor_Cs)
        if ip.status_Eparam=='Cylindrical_Cell2' or ip.status_Eparam=='Prismatic_Cell1':
           #-----#New lines when parameters were scaled by scaling the electrode area of some real cell to reflect the behaviour of unknown desired cell
            LUT_R0_Cell=LUT_R0_Cell*(ip.LUT_Scale_Factor_Rs_area)
            LUT_Ri_Cell=LUT_Ri_Cell*(ip.LUT_Scale_Factor_Rs_area)
            LUT_Ci_Cell=LUT_Ci_Cell*(ip.LUT_Scale_Factor_Cs_area)
        
        if self.status_R0minusRcc=='Yes':
            LUT_R0_Cell=LUT_R0_Cell-self.Rcc
        
        if nRC_local==3:
            LUT_R0_Cell_min=LUT_R0_Cell[:,:,0].min(); LUT_R0_Cell_max=LUT_R0_Cell[:,:,0].max()
            LUT_R1_Cell_min=LUT_Ri_Cell[:,:,0].min(); LUT_R1_Cell_max=LUT_Ri_Cell[:,:,0].max(); LUT_R2_Cell_min=LUT_Ri_Cell[:,:,1].min(); LUT_R2_Cell_max=LUT_Ri_Cell[:,:,1].max(); LUT_R3_Cell_min=LUT_Ri_Cell[:,:,2].min(); LUT_R3_Cell_max=LUT_Ri_Cell[:,:,2].max()
            LUT_C1_Cell_min=LUT_Ci_Cell[:,:,0].min(); LUT_C1_Cell_max=LUT_Ci_Cell[:,:,0].max(); LUT_C2_Cell_min=LUT_Ci_Cell[:,:,1].min(); LUT_C2_Cell_max=LUT_Ci_Cell[:,:,1].max(); LUT_C3_Cell_min=LUT_Ci_Cell[:,:,2].min(); LUT_C3_Cell_max=LUT_Ci_Cell[:,:,2].max()
            LUT_Ri_Cell_min=np.array([LUT_R1_Cell_min,LUT_R2_Cell_min,LUT_R3_Cell_min]); LUT_Ri_Cell_max=np.array([LUT_R1_Cell_max,LUT_R2_Cell_max,LUT_R3_Cell_max])
            LUT_Ci_Cell_min=np.array([LUT_C1_Cell_min,LUT_C2_Cell_min,LUT_C3_Cell_min]); LUT_Ci_Cell_max=np.array([LUT_C1_Cell_max,LUT_C2_Cell_max,LUT_C3_Cell_max])
        else:
            LUT_R0_Cell_min=LUT_R0_Cell[:,:,0].min(); LUT_R0_Cell_max=LUT_R0_Cell[:,:,0].max()
            LUT_R1_Cell_min=LUT_Ri_Cell[:,:,0].min(); LUT_R1_Cell_max=LUT_Ri_Cell[:,:,0].max(); LUT_R2_Cell_min=LUT_Ri_Cell[:,:,1].min(); LUT_R2_Cell_max=LUT_Ri_Cell[:,:,1].max()
            LUT_C1_Cell_min=LUT_Ci_Cell[:,:,0].min(); LUT_C1_Cell_max=LUT_Ci_Cell[:,:,0].max(); LUT_C2_Cell_min=LUT_Ci_Cell[:,:,1].min(); LUT_C2_Cell_max=LUT_Ci_Cell[:,:,1].max()
            LUT_Ri_Cell_min=np.array([LUT_R1_Cell_min,LUT_R2_Cell_min]); LUT_Ri_Cell_max=np.array([LUT_R1_Cell_max,LUT_R2_Cell_max])
            LUT_Ci_Cell_min=np.array([LUT_C1_Cell_min,LUT_C2_Cell_min]); LUT_Ci_Cell_max=np.array([LUT_C1_Cell_max,LUT_C2_Cell_max])
            
        LUT_R0_PerA=LUT_R0_Cell*self.A_electrodes_real;       LUT_Ri_PerA=LUT_Ri_Cell*self.A_electrodes_real;   LUT_Ci_PerA=LUT_Ci_Cell/self.A_electrodes_real         #transform whole cell Rs and Cs into Rs and Cs per layer thickness per area
        LUT_R0_PerA=LUT_R0_PerA/self.scalefactor_z;           LUT_Ri_PerA=LUT_Ri_PerA/self.scalefactor_z;       LUT_Ci_PerA=LUT_Ci_PerA*self.scalefactor_z             #transform above Rs and Cs per layer thickness per area into scaled layer thickness per area see ppt p115
    
        LUT_R0_PerA_min=LUT_R0_Cell_min*self.A_electrodes_real/self.scalefactor_z; LUT_R0_PerA_max=LUT_R0_Cell_max*self.A_electrodes_real/self.scalefactor_z
        LUT_Ri_PerA_min=LUT_Ri_Cell_min*self.A_electrodes_real/self.scalefactor_z; LUT_Ri_PerA_max=LUT_Ri_Cell_max*self.A_electrodes_real/self.scalefactor_z
        LUT_Ci_PerA_min=LUT_Ci_Cell_min/self.A_electrodes_real*self.scalefactor_z; LUT_Ci_PerA_max=LUT_Ci_Cell_max/self.A_electrodes_real*self.scalefactor_z
        #-----------------------------------------------------
        LUT_R0_PerA=np.concatenate( (LUT_R0_PerA[:,0,0].reshape(self.nT,1,1),LUT_R0_PerA,LUT_R0_PerA[:,-1,0].reshape(self.nT,1,1)),axis=1 )               # for cut-off extrapolating; duplicate the first and last layer, along SoC axis
        LUT_R0_PerA=np.concatenate( (LUT_R0_PerA[0,:,:].reshape(1,self.nSoC+2,1), LUT_R0_PerA, LUT_R0_PerA[-1,:,:].reshape(1,self.nSoC+2,1)),axis=0 )     # duplicate the first and last layer, along the T axis
        LUT_Ri_PerA=np.concatenate( (LUT_Ri_PerA[:,0,:].reshape(self.nT,1,nRC_local),LUT_Ri_PerA,LUT_Ri_PerA[:,-1,:].reshape(self.nT,1,nRC_local)),axis=1 )           # for cut-off extrapolating; duplicate the first and last layer, along SoC axis
        LUT_Ri_PerA=np.concatenate( (LUT_Ri_PerA[0,:,:].reshape(1,self.nSoC+2,nRC_local), LUT_Ri_PerA, LUT_Ri_PerA[-1,:,:].reshape(1,self.nSoC+2,nRC_local)),axis=0 ) # duplicate the first and last layer, along the T axis
        LUT_Ci_PerA=np.concatenate( (LUT_Ci_PerA[:,0,:].reshape(self.nT,1,nRC_local),LUT_Ci_PerA,LUT_Ci_PerA[:,-1,:].reshape(self.nT,1,nRC_local)),axis=1 )           # for cut-off extrapolating; duplicate the first and last layer, along SoC axis
        LUT_Ci_PerA=np.concatenate( (LUT_Ci_PerA[0,:,:].reshape(1,self.nSoC+2,nRC_local), LUT_Ci_PerA, LUT_Ci_PerA[-1,:,:].reshape(1,self.nSoC+2,nRC_local)),axis=0 ) # duplicate the first and last layer, along the T axis 
        
        if nRC_local==3:
            (
            self.LUT_R0_PerA, self.LUT_Ri_PerA, self.LUT_Ci_PerA,                             
            self.LUT_R0_Cell_min, self.LUT_R0_Cell_max,                                       
            self.LUT_R1_Cell_min, self.LUT_R1_Cell_max, self.LUT_R2_Cell_min, self.LUT_R2_Cell_max, self.LUT_R3_Cell_min, self.LUT_R3_Cell_max,                                       
            self.LUT_C1_Cell_min, self.LUT_C1_Cell_max, self.LUT_C2_Cell_min, self.LUT_C2_Cell_max, self.LUT_C3_Cell_min, self.LUT_C3_Cell_max,                                       
            self.LUT_Ri_Cell_min, self.LUT_Ri_Cell_max,                                       
            self.LUT_Ci_Cell_min, self.LUT_Ci_Cell_max,                                       
            self.LUT_R0_PerA_min, self.LUT_R0_PerA_max,                                       
            self.LUT_Ri_PerA_min, self.LUT_Ri_PerA_max,                                       
            self.LUT_Ci_PerA_min, self.LUT_Ci_PerA_max                                        
            )=(                                                                               
            LUT_R0_PerA, LUT_Ri_PerA, LUT_Ci_PerA,                                            
            LUT_R0_Cell_min, LUT_R0_Cell_max,                                                 
            LUT_R1_Cell_min, LUT_R1_Cell_max, LUT_R2_Cell_min, LUT_R2_Cell_max, LUT_R3_Cell_min, LUT_R3_Cell_max,                                       
            LUT_C1_Cell_min, LUT_C1_Cell_max, LUT_C2_Cell_min, LUT_C2_Cell_max, LUT_C3_Cell_min, LUT_C3_Cell_max,                                       
            LUT_Ri_Cell_min, LUT_Ri_Cell_max,                                                 
            LUT_Ci_Cell_min, LUT_Ci_Cell_max,                                                 
            LUT_R0_PerA_min, LUT_R0_PerA_max,                                                 
            LUT_Ri_PerA_min, LUT_Ri_PerA_max,                                                
            LUT_Ci_PerA_min, LUT_Ci_PerA_max
            )
        else:
            (
            self.LUT_R0_PerA, self.LUT_Ri_PerA, self.LUT_Ci_PerA,                             
            self.LUT_R0_Cell_min, self.LUT_R0_Cell_max,                                       
            self.LUT_R1_Cell_min, self.LUT_R1_Cell_max, self.LUT_R2_Cell_min, self.LUT_R2_Cell_max,                                       
            self.LUT_C1_Cell_min, self.LUT_C1_Cell_max, self.LUT_C2_Cell_min, self.LUT_C2_Cell_max,                                       
            self.LUT_Ri_Cell_min, self.LUT_Ri_Cell_max,                                       
            self.LUT_Ci_Cell_min, self.LUT_Ci_Cell_max,                                       
            self.LUT_R0_PerA_min, self.LUT_R0_PerA_max,                                       
            self.LUT_Ri_PerA_min, self.LUT_Ri_PerA_max,                                       
            self.LUT_Ci_PerA_min, self.LUT_Ci_PerA_max                                        
            )=(                                                                               
            LUT_R0_PerA, LUT_Ri_PerA, LUT_Ci_PerA,                                            
            LUT_R0_Cell_min, LUT_R0_Cell_max,                                                 
            LUT_R1_Cell_min, LUT_R1_Cell_max, LUT_R2_Cell_min, LUT_R2_Cell_max,                                       
            LUT_C1_Cell_min, LUT_C1_Cell_max, LUT_C2_Cell_min, LUT_C2_Cell_max,                                       
            LUT_Ri_Cell_min, LUT_Ri_Cell_max,                                                 
            LUT_Ci_Cell_min, LUT_Ci_Cell_max,                                                 
            LUT_R0_PerA_min, LUT_R0_PerA_max,                                                 
            LUT_Ri_PerA_min, LUT_Ri_PerA_max,                                                
            LUT_Ci_PerA_min, LUT_Ci_PerA_max
            )

    
