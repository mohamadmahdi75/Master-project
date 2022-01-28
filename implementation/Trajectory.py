from IPython.display import clear_output
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform

# np.random.seed(1991)


class Trajectory():

  def __init__(self,*type_of_motion):

    self.__transition_model=[]

    self.func_x=None
    self.func_y = None
    self.func_z =None

    self.model_types={"cv":self.Costant_Vel , "ca":self.Constat_Acc}

    

    for motion_model in type_of_motion:

      self.__transition_model.append(self.model_types[motion_model]())
    


    self.__transition_model = self.DIAG(self.__transition_model)

  def Transition_Model(self,time):

    t1 =np.where(self.__transition_model=='t',time,self.__transition_model)

    t2 = np.where(t1=="t**2"  , time**2 ,t1)
    return t2.astype(np.float32)

  

  def nonlinear_pos_vel(self,function_x,function_y):
    
    self.func_x = function_x
    self.func_y = function_y
    

  def nonlinear_pos_vel_acc(self,function_x,function_y,function_z):
    
    self.func_x = function_x
    self.func_y = function_y
    self.func_z = function_z

  def Transition_Model_nonlinear(self,time):
    pass


  def noise_type(self,noise_name,*args):
    
 
    noise_type={
                    "normal":np.random.normal(loc=args[0],scale=args[1] if len(args)>1 else 0.1,size=(self.__transition_model.shape[0],1)),
                    "uniform":np.random.uniform(low=args[0] , high=args[1] if len(args)>1 else 0.1,size=(self.__transition_model.shape[0],1)),
                    "poisson":np.random.poisson(lam=args[0],size=(self.__transition_model.shape[0],1))

                    }
    
    return noise_type[noise_name]



  def Costant_Vel(self):

    return np.array([[1,'t'],[0,1]])

  def Constat_Acc(self):

    return np.array([[1,'t','t**2'],[0 , 1 ,'t'],[0,0,1]])

  def DIAG(self,list_):

    n_rows,n_cols = np.sum(np.array([list(i.shape) for i in list_]),axis=0)

    Result = np.zeros((n_rows,n_cols)).astype(str)

    start_row , start_col = 0,0
    for i in list_:

      index_row = slice(start_row, start_row + i.shape[0] )
      index_col=slice(start_col, start_col+i.shape[1] )
      
      Result[index_row,index_col] = i

      start_row += i.shape[0]
      start_col += i.shape[1]
    
    return Result

  def __call__(self,X,B,U,time,noise_type,*args):
   

    return  self.Transition_Model(time) @ X + B @ U + self.noise_type(noise_type,*args)

  

  
if __name__ == "__main__":

    tr = Trajectory('cv')

    print(tr.Transition_Model(1))