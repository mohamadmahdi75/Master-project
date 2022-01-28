from tkinter import N
import numpy as np

class KALMANFILTER():

  def __init__(self,A,B,C,cov_state ,cov_sensor):

    self.A = A.Transition_Model 
    self.B = B
    self.C = C 
    self.cov_state = cov_state
    self.cov_sensor = cov_sensor

  def predict(self,prior_state ,prior_cov ,U,time):

    P=  self.A(time) @ prior_state + self.B @ U
    C=  self.A(time) @ prior_cov @ self.A(time).T + self.cov_state

    return P , C


  def correct(self,prior_state ,prior_cov,measurement):

    K = prior_cov @ self.C.T  @  np.linalg.inv(self.C  @  prior_cov @ self.C.T  + self.cov_sensor )


    P = prior_state + K @ ( measurement - self.C @ prior_state )

    C = prior_cov - K @ self.C @ prior_cov

    return P ,C , K




beta = 0.125
#   این تابع نیاز به بروزرسانی داره فقط وقتی کار میکنه که همه ی دیکشنری ها پر باشن
def joint_pro(number_of_measurements,*dicts):

  if len(dicts)==1:
    return dicts
  
  elif len(dicts)==2:

    a,b = dicts[0],dicts[-1]
    new={}
    for k1,v1 in a.items():
      for k2,v2 in b.items():

        if (k1==0 and k2==0) or (k1 != k2):
          
          new[f"{k1}{k2}"] = v1 * v2 
    
    for k,v in new.items():
      M = k.count('0')
      n = (number_of_measurements -M) if (number_of_measurements-M)>0 else 0

      new[k] =v * beta**n

    return new
  
  elif len(dicts)>2:

    c = dicts[-1]

    LEN = len(dicts) -1
    
    new_ = joint_pro(number_of_measurements,*tuple(dicts[:LEN]))

    
    R={}
    for k1,v1 in new_.items():
      for k2,v2 in c.items():
        
        if k2==0 or str(k2) not in k1:

          b = beta if k2==0 else 1

          R[k1 + str(k2)] = v1*v2*b

    return R 



















if __name__ =="__main__":


    a={0:0.1 , 2:0.9, 3:0.1,5:0.01}
    b={0:0 , 2:0.9 , 3:0.1}
    c ={0:0.2 ,2:0.8}
    d = {0:0.1 , 3:0.9}
    e = {0:0.5 , 3:0.5}
    f = {0:0.7 , 3:0.3}

    for k,v in joint_pro(10,a,b,c,d,e,f).items():
      print(k,':',v)