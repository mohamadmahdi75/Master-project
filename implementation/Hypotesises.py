import numpy as np

from numpy.linalg import det, inv
from JPDA_PRO import JPDA
# import tensorflow as tf


class GNN():

  def __init__(self,margine ,sensor_cov ):

    self.margin =margine
    self.sensor_cov = sensor_cov



  def Mahalanobis(self,Z,Z_hat,S):
    
    d = np.sum(((Z-Z_hat).T @ inv(S)).T * (Z-Z_hat),axis=0)

    return d

  


  def get_global_neighbor(self,Z , Z_hat , P ,C):

    S = C @ P @ C.T + self.sensor_cov

    Z = Z[~np.isnan(Z)].reshape((Z.shape[0],-1))

    if Z.size==0:

      return np.nan * np.ones_like(Z_hat)

    else:

      distances = self.Mahalanobis(Z,Z_hat,S)

      index = np.where(distances<=self.margin)[-1]

      if index.size==0:
        return np.nan * np.ones_like(Z_hat)
      
      else :

        I = np.argmin(distances)

        return Z[:,[I]]








class PDA_Filter():

  def __init__(self,Gate ,C ,sensor_cov , PD):

    self.Gate = Gate
    self.C = C
    self.sensor_cov = sensor_cov
    self.PD = PD
    
  
  

  def Correct(self,state, P ,Valid_Measurements ,Z_hat ,Beta):
    

    S = self.C @ P @ self.C.T + self.sensor_cov

    valid_measurements  =  Valid_Measurements[~np.isnan(Valid_Measurements)].reshape((Valid_Measurements.shape[0],-1))

    
    
    if valid_measurements.size==0:
 
      return state ,P
    
    else:

      Beta = Beta.flatten()

      (beta0 ,betai) = (Beta[0] ,Beta[1:])

      print('beta0:',beta0)
      print("betai:",betai)
      assert betai.shape[-1] == valid_measurements.shape[-1]


      E  = valid_measurements - Z_hat

      etot = E @ betai[np.newaxis,:].T

      W = P @ self.C.T @ inv(S)

      correcyed_state = state + W @ etot

      beta_v = ((E * (np.ones_like(Z_hat) * betai)) @ E.T) - (etot @ etot.T)

      Pc = P - W @ S @ W.T

      P_hat = W @ beta_v @ W.T

      corrected_P = beta0*P + (1-beta0)*Pc + P_hat
      
      return correcyed_state , corrected_P
















class PDA():

  def __init__(self, Gate_threshold ,state_cov, sensor_cov ,PD, SNR ,use_constant_p_false_alarm=False):

    self.use_constant_p_f_alarm = use_constant_p_false_alarm
    self.Gate = Gate_threshold

    self.SNR = SNR

    self.Cnz= [1,2 , np.pi , 4/3 * np.pi , np.pi**2 /2,
    (8 *np.pi**2)/15 ,np.pi**3/6 ,(16*np.pi**3)/105,

    np.pi**4/24 ,(32*np.pi**4)/945 , np.pi**5/120,

    (64*np.pi**5)/10395
    
    ]
 
    self.sensor_cov = sensor_cov
    self.state_cov = state_cov
    self.PD = PD
    
  
  def Validated_measurements(self,Z ,Z_hat , S):

    Z = Z[~np.isnan(Z)].reshape((Z_hat.shape[0],-1))
    distances =  np.sqrt(np.abs(np.sum(((Z-Z_hat).T @ inv(S)).T * (Z-Z_hat),axis=0).flatten()))
    index = np.where(distances<=self.Gate)[0]
    valid = Z[:,index]

    return valid

  def Normal(self,Z,Z_hat,S):

    # d= np.sqrt(np.abs(np.sum(((Z-Z_hat).T @ inv(S)).T * (Z-Z_hat),axis=0).flatten()))
    # d = d/np.sum(d)
    # M = Z_hat.size
    # num = np.exp(-0.5 * d**2)
    # den = (2*np.pi)**(M/2) * np.sqrt(np.abs(det(S)))
    # den = den if den != 0 else 1

    # result = num/den
    # return result

    dist = np.sum(((Z-Z_hat).T @ inv(S)).T * (Z-Z_hat),axis=0).flatten()

    num =  np.exp( -1*dist)
    den = np.sum(num)




    return num/den


  def BETA(self,Z,Z_hat,S):

    valid = self.Validated_measurements(Z,Z_hat,S)

    if not self.use_constant_p_f_alarm:
      nz = Z_hat.size 
      Volum = self.Cnz[nz] * self.Gate**(nz/2) * np.sqrt(np.abs(det(S)))
      phi = valid.shape[-1]  
      self.SNR = phi/Volum


  
    b = (self.SNR * np.sqrt(np.abs(det(2*np.pi*S))) * (1-self.PD))/self.PD
    distance = np.sum(((valid-Z_hat).T @ inv(S)).T * (valid-Z_hat),axis=0) 
    e = np.exp(-0.5 * distance)
    den = b + np.sum(e.flatten())

    betai = e/den 
    beta0 = b/den

    # L = (self.Normal(valid ,Z_hat ,S)*self.PD)/self.SNR

    # PG = 1/valid.shape[-1]

    # sumL = np.sum(L)
    # den = 1 - self.PD * PG  + sumL
    # betai = L/den
    # beta0 = (1 - self.PD*PG)/den

    return beta0 ,betai




  def Correcet(self,S,P,C,Z,Z_hat):

    
    S_M = C @ P @ C.T + self.sensor_cov
    valid = self.Validated_measurements(Z,Z_hat,S_M)

    if valid.size==0:
      
      v= [np.nan]*Z_hat.size
      v = np.array(v).reshape((Z_hat.shape))
      
      return S,P,v,np.zeros((1,))

    beta0,betai = self.BETA(Z,Z_hat,S_M)

    E = valid-Z_hat
    etot = (valid-Z_hat)@ betai[np.newaxis,:].T
    beta_v = (betai*E)@ E.T - etot@ etot.T


    W = P @ C.T @ inv(S_M)
    Pc = P - W @ S_M @ W.T
    P_hat = W @ beta_v @ W.T


    S = S + W @ etot
    P = beta0*P + (1-beta0)*Pc + P_hat

    return (S,P,valid , betai)
    
    


  












if __name__ =="__main__":

  Z = np.array([
        [1.5,3,5,7,9,np.nan],
        [2.2,4,6,8,10,np.nan]
  ])
  state_cov = np.identity(4)*5

  sensor_cov = np.diag([1.1**2]*2)

  gnn = GNN(10,sensor_cov)


  
  Z_hat = np.array([
            [1],
            [2]
  ])

  P = np.identity(4)*1

  C = np.array([
            [1,0,0,0],
            [0,0,1,0]
  ])

  # print('global neighbor',gnn.get_global_neighbor(Z,Z_hat,P,C))

  

  S = C @ P @ C.T + sensor_cov

  gate = 10

  dencity_of_clutter= 0.5
  PD=1
  SNR = 0.25
  pda = PDA(gate,state_cov,sensor_cov,PD ,SNR)
  jpda = JPDA(SNR ,sensor_cov)

  valid = pda.Validated_measurements(Z,Z_hat,S)

  print('JPDA likelihood',jpda.LIKELIHOOD(valid,Z_hat,S,PD))

  print('PDA LIKELIHOOD:',pda.BETA(Z,Z_hat,S))




 



 
  

  
