import numpy as np
from numpy.linalg import inv ,det
from numpy import sqrt

#رو بگیریم   SNR آخرین اصلاحیه: لزومی نداره از کاربر 

class JPDA():

  def __init__(self, p_false_alarm , sensor_cov,use_constant_p_false_alarm=False):
    self.use_constant_p_false_alarm = use_constant_p_false_alarm
    self.SNR = p_false_alarm
    self.sensor_cov = sensor_cov
    self.Cnz= [1,2 , np.pi , 4/3 * np.pi , np.pi**2 /2,
    (8 *np.pi**2)/15 ,np.pi**3/6 ,(16*np.pi**3)/105,

    np.pi**4/24 ,(32*np.pi**4)/945 , np.pi**5/120,

    (64*np.pi**5)/10395
    
    ]

  #  در اینجا قصد دارم به دو روش ، فرضیه ها رو حساب کنم اول به صورت ثات با زمان و دومی متغییر
  def joint_pro(self,number_of_measurements ,sum_volum, *dicts):

    def joint_pro_(number_of_measurements,*dicts):

      if len(dicts)==1:
        d = {str(k):v  for k,v in dicts[0].items()}
        return d
      
      elif len(dicts)==2:

        a,b = dicts[0],dicts[-1]
        new={}
        for k1,v1 in a.items():
          for k2,v2 in b.items():

            if (k1==0 and k2==0) or (k1 != k2):
              
              new[f"{k1},{k2}"] = v1 * v2 
        
        return new
      
      elif len(dicts)>2:

        c = dicts[-1]
        LEN = len(dicts) -1
        new_ = joint_pro_(number_of_measurements,*tuple(dicts[:LEN]))
        R={}
        for k1,v1 in new_.items():
          for k2,v2 in c.items():
            
            keys= k1.split(',')

            if k2==0 or str(k2) not in keys:

              R[k1 + ',' + str(k2)] = v1*v2

        return R 

    result = joint_pro_(number_of_measurements ,*dicts)
    for k,v in result.items():
      keys = k.split(',')

      number_of_assigned_m = len(keys) - keys.count('0')
      number_of_Unassigned_m = number_of_measurements - number_of_assigned_m

      number_of_Unassigned_m = number_of_Unassigned_m if number_of_Unassigned_m>0 else 0
      phi = np.math.factorial(number_of_Unassigned_m)
      
      if self.use_constant_p_false_alarm:
        b = self.SNR**number_of_Unassigned_m
        # print(' costant  used '*100,'number of unassigned m=',number_of_Unassigned_m)
       
      else:
        b = phi/sum_volum
        # print(' variable is used '*100,'number of unassigned m=',number_of_Unassigned_m)
       
      
      result[k] = v * b
    

    return result

  def calculate_joint_L_for_each_target(self,new):
      if new.__class__.__name__ == 'tuple':
          new = new[0]


      sum_new = np.sum(list(new.values()))
      sum_new = sum_new if sum_new!=0 else  1
      new = {k:v/sum_new  for k,v in new.items()}

    
      n_targtes = len((list(new.keys())[0]).split(','))

      result=[]

      if n_targtes>1:

          for i in range(n_targtes):

              # targte_index = list(set([k1[i]   for k1,v1 in new.items() ]))
              #
              targte_index_of_m = list(set( [list(k1.split(','))[i]       for k1,v1 in new.items() ] ))
              #

              R = {j: np.sum([ v2  for k2,v2 in new.items() if list(k2.split(','))[i]==j ])      for j in targte_index_of_m  }

              result.append(R)

          return result
      else:
          return [new,]
          
  #   دو روش در محاسبه ی این تابع وجود داره
  def JOINT_LIKELIHOOD(self,n_measure ,sum_volum , *dicts):



      D = dict(enumerate(dicts))

      OK_dicts = [ v for k,v  in D.items() if v]

      if len(OK_dicts)==0 :
          
          return list(dicts)
      
      elif len(OK_dicts)==1:
          n = OK_dicts[0]
          sum_n = np.sum(list(n.values()))
          sum_n  = sum_n if sum_n !=0 else 1
          n = dict(sorted({int(k):v/sum_n for k,v in n.items()}.items()))

          for k,v in D.items():
              if v:
                  D[k] = n

          D = list(D.values())
          D = [  dict(sorted({ int(k):v   for k,v in i.items()}.items()) )         for i in D]

          return D
      

      
      elif len(OK_dicts)>1:

          
          new = self.joint_pro(n_measure ,sum_volum,*tuple(OK_dicts)) 

          targtes = self.calculate_joint_L_for_each_target(new)

          for k,v in D.items():

              if v:
                  D[k]=targtes.pop(0)
              else:
                  continue

          
          D = list(D.values())
          D = [  dict(sorted({ int(k):v   for k,v in i.items()}.items()) )         for i in D]
          
          return D



  #باید تابعی بسازم که شباهت هر چندتا هدف که بهش دادم رو حساب بکنه
  def distances(self,Z,Z_hat ,S):

    return np.sqrt(np.abs(np.sum(((Z-Z_hat).T @ inv(S)).T * (Z-Z_hat),axis=0)))
    




  def LIKELIHOOD(self,Z ,Z_hat ,S ,PD):

    distance = np.sum(((Z-Z_hat).T @ inv(S)).T * (Z-Z_hat),axis=0)
    # b = (self.SNR * np.sqrt(np.abs(det(2*np.pi*S))) * (1-PD))/ PD
    b = ( np.sqrt(np.abs(det(2*np.pi*S))) * (1-PD) )/PD
    e = np.exp(-0.5 * distance)
    # 

    den = b + np.sum(e.flatten())
    den = den if den!=0 else 1

    Pi = e /den
    P0 = b/den
    BETA = np.hstack([P0 ,Pi])

    return BETA
  
    
   
  
  #به صورت یک دیکشنری باشه P , Z_hat, C ,gate ,PD  توی این تابع مشخصات هدف باید شامل 
  def valid_measurements_and_Likelihood(self , Z ,*targets_characteristics):

    # example :   {'P':array , 'Z_hat':array , C:'array' , 'gate':number , 'PD':number }
    Z = Z[~np.isnan(Z)].reshape((Z.shape[0],-1))
    Z_dict = {i+1 : Z[:,[i]]  for i in range(Z.shape[-1])}
    number_of_targtes = len(targets_characteristics)

    if Z.size==0:
      d = [{'likelihood':{} ,'valid_m':{} },] * number_of_targtes
      return d

    if len(targets_characteristics)==1:

      target = targets_characteristics[0]
      
      S = target['C'] @ target['P'] @ (target['C']).T  + self.sensor_cov
      nz = target['Z_hat'].size
      volum =self.Cnz[target['Z_hat'].size] * np.sqrt(target['gate']**nz) * np.sqrt(np.abs(det(S)))

      distances = {k+1:v       for k,v in enumerate(self.distances(Z, target['Z_hat'] ,S))}

      likelihood={k:v    for k,v in enumerate (self.LIKELIHOOD(Z,target['Z_hat'],S,target['PD']).flatten())}

      valid_m = {k:v   for k,v in Z_dict.items() if distances[k]<=target['gate'] }

      valid_m = dict(sorted(valid_m.items()))
      # رو برداشتم چون قبلا توی محاسبه شباهت لحاظ شده PD در فرمول زیر، من 
      likelihood = {k:v  for k,v in likelihood.items() if k in valid_m or k==0 }
      if len(likelihood)==1:
        likelihood = {}
      
      likelihood = dict(sorted(likelihood.items()))

      return [{'likelihood':likelihood ,'valid_m':valid_m ,'volum':volum},]
    
    elif len(targets_characteristics)>1:

      number_of_targtes = len(targets_characteristics)-1

      
      prior_targtes = targets_characteristics[:number_of_targtes]
      
      target_new = targets_characteristics[-1]
      
      prior_result = self.valid_measurements_and_Likelihood(Z,*prior_targtes )
     
      S_new = target_new['C'] @ target_new['P'] @ (target_new['C']).T  + self.sensor_cov
      nz_new = target_new['Z_hat'].size
      volum_new =self.Cnz[target_new['Z_hat'].size] * np.sqrt(target_new['gate']**nz_new) * np.sqrt(np.abs(det(S_new)))

      distances_new = {k+1:v       for k,v in enumerate(self.distances(Z, target_new['Z_hat'] ,S_new))}
      likelihood_new={ k:v    for k,v in enumerate (self.LIKELIHOOD(Z,target_new['Z_hat'],S_new,target_new['PD']).flatten())}
      valid_m_new = {k:v   for k,v in Z_dict.items() if distances_new[k]<=target_new['gate'] }
      valid_m_new = dict(sorted(valid_m_new.items()))
      likelihood_new = {k:v  for k,v in likelihood_new.items() if k in valid_m_new or k==0 }

      if len(likelihood_new)==1:
        likelihood_new = {}

      likelihood_new = dict(sorted(likelihood_new.items()))

      return prior_result + [{'likelihood':likelihood_new ,'valid_m':valid_m_new,'volum':volum_new},]


  def get_joint_probabilities_and_valid_mesurements(self ,Z ,*targets_characteristics):

    Z = Z[~np.isnan(Z)].reshape((Z.shape[0],-1))
    # n_measurements = Z.shape[-1]
    
    V_and_L = self.valid_measurements_and_Likelihood(Z ,*targets_characteristics)
    n_M = (np.unique(np.hstack([list(i['valid_m'].keys())   for i in V_and_L])).flatten()).size
    # print('Valid_M==',n_M)
    likelihoods = [ i['likelihood']    for i in V_and_L ]
    sum_volum = np.sum([i['volum']  for i in V_and_L])
    
    likelihoods = self.JOINT_LIKELIHOOD(n_M ,sum_volum,*tuple(likelihoods))
  
    for i in V_and_L:
      i['likelihood'] = likelihoods.pop(0)

    return V_and_L


  #  فاز اصلاح الگوریتم پی دی ای رو فقط برای یک هدف انجام میده
  def PDA_correction(self,state , P  , C , Z_hat ,valid_m ,likelihood):
    
    valid_m = valid_m[~np.isnan(valid_m)].reshape((valid_m.shape[0],-1))
    b__ = np.array(list(likelihood.values()))

    if valid_m.size==0 or b__.size==0:
      return state ,P
    else:
     
      # beta0,betai = self.BETA(Z,Z_hat,S_M)

      # E = valid-Z_hat
      # etot = (valid-Z_hat)@ betai[np.newaxis,:].T
      # beta_v = (betai*E)@ E.T - etot@ etot.T


      # W = P @ C.T @ inv(S_M)
      # Pc = P - W @ S_M @ W.T
      # P_hat = W @ beta_v @ W.T


      # S = S + W @ etot
      # P = beta0*P + (1-beta0)*Pc + P_hat



      beta0 = np.array([likelihood.pop(0)])
      betai = np.hstack(list(likelihood.values()))
      E = valid_m - Z_hat
      S  = C @ P @ C.T  + self.sensor_cov

      etot = E @ betai[np.newaxis,:].T
      beta_v = (betai * E) @ E.T - etot @ etot.T
      W = P @ C.T @ inv(S)
      Pc = P - W @ S @ W.T
      P_hat = W @ beta_v @ W.T

      new_state = state + W @ etot
      new_P = beta0*P  + (1-beta0)*Pc + P_hat
      return new_state ,new_P



  # نیاز دارم بهم میده PDA   این تابع همه ی اطلاعاتی رو که من برای فاز اصلاح در الگوریتم 
  def create_data_needed_for_correction(self,Z,*targets_characteristics):

    P_M = self.get_joint_probabilities_and_valid_mesurements(Z,*targets_characteristics)
    TARGETS=list(targets_characteristics)
    result =[]
    for T in range(len(TARGETS)):
      T_I ={'state' : TARGETS[T]['state'] , 'P':TARGETS[T]['P'] ,
          'C':TARGETS[T]['C'] , 'Z_hat':TARGETS[T]["Z_hat"] , 'likelihood':P_M[T]['likelihood']}

      valid = P_M[T]['valid_m']
      valid = np.hstack(list(valid.values())) if valid else (np.nan *np.ones_like(T_I['Z_hat']))
      T_I['valid_m'] = valid

      result.append(T_I)


    
    return result

  def Correct(self,Z,*targets_characteristics):

    data = self.create_data_needed_for_correction(Z,*targets_characteristics)
    RESULT=[]

    for t in range(len(data)):
        
      target = data[t]
      state , P = self.PDA_correction(target['state'] , target['P'] ,target['C'],
                                        target['Z_hat'] ,target['valid_m'] ,target['likelihood'])
      target['state'] = state
      target['P']=P
      RESULT.append(  { 'state':target['state'] ,'P':target['P'] 
                        ,'likelihood':target['likelihood']  , 'valid_m':target['valid_m'] } )
  
    

    return RESULT
    

    
    

  
        
        

        


      

      



















if __name__ =="__main__":

  sensor_cov = np.diag([1.1**2]*2)

  Z = np.array([

                [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,np.nan,np.nan,],    # X
                [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,np.nan,np.nan,]    # Y
                
                
                ])

  C = np.array([[1,0,0,0],[0,0,1,0]])

  target1 ={'state':np.array([[25],[0],[26],[0]]),'P':np.identity(4) , 'Z_hat':np.array([[25],[26]]) , 'C':C , 'gate':10 ,'PD':0.99}

  target2 ={'state':np.array([[1],[5],[2],[3]]),'P':np.identity(4) , 'Z_hat':np.array([[1],[2]]) , 'C':C , 'gate':20 ,'PD':0.99}

  target3 ={'state':np.array([[5333],[0],[6333],[0]]),'P':np.identity(4),'Z_hat':np.array([[5333],[6333]]),'C':C ,'gate':10 ,'PD':0.7}

  target4 ={'state':np.array([[78],[0],[88],[0]]),'P':np.identity(4)*100 , 'Z_hat':np.array([[78],[88]]) , 'C':C , 'gate':10 ,'PD':0.85}
  
  target5 ={'state':np.array([[3.5],[0],[1.75],[0]]),'P':np.identity(4) , 'Z_hat':np.array([[3.5],[4.75]]) , 'C':C , 'gate':10 ,'PD':0.1}


  beta = 0.5
  jpda = JPDA(beta ,sensor_cov ,False)



  print((jpda.Correct(Z , *(  target3,target3))))

  
 
   
  
 