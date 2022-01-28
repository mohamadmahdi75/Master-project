import numpy as np
from numpy.linalg import inv ,det


#رو بگیریم   SNR آخرین اصلاحیه: لزومی نداره از کاربر 

class JPDA():

  def __init__(self,sensor_cov,SNR=0.25,use_constant_p_false_alarm=False):
    
    self.sensor_cov = sensor_cov
    self.SNR=SNR
    self.use_constant_p_false_alarm=use_constant_p_false_alarm
    self.Cnz= [1,2 , np.pi , 4/3 * np.pi , np.pi**2 /2,
    (8 *np.pi**2)/15 ,np.pi**3/6 ,(16*np.pi**3)/105,

    np.pi**4/24 ,(32*np.pi**4)/945 , np.pi**5/120,

    (64*np.pi**5)/10395
    
    ]
  # محاسبه ی فاصله ی ماهالانوبیس
  def distances(self,Z,Z_hat,S):
    return np.sqrt(np.abs( np.sum( ((Z-Z_hat).T @ inv(S)).T * (Z-Z_hat),axis=0)    )).flatten()
  # محاسبه ی احتمالات شرطی 
  def BETA(self,valid_z,Z_hat,S,gate,PD):

    # if not self.use_constant_p_false_alarm:
    #   nz = Z_hat.size 
    #   Volum = self.Cnz[nz] * gate**(nz/2) * np.sqrt(det(np.abs(S)))
    #   phi = valid_z.shape[-1]  
    #   self.SNR = phi/Volum


  
    # b = (self.SNR * np.sqrt(np.abs(det(2*np.pi*S))) * (1-PD))/PD
    # distance = np.sum(((valid_z-Z_hat).T @ inv(S)).T * (valid_z-Z_hat),axis=0) 
    # e = np.exp(-0.5 * distance)
    # den = b + np.sum(e.flatten())

    # # den = den if den!=0 else 1

    # betai = e/den 
    # beta0 = b/den
    # return np.hstack([ beta0 ,betai ])




    if not self.use_constant_p_false_alarm:
      nz = Z_hat.size 
      Volum = self.Cnz[nz] * np.sqrt(np.abs(gate**(nz))) * np.sqrt(np.abs(det(S)))
      phi = valid_z.shape[-1]  
      SNR = phi/Volum
      b = (SNR * np.sqrt(np.abs(det(2*np.pi*S))) * (1-PD))/PD
       
      distance = np.sum(((valid_z-Z_hat).T @ inv(S)).T * (valid_z-Z_hat),axis=0) 
      e = np.exp(-0.5 * distance)
      den = b + np.sum(e.flatten())
      # print('den=',den)
      if den==0:
        print("JPDA Stopped!!!")
      # den  = den if den!=0 else 1

      betai = e/den 
      beta0 = b/den

      return np.hstack([beta0.flatten() ,betai.flatten()])

    else:
  
      b = (self.SNR * np.sqrt(np.abs(det(2*np.pi*S))) * (1-PD))/PD
      distance = np.sum(((valid_z-Z_hat).T @ inv(S)).T * (valid_z-Z_hat),axis=0) 
      e = np.exp(-0.5 * distance)
      den = b + np.sum(e.flatten())
      if len(den.shape)>1:
        print('b=',b)
        den = (den.flatten())[0]
      
      if den==0:
        print('JPDA stopped!!!')

      betai = e/den 
      beta0 = b/den
      return np.hstack([beta0.flatten() ,betai.flatten()])



  # محاسبه ی مشاهدات موثق برای هر هدف به همراه احتمالات شرطی
  def valid_z_likelihood(self,Z,*targets_characteristics):
    Z = (Z[~np.isnan(Z)]).reshape((Z.shape[0] , -1))
    Z_dict = {k+1:Z[:,[k]]   for k in range(Z.shape[-1])}

    result=[]

    if Z.size==0:
      return [ { 'likelihood':{},"valid_m":{}},]*len(targets_characteristics)


    for target in targets_characteristics:

      S = target['C'] @ target['P'] @ target['C'].T + self.sensor_cov

      distance = {k+1:v      for k,v in enumerate(self.distances(Z,target['Z_hat'],S)) }

      valid_m = {k:v  for k,v in Z_dict.items() if distance[k]<=target['gate']}

      
      if valid_m:
        VM = np.hstack(list(valid_m.values()))
        likelihood=self.BETA(VM ,target['Z_hat'],S,target['gate'],target['PD'])
        l0 = likelihood[0]
        likelihood = {k:v   for k,v in zip( list(valid_m.keys()) , likelihood[1:] )}
        likelihood[0] =l0
      else:
        likelihood ={}
    

      likelihood = dict(sorted(likelihood.items()))

      result.append({ 'valid_m':valid_m,"likelihood": likelihood })
    
    return result

  # تمامی داده‌های مورد نیاز برای مرحله‌ی اصلاح
  def create_data_needed_for_correction(self,Z,*targets_characteristics):

    V_M = self.valid_z_likelihood(Z,*targets_characteristics)

    likelihoods = [i['likelihood']  for i in V_M]

    likelihoods =list((self.MARGINAL_LIKELIHOOD(*likelihoods)).values())

    for i in range(len(V_M)):
      V_M[i]['likelihood'] = likelihoods[i]

    result=[]
    for index in range(len(targets_characteristics)):

      T = {**targets_characteristics[index] ,**V_M[index]}

      result.append(T)
    
    return result
      
  
  def PDA_correction(self,state , P  , C , Z_hat ,valid_m ,likelihood):
    
  

    if not valid_m or not likelihood:
      return state ,P
    else:
      
      valid_m = np.hstack(list(valid_m.values()))
      
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



  def Correct(self,Z,*targets_characteristics):

    data = self.create_data_needed_for_correction(Z,*targets_characteristics)

    for target in data:
      
    
      S,P = self.PDA_correction(target['state'],target['P'],target['C'],
                                  target['Z_hat'],target['valid_m'],target['likelihood'])
      target['state']=S
      target['P']=P

    return data


  


    








  # پیدا کردن اهدافی که مشاهده‌ی مشترک دارند، تا از آن برای محاسبه‌ی احتمال توام استفاده شود
  def find_Subscription(self,*dicts):
    all_dicts = dict(enumerate(dicts))
    
    pack={}
    for k,v in all_dicts.items():
      a={k:v} ; other_dicts= {k2:v2  for k2,v2 in all_dicts.items() if k2!=k }
      a_keys = set(a[k].keys())-{0}

      for k3,v3 in other_dicts.items(): 
        other_keys = set(v3.keys())-{0}
        if len( a_keys & other_keys )>0:
          a[k3]=v3
      pack[k]=a
    return pack        
 
  #  ساخت تمامی فرضیه‌های ممکن برای اهدافی که مشاهدات مشترک دارند
  def marginal_p(self,*dicts):

    if len(dicts)==0:
      return
    elif len(dicts)==1:
      return{str(k1):v1 for k1,v1 in dicts[0].items()}

    elif len(dicts)==2:
      a,b = dicts[0],dicts[-1]
      pack={}
      for k,v in a.items():
        for k2,v2 in b.items():

          if k2==0 or k2!=k:
            pack[str(k)+','+str(k2)] = v*v2
          else:
            continue
      return pack
    else:
      past_d = dicts[:len(dicts)-1]
      past_r = self.marginal_p(*past_d)
      pack={}
      for k,v in past_r.items():
        for k2,v2 in dicts[-1].items():

          keys = k.split(',')
          if k2==0 or str(k2) not in keys:
            pack[k+','+str(k2)] = v*v2
          else:
            continue
      return pack

  # محاسبه‌ی احتمال توام برای یک هدف با استفاده از تابع فوق   
  def MARGINAL_P(self,*dicts):

    new=self.marginal_p(*dicts)
  
    sum_new = np.sum(list(new.values())); sum_new =sum_new if sum_new!=0 else 1

    new = {k:v/sum_new  for k,v in new.items()}
    
    first_target_M=list(set(  [ k.split(',')[0]    for k,v in new.items() ]  ))

    result = {int(k):np.sum([v  for k2,v in new.items() if k2.split(',')[0]==k   ])     for k in first_target_M}
    
    return result

  # تابع اصلی که باید استفاده کرد برای محاسبه‌ی احتمالات توام
  def MARGINAL_LIKELIHOOD(self,*dicts):

    targets = self.find_Subscription(*dicts)

    for k,v in targets.items():

      new = dict(sorted(self.MARGINAL_P(*tuple(v.values())).items()))
      targets[k] = new

    return targets



        
        

        


      

      



















if __name__ =="__main__":

  
  # a={0:0,  134:0.8 ,  2:0.1  ,3:0.05 ,50: 0.05}
  # b={}
  # c={}
  # d={0:0.1,  134:0.2,  2: 0.7 }
  # e={0:0.1,  1: 0.8,  5:0.05,  134:0.05}

  sensor_cov = np.identity(2)*25**2

  C = np.array([[1,0,0,0],[0,0,1,0]])
  jpda=JPDA(sensor_cov,use_constant_p_false_alarm=True)

  Z = np.array([

                [1,3,5,7,9,11,13,15,17,19,21,23,25.5,27,29,31,33,35,102,1500,np.nan,np.nan,],    # X
                [2,4,6,8,10,12,14,16,18,20,22,24,26.3,28,30,32,34,36,136,1600,np.nan,np.nan,]    # Y
                
                
                ])

  target1 ={'state':np.array([[25],[0],[26],[0]]),'P':np.identity(4) , 'Z_hat':np.array([[25],[26]]) , 'C':C , 'gate':50 ,'PD':0.99}

  target2 ={'state':np.array([[28],[5],[29],[3]]),'P':np.identity(4) , 'Z_hat':np.array([[28],[29]]) , 'C':C , 'gate':50 ,'PD':0.99}

  target3 ={'state':np.array([[5333],[0],[6333],[0]]),'P':np.identity(4),'Z_hat':np.array([[5333],[6333]]),'C':C ,'gate':10 ,'PD':0.7}

  target4 ={'state':np.array([[78],[0],[88],[0]]),'P':np.identity(4)*100 , 'Z_hat':np.array([[78],[88]]) , 'C':C , 'gate':10 ,'PD':0.85}
  
  target5 ={'state':np.array([[3.5],[0],[1.75],[0]]),'P':np.identity(4) , 'Z_hat':np.array([[3.5],[4.75]]) , 'C':C , 'gate':10 ,'PD':0.1}


  track1 , track2 = jpda.Correct(Z,target1,target2) 

  print(jpda.Correct(Z,target1,target2))

  
 
   
  
 