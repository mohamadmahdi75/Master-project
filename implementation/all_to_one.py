
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Filters import KALMANFILTER
from Hypotesises import PDA ,GNN
from JPDA_PRO import JPDA
from Trajectory import Trajectory

from scipy.stats import poisson,  norm ,uniform

tr1 = Trajectory('cv','cv')
tr2 = Trajectory("cv",'cv')

B = np.zeros((4,2))
U = np.zeros((2,1))

ground_truth1 = np.array([[0],[1],[20],[-1]])
ground_truth2 = np.array([[0],[1],[0],[+1]])

time = 15

std1 = 0.1
std2 = 0.1

for T in range(time-1):

    ground_truth1 = np.append( ground_truth1 ,tr1(ground_truth1[:,[-1]], B,U,T,'normal' ,0, std1  ),axis=-1)
    ground_truth2 = np.append( ground_truth2 ,tr2(ground_truth2[:,[-1]], B,U,T,'normal' ,0, std2 ),axis=-1)


plt.figure(figsize=(20,8))

plt.plot(ground_truth1[0],ground_truth1[2],'--g',label="ground truth 1")
plt.plot(ground_truth2[0],ground_truth2[2],'--r',label="ground truth 2")

plt.plot(ground_truth1[0][0],ground_truth1[2][0],'ks',)
plt.plot(ground_truth2[0][0],ground_truth2[2][0],'ks',label="start")

plt.plot(ground_truth1[0][-1],ground_truth1[2][-1],'bo',)
plt.plot(ground_truth2[0][-1],ground_truth2[2][-1],'bo',label="Finish")

plt.legend(fontsize=15)
# plt.show()



def sensor(x ,bias ,std_):
    return np.array([ x[0].flatten(), x[2].flatten()]) + np.random.normal(loc= 0 ,scale=std_)


pd1 = 1
pd2 = 1
bias = 0 ;std=0.01
max_number_of_clutters= 10

ALL_measurements=[]
for T in range(time):

    pack={'clutter':np.empty((2,0))}

    if np.random.choice([0 , 1], p=[1-pd1 ,pd1]):

        real1 = sensor(ground_truth1[:,[T]] , bias , std)
        pack['real1'] = real1
    else:

        pack['real1'] = np.nan * np.ones((2,1))

    if np.random.choice([0 , 1], p=[1-pd2 ,pd2]):

        real2 = sensor(ground_truth2[:,[T]] , bias , std)
        pack['real2'] = real2
    else:

        pack['real2'] = np.nan * np.ones((2,1))

    
    # create clutters

    n_clutter = np.random.randint(max_number_of_clutters) if max_number_of_clutters!=0 else 0

    if n_clutter==0:
        pack['clutter'] = np.nan * np.ones((2,2))
        pack['time'] = [T]*2
    else:

        X1 = ground_truth1[ [0] , [T]]
        X2 = ground_truth2[ [0] ,[T]]
        Y1 = ground_truth1[ [2] ,[T]]
        Y2 = ground_truth2[[2] , [T]]

        for _ in range(n_clutter):

            cx1 = uniform.rvs(X1-10 ,20)
            cx2 = uniform.rvs(X2-10 ,20)
            cy1 = uniform.rvs(Y1-10 ,20)
            cy2 = uniform.rvs(Y2-10 ,20)

            clutter1 = np.array([[cx1],[cy1]])
            clutter2 = np.array([[cx2],[cy2]])

            pack['clutter'] = np.append(pack['clutter'] ,np.hstack([clutter1 ,clutter2]) ,axis=-1)

        pack['time'] = [T] * n_clutter*2
        
    ALL_measurements.append(pack)


ALL_measurements = pd.DataFrame(ALL_measurements )
# ALL_measurements

real1 =np.hstack(ALL_measurements['real1'])
real2 =np.hstack(ALL_measurements['real2'])
clutters = np.hstack(ALL_measurements['clutter'])
time_for_clutter = np.hstack(ALL_measurements['time'])

plt.figure(figsize= (20,10))
plt.plot(ground_truth1[0],ground_truth1[2],'g--',label ='ground truth1')
plt.plot(ground_truth2[0],ground_truth2[2],'r--',label='ground truth2')

plt.plot(ground_truth1[0,[0]],ground_truth1[2,[0]],'cs',label='start')
plt.plot(ground_truth2[0,[0]],ground_truth2[2,[0]],'cs')
plt.plot(ground_truth1[0][-1],ground_truth1[2][-1],'b*',label='finish')
plt.plot(ground_truth2[0][-1],ground_truth2[2][-1],'b*')

plt.plot(real1[0],real1[1],'ko',label='sensor')
plt.plot(real2[0],real2[1],'ko')
plt.plot(clutters[0],clutters[1],'^y',label='clutter')



plt.legend(fontsize=15)
# plt.show()




p_false_alarm = 0.01 ; 
sensor_cov = np.identity(2) * 0.01**2
jpda = JPDA(p_false_alarm , sensor_cov)


S1 = ground_truth1[:,[0]]
S2 = ground_truth2[:,[0]]
P1 = np.identity(4)
P2 = np.identity(4)

state_cov1 = np.diag([1.5,0.5,1.5,0.5])
state_cov2 = np.diag([1.5,0.5,1.5,0.5])

B = np.zeros((4,2)) ;  U  = np.zeros((2,1)) ; C =np.array([[1,0,0,0],[0,0,1,0]])

kalman1 = KALMANFILTER(tr1 ,B, C ,state_cov1 , sensor_cov)
kalman2 = KALMANFILTER(tr2 ,B, C ,state_cov2 , sensor_cov)
#        گیتهایی که درست بودن 
gate1 = 1  # 5 , 25 ,24 ,20 ,18
gate2 = 0.7  # 8  , 5  ,5  ,5  ,5

track1=[] ; track2=[] ; weight1 =[] ;weight2=[] ; valid1=[] ; valid2=[] ; time_for_valid1=[] ; time_for_valid2=[]; 



for T in range(time):

    S1 , P1 = kalman1.predict(S1,P1,U,T)

    S2 , P2 = kalman2.predict(S2,P2,U,T)

    target1 = {'state':S1 ,'P':P1, 'Z_hat': C@S1 ,'C':C, 'gate':gate1 ,'PD':pd1 }
    target2 = {'state':S2 ,'P':P2, 'Z_hat': C@S2 ,'C':C, 'gate':gate2 ,'PD':pd2 }

    Z = np.hstack(ALL_measurements[['real1','real2','clutter']].loc[T])

    t1,t2 = jpda.Correct(Z,*(target1 ,target2))

    S1,P1 = t1['state'] ,t1['P']
    S2,P2 = t2['state'] ,t2['P']
    
    weight1.append(list(t1['likelihood'].values()) if t1['likelihood'] else np.nan  ) ; 
    weight2.append(list(t2['likelihood'].values()) if t2['likelihood'] else np.nan  ); 

    valid1.append((t1['valid_m']))  ; valid2.append((t2['valid_m'])) ; 

    time_for_valid1.extend([T] *  valid1[-1].shape[-1])
    time_for_valid2.extend( [T] * valid2[-1].shape[-1])

    track1.append(S1) ;track2.append(S2)




track1 = np.hstack(track1)
track2 = np.hstack(track2)

weight1 = np.hstack(weight1) ; weight2 = np.hstack(weight2); 
valid1 = np.hstack(valid1)  ; valid2 =np.hstack(valid2)

real1 =np.hstack(ALL_measurements['real1'])
real2 =np.hstack(ALL_measurements['real2'])
clutters = np.hstack(ALL_measurements['clutter'])
time_for_clutter = np.hstack(ALL_measurements['time'])

plt.figure(figsize= (20,10))
plt.plot(ground_truth1[0],ground_truth1[2],'g--',label ='ground truth1')
plt.plot(ground_truth2[0],ground_truth2[2],'r--',label='ground truth2')

plt.plot(ground_truth1[0,[0]],ground_truth1[2,[0]],'cs',label='start')
plt.plot(ground_truth2[0,[0]],ground_truth2[2,[0]],'cs')
plt.plot(ground_truth1[0][-1],ground_truth1[2][-1],'b*',label='finish')
plt.plot(ground_truth2[0][-1],ground_truth2[2][-1],'b*')

plt.plot(real1[0],real1[1],'ko',label='sensor')
plt.plot(real2[0],real2[1],'ko')
plt.plot(clutters[0],clutters[1],'^y',label='clutter')

plt.plot(track1[0],track1[2],'g',label='track1')
plt.plot(track2[0],track2[2],'r',label='track2')

plt.scatter(valid1[0],valid1[1],s=np.where(weight1>0.6 ,weight1*900 , weight1*900),color='g',label='valid measurements track 1')
plt.scatter(valid2[0],valid2[1],s=np.where(weight2>0.6 ,weight2*900 , weight2*900),color='r',label='valid measurements track 2')
plt.title('JPDA Algorithm',fontdict={'fontsize':25})
plt.legend(fontsize=15)
plt.xlabel('X',fontdict={'fontsize':19})
plt.ylabel('Y',fontdict={"fontsize":19})
# plt.show()




fig ,ax = plt.subplots(4,1,figsize=(30,50))

ax[0].set_title("Pos X",fontsize=20)
ax[0].plot(ground_truth1[0],'g--',label='ground truth1')
ax[0].plot(ground_truth2[0],'r--',label='ground truth2')
ax[0].plot(time_for_clutter ,clutters[0],'y^',label='clutter')
ax[0].plot(track1[0],'g',label='track1')
ax[0].plot(track2[0],'r',label='track2')
ax[0].plot(real1[0],'ko',label='sensor')
ax[0].plot(real2[0],'ko')
ax[0].scatter(time_for_valid1,valid1[0],s=np.where(weight1>0.6 ,weight1*600 , weight1*400),color='g',label='valid measurements track 1')
ax[0].scatter(time_for_valid2,valid2[0],s=np.where(weight2>0.6 ,weight2*600 , weight2*400),color='r',label='valid measurements track 2')
ax[0].legend(fontsize=5)

ax[1].set_title("Pos Y",fontsize=20)
ax[1].plot(ground_truth1[2],'g--',label='ground truth1')
ax[1].plot(ground_truth2[2],'r--',label='ground truth2')
ax[1].plot(time_for_clutter ,clutters[1],'y^',label='clutter')
ax[1].plot(track1[2],'g',label='track1')
ax[1].plot(track2[2],'r',label='track2')
ax[1].plot(real1[1],'ko',label='sensor')
ax[1].plot(real2[1],'ko')
ax[1].scatter(time_for_valid1,valid1[1],s=np.where(weight1>0.6 ,weight1*600 , weight1*400),color='g',label='valid measurements track 1')
ax[1].scatter(time_for_valid2,valid2[1],s=np.where(weight2>0.6 ,weight2*600 , weight2*400),color='r',label='valid measurements track 2')
ax[1].legend(fontsize=5)

ax[2].set_title("Vel X",fontsize=20)
ax[2].plot(ground_truth1[1],'g--',label='ground truth1')
ax[2].plot(ground_truth2[1],'r--',label='ground truth2')
ax[2].plot(track1[1],'g',label='track1')
ax[2].plot(track2[1],'r',label='track2')
ax[2].legend(fontsize=5)

ax[3].set_title("Vel Y",fontsize=20)
ax[3].plot(ground_truth1[-1],'g--',label='ground truth1')
ax[3].plot(ground_truth2[-1],'r--',label='ground truth2')
ax[3].plot(track1[-1],'g',label='track1')
ax[3].plot(track2[-1],'r',label='track2')
ax[3].legend(fontsize=5)
plt.tight_layout()



#    PDA______________
#    _________________


pda1 = PDA(gate1 , state_cov1 ,sensor_cov, pd1 , p_false_alarm)
pda2 = PDA(gate2 , state_cov2 ,sensor_cov, pd2 , p_false_alarm)

S1 = ground_truth1[:,[0]]
S2 = ground_truth2[:,[0]]
P1 = np.identity(4)
P2 = np.identity(4)

state_cov1 = np.diag([1.5,0.5,1.5,0.5])
state_cov2 = np.diag([1.5,0.5,1.5,0.5])

B = np.zeros((4,2)) ;  U  = np.zeros((2,1)) ; C =np.array([[1,0,0,0],[0,0,1,0]])

kalman1 = KALMANFILTER(tr1 ,B, C ,state_cov1 , sensor_cov)
kalman2 = KALMANFILTER(tr2 ,B, C ,state_cov2 , sensor_cov)


track1=[] ; track2=[] ; weight1 =[] ;weight2=[] ; valid1=[] ; valid2=[] ; time_for_valid1=[] ; time_for_valid2=[]; 

for T in range(time):

    S1 , P1 = kalman1.predict(S1,P1,U,T)

    S2 , P2 = kalman2.predict(S2,P2,U,T)

    # target1 = {'state':S1 ,'P':P1, 'Z_hat': C@S1 ,'C':C, 'gate':gate1 ,'PD':pd1 }
    # target2 = {'state':S2 ,'P':P2, 'Z_hat': C@S2 ,'C':C, 'gate':gate2 ,'PD':pd2 }

    Z = np.hstack(ALL_measurements[['real1','real2','clutter']].loc[T])

    S1, P1 , v1 ,b1 = pda1.Correcet(S1,P1,C,Z, C @ S1)
    S2, P2 , v2, b2 = pda2.Correcet(S2,P2,C,Z, C @ S2)
    
    weight1.append( b1 ) ; 
    weight2.append( b2 ); 

    valid1.append( v1 )  ; valid2.append(v2) ; 

    time_for_valid1.extend([T] *  valid1[-1].shape[-1])
    time_for_valid2.extend( [T] * valid2[-1].shape[-1])

    track1.append(S1) ;track2.append(S2)

track1 = np.hstack(track1)
track2 = np.hstack(track2)

weight1 = np.hstack(weight1) ; weight2 = np.hstack(weight2); 
valid1 = np.hstack(valid1)  ; valid2 =np.hstack(valid2)


real1 =np.hstack(ALL_measurements['real1'])
real2 =np.hstack(ALL_measurements['real2'])
clutters = np.hstack(ALL_measurements['clutter'])
time_for_clutter = np.hstack(ALL_measurements['time'])

plt.figure(figsize= (20,10))
plt.plot(ground_truth1[0],ground_truth1[2],'g--',label ='ground truth1')
plt.plot(ground_truth2[0],ground_truth2[2],'r--',label='ground truth2')

plt.plot(ground_truth1[0,[0]],ground_truth1[2,[0]],'cs',label='start')
plt.plot(ground_truth2[0,[0]],ground_truth2[2,[0]],'cs')
plt.plot(ground_truth1[0][-1],ground_truth1[2][-1],'b*',label='finish')
plt.plot(ground_truth2[0][-1],ground_truth2[2][-1],'b*')

plt.plot(real1[0],real1[1],'ko',label='sensor')
plt.plot(real2[0],real2[1],'ko')
plt.plot(clutters[0],clutters[1],'^y',label='clutter')

plt.plot(track1[0],track1[2],'g',label='track1')
plt.plot(track2[0],track2[2],'r',label='track2')

plt.scatter(valid1[0],valid1[1],s=np.where(weight1>0.6 ,weight1*900 , weight1*900),color='g',label='valid measurements track 1')
plt.scatter(valid2[0],valid2[1],s=np.where(weight2>0.6 ,weight2*900 , weight2*900),color='r',label='valid measurements track 2')
plt.title("PDA Algorithm",fontdict={'fontsize':25})
plt.legend(fontsize=15)
plt.xlabel('X',fontdict={'fontsize':19})
plt.ylabel('Y',fontdict={"fontsize":19})
# plt.show()


fig ,ax = plt.subplots(4,1,figsize=(30,50))

ax[0].set_title("Pos X PDA",fontsize=20)
ax[0].plot(ground_truth1[0],'g--',label='ground truth1')
ax[0].plot(ground_truth2[0],'r--',label='ground truth2')
ax[0].plot(time_for_clutter ,clutters[0],'y^',label='clutter')
ax[0].plot(track1[0],'g',label='track1')
ax[0].plot(track2[0],'r',label='track2')
ax[0].plot(real1[0],'ko',label='sensor')
ax[0].plot(real2[0],'ko')
ax[0].scatter(time_for_valid1,valid1[0],s=np.where(weight1>0.6 ,weight1*600 , weight1*400),color='g',label='valid measurements track 1')
ax[0].scatter(time_for_valid2,valid2[0],s=np.where(weight2>0.6 ,weight2*600 , weight2*400),color='r',label='valid measurements track 2')
ax[0].legend(fontsize=5)

ax[1].set_title("Pos Y PDA",fontsize=20)
ax[1].plot(ground_truth1[2],'g--',label='ground truth1')
ax[1].plot(ground_truth2[2],'r--',label='ground truth2')
ax[1].plot(time_for_clutter ,clutters[1],'y^',label='clutter')
ax[1].plot(track1[2],'g',label='track1')
ax[1].plot(track2[2],'r',label='track2')
ax[1].plot(real1[1],'ko',label='sensor')
ax[1].plot(real2[1],'ko')
ax[1].scatter(time_for_valid1,valid1[1],s=np.where(weight1>0.6 ,weight1*600 , weight1*400),color='g',label='valid measurements track 1')
ax[1].scatter(time_for_valid2,valid2[1],s=np.where(weight2>0.6 ,weight2*600 , weight2*400),color='r',label='valid measurements track 2')
ax[1].legend(fontsize=5)

ax[2].set_title("Vel X PDA",fontsize=20)
ax[2].plot(ground_truth1[1],'g--',label='ground truth1')
ax[2].plot(ground_truth2[1],'r--',label='ground truth2')
ax[2].plot(track1[1],'g',label='track1')
ax[2].plot(track2[1],'r',label='track2')
ax[2].legend(fontsize=5)

ax[3].set_title("Vel Y PDA",fontsize=20)
ax[3].plot(ground_truth1[-1],'g--',label='ground truth1')
ax[3].plot(ground_truth2[-1],'r--',label='ground truth2')
ax[3].plot(track1[-1],'g',label='track1')
ax[3].plot(track2[-1],'r',label='track2')
ax[3].legend(fontsize=5)
plt.tight_layout()

plt.show()