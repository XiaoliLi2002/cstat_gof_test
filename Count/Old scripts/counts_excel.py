import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
N=300
A=-0.600716
B=-2.66890
C=-2.360850
D=0.514446
E=0.331258
F=1.017396
beta=0.484436
alpha=3.937691
VA=-0.59488
VB=-1.0919
VC=0.85073
Valpha=0.94111
file_path = r'../data/Data_19.5~22.5_all.xlsx'
raw_data = pd.read_excel(file_path)
data = raw_data.values


count = [0 for x in range(300)]
wavelength = [0 for x in range(300)]
i=0
for x in range(300):
    wavelength[i] = 19.5 + 0.01*i
    i=i+1

i=0
for x in data:
    count[int((x-19.5)/0.01)]+=1

mean = np.mean(count)
print(mean)
C_stat = 0

for x in count:
    if x==0:
        C_stat+=2*mean
    else:
        print(x)
        C_stat+= 2*(mean-x*math.log(mean,math.e)-x+x*math.log(x,math.e))
        print(2*(mean-x*math.log(mean,math.e)-x+x*math.log(x,math.e)))
print(C_stat)

E_C_min=N*(F+E*pow(math.e,-beta*mean)+pow(math.e,-alpha*mean)*(A+B*mean+C*pow(mean-D,2)))-1
print(E_C_min)

Var_C_min=2*(N-1)*(1+pow(math.e,-Valpha*mean)*(VA+VB*mean+VC*pow(mean,2)))
print(Var_C_min)
q=1
C_min_crit=E_C_min+q*math.sqrt(Var_C_min)
print(C_min_crit)

plt.figure()
plt.scatter(wavelength,count,c='blue',s=1,label='Source')
plt.xticks(range(20,23,1))
plt.yticks(range(45,90,5))
plt.xlabel("Wavelength",fontdict={'size':16})
plt.ylabel("Counts",fontdict={'size':16})
plt.legend(loc='best')
plt.show()
