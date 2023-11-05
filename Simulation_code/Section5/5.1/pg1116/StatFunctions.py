import numpy as np
# ymodel: model
# ydata: data
def cstat(ydata,ymodel):
    N=len(ymodel)
    cstat=np.zeros(N)
    for i in range(N):
        if(ydata[i]>0):
            cstat[i]=2.*(ymodel[i]-ydata[i]-ydata[i]*np.log(ymodel[i]/ydata[i]))
        if(ydata[i]==0):
            cstat[i]=2.*ymodel[i]
    return cstat # this is cstat for each bin

