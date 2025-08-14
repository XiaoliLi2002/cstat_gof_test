import numpy as np

from matplotlib import pyplot as plt

from sherpa.astro import io
from sherpa.astro import instrument
from sherpa.astro.plot import ARFPlot
from sherpa.utils.testing import get_datadir

################### ARF ############################

arf = io.read_arf("pg1116-newobs_new_leg_abs1.arf")
print(arf)
arf
# prepare the arf plot object
aplot=ARFPlot()
aplot.prepare(arf)
print(aplot)
# this plot doesn't seem to work b/c of missing backend ?!
aplot.plot()

# usual way to plot
fig,ax=plt.subplots(figsize=(8,6))
plt.plot(aplot.xlo,aplot.y)
plt.savefig('arf.pdf')

#quit()
############### RMF ##############################

rmf = io.read_rmf("pg1116-newobs_new_leg_abs1.rmf")
print(rmf.matrix.shape)
print(rmf.matrix)

## Access the matrix information

matinfo = instrument.rmf_to_matrix(rmf)
print(matinfo.matrix.shape)
indx=200
row=matinfo.matrix[:,indx]
print('length of row %d: %d, sum of prob.: %.4f'%(indx,len(row),sum(row)))
