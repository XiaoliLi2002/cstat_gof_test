import numpy as np
import pandas as pd
import seaborn as sns

n=100
uncondata=np.mat(pd.read_excel("power_model_C.xlsx",sheet_name='uncondata',header=None))
condata=np.mat(pd.read_excel("power_model_C.xlsx",sheet_name='condata',header=None))
power_improved=(uncondata-condata)/uncondata

sqrtalpha=[0.1*(i+1) for i in range(100)]
ns=[5,10,15,25,35,50,75,100]
xticks=['' for i in range(100)]
xticks[0]=0.1
xticks[24]=2.5
xticks[49]=5
xticks[74]=7.5
xticks[99]=10

ax=sns.heatmap(power_improved,xticklabels=xticks,yticklabels=ns)
ax.set_title('Improved Power Percentage')
ax.set_xlabel('mean')
ax.set_ylabel('n')
figure=ax.get_figure()
figure.savefig('heatmap.png')

