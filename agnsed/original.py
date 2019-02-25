from xspec import *
import numpy as np
import matplotlib.pyplot as plt

m = Model("qsosed", setPars=(1.e8, 100., -0.3, 0., 0.5, 0., 0.))

AllData.dummyrsp(lowE=0.001, highE=100)
    
Plot.device="/null"
Plot("model")
chans = np.array(Plot.x())
model1 = np.array(Plot.model())
print(model1)

plt.loglog(chans, model1*chans**2)


plt.xlabel("keV")
plt.ylabel("counts/cm^2/s")
plt.legend()
plt.show()
