
# How to save the file
'''
import numpy as np
x = np.arange(0.0,5.0,1.0)
np.save('daro.out', x)'''

import numpy as np
x, y, z = np.genfromtxt('file.txt',
                    unpack=True)
