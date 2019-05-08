import numpy as np
from findfundmat import findfundmat
from prewarp import find_prewarp

image1 = 'einstein1.jpg'
image2 = 'einstein3.jpg'

F = findfundmat(image1, image2)
print('F =', F)

[H1, H2] = find_prewarp(F)
print('H1 =', H1)
print('H2 =', H2)
