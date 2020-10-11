

import libmr

import numpy as np
import matplotlib.pyplot as plt

mr = libmr.MR()
data = np.random.randn(100)
xs = np.linspace(-5,5, 100)

# Plot two plots
fig,(ax1,ax2) = plt.subplots(2,1)
ax2.set_title("Fitting high")
ax2.hist(data,bins=20,density=True)
for tailsize in [10,30,50]:
    mr.fit_high(data, tailsize)
    assert mr.is_valid
    print("scale shape sign translate score  " , mr.get_params());
    # print("scale lb up shape lb up  " , mr.get_confidene());
    ax2.plot(xs, mr.w_score_vector(xs), label="Tailsize: %d"%tailsize)
ax2.legend()

plt.tight_layout()
plt.show()
# for tailsize in [10]:
# 	mr.fit_high(data,tailsize)

# 	print ("Scale shape sign translate score ", mr.get_params())
# 	print ("Scale lb up shape lb up ", mr.get_confidence())