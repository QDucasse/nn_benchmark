
import matplotlib.pyplot as plt
import numpy as np

results = "/home/quentin/Desktop/GitProjects/AI/finn/trained_models/results.txt"

x = np.arange(2,9)
y = np.arange(2,9)

epochs01 = np.zeros((7,7))
epochs10 = np.zeros((7,7))
epochs20 = np.zeros((7,7))
epochs30 = np.zeros((7,7))
epochs39 = np.zeros((7,7))
with open(results, "r") as f:
    content = f.readlines()

    for line  in content:
        acq = int(line[10]) - 2 # map 2-8 to 0-6 for indexes
        weq = int(line[12]) - 2
        acc = 100 - float(line[-7:])
        if "Epoch-1 " in line:
            epochs01[acq][weq] = acc
        elif "Epoch-10" in line:
            epochs10[acq][weq] = acc
        elif "Epoch-20" in line:
            epochs20[acq][weq] = acc
        elif "Epoch-30" in line:
            epochs30[acq][weq] = acc
        elif "Epoch-39" in line:
            epochs39[acq][weq] = acc

print(epochs01)
print(epochs10)
print(epochs20)
print(epochs30)
print(epochs39)

X, Y = np.meshgrid(x,y)
Z_e01 = epochs01[X-2,Y-2]
Z_e10 = epochs10[X-2,Y-2]
Z_e20 = epochs20[X-2,Y-2]
Z_e30 = epochs30[X-2,Y-2]
Z_e39 = epochs39[X-2,Y-2]


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_ylim(8,2)
# ax.plot_surface(X, Y, Z_e01, rstride=1, cstride=1,
#                 edgecolor='none')

ax.plot_surface(X, Y, Z_e10, rstride=1, cstride=1,
                edgecolor='none')

ax.plot_surface(X, Y, Z_e20, rstride=1, cstride=1,
                edgecolor='none')

ax.plot_surface(X, Y, Z_e30, rstride=1, cstride=1,
                edgecolor='none')

ax.plot_surface(X, Y, Z_e39, rstride=1, cstride=1,
                edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()
