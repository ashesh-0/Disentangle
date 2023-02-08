import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
from tifffile import imsave, imread



def open_img(dir):
    array = np.loadtxt(dir)
    return array

def G_0(k,l,m,n):

    if k == m and n == l + 1:
        x = np.mean(np.subtract(p1[k * patch_size: (k + 1) * patch_size, (l + 1) * patch_size - 2], p1[k * patch_size: (k + 1) * patch_size, (l + 1) * patch_size - 1]))
        y = np.mean(np.subtract(p1[k * patch_size: (k + 1) * patch_size, (l + 1) * patch_size], p1[k * patch_size:(k + 1) * patch_size, (l + 1) * patch_size + 1]))
        z = np.mean(np.subtract(p1[k * patch_size:(k + 1) * patch_size, (l + 1) * patch_size - 1], p1[k * patch_size:(k + 1) * patch_size, (l + 1) * patch_size]))

    elif l == n and m == k + 1:
        x = np.mean(np.subtract(p1[(k + 1) * patch_size - 2, l * patch_size: (l + 1) * patch_size], p1[(k + 1) * patch_size - 1, l * patch_size: (l + 1) * patch_size]))
        y = np.mean(np.subtract(p1[(k + 1) * patch_size, l * patch_size: (l + 1) * patch_size], p1[(k + 1) * patch_size + 1, l * patch_size:(l + 1) * patch_size]))
        z = np.mean(np.subtract(p1[(k + 1) * patch_size - 1, l * patch_size:(l + 1) * patch_size], p1[(k + 1) * patch_size, l * patch_size:(l + 1) * patch_size]))
    return x, y, z

def G_1(k,l,m,n):

    if k == m and n == l + 1:
        x = np.mean(np.subtract(p2[k * patch_size: (k + 1) * patch_size, (l + 1) * patch_size - 2], p2[k * patch_size: (k + 1) * patch_size, (l + 1) * patch_size - 1]))
        y = np.mean(np.subtract(p2[k * patch_size: (k + 1) * patch_size, (l + 1) * patch_size], p2[k * patch_size:(k + 1) * patch_size, (l + 1) * patch_size + 1]))
        z = np.mean(np.subtract(p2[k * patch_size:(k + 1) * patch_size, (l + 1) * patch_size - 1], p2[k * patch_size:(k + 1) * patch_size, (l + 1) * patch_size]))

    elif l == n and m == k + 1:
        x = np.mean(np.subtract(p2[(k + 1) * patch_size - 2, l * patch_size: (l + 1) * patch_size], p2[(k + 1) * patch_size - 1, l * patch_size: (l + 1) * patch_size]))
        y = np.mean(np.subtract(p2[(k + 1) * patch_size, l * patch_size: (l + 1) * patch_size], p2[(k + 1) * patch_size + 1, l * patch_size:(l + 1) * patch_size]))
        z = np.mean(np.subtract(p2[(k + 1) * patch_size - 1, l * patch_size:(l + 1) * patch_size], p2[(k + 1) * patch_size, l * patch_size:(l + 1) * patch_size]))
    return x, y, z

def grbi(x,y):

    tiling_model = gp.Model('Tiling')
    p1 = x
    p2 = y
    Cells_0 = {}
    Cells_1 = {}


    h = len(p1)
    w = len(p1[0])
    hh = int(h / patch_size)
    ww = int(w / patch_size)

    for k in range(hh):
        for l in range(ww):
            Cells_0[k, l] = tiling_model.addVar(vtype=GRB.CONTINUOUS)
            Cells_1[k, l] = tiling_model.addVar(vtype=GRB.CONTINUOUS)

    tiling_model.update()
    for k in range(hh):
        for l in range(ww):
            tiling_model.addConstr(Cells_0[k, l] + Cells_1[k, l] == 0)

            if l != ww - 1:
                (x, y, z) = G_0(k, l, k, l + 1)
                if y >= x:
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k, l + 1] <= y - z)
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k, l + 1] >= x - z)
                elif x >= y:
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k, l + 1] >= y - z)
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k, l + 1] <= x - z)
            if k != hh - 1:
                x, y, z = G_0(k, l, k + 1, l)
                if y >= x:
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k + 1, l] <= y - z)
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k + 1, l] >= x - z)
                elif x >= y:
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k + 1, l] >= y - z)
                    tiling_model.addConstr(Cells_0[k, l] - Cells_0[k + 1, l] <= x - z)

            if l != ww - 1:
                (x, y, z) = G_1(k, l, k, l + 1)
                if y >= x:
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k, l + 1] <= y - z)
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k, l + 1] >= x - z)
                elif x >= y:
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k, l + 1] >= y - z)
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k, l + 1] <= x - z)
            if k != hh - 1:
                x, y, z = G_1(k, l, k + 1, l)
                if y >= x:
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k + 1, l] <= y - z)
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k + 1, l] >= x - z)
                elif x >= y:
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k + 1, l] >= y - z)
                    tiling_model.addConstr(Cells_1[k, l] - Cells_1[k + 1, l] <= x - z)

    tiling_model.setObjective( \
        quicksum(
            ((G_0(i, j, k, l)[0] + G_0(i, j, k, l)[1]) / 2 - G_0(i, j, k, l)[2] - Cells_0[i, j] + Cells_0[k, l]) \
            for (i, j) in Cells_0 for (k, l) in Cells_0 if i == k and j + 1 == l) \
        + quicksum(
            ((G_0(i, j, k, l)[0] + G_0(i, j, k, l)[1]) / 2 - G_0(i, j, k, l)[2] - Cells_0[i, j] + Cells_0[k, l]) \
            for (i, j) in Cells_0 for (k, l) in Cells_0 if i + 1 == k and j == l) \
        + quicksum(
            ((G_1(i, j, k, l)[0] + G_1(i, j, k, l)[1]) / 2 - G_1(i, j, k, l)[2] - Cells_1[i, j] + Cells_1[k, l]) \
            for (i, j) in Cells_1 for (k, l) in Cells_1 if i == k and j + 1 == l) \
        + quicksum(
            ((G_1(i, j, k, l)[0] + G_1(i, j, k, l)[1]) / 2 - G_1(i, j, k, l)[2] - Cells_1[i, j] + Cells_1[k, l]) \
            for (i, j) in Cells_1 for (k, l) in Cells_1 if i + 1 == k and j == l), GRB.MINIMIZE)


    tiling_model.optimize()

    if tiling_model.status == GRB.INFEASIBLE:
        tiling_model.Params.DualReductions = 0
        tiling_model.reset()
        tiling_model.optimize()
    elif tiling_model.status != GRB.OPTIMAL:

        tiling_model.computeIIS()

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(p1)
    ax[0, 0].set_title('Channel I - before')
    ax[1, 0].imshow(p2)
    ax[1, 0].set_title('Channel II - before')

    count = 0
    for k in range(hh):
        for l in range(ww):
            if Cells_1[k, l].x != 0:
                count += 1
            p1[k * patch_size: (k + 1) * patch_size, l * patch_size: (l + 1) * patch_size] += Cells_0[k, l].x
            p2[k * patch_size: (k + 1) * patch_size, l * patch_size: (l + 1) * patch_size] += Cells_1[k, l].x

    print(count, " Patches have nonzero offset")
    ax[0, 1].imshow(p1)
    ax[0, 1].set_title('Channel I - after')
    ax[1, 1].imshow(p2)
    ax[1, 1].set_title('Channel II - after')

    plt.show()
    plt.close()

    tiling_model.reset(0)
    return p1,p2


if __name__ == '__main__':

    # p1 = open_img('./Ashesh/small_pred0_0-0-400.npy')
    # p2 = open_img('./Ashesh/small_pred1_0-0-400.npy')

    img = imread("/group/jug/ashesh/sheida_data/prediction_baseline_16grid.tif")
    img = np.swapaxes(img, 1, 3)

    patch_size = 16

    for i in range(len(img)):
        p1 = img[i][0].astype('float64')
        p2 = img[i][1].astype('float64')
        # imsave('Channel_I_before'+str(i)+'.tif', p1)
        # imsave('Channel_II_before'+str(i)+'.tif', p2)
        p1, p2 = grbi(p1,p2)
        # imsave('Channel_I_after'+str(i)+'.tif', p1)
        # imsave('Channel_II_after'+str(i)+'.tif', p2)