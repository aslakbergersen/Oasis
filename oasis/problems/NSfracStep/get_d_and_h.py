import numpy as np

def get_d_and_h(N):
    area = np.loadtxt("a.txt", delimiter=", ")
    ratio = np.loadtxt("ratio.txt", delimiter=", ")

    # 10 000 timesteps per cycle
    t = np.linspace(0, 1.06, N)

    area_new = np.zeros(t.shape[0])
    ratio_new = np.zeros(t.shape[0])

    for i in range(t.shape[0]):
        if t[i] == 1.06 or t[i] < max(area[:,0].min(), ratio[:,0].min()):
            area_new[i] = np.pi * 3.4 * 6.8 / 4
            ratio_new[i] = 2
            continue
        else:
            #from IPython import embed; embed()
            area_x_larger = area[area[:,0] > t[i], 0].min()
            area_x_smaller = area[area[:,0] <= t[i], 0].max()
            ratio_x_larger = ratio[ratio[:,0] > t[i], 0].min()
            ratio_x_smaller = ratio[ratio[:,0] <= t[i], 0].max()

            area_index_smaller = np.where(area[:,0] == area_x_smaller)[0][0]
            area_index_larger = np.where(area[:,0] == area_x_larger)[0][0]
            ratio_index_smaller = np.where(ratio[:,0] == ratio_x_smaller)[0][0]
            ratio_index_larger = np.where(ratio[:,0] == ratio_x_larger)[0][0]

            area_x_dist = t[i] - area_x_smaller
            ratio_x_dist = t[i] - ratio_x_smaller

            a_area = ((area[area_index_larger, 1] - area[area_index_smaller, 1]) /
                        (area_x_larger - area_x_smaller))
            a_ratio = ((ratio[ratio_index_larger, 1] - ratio[ratio_index_smaller, 1]) /
                        (ratio_x_larger - ratio_x_smaller))

            area_extrapolate = area_x_dist * a_area
            ratio_extrapolate = ratio_x_dist * a_ratio

            area_new[i] = area[area_index_smaller, 1] + area_extrapolate
            ratio_new[i] = ratio[ratio_index_smaller, 1] + ratio_extrapolate

    ratio_new[ratio_new > 2] = 2
    area_new[area_new < np.pi * 3.4 * 6.8 / 4] = np.pi * 3.4 * 6.8 / 4

    H = np.sqrt(4 * area_new * ratio_new / np.pi) / H[0]
    D = H / area_new / D[0]

    return D, H
