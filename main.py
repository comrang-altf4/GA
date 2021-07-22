import cv2
import random
import imutils
from scipy import spatial
from scipy.spatial import ConvexHull
import scipy.sparse
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from itertools import cycle
import numpy as np
import math
from ConcaveHull import ConcaveHull

n = 3  # size of individual (chromosome)
m = 200 # size of population
d_max = 250
a_max = 0
n_generations = 5  # number of generations
order = 0
local_max = [-1] * m
order_of_individual = [-1] * m
fitnesses = []


def cal_dist(a, b):
    angle = min(abs(a[2] - b[2]), 360 - abs(a[2] - b[2]))
    dist = math.sqrt(pow(abs(a[0] - b[0]), 2) + pow(abs(a[1] - b[1]), 2))
    # if angle > a_max or dist > d_max:
    #     return 10000
    angle=0
    if dist > d_max:
        return 10000
    return angle + dist


def crop_img(individual2, rotated, mode=0, k=0, l=0):
    individual=[0,0]
    individual[0]=int(individual2[0])
    individual[1]=int(individual2[1])
    h = int(rotated.shape[0] / 2)
    w = int(rotated.shape[1] / 2)
    t_h = 0
    t_w = 0
    if rotated.shape[0] % 2 == 1:
        t_h = 1
    if rotated.shape[1] % 2 == 1:
        t_w = 1
    crop = src2[individual[0] - h:individual[0] + h + t_h, individual[1] - w:individual[1] + w + t_w]
    if mode == 1:
        cv2.imwrite("{}clustercrop{}.jpg".format(k, l), crop)
    hei = h
    wid = w
    if mode == 1:
        return crop, wid, hei
    return crop


def compute_fitness(individual):
    rotated = imutils.rotate_bound(image, individual[2])
    ret, rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
    crop = crop_img(individual, rotated)
    ret, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
    sA = scipy.sparse.csr_matrix(crop.flatten()).toarray()
    sB = scipy.sparse.csr_matrix(rotated.flatten()).toarray()
    if 255 not in sA[0]:
        ttt = 1
    else:
        ttt = spatial.distance.cosine(sA, sB)
    individual[3] = (1 - ttt)
    return 1 - ttt


def share_function(d):
    if (d > d_max + a_max):
        return 0
    return 1 - pow((d / (d_max + a_max)), 1)


def compute_share_fitness(index, population):
    individual = population[index]
    m = 0
    for i in population:
        t = share_function(cal_dist(individual, i))
        m += t
    individual[4] =individual[3]
    individual[3] /= m
    return individual[3]

def create_individual():
    return ([random.randint(int(h_max / 2), src.shape[0] - int(h_max)),
             random.randint(int(w_max / 2), src.shape[1] - int(w_max)), random.randint(0, 360), 0, -1])


def create_color():
    return ([random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)])


def crossover(individual1, individual2, crossover_rate=0.6):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(n):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]

    return individual1_new, individual2_new


def mutate(individual, mutation_rate=0.001):
    if geni - n_generations > 0:
        mutation_rate = 0.1 * int((geni - n_generations) / 10 + 1)
    individual_m = individual.copy()

    for i in range(n):
        temp = create_individual()
        if random.random() < mutation_rate:
            if i == 0:
                individual_m[i] = temp[0]
            elif i == 1:
                individual_m[i] = temp[1]
            else:
                individual_m[i] = temp[2]
    return individual_m


def selection(sorted_old_population):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if (index2 != index1):
            break
    return sorted_old_population[index1] if sorted_old_population[index1][3] > sorted_old_population[index2][3] else \
        sorted_old_population[index2]


def remove(crop, rotated):
    h = int(rotated.shape[0] / 2)
    w = int(rotated.shape[1] / 2)
    t_h = 0
    t_w = 0
    if rotated.shape[0] % 2 == 1:
        t_h = 1
    if rotated.shape[1] % 2 == 1:
        t_w = 1
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            if crop[i, j] == rotated[i, j] and crop[i, j] != 0:
                crop[i, j] = 0
            elif crop[i, j] != rotated[i, j] and crop[i, j] == 1:
                crop[i, j] = 1
    src2[individual[0] - h:individual[0] + h + t_h, individual[1] - w:individual[1] + w + t_w] = crop
    return 0


def bfs(x, y, mode):
    bfs_coord = [[x, y]]
    l = 0
    bfs_bool_arr = [[True for i in range(src2.shape[1])] for j in range(src2.shape[0])]
    temp = src2[x, y]
    bfs_bool_arr[x][y] = False
    sum_pix = 0
    if temp != 0 and mode == 0:
        return x, y
    while l <= len(bfs_coord) - 1:
        x, y = bfs_coord[l]
        aaa = [6, 12]
        if mode==0:
            aaa=[0]
        # aaa=[0]
        for skamtt in aaa:
            skamt = skamtt + 1
            if x + skamt in range(src2.shape[0]) and bfs_bool_arr[x + skamt][y] == True:
                if src2[x + skamt, y] != 0:
                    if mode == 0:
                        bfs_coord.append([x + skamt, y])
                        sum_pix += 1
                        return x+skamt,y
                    else:
                        src2[x + skamt, y] = 0
                        bfs_coord.append([x + skamt, y])
                else:
                    if mode == 0:
                        bfs_coord.append([x + skamt, y])
                bfs_bool_arr[x + skamt][y] = False
            if x - skamt in range(src2.shape[0]) and bfs_bool_arr[x - skamt][y] == True:
                if src2[x - skamt, y] != 0:
                    if mode == 0:
                        bfs_coord.append([x - skamt, y])
                        sum_pix += 1
                        return x - skamt, y
                    else:
                        src2[x - skamt, y] = 0
                        bfs_coord.append([x - skamt, y])
                else:
                    if mode == 0:
                        bfs_coord.append([x - skamt, y])
                bfs_bool_arr[x - skamt][y] = False
            if y + skamt in range(src2.shape[1]) and bfs_bool_arr[x][y + skamt] == True:
                if src2[x, y + 1] != 0:
                    if mode == 0:
                        bfs_coord.append([x, y + skamt])
                        sum_pix += 1
                        return x, y+skamt
                    else:
                        src2[x, y + 1] = 0
                        bfs_coord.append([x, y + skamt])
                else:
                    if mode == 0:
                        bfs_coord.append([x, y + skamt])
                bfs_bool_arr[x][y + skamt] = False
            if y - skamt in range(src2.shape[1]) and bfs_bool_arr[x][y - skamt] == True:
                if src2[x, y - skamt] != 0:
                    if mode == 0:
                        bfs_coord.append([x, y - skamt])
                        sum_pix += 1
                        return x, y-skamt
                    else:
                        src2[x, y - skamt] = 0
                        bfs_coord.append([x, y - skamt])
                else:
                    if mode == 0:
                        bfs_coord.append([x, y - skamt])
                bfs_bool_arr[x][y - skamt] = False
            if mode == 1 and len(bfs_coord) >= 37887:
                return 0
            if mode == 0 and sum_pix >= 50:
                return bfs_coord[-1]
        l += 1


def remove2(individual2, rotated):
    individual = [0, 0, 0]
    individual[0] = int(individual2[0])
    individual[1] = int(individual2[1])
    individual[2] = int(individual2[2])
    x_t = individual[0]
    y_t = individual[1]
    x_t, y_t = bfs(x_t, y_t, 0)
    bfs(x_t, y_t, 1)
    #cv2.rectangle(src2, (y_t - 10, x_t - 10), (y_t + 10, x_t + 10), (200, 0, 0), 3)
    #cv2.circle(src2, (y_t, x_t), 10,(200, 0, 0), 3)


def create_new_population(old_population, elitism=2, gen=1):
    # sorted_population = sorted(old_population, key=compute_fitness)
    max_fitness = -1
    for i in old_population:
        compute_fitness(i)
    for i in range(len(old_population)):
        max_fitness = max(max_fitness, compute_share_fitness(i, old_population))
    sorted_population = sorted(old_population, key=lambda ind: ind[3])
    if gen % 1 == 0:
        # fitnesses.append(compute_fitness(sorted_population[m - 1]))
        print("BEST:", compute_fitness(sorted_population[m - 1]))

    new_population = []
    while len(new_population) < m - elitism:
        # selection
        individual_s1 = selection(sorted_population)
        individual_s2 = selection(sorted_population)  # duplication

        # crossover
        individual_c1, individual_c2 = crossover(individual_s1, individual_s2)

        # mutation
        individual_m1 = mutate(individual_c1)
        individual_m2 = mutate(individual_c2)

        new_population.append(individual_m1)
        new_population.append(individual_m2)

    for ind in sorted_population[m - elitism:]:
        new_population.append(ind.copy())
    return new_population

def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area
def getwhitepixel(individual, k, l):
    individual = [int(individual[0]), int(individual[1]), int(individual[2])]
    crop, wid, hei = crop_img(individual, imutils.rotate_bound(image, individual[2]), 1, k, l)
    whitepixel = []
    print(crop.shape)
    for ii in range(crop.shape[0]):
        for jj in range(crop.shape[1]):
            if crop[ii][jj] != 0:
                whitepixel.append([ii, jj])
    whitepixel = np.array(whitepixel)
    clustering = DBSCAN(eps=14, min_samples=200).fit(whitepixel)
    labels = clustering.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    maxx = 100000
    colors = [create_color() for _ in range(m)]
    blank_image3 = np.zeros((crop.shape[0], crop.shape[1], 3), np.uint8)
    for ll,col in zip(range(n_clusters_),colors):
        my_members = labels == ll
        if (len(whitepixel[my_members])<=2):
            continue
        ch = ConcaveHull()
        ch.loadpoints(whitepixel[my_members])
        ch.calculatehull()
        pts = np.vstack(ch.boundary.exterior.coords.xy).T
        area=PolyArea2D(pts)
        density=len(whitepixel[my_members])/area
        print(col)
        if density<maxx:
            maxx = density
            crowded = whitepixel[my_members]
            print("density: ",density,len(whitepixel[my_members]),area)
        for lll in whitepixel[my_members]:
            cv2.circle(blank_image3, (lll[1], lll[0]), 1, col, 1)
    print(maxx)

    for lll in crowded:
        src2[individual[0] - hei + lll[0]][individual[1] - wid + lll[1]] = 0
        cv2.circle(blank_image3, (lll[1], lll[0]), 1, (255,255,255), 1)
    cv2.rectangle(src2, (individual[1] - 10, individual[0] - 10), (individual[1] + 10, individual[0] + 10), (200, 0, 0), 3)
    cv2.imwrite("whitepixel_{}_{}.jpg".format(k,l),blank_image3)
    #cv2.circle(src2, (individual[1], individual[0]), 30, 255, 10)


#colors = [create_color() for _ in range(m)]
colors="abcdefghiklmno"
image = cv2.imread("template_crop.png", 0)
w_max = 0
h_max = 0
for i in range(360):
    temp = imutils.rotate_bound(image, i)
    w_max = max(w_max, temp.shape[1])
    h_max = max(h_max, temp.shape[0])
src = cv2.imread("template2.jpg", 0)
borderType = cv2.BORDER_CONSTANT
src = cv2.copyMakeBorder(src, h_max, h_max, w_max, w_max, borderType, None, 0)
src2 = src
ret, src2 = cv2.threshold(src2, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite("original_binary.jpg", src2)
geni = 0
for k in range(6):
    population = [create_individual() for _ in range(m)]

    # for i in population:
    #     if i[0] + h_max > src.shape[0] or i[1] + w_max > src.shape[1]:
    #         print(i)
    geni = 0
    while geni < n_generations:  # or compute_fitness(population[m-1])<0.5:
        population = create_new_population(population, 2, geni)
        blank_image = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
        cccc = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in population:
            cv2.putText(blank_image, str(cccc), (i[1], i[0]), font, 1, (0, 255, 0), 2)
            cccc += 1
        cv2.imwrite("iter{}_gen{}.jpg".format(k, geni), blank_image)
        geni += 1
    blank_image2 = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
    blank_image2.fill(255)
    population = np.array(population)
    population2 = [i[0:2] for i in population]

    # clustering = MeanShift(bandwidth=200).fit(population2)
    clustering = DBSCAN(eps=100, min_samples=10).fit(population2)
    labels = clustering.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print(n_clusters_)
    cccc = 0
    peaks = []
    for ll, col in zip(range(n_clusters_), colors):
        my_members = labels == ll
        maxx = -10
        ind = []
        for lll in population[my_members]:
            if (lll[4] > maxx):
                maxx = lll[4]
                ind = lll
            cv2.putText(blank_image2, str(col), (int(lll[1]), int(lll[0])), font, 1, (100,100,100), 2)
            cccc += 1
        if maxx > 0.4:
            peaks.append([ind, len(population[my_members])])
    peaks.sort(reverse=True, key=lambda ind: ind[1])
    print(peaks)
    cccc = 0
    prev=-10
    for indi in peaks[0:min(len(peaks), 2)]:
        ind = indi[0]
        # if prev==-10:
        #     prev=indi[1]
        # elif indi[1] < prev/2:
        #     continue
        rotated = imutils.rotate_bound(image, ind[2])
        ret, rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
        crop = crop_img(ind, rotated)
        ret, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
        remove2(ind, rotated)
        getwhitepixel(ind, k, cccc)
        cccc += 1
    cv2.imwrite("res_iter{}.jpg".format(k), src2)
    cv2.imwrite("label_iter{}.jpg".format(k), blank_image2)
    continue
