import cv2
import random
import imutils
from scipy import spatial
import scipy.sparse
import numpy as np
n = 3  # size of individual (chromosome)
m = 100  # size of population
n_generations = 10 # number of generations
fitnesses = []
def crop_img(individual,rotated):
    h = int(rotated.shape[0] / 2)
    w = int(rotated.shape[1] / 2)
    t_h=0
    t_w=0
    if rotated.shape[0]%2==1:
        t_h=1
    if rotated.shape[1]%2==1:
        t_w=1
    crop = src2[individual[0] - h:individual[0] + h+t_h, individual[1] - w:individual[1] + w+t_w]
    return crop
def compute_fitness(individual):
    rotated = imutils.rotate_bound(image, individual[2])
    ret,rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
    crop = crop_img(individual,rotated)
    ret, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
    sA = scipy.sparse.csr_matrix(crop.flatten()).toarray()
    sB = scipy.sparse.csr_matrix(rotated.flatten()).toarray()
    if 255 not in sA[0]:
        ttt=1
    else:
        ttt = spatial.distance.(sA, sB)
    return 1-ttt
def create_individual():
    return ([random.randint(int(h_max/2), src.shape[0]-int(h_max)),random.randint(int(w_max/2) , src.shape[1]-int(w_max)),random.randint(0, 360)])

def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(n):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]

    return individual1_new, individual2_new


def mutate(individual, mutation_rate=0.4):
    if geni-n_generations>0:
        mutation_rate=0.1*int((geni-n_generations)/10+1)
    individual_m = individual.copy()

    for i in range(n):
        temp=create_individual()
        if random.random() < mutation_rate:
            if i==0:
                individual_m[i] = temp[0]
            elif i==1:
                individual_m[i]=temp[1]
            else:
                individual_m[i]=temp[2]
    return individual_m


def selection(sorted_old_population):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if (index2 != index1):
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s

def remove(crop,rotated):
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
            if crop[i,j]==rotated[i,j] and crop[i,j]!=0:
                crop[i,j]=0
    src2[individual[0] - h:individual[0] + h + t_h, individual[1] - w:individual[1] + w + t_w]=crop
    return 0
def bfs(x,y,mode):
    bfs_coord=[[x,y]]
    l=0
    bfs_bool_arr=[[True for i in range(src.shape[0])] for j in range(src.shape[1])]
    temp=src2[x,y]
    bfs_bool_arr[x][y]=False
    if temp!=0 and mode==0:
        return x,y
    while l<=len(bfs_coord)-1:
        x,y=bfs_coord[l]
        for skamtt in range(6):
            skamt=skamtt+1
            if bfs_bool_arr[x+skamt][y]==True:
                if src2[x+skamt,y]!=0:
                    if mode==0:
                        return x+skamt,y
                    else:
                        src2[x+skamt,y]=0
                        bfs_coord.append([x + skamt, y])
                else:
                    if mode==0:
                        bfs_coord.append([x + skamt, y])
                bfs_bool_arr[x + skamt][y]=False
            if bfs_bool_arr[x-skamt][y]==True:
                if src2[x - skamt, y] != 0:
                    if mode == 0:
                        return x - skamt, y
                    else:
                        src2[x - skamt, y] = 0
                        bfs_coord.append([x - skamt, y])
                else:
                    if mode == 0:
                        bfs_coord.append([x - skamt, y])
                bfs_bool_arr[x - skamt][y] = False
            if bfs_bool_arr[x][y+skamt]==True:
                if src2[x, y+1] != 0:
                    if mode == 0:
                        return x, y+skamt
                    else:
                        src2[x , y+ 1] = 0
                        bfs_coord.append([x , y+ skamt])
                else:
                    if mode == 0:
                        bfs_coord.append([x , y+ skamt])
                bfs_bool_arr[x][y + skamt] = False
            if bfs_bool_arr[x][y-skamt]==True:
                if src2[x , y-skamt] != 0:
                    if mode == 0:
                        return x, y-skamt
                    else:
                        src2[x , y-skamt] = 0
                        bfs_coord.append([x, y-skamt])
                else:
                    if mode == 0:
                        bfs_coord.append([x , y-skamt])
                bfs_bool_arr[x ][y-skamt] = False
        l+=1
def remove2(individual,rotated):
    h = int(rotated.shape[0] / 2)
    w = int(rotated.shape[1] / 2)
    t_h = 0
    t_w = 0
    if rotated.shape[0] % 2 == 1:
        t_h = 1
    if rotated.shape[1] % 2 == 1:
        t_w = 1
    x_t=individual[0]+int(h/2)+t_h
    y_t=individual[1]+int(w/2)+t_w
    print(x_t,y_t)
    x_t,y_t=bfs(x_t,y_t,0)
    print(src2[x_t,y_t])
    print(x_t,y_t)
    bfs(x_t,y_t,1)
def create_new_population(old_population, elitism=2, gen=1):
    sorted_population = sorted(old_population, key=compute_fitness)

    if gen % 1 == 0:
        fitnesses.append(compute_fitness(sorted_population[m - 1]))
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


image = cv2.imread("template_crop.png",0)
w_max=0
h_max=0
for i in range(360):
    temp=imutils.rotate_bound(image,i)
    w_max=max(w_max,temp.shape[1])
    h_max=max(h_max,temp.shape[0])
src=cv2.imread("template2.jpg",0)
borderType = cv2.BORDER_CONSTANT
src = cv2.copyMakeBorder(src, h_max, h_max, w_max, w_max, borderType, None, 0)
src2 = src
ret, src2 = cv2.threshold(src2, 127, 255, cv2.THRESH_BINARY)
geni=0
for k in range(13):
    population = [create_individual() for _ in range(m)]

    for i in population:
        if i[0]+h_max>src.shape[0] or i[1]+w_max>src.shape[1]:
            print(i)
    geni=0
    while geni<n_generations or compute_fitness(population[m-1])<0.5:
        population = create_new_population(population, 2, geni)
        geni+=1
        if geni>n_generations+40:
            break

    individual=population[m-1]
    rotated = imutils.rotate_bound(image, individual[2])
    ret,rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
    crop = crop_img(individual,rotated)
    ret,crop=cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('b',rotated)
    #remove(crop,rotated)
    remove2(individual,rotated)
    #cv2.rectangle(src, (individual[1]-int(crop.shape[1]/2),individual[0]-int(crop.shape[0]/2)), (individual[1] + int(crop.shape[1]/2),individual[0]+int(crop.shape[0]/2)), (200, 0, 0), 3)
    cv2.imwrite("found{}.jpg".format(k),src2)
    print("{} iteration".format(k+1))
    # cv2.imshow('a', crop)
    # cv2.waitKey()
