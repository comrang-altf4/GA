import cv2
import random
import imutils
from scipy import spatial
import scipy.sparse
n = 3  # size of individual (chromosome)
m = 100  # size of population
n_generations = 40  # number of generations
fitnesses = []

def generate_random_value():
    return ([random.randint(0, src.shape[0]),random.randint(0, src.shape[1]),random.randint(0, 360)])
def crop_img(individual,rotated):
    h = int(rotated.shape[0] / 2)
    w = int(rotated.shape[1] / 2)
    if individual[0] - h < 0:
        h = individual[0]
    elif individual[0] + h > src.shape[0]:
        h = src.shape[0] - individual[0]
    if individual[1] - w < 0:
        w = individual[1]
    elif individual[1] + w > src.shape[1]:
        w = src.shape[1] - individual[1]
    crop = src[individual[0] - h:individual[0] + h, individual[1] - w:individual[1] + w]
    top = 0
    bottom = top
    left = 0  # shape[1] = cols
    right = left
    if crop.shape[0] < rotated.shape[0]:
        h = rotated.shape[0] - crop.shape[0]
        top = int(h / 2)
        bottom = int(h / 2)
        if h % 2 == 1:
            bottom = int(h / 2) + 1
    if crop.shape[1] < rotated.shape[1]:
        h = rotated.shape[1] - crop.shape[1]
        left = int(h / 2)
        right = int(h / 2)
        if h % 2 == 1:
            right = int(h / 2) + 1
    borderType = cv2.BORDER_CONSTANT
    crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType, None, 0)
    ret, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
    return crop
def compute_fitness(individual):
    rotated = imutils.rotate_bound(image, individual[2])
    ret,rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
    crop = crop_img(individual,rotated)
    ret, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)

    sA = scipy.sparse.csr_matrix(crop.flatten())
    sB = scipy.sparse.csr_matrix(rotated.flatten())
    return spatial.distance.cosine(sA.toarray(), sB.toarray())


def create_individual():
    return ([random.randint(0, src.shape[0]), random.randint(0, src.shape[1]), random.randint(0, 360)])

def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(n):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]

    return individual1_new, individual2_new


def mutate(individual, mutation_rate=0.05):
    individual_m = individual.copy()

    for i in range(n):
        if random.random() < mutation_rate:
            if i==0:
                individual_m[i] = random.randint(0, src.shape[0])
            elif i==1:
                individual_m[i]=random.randint(0, src.shape[1])
            else:
                individual_m[i]=random.randint(0, 360)
    return individual_m


def selection(sorted_old_population):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if (index2 != index1):
            break

    individual_s = sorted_old_population[index2]
    if index2 > index1:
        individual_s = sorted_old_population[index1]

    return individual_s


def create_new_population(old_population, elitism=2, gen=1):
    sorted_population = sorted(old_population, key=compute_fitness)
    if gen % 1 == 0:
        fitnesses.append(compute_fitness(sorted_population[0]))
        print("BEST:", compute_fitness(sorted_population[0]))
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

    return sorted(new_population,key=compute_fitness)

c=0
image = cv2.imread("template_crop.png",0)
src=cv2.imread("image1.jpg",0)
population = [create_individual() for _ in range(m)]
#print(population)
for i in range(n_generations):
    population = create_new_population(population, 2, i)
#print(population)
individual=population[0]
rotated = imutils.rotate_bound(image, individual[2])
ret,rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
crop = crop_img(individual,rotated)
ret,crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('a',crop)
cv2.imshow('b',rotated)
cv2.waitKey()
