# -*- coding: UTF-8 -*-
import numpy as np # type: ignore
import random
from argparse import ArgumentParser
from matplotlib import pyplot as plt # type: ignore
import math

def calculate_agreement(population, row, col, external=0.0):
    if row < 0 or col < 0:
        raise ValueError("row and col must be >= 0")

    num_rows = population.shape[0]
    num_cols = population.shape[1]

    neighbours = []

    if row > 0:
        neighbours.append((row-1, col))

    if row < num_rows-1:
        neighbours.append((row+1, col))

    if col > 0:
        neighbours.append((row, col-1))

    if col < num_cols-1:
        neighbours.append((row, col+1))

    D = 0
    H = external

    S_i = population[row][col]
    for neighbour in neighbours:

        S_j = population[neighbour[0]][neighbour[1]]
        D = D + (S_i * S_j)
    D = D + (H * S_i)


    print("D 值为%s" % D)
    return D

def ising_step(population, external, alpha):
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)
    agreement = calculate_agreement(population, row, col, external)

    if agreement < 0:
        population[row, col] *= -1
    else:

        prob = math.exp(-agreement / alpha)

        if random.random() < prob:
            population[row, col] *= - 1


def plot_ising(im, population):

    new_im = np.array([[255 if val == 1 else 1 for val in rows] for rows in population])
    im.set_data(new_im)
    plt.pause(0.1)

def test_ising():
    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1,1] = 1
    assert(calculate_agreement(population, 1, 1)==-4), "Test 2"

    population[0, 1] = 1
    assert(calculate_agreement(population, 1, 1)==-2), "Test 3"

    population[1, 0] = 1
    assert(calculate_agreement(population, 1, 1)==0), "Test 4"

    population[2, 1] = 1
    assert(calculate_agreement(population, 1, 1)==2), "Test 5"

    population[1, 2] = 1
    assert(calculate_agreement(population, 1, 1)==4), "Test 6"

    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1, 1,1)==3), "Test 7"
    assert(calculate_agreement(population, 1, 1, -1)==5), "Test 8"
    assert(calculate_agreement(population, 1, 1, -10)==14), "Test 9"
    assert(calculate_agreement(population, 1, 1, +10)==-6), "Test 10"

    print("Tests passed")

def ising_main(population, alpha, external):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    new_im = np.array([[255 if val == 1 else 1 for val in rows] for rows in population])

    im = ax.imshow(new_im, interpolation='none', cmap='RdPu_r')

    for frame in range(100):
        for step in range(1000):
            ising_step(population, external, alpha)
        print('Step:', frame, end='/r')
        plot_ising(im, population)

parser = ArgumentParser()

parser.add_argument("-ising_model", action='store_true')
parser.add_argument("-test_ising", action='store_true')
parser.add_argument("-alpha", type=float)
parser.add_argument("-external", type=float)

args = parser.parse_args()

if args.alpha is None:
    args.alpha = 1.0
if args.external is None:
    args.external = 0.0

if args.test_ising:
    test_ising()
elif args.ising_model:

    population = np.random.randint(2, size=(100,100))

    population[population == 0] = -1
    ising_main(population, args.alpha, args.external)

test_ising()
