import math
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

class Network:

    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
        self.queue = Queue()  # Create a queue for network operations

    def get_mean_degree(self):
        total_degree = sum(sum(node.connections) for node in
                           self.nodes)  # Calculate the total degree, which is the sum of connections of all nodes
        mean_degree = total_degree / len(self.nodes)  # Calculate the mean degree, which is the total degree divided by the number of nodes
        return mean_degree  # Return the mean degree

    def get_mean_clustering(self):
        total_clustering = 0  # Initialize the total clustering coefficient
        for node in self.nodes:
            neighbour_indices = [i for i, is_connected in enumerate(node.connections) if
                                 is_connected]  # Get the indices of all neighboring nodes of the node
            neighbour_count = len(neighbour_indices)
            if (neighbour_count < 2):
                continue
            possible_triangles = neighbour_count * (neighbour_count - 1) / 2  # Calculate the possible number of triangles, which is the combination of the number of neighbors
            actual_triangles = sum(sum(self.nodes[neighbour_index].connections[j] for j in neighbour_indices[i + 1:])
                                   # Calculate the actual number of triangles formed by the node
                                   for i, neighbour_index in enumerate(neighbour_indices))
            clustering_coefficient = actual_triangles / possible_triangles if possible_triangles > 0 else 0  # Calculate the clustering coefficient of the node
            total_clustering += clustering_coefficient  # Add the clustering coefficient of the node to the total clustering coefficient
        mean_clustering = total_clustering / len(
            self.nodes)  # Calculate the mean clustering coefficient, which is the total clustering coefficient divided by the number of nodes
        return mean_clustering  # Return the mean clustering


    def get_mean_path_length(self):
        total_path_length = 0  # Initialize the total path length
        for start_node in self.nodes:  # Iterate over each node
            distances = {}  # Dictionary to store distances from the start node to other nodes
            self.queue.enqueue((start_node, 0))  # Perform breadth-first search using a queue, starting node with distance 0
            while not self.queue.is_empty():  # Continue searching while the queue is not empty
                node, distance = self.queue.dequeue()
                if node not in distances:
                    distances[
                        node] = distance  # Add the node to distances dictionary and update the distance if not already present
                    for neighbour_index, is_connected in enumerate(
                            node.connections):  # For each neighbor node of the node, enqueue it and increase the distance by 1
                        if is_connected:
                            neighbour_node = self.nodes[neighbour_index]
                            self.queue.enqueue((neighbour_node, distance + 1))  # Using self-implemented queue
            total_path_length += sum(
                distances.values())  # Calculate the sum of distances from each node to other nodes and add to total path length
        mean_path_length = total_path_length / (len(self.nodes) * (
                    len(self.nodes) - 1))  # Calculate the mean path length, which is the total path length divided by the number of node pairs
        return mean_path_length  # Return the mean path length


    def make_random_network(self, N, connection_probability):
        '''
            This function makes a *random* network of size N.
            Each node is connected to each other node with probability p
            '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=2):
        '''
        This function makes a *ring* network of size N.
        Each node is connected to nearest neighbor node
        '''

        self.nodes = []
        for node_number in range(N):
            connections = [0 for val in range(N)]
            for i in range(1, neighbour_range + 1):
                connections[(node_number - i) % N] = 1
                connections[(node_number + i) % N] = 1
            new_node = Node(value=random.random(), number=node_number, connections=connections)
            self.nodes.append(new_node)


    def make_small_world_network(self, N, re_wire_prob=0.2):
        self.make_ring_network(N, neighbour_range=2)  # Call ring network
        for node in self.nodes:
            # This for loop is used to iterate over the list of connections of the node node.connections,
            # Obtain the neighbour index and connection status of each neighbor
            for neighbour_index, connected in enumerate(node.connections):
                if connected and np.random.random() < re_wire_prob:
                    # If the current neighbor connection exists and the probability of random generation (np.random.random) is less than the given probability of reconnection
                    new_neighbour_index = np.random.randint(0, N)
                    # A new neighbor index is randomly selected, ranging from 0 to N
                    while new_neighbour_index == node.index or node.connections[new_neighbour_index]:
                        new_neighbour_index = np.random.randint(0, N)
                    # When the new neighbor index is equal to the index of the current node or the new neighbor is already connected to the current node,
                    # A new neighbor node index is selected again until neither is satisfied, and the following operations are performed out of the loop
                    node.connections[neighbour_index] = 0
                    node.connections[new_neighbour_index] = 1
        # To reconnect, cancel the current neighbor connection and establish a new connection

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
        plt.show()

def test_networks():   #Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1)%num_nodes] = 1
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert(network.get_mean_degree()==2), network.get_mean_degree()
    assert(network.get_mean_clustering()==0), network.get_mean_clustering()
    assert round(network.get_mean_path_length(), 6) == 2.777778, network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert(network.get_mean_degree()==1), network.get_mean_degree()
    assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
    assert(network.get_mean_path_length()==5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
    assert(network.get_mean_path_length()==1), network.get_mean_path_length()

    print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
    if row < 0 or col < 0:
        raise ValueError("row and col must be >= 0")
    #return np.random.random() * population
    num_rows = population.shape[0]
    num_cols = population.shape[1]
    neighbours = []

    if row > 0:
        neighbours.append((row-1, col)) # record the last coordinates

    if row < num_rows-1:
        neighbours.append((row+1, col))
    # not the left side row
    if col > 0:
        neighbours.append((row, col-1)) # record the last coordinates
    # not the right side row
    if col < num_cols-1:
        neighbours.append((row, col+1))

    D = 0
    H = external

    S_i = population[row][col]
    for neighbour in neighbours:

        S_j = population[neighbour[0]][neighbour[1]]
        D = D + (S_i * S_j)
    D = D + (H * S_i)

    # the caculation effet the external influence
    print("D 值为%s" % D)
    return D

def ising_step(population, external, alpha):
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)
    agreement = calculate_agreement(population, row, col, external)

    if agreement < 0:
        population[row, col] *= -1 # change the opinion
    else:

        prob = math.exp(-agreement / alpha)
        # get a random percentage between 0 and 1
        if random.random() < prob:
            population[row, col] *= - 1



def plot_ising(im, population):
    #use two different color
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


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main(Threshold = 0.2, Coupling = 0.2, testTimes = 100):
    array = np.random.uniform(low = 0,high = 1,size=100)
    array_2d = []
    for i in range(testTimes*100 ):
        position_random_element = random.randrange(len(array))
        if position_random_element == 0:
            random_element_neighbor = position_random_element + 1
            # print('the random element is the first number and it has no left side neighbor')
        elif position_random_element == len(array) -1:
            random_element_neighbor= position_random_element - 1
            # print('the random element is the final number and it has no right side neighbor')
        else:
            random_element_neighbor = position_random_element +random.choice([1,-1])
        if abs(array[random_element_neighbor] - array[position_random_element]) < Threshold:
            temp = array[position_random_element]
            array[position_random_element] = array[position_random_element] +Coupling*(array[random_element_neighbor] - array[position_random_element])
            array[random_element_neighbor] = array[random_element_neighbor] + Coupling*(temp - array[random_element_neighbor])
        if i % 100 == 0:
            array_2d.append(array.copy())
        # elif abs(array[random_element_neighbor] - array[random_element]) > Threshold:
        #     print('the two selected people have a difference of opinion greater than a threshold')



    rows = len(array_2d)
    cols = len(array_2d[0])



    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,5))
    ax1.hist(array, bins=10, range = (0.0, 1.0), edgecolor='black')

    ax1.set_xlabel('opinion')
    ax1.set_ylabel('time')
    ax1.set_xlim(0, 1)
    rows = len(array_2d)
    cols = len(array_2d[0])
    for row in range(rows):
        plt.scatter([row] * cols, array_2d[row], c='red')
    ax2.set_ylim(0.0,1.0)
    ax2.set_xlim(1,testTimes)
    ax2.set_xlabel('Times')
    ax2.set_ylabel('Opinion')
    plt.suptitle('Coupling ='+ str(Coupling)+ ', Threshold =' +str(Threshold))
    plt.show()
    pass

#Task 5
def defuant_main_network(Threshold = 0.5, Coupling = 0.5, N=20, re_wire_prob=0.2):

    # create a network
    population = Network()
    population.make_small_world_network(N, re_wire_prob)
    array = []
    # for loop
    for n in population.nodes:
        # choose random node in network (network.node_list[random)
        position_random_element = random.choice(population.nodes)
        #print(position_random_element.value)
        # choose random neighbour from Node (node.connections)[random]
        random_element_neighbor = random.choice(position_random_element.connections)
        # calculate agreement from neighbours
        if abs(population.nodes[random_element_neighbor].value - position_random_element.value) < Threshold:
            temp = population.nodes[random_element_neighbor].value
            population.nodes[random_element_neighbor].value = population.nodes[random_element_neighbor].value + Coupling * (temp - position_random_element.value)
            position_random_element.value = position_random_element.value + Coupling * (position_random_element.value - temp)
        # update node value using defuant equations...
        # display the figure of the network
        sum = 0
        for i in population.nodes:
            sum += i.value
        array.append(sum / N)

    plt.scatter([i+1 for i in range(len(array))], array, c='red')
    plt.xlabel('Row Number')
    plt.ylabel('Value')
    plt.title('Scatter Plot of 2D Array Data')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.show()

def test_defuant():
    defuant_main(Coupling=0.5,Threshold=0.5)
    defuant_main(Coupling=0.5,Threshold=0.1)
    defuant_main(Coupling=0.1,Threshold=0.5)
    defuant_main(Coupling=0.1,Threshold=0.1)
    pass


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
    parser = argparse.ArgumentParser()
    
    # Task 1
    parser.add_argument("-ising_model", action='store_true')
    parser.add_argument("-test_ising", action='store_true')
    parser.add_argument("-alpha", type=float)
    parser.add_argument("-external", type=float)

    # Task 2
    parser.add_argument("-defuant", action='store_true')
    parser.add_argument("-threshold", type=float)
    parser.add_argument("-beta", type=float)
    parser.add_argument("-test_defuant", action='store_true')

    # Task 3
    parser.add_argument('-network', type=int, help='Creates and plots a random network of size N', metavar='N')
    parser.add_argument('-test_network',action='store_true')

    # Task 4
    parser.add_argument('-ring_network', type=int)
    parser.add_argument('-small_world', type=int)
    parser.add_argument('-re_wire', type=float)

    # Task 5
    parser.add_argument('-use_network', type=int)
    parser.add_argument("-N", type=int, default=20, help="Number of nodes in the network")

    args = parser.parse_args()

    #Task 1
    if args.alpha is None:
        args.alpha = 1.0
    if args.external is None:
        args.external = 0.0

    if args.test_ising:  # test
        test_ising()
    elif args.ising_model:  # print it
        # print a random a 100x100 matrix
        # use a 0 or 1
        population = np.random.randint(2, size=(100, 100))

        population[population == 0] = -1
        ising_main(population, args.alpha, args.external)


    #Task 2
    if args.test_defuant:
        test_defuant()
    elif args.defuant:
        # Task 5
        if args.use_network:
            if args.N is None:
                args.N = 20
            defuant_main_network(N=args.N)
        else:
            if args.threshold is None:
                args.threshold = 0.2
            if args.beta is None:
                args.beta = 0.2
            defuant_main(Threshold=args.threshold, Coupling=args.beta)

    #Task 3
    if args.network:
        network = Network()
        network.make_random_network(args.network, 0.5)
        print("Mean degree:", network.get_mean_degree())
        print("Average path length:", network.get_mean_path_length())
        print("Clustering co-efficient:", network.get_mean_clustering())
    if args.test_network:
        test_networks()

    #Task 4
    if args.small_world is not None:
        N = args.small_world if args.small_world is not None else args.N
        print("Creating a small world network.")
        network = Network()
        network.make_small_world_network(N, args.re_wire)
        network.plot()
    elif args.ring_network:
        N = args.ring_network if args.ring_network is not None else args.N
        print("Creating a ring network.")
        network = Network()
        network.make_ring_network(N)
        network.plot()
    else:
        print("No small world network requested.")

if __name__=="__main__":
    main()
