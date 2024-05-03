import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import argparse
import random


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

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes,  color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

        plt.show()
        plt.pause(0.1)




def main():
    parser = argparse.ArgumentParser(description="Generate and plot networks.")

    parser.add_argument('-small_world', nargs='?', const=10, type=int, default=None,
                        help='Activate small world model and optionally set number of nodes')
    parser.add_argument('-ring_network', nargs='?', const=10, type=int, default=None,
                        help='Activate ring network model and optionally set number of nodes')
    parser.add_argument('-N', type=int, default=20, help="Number of nodes in the network if not set by --small_world")
    parser.add_argument('-re_wire', '-p', type=float, default=0.2, help="Probability of rewiring connections")

    args = parser.parse_args()



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

    
if __name__ == "__main__":
    main()