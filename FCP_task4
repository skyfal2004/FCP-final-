import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import argparse

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
            new_node = Node(0, node_number, connections=connections)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot small-world networks.")
    parser.add_argument("-N", type=int, default=20, help="Number of nodes in the network")
    parser.add_argument("-p", "--rewire_prob", type=float, default=0.2, help="Probability of rewiring connections")

    args = parser.parse_args()

    network = Network()
    network.make_small_world_network(N=args.N, re_wire_prob=args.rewire_prob)
    network.plot()
