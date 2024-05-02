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
		self.queue = Queue()   # Create a queue for network operations

	def get_mean_degree(self):
		total_degree = sum(sum(node.connections) for node in self.nodes)   # Calculate the total degree, which is the sum of connections of all nodes
		mean_degree = total_degree / len(self.nodes)   # Calculate the mean degree, which is the total degree divided by the number of nodes
		return mean_degree   # Return the mean degree

	def get_mean_clustering(self):
		total_clustering = 0   # Initialize the total clustering coefficient
		for node in self.nodes:
			neighbour_indices = [i for i, is_connected in enumerate(node.connections) if is_connected]   # Get the indices of all neighboring nodes of the node
			neighbour_count = len(neighbour_indices)
			if (neighbour_count < 2):
				continue
			possible_triangles = neighbour_count * (neighbour_count - 1) / 2   # Calculate the possible number of triangles, which is the combination of the number of neighbors
			actual_triangles = sum(sum(self.nodes[neighbour_index].connections[j] for j in neighbour_indices[i + 1:])   # Calculate the actual number of triangles formed by the node
								   for i, neighbour_index in enumerate(neighbour_indices))
			clustering_coefficient = actual_triangles / possible_triangles if possible_triangles > 0 else 0   # Calculate the clustering coefficient of the node
			total_clustering += clustering_coefficient   # Add the clustering coefficient of the node to the total clustering coefficient
		mean_clustering = total_clustering / len(self.nodes)   # Calculate the mean clustering coefficient, which is the total clustering coefficient divided by the number of nodes
		return mean_clustering   # Return the mean clustering


	def get_mean_path_length(self):
		total_path_length = 0   # Initialize the total path length
		for start_node in self.nodes:   # Iterate over each node
			distances = {}   # Dictionary to store distances from the start node to other nodes
			self.queue.enqueue((start_node, 0))   # Perform breadth-first search using a queue, starting node with distance 0
			while not self.queue.is_empty():   # Continue searching while the queue is not empty
				node, distance = self.queue.dequeue()
				if node not in distances:
					distances[node] = distance   # Add the node to distances dictionary and update the distance if not already present
					for neighbour_index, is_connected in enumerate(node.connections):   # For each neighbor node of the node, enqueue it and increase the distance by 1
						if is_connected:
							neighbour_node = self.nodes[neighbour_index]
							self.queue.enqueue((neighbour_node, distance + 1))   # Using self-implemented queue
			total_path_length += sum(distances.values())   # Calculate the sum of distances from each node to other nodes and add to total path length
		mean_path_length = total_path_length / (len(self.nodes) * (len(self.nodes) - 1))  # Calculate the mean path length, which is the total path length divided by the number of node pairs
		return mean_path_length   # Return the mean path length



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
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1


	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-network', type=int, help='Creates and plots a random network of size N', metavar='N')
	parser.add_argument('-test', choices=['network'], help='Run test functions')
	args = parser.parse_args()

	if args.network:
		network = Network()
		network.make_random_network(args.network, 0.5)
		print("Mean degree:", network.get_mean_degree())
		print("Average path length:", network.get_mean_path_length())
		print("Clustering co-efficient:", network.get_mean_clustering())

	elif args.test:
		test_networks()

if __name__=="__main__":
	main()
