import numpy as np
from collections import defaultdict # Imports defaultdict from collections to store adjacency lists

# Assigns each pixel to a cluster


# This function creates cluster groups based on the given edges
def _createrClusterGroups(edges): 
    graph = defaultdict(list) # Creates a dictionary where each node has a list of neighbors -> example: {1: [2], 2: [1]}

    # Creates an undirected adjacency list from the given edges
    for edge in edges: # example: edge = (1, 2)
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0]) 

    
    # Depth-First Search (DFS) to find all connected nodes in a component
    def dfs(node, visited):
        component = [] # List to store the current connected component
        stack = [node] # Stack for DFS: add the current node
        visited.add(node) # Mark the current node as visited

        while stack: # While there are still nodes in the stack, continue DFS
            current = stack.pop() # Pop the top node from the stack
            component.append(current) # Add the node to the current component

            for neighbor in graph[current]: # Iterate over all neighbors of the current node
                if neighbor not in visited: # If neighbor hasn't been visited yet
                    visited.add(neighbor) # Mark the neighbor as visited
                    stack.append(neighbor) # Add the neighbor to the stack
                    
        return component # Return the list of connected nodes -> graph theory: connected component = all nodes connected by some path

    # Finds all connected node groups in the graph
    def find_all_connected_groups():
        visited = set() # set() to store already visited nodes
        all_groups = [] # List to store all found clusters

        # Iterate over all nodes in the graph
        for node in graph:
            if node not in visited: # If node hasn't been visited
                component = dfs(node, visited) # Find all connected nodes in the current component
                all_groups.append(np.array(component)) # Store the component as a NumPy array in the list

        return all_groups # Return the list of all groups

    # Return the found clusters: find_all_connected_groups() returns all_groups, _createrClusterGroups() returns that function -> effectively returns all_groups
    return find_all_connected_groups() 

# Creates a mapping from nodes to cluster IDs
def _createClusterNodeMap(connected_nodes, finalNodes):
    node_cluste_map = np.zeros(len(finalNodes)) # Create an array of zeros of length finalNodes -> stores the cluster IDs of nodes

    # Iterate over found clusters and assign a cluster ID to each node
    for cluster_index, cluster in enumerate(connected_nodes):
          for node in cluster:
              node_cluste_map[node] = cluster_index # Store the cluster number for the respective node
    return node_cluste_map # Return the node-to-cluster mapping           
      

# Main function for cluster assignment based on the data structure ###
def cluster(datum_node_map, finalNodes, edges):
    connected_nodes = _createrClusterGroups(edges) # Find connected groups of nodes
    node_cluster_map = _createClusterNodeMap(connected_nodes, finalNodes) # Create a cluster mapping for the nodes
    pixel_cluster_map = list() # List to store the cluster assignment for pixels

    # Assign each data point from datum_node_map to a cluster
    for node_index in datum_node_map:
        pixel_cluster_map.append(node_cluster_map[node_index])

    return pixel_cluster_map, node_cluster_map # Return pixel and node cluster assignments
