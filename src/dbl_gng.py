#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import constants
"""
Created on Fri Jun 21 16:21:48 2024

@author: Corner
@modified: Paul Schönbrunner
"""

class DBL_GNG():
    # Initialize the class with various hyperparameters
    def __init__(self, feature_number, maxNodeLength, L1=0.5, L2=0.01, 
                 errorNodeFactor = 0.5, newNodeFactor = 0.5):
       
        self.feature_number = feature_number # Number of features
        self.M = maxNodeLength # Maximum number of nodes
        self.alpha = L1 # Learning rate for the first adjustment step
        self.beta = L2 # Learning rate for the second adjustment step          
        self.delta = errorNodeFactor # Error reduction factor
        self.rho = newNodeFactor # Factor for new nodes
        self.finalDistMap = 0 # Stores mapping from each datum to nearest node (initialized to 0)
        
        self.eps = 1e-04 # Small value to prevent division by zero
        
    # Resets batch learning variables to zero to avoid outdated values from previous batches -> correct learning + stable network update
    def resetBatch(self):    
        self.Delta_W_1 = np.zeros_like(self.W) # Weight changes for step 1
        self.Delta_W_2 = np.zeros_like(self.W) # Weight changes for step 2
        self.A_1 = np.zeros(len(self.W)) # Activation count for step 1 (how many times each node was activated)
        self.A_2 = np.zeros(len(self.W)) # Activation count for step 2
        self.S = np.zeros((len(self.W),len(self.W))) # Connection strength matrix             

    # Initialize the network with randomly selected starting nodes    
    def initializeDistributedNode(self, data, number_of_starting_points = 1):
        
        data = data[:,:self.feature_number].copy() # Reduce data to relevant features
        np.random.shuffle(data) # Shuffle the data randomly
        
        nodeList = np.empty((0,self.feature_number),dtype=np.float32) # List of nodes
        edgeList = np.empty((0,2),dtype=int) # List of edges

        # copy the data, ready for batch processing
        tempData = data.copy()

        # define the batch size
        batchSize = len(data) // number_of_starting_points

        for i in range(number_of_starting_points):
            idx = np.arange(len(tempData), dtype=int) # Index list for data
            
            # randomly select a node
            selectedIndex = np.random.choice(idx[-batchSize:]) # Randomly select an index
            currentNode = tempData[selectedIndex] # Choose a starting node
            
            # insert the node into the list
            nodeList = np.append(nodeList, [currentNode],axis=0)
            
            # calculate distance from all data to current node
            y2 = np.sum(np.square(tempData), axis=1)
            dot_product = 2 * np.matmul(currentNode,tempData.T)
            dist =  y2 - dot_product 
            idx = np.argsort(dist) # Sort distances to find nearest and farthest points
            
            # select the third closest node as neighbor to leave some space
            neighborNode = tempData[idx[2]] 
            nodeList = np.append(nodeList, [neighborNode],axis=0) # Add neighbor to node list
            
            # connect them and add to edge list
            edgeList = np.append(edgeList,[[i*2, i*2 + 1]],axis=0)
            
            # randomly select a node from the farthest nodes within batch size
            selectedIndex = np.random.choice(idx[-batchSize:])            
            currentNode = tempData[selectedIndex,:2]
            
            # remove the current area so it won't repeat in the next search
            idx = idx[batchSize:]
            tempData = tempData[idx]
        
        self.W = nodeList # Store nodes
        self.C = edgeList # Store edges
        self.E = np.zeros(len(self.W)) # Initialize error values
    
    # Input batch, feature -> performs one learning cycle with a batch of input data
    def batchLearning(self, X):
        X = X[:,:self.feature_number] # Use only relevant features
        
        # identity matrix
        i_adj = np.eye(len(self.W)) # Identity matrix for neighbor relations
        
        adj = np.zeros((len(self.W),len(self.W))) # Adjacency matrix (connections)

        # Set edges in both directions for undirected graph
        adj[self.C[:,0],self.C[:,1]] += 1 
        adj[self.C[:,1],self.C[:,0]] += 1 

        adj[adj > 0] = 1 # Make binary to get adjacency matrix
        
        batchIndices = np.arange(len(X))
        
        # compute distance
        x2 = np.sum(np.square(X), axis=1) # Squared norms for input
        y2 = np.sum(np.square(self.W), axis=1) # Squared norms for nodes            
        dot_product = 2 * np.matmul(X, self.W.T) # Dot product
      
        dist = np.clip(np.expand_dims(x2, axis=1) + y2 - dot_product, a_min=0, a_max = None) # Distance computation
        dist = np.sqrt(dist  + self.eps) # Prevent zero-division
        
        # get first and second winner nodes
        tempDist = dist.copy()                
        s1 = np.argmin(tempDist,axis=1) # Nearest node
        tempDist[batchIndices,s1] = 99999 # Set nearest distance to high value to find second nearest              
        s2 = np.argmin(tempDist,axis=1) # Second nearest node
        
        # add error to s1
        self.E += np.sum(i_adj[s1] * dist, axis=0) * self.alpha # Update error values
        
        # Update s1 position
        self.Delta_W_1 += (np.matmul(i_adj[s1].T, X) - (self.W.T * np.sum(i_adj[s1],axis=0)).T)  * self.alpha
        
        # Update s1 neighbor position        
        self.Delta_W_2 += (np.matmul(adj[s1].T, X) - np.multiply(self.W.T, adj[s1].sum(0)).T) * self.beta
    
        # Add 1 to s1 node activation
        self.A_1 += np.sum(i_adj[s1], axis=0) 
                
        # Add 1 to neighbor node activation
        self.A_2 += np.sum(adj[s1], axis=0) 
        
        # Count the important edge (s1 and s2)
        connectedEdge = np.zeros_like(self.S)              
        connectedEdge[s1,s2] = 1 
        connectedEdge[s2,s1] = 1 # undirected graph

        t = i_adj[s1] + i_adj[s2]
        connectedEdge *= np.matmul(t.T,t) # Update connection strength
        
        self.S += connectedEdge # Store new connections
        
    # Update the network after a learning cycle    
    def updateNetwork(self):
        self.W += (self.Delta_W_1.T * (1 / (self.A_1 + self.eps))).T + (self.Delta_W_2.T * (1 / (self.A_2 + self.eps))).T # Update node positions       
        self.C = np.asarray(self.S.nonzero()).T # Update edge list
        self.removeIsolatedNodes() # Remove isolated nodes
        self.E *= self.delta # Reduce error values
        if random.random() > 0.9: # Random condition for controlled cleanup
            self.removeNonActivatedNodes() # Remove non-activated nodes
    
    def removeIsolatedNodes(self):
        # Create adjacency matrix
        adj = np.zeros((len(self.W),len(self.W)))
        adj[self.C[:,0],self.C[:,1]] = 1
        adj[self.C[:,1],self.C[:,0]] = 1
        
        # Find isolated nodes
        isolatedNodes = (np.sum(adj, axis=0) + np.sum(adj, axis=1) == 0).nonzero()[0]
        finalDelete = list(np.unique(isolatedNodes))
        if len(finalDelete) > 1:            
            finalDelete.sort(reverse=True) # Sort descending to avoid index shift  
        
        # Remove isolated nodes from edge list
        for v in finalDelete:                       
            self.C = self.C[np.logical_not(np.logical_or(self.C[:,0] == v, self.C[:,1] == v))]
            self.C[self.C[:,0] > v,0] -= 1
            self.C[self.C[:,1] > v,1] -= 1       
            
        # Delete nodes from all relevant data structures
        if len(finalDelete) > 0:
            self.S = np.delete(self.S, finalDelete,axis=0)
            self.S = np.delete(self.S, finalDelete,axis=1)
            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)
        
    def removeNonActivatedNodes(self):
        # Find nodes that were not activated
        nodeActivation = self.A_1 
        nonActivatedNodes = (nodeActivation == 0).nonzero()[0]
        finalDelete = list(nonActivatedNodes)
        if len(finalDelete) > 1:                
            finalDelete.sort(reverse=True) # Sort descending
        
        # Remove non-activated nodes from edge list
        for v in finalDelete:    
            self.C = self.C[np.logical_not(np.logical_or(self.C[:,0] == v, self.C[:,1] == v))]
            self.C[self.C[:,0] > v,0] -= 1
            self.C[self.C[:,1] > v,1] -= 1           

        # Delete non-activated nodes from all relevant data structures    
        if len(finalDelete) > 0:
            self.S = np.delete(self.S, finalDelete,axis=0)
            self.S = np.delete(self.S, finalDelete,axis=1)
            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)
  
    def addNewNode(self, gng):
        # Calculate how many new nodes should be added based on error 'E'
        g = np.sum(self.E > np.quantile(gng.E,0.85)) # Number of nodes above 85th percentile
        
        for _ in range(g):
            if len(self.W) >= self.M:
                return
            
            q1 = np.argmax(self.E) # Node with highest error
            if self.E[q1] <= 0:          
                return
            
            # get connected nodes
            connectedNodes = np.unique(np.concatenate((self.C[self.C[:,0] == q1,1], self.C[self.C[:,1] == q1,0])))
            if len(connectedNodes) == 0:
                return
           
            # get neighbor with maximum error
            q2 = connectedNodes[np.argmax(self.E[connectedNodes])]
            if self.E[q2] <= 0:              
                return
          
            # insert new node between q1 and q2
            q3 = len(self.W) 
            new_w = (self.W[q1] + self.W[q2]) * 0.5       
            self.W = np.vstack((self.W, new_w))
            self.E = np.concatenate((self.E,np.zeros(1)),axis=0)
            
            # update error
            self.E[q1] *= self.rho
            self.E[q2] *= self.rho        
            self.E[q3] = (self.E[q1] + self.E[q2]) * 0.5 
            
            # remove original edge
            self.C = self.C[np.logical_not(np.logical_and(self.C[:,0] == q1, self.C[:,1] == q2))]
            self.C = self.C[np.logical_not(np.logical_and(self.C[:,0] == q2, self.C[:,1] == q1))]
            
            # add new edges
            self.C = np.vstack((self.C,np.asarray([q1,q3])))
            self.C = np.vstack((self.C,np.asarray([q2,q3])))
            self.S = np.pad(self.S, pad_width=((0, 1), (0, 1)), mode='constant') 
            self.S[q1,q2] = 0
            self.S[q2,q1] = 0
            self.S[q1,q3] = 1
            self.S[q3,q1] = 1
            self.S[q2,q3] = 1
            self.S[q3,q2] = 1  
            
            # update activations
            self.A_1 = np.concatenate((self.A_1,np.ones(1)),axis=0)  
            self.A_2 = np.concatenate((self.A_2,np.ones(1)),axis=0)  
        
    def cutEdge(self):
        # Remove non-activated nodes  
        self.removeNonActivatedNodes()
        
        # Create mask for existing edges
        mask = self.S > 0

        # Determine threshold for cutting edges
        filterV = np.quantile(self.S[mask], constants.EDGE_CUTTING) 

        # Copy adjacency matrix
        temp = self.S.copy()

        # Set weak connections to 0
        temp[self.S < filterV] = 0

        # Extract remaining edges
        self.C = np.asarray(temp.nonzero()).T
        
        # Remove isolated nodes created by cutting
        self.removeIsolatedNodes()
    
    # Compute the nearest node for each input datum
    def finalNodeDatumMap(self, X):
        X = X[:,:self.feature_number] # Keep only relevant features        
 
        # compute distance
        x2 = np.sum(np.square(X), axis=1)
        y2 = np.sum(np.square(self.W), axis=1)          
        dot_product = 2 * np.matmul(X, self.W.T)  

        # Euclidean distance matrix
        dist = np.clip(np.expand_dims(x2, axis=1) + y2 - dot_product, a_min=0, a_max = None)
        dist = np.sqrt(dist  + self.eps) 
        # dist: rows = data, columns = nodes
        
        tempDist = dist.copy()                
        s1 = np.argmin(tempDist,axis=1) # Index of nearest node
        self.finalDistMap = s1 # Save mapping from each datum to nearest node
