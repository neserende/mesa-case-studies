import mesa
from mesa.experimental.cell_space import CellAgent, Network 
import networkx as nx
import numpy as np
from enum import Enum
import os
import json

def number_state(model, state):
    return len(model.agents.select(lambda a: a.state == state))

def number_spreader(model):
    return number_state(model, State.ACTIVE_SPREADER)

def number_active(model):
    return number_state(model, State.ACTIVE)

def number_inactive(model):
    return number_state(model, State.INACTIVE)

def calculate_total_active(model):
    return number_spreader(model) + number_active(model)

def save_graph(epoch_num, current_graph, file_name):
        path = os.path.join(os.getcwd(), file_name)
        data = nx.node_link_data(current_graph, edges="links")

        # Load existing data if the file already exists
        try:
            with open(path, 'r') as f:
                existing_data = json.load(f)
                # print(existing_data)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # Add new data to the existing data
        existing_data[epoch_num] = data

        # Write back to file
        with open(path, 'w') as f:
            json.dump(existing_data, f, indent=4)

def save_datacollector(data_frame):
        path = os.path.join(os.getcwd(), "datacollector.json")
        data = data_frame.to_json(orient="split")

        # Write back to file
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

class State(Enum):
    ACTIVE_SPREADER = 0
    ACTIVE = 1
    INACTIVE = 2

class IndependentCascadeModel(mesa.Model):
    def __init__(
            self,
            initial_active_count = 100,
            initialization_method = "pagerank",
            treshold = 1,
            input_network=None,
            ):
        # Step 1: Initialize the Network environment
        super().__init__()
        self.grid = Network(input_network, random=self.random)

        # Step 2: Set the edge parameter using in degree of nodes
        # Using weighted cascade for probability assignment where
        # p(u, v) = 1/in_degree(v) or treshold/in_degree(v)

        graph = self.grid.G
        for u, v in graph.edges():
            graph[u][v]["activation_prob"] = treshold/graph.degree[v]

        # Step 3: Select the seed nodes with the given metric and k 
        seed_nodes = self.select_seed_nodes(initialization_method, initial_active_count)
        
        # Step 4: Create agents and assign the seed nodes as active spreaders, others are inactive
        for position in list(graph):
            if position in seed_nodes:
                agent = ICMAgent(
                    model=self,
                    initial_state=State.ACTIVE_SPREADER,
                    # nx_index = position
                )
            else:
                agent = ICMAgent(
                    model=self,
                    initial_state=State.INACTIVE,
                    # nx_index = position
                )
            
            # Add the agent to grid
            agent.move_to(self.grid[position])

        # Initialize data collector 
        self.datacollector = mesa.DataCollector(
            {
                "active_spreader": number_spreader,
                "active": number_active, 
                "inactive": number_inactive,
                "total_active": calculate_total_active
            }
        )

        self.activated_agents = []

        # Others
        self.running = True
        self.datacollector.collect(self)
        
        
    def select_seed_nodes(self, metric, k):
        graph = self.grid.G

        if metric == 'degree':
            centrality = nx.degree_centrality(graph)
        elif metric == 'pagerank':
            # print("Before pagerank")
            centrality = nx.pagerank(graph)
            # print("Page rank results", centrality)
        elif metric == 'betweenness':
            centrality = nx.betweenness_centrality(graph)
        elif metric == 'closeness':
            centrality = nx.closeness_centrality(graph)
        elif metric == 'eigenvector':
            centrality = nx.eigenvector_centrality(graph)
        elif metric == 'katz':
            centrality = nx.katz_centrality(graph)
        else:
            print("Unknown metric", metric)
            raise ValueError(f"Unknown metric: {metric}")

        # print("Before sorted nodes")
        sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        # print("sorted nodes", sorted_nodes)
        top_k_nodes = [node for node, _ in sorted_nodes[:k]]

        # print("Top k nodes:", top_k_nodes)
        return top_k_nodes
      
    
    def step(self, epoch):
        # spread information
        # self.spread_information()
        self.activated_agents = []
        self.agents.shuffle_do("step")
        self.agents.do("update_agent_state")

        # collect data
        self.datacollector.collect(self)

        # write graph to file
        # currently, since the states are not saved in graph nodes but in agent variables, this does not hold much value
        # but we still do this operation to compare the execution time with crowd
        # save_graph(epoch, self.grid.G, "graph.json")
    
    def run(self, n):
        """Run the model for n steps."""
        save_graph(0, self.grid.G, "graph.json")
        for epoch in range(n):
            self.step(epoch)  
        # After the run is completed, save the values from data collector
        save_graph(n-1, self.grid.G, "graph.json")
        save_datacollector(self.datacollector.get_model_vars_dataframe())


    
class ICMAgent(CellAgent):
    def __init__(self, 
                 model, 
                 initial_state, 
                #  nx_index
                ):
        super().__init__(model)
        self.state = initial_state
        # self.nx_index = nx_index
        # if self.state == State.ACTIVE_SPREADER:
        #     self.countdown = 1
        # else:
        #     self.countdown = -1

    def update_agent_state(self):
        if self.state == State.ACTIVE_SPREADER:
            # if self.countdown == 0:
            self.state = State.ACTIVE
            # else:
            #     self.countdown -= 1
        elif self.state == State.INACTIVE and self in self.model.activated_agents:
            self.state = State.ACTIVE_SPREADER

    def step(self):
        # print("pos: ", self.pos, "typeof pos: ", type(self.pos))
        # print("unique_id: ", self.unique_id)
        # print("coordinate:", self.cell.coordinate, " type of coordinate:", type(self.cell.coordinate))
        # print("self.state", self.state, "graph[coordinate].state", self.model.grid.G[self.cell.coordinate])
        # print("self.nx_index", self.nx_index)
        
        # neighbors try to influence this node
        neighbors = [agent for agent in self.cell.neighborhood.agents if agent is not self]

        graph = self.model.grid.G

        # loop over neighbors
        for v in neighbors:
            if v.state == State.ACTIVE_SPREADER:                
                if (self.cell.coordinate, v.cell.coordinate) in graph.edges():
                    activation_prob = graph[self.cell.coordinate][v.cell.coordinate]["activation_prob"] 
                else:
                    activation_prob = graph[v.cell.coordinate][self.cell.coordinate]["activation_prob"]

                #Generate a random number
                rand = self.random.random()

                if activation_prob >= rand: 
                    self.model.activated_agents.append(self)
                    return 

