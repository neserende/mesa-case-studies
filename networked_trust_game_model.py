import mesa
from mesa.experimental.cell_space import CellAgent, Network 
import networkx as nx
import numpy as np
from enum import Enum
import os
import json

def number_state(model, state):
    return len(model.agents.select(lambda a: a.state == state))

def number_investors(model):
    return number_state(model, State.INVESTOR)

def number_trustable(model):
    return number_state(model, State.TRUSTABLE)

def number_untrustable(model):
    return number_state(model, State.UNTRUSTABLE)

def get_global_payoff(model):
    return model.global_payoff

class State(Enum):
    INVESTOR = 0
    TRUSTABLE = 1
    UNTRUSTABLE = 2

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

class TrustGameModel(mesa.Model):
    
    def __init__(
            self, 
            initial_i_ratio=0.3, 
            initial_t_ratio=0.25, 
            initial_u_ratio=0.45, 
            r_T=6, 
            r_UT=0.4,
            input_network=None
            ):
        super().__init__()
        # self.G = input_network
        # input_network = nx.read_edgelist("m3.edgelist", nodetype=int)
        # input_network = nx.read_edgelist("SF-4900nodes_m_3_0.edgelist", nodetype=int)
        input_network = nx.read_edgelist("SF-14400nodes_m_3_0.edgelist", nodetype=int)
        self.grid = Network(input_network, random=self.random)
        
        # Node types' initial ratios, sums up to 1
        self.initial_i_ratio = initial_i_ratio
        self.initial_t_ratio = initial_t_ratio
        self.initial_u_ratio = initial_u_ratio

        # Initializing model level parameters (network parameters)
        self.r_T = r_T # Multiplier of what is recieved by T from I
        self.r_UT = r_UT # Temptation to defect ratio. Range: (0, 1). 
        self.r_U = (1 + self.r_UT) * self.r_T
        self.min_payoff = -1
        self.max_payoff = self.init_max_payoff()
        self.global_payoff = 0

        # Initialize data collector
        self.datacollector = mesa.DataCollector(
            {
                "I": number_investors, 
                "T": number_trustable, 
                "U": number_untrustable, 
                "global_payoff": get_global_payoff,
            }
        )

        # Create agents
        self.node_count = self.grid.G.number_of_nodes()
        random_nodetypes = []
        init_i_count = int(round(self.initial_i_ratio * self.node_count))
        init_t_count = int(round(self.initial_t_ratio * self.node_count))
        init_u_count = int(round(self.initial_u_ratio * self.node_count))

        # if less than node count assigned, add 1 of last type
        if (init_i_count + init_t_count + init_u_count) < self.node_count: 
            init_u_count += 1
        # if more than node count assigned, take out the last element
            init_u_count -= 1

        random_nodetypes.extend([ State.INVESTOR ] * init_i_count)
        random_nodetypes.extend([ State.TRUSTABLE ] * init_t_count)
        random_nodetypes.extend([ State.UNTRUSTABLE ] * init_u_count)
        
        self.random.shuffle(random_nodetypes)

        list_of_random_nodes = self.random.sample(list(input_network), input_network.number_of_nodes())

        # Create agents
        i = 0
        for position in list_of_random_nodes:
            agent = TrustGameAgent(
                    model=self, 
                    initial_state=random_nodetypes[i]
                )

            # Add the agent to a random node
            agent.move_to(self.grid[position])
            i+= 1 # Increment i for next iteration
       
        # Others
        self.running = True
        self.datacollector.collect(self)

    # Helper methods for initialization
    def init_max_payoff(self):
        avg_degree = sum( dict( self.grid.G.degree() ).values() ) / self.grid.G.number_of_nodes() if self.grid.G.number_of_nodes() > 0 else 0
        max_payoff = avg_degree * self.r_U
        return max_payoff


    def step(self):
        self.global_payoff = 0
        self.agents.shuffle_do("step")
        # After we finish proportional imitation for each agent, then we do payoff calculation
        # Order of execution actually doesnt matter here, since we are not changing any of the current states
        self.agents.do("calculate_payoffs_with_neighbors")
        # self.datacollector.collect(self)

    def run(self, n):
        """Run the model for n steps."""
        save_graph(0, self.grid.G, "graph.json")

        for _ in range(n):
            self.step()
        # After the run is completed, save the network and the values from data collector
        self.datacollector.collect(self)
        save_graph(n-1, self.grid.G, "graph.json")
        save_datacollector(self.datacollector.get_model_vars_dataframe())

class TrustGameAgent(CellAgent):
    def __init__(self, 
                 model, 
                 initial_state
                ):
        super().__init__(model)
        self.state = initial_state
        self.prev_strategy = "undefined_strategy"
        self.current_payoff = -100 # initially setting as a small value
        self.previous_payoff = -100 # initially setting as a small value

    def step(self):
        self.proportional_imitation()

    def proportional_imitation(self):
        neighbors = [agent for agent in self.cell.neighborhood.agents if agent is not self]

        if len(neighbors) > 0:

            # Choose a random neighbor, j
            j = self.random.choice(neighbors)

            neigh_payoff = j.previous_payoff
            focal_agent_payoff = self.previous_payoff

            # max payoff is a network level parameter we have set previously
            # If neighbor's payoff is higher than the limit, set it back to max possible
            if neigh_payoff > self.model.max_payoff:
                neigh_payoff = self.model.max_payoff          

            # Compare neighbor's agent current agent's payoffs in the previous step/epoch
            if neigh_payoff > focal_agent_payoff:
            
                # The neighbor's strategy was better so we change the strategy with a probability
                prob = (neigh_payoff - focal_agent_payoff) / (self.model.max_payoff - self.model.min_payoff)

                # Generate a random number
                rand = self.random.random()

                # If number < probability, we current agent changes its strategy
                if rand < prob:
                    # In this simulation, strategies are saved as the states of the agent, in other words, the current node type
                    # So, we set the current state to neighbor agent's state
                    self.state = j.state


    def calculate_payoffs_with_neighbors(self):
        # increase global payoff of model here
        # include curr agent in neighbors
        neighbors = [agent for agent in self.cell.neighborhood.agents]

        count_I = len([
            agent
            for agent in neighbors
            if agent.state is State.INVESTOR
        ])

        count_T = len([
            agent
            for agent in neighbors
            if agent.state is State.TRUSTABLE
        ])

        count_U = len([
            agent
            for agent in neighbors
            if agent.state is State.UNTRUSTABLE
        ])

        curr_payoff = 0
        denom = count_T + count_U

        if denom > 0: # if there are any trustees
            match self.state:
                case State.INVESTOR:
                    curr_payoff = (self.model.r_T * (count_T / denom)) - 1
                case State.TRUSTABLE:
                    curr_payoff = self.model.r_T * (count_I / denom)
                case State.UNTRUSTABLE:
                    curr_payoff = self.model.r_U * (count_I / denom)
        
        # Set the previous and current payoff of this agent
        self.previous_payoff = self.current_payoff
        self.current_payoff = curr_payoff

        # Setup for the next iteration
        self.prev_strategy = self.state 

        # Update the model's global payoff parameter
        self.model.global_payoff += self.current_payoff
        