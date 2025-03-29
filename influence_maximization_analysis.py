from influence_maximization.inf_max_model import IndependentCascadeModel
import pandas as pd
import networkx as nx
import time


def run_model(model):
    """
    Run an experiment with a given model, and plot the results.
    """
    # print("Initial graph")
    # draw_space(model.grid, agent_portroyal)

    model.run(20)

start = time.time()
# network_df = pd.read_csv("facebook_combined.txt", sep=" ")
network_df = pd.read_csv("musae_git_edges.csv", sep=",")
# network_df = pd.read_csv("large_twitch_edges.csv", sep=",")
headers = network_df.columns.to_list()
network = nx.from_pandas_edgelist(network_df, source=headers[0], target=headers[1], create_using=nx.Graph())


ic_model = IndependentCascadeModel(
    initial_active_count=100,
    initialization_method="pagerank",
    treshold=1,
    input_network = network
)

run_model(ic_model)
end = time.time()
print("Total time to run the simulation: ", end-start, " seconds")

