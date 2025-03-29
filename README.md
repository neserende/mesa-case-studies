# mesa-case-studies

This repository includes the implementation of 3 case studies included in the paper, "Crowd: A Social Network Simulation Framework".

Citations to the original work of case studies 1 and 3:

- R. Williams, N. Hosseinichimeh, A. Majumdar and N. Ghaffarzadegan,
  ”Epidemic modeling with generative agents,” 2023, arXiv:2307.04986.

- M. Chica, R. Chiong, M. Kirley and H. Ishibuchi, ”A networked Nplayer
  trust game and its evolutionary dynamics,” in IEEE Trans. Evol.
  Comput., vol. 22, no. 6, 2017, pp. 866-878.

In our paper, we provide a comparison of execution times between Crowd and Mesa. The features such as writing the graph and collected data to a file are features of Crowd that are not optional in the current version (v0.9). Therefore, we also perform these operations in the Mesa implementation to provide a comparison as close as possible.

The original code for case study 1 was also implemented with Mesa. We have changed and simplified various parts according to the previous implementation made on Crowd framework.
