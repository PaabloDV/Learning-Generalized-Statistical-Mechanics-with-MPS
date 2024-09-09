# Learning-Generalized-Statistical-Mechanics-with-MPS
Generative algorithm based on Matrix Product States to reproduce generalized statistical mechanics. The code in this repository serves as complementary material to the letter ‘Learning Generalized Statistical Mechanics with Matrix Product States’, where more details and results of the algorithm can be found.

This repository contains the following material
  - Folder "Q2MPS": contains all the code to run the variational algorithm. It also includes functions and classes that allow us to reproduce the analytical and numerical approximations of the Tsallis statistic and other results of the paper. 
  - Folder "Problems": contains the randomly generated J-matrices of the Ising model instances used to obtain the results of the manuscript.
  - Jupyter notebook "Algorithm_execution_example.ipynb": example of how to generate generalized thermal distributions using the algorithm.

Main libraries used (python 3.11.4):
  - numpy==1.25.2
  - opt-einsum==3.3.0 
  - scipy==1.11.2
  - seemps==1.2
