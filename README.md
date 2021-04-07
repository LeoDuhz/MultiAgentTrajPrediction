# RCAR2021_MultiAgentTrajPrediction
code and materials for Multi-Agent Trajectory Predition Based on Graph Neural Network



## Heterogeneous GNN

- `main.py`: main function of LOG Analysis model training process

- `parameters.py`: parameters management of the training process

- `dataFormat.py`: basic data format of player, ball and game data

- `referee.py`: referee command defined in proto file

- `dataPreprocess.py`: preprocess the text file we get from our vision module, converting into formatted data we can use and doing Min-Max normalization

- `SSLDataset.py`: construct the graph structure for future gnn training

- `mys2v.py`: basic graph neural network we use 

- `Net.py`: neural network we construct

- `visualize.py`: draw pictures of our training result

- `testOurModel.py`: load torch model and see results and also visualiztion

  