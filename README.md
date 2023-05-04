# On the Expressivity Role of LayerNorm in Transformers' Attention

This repository contains the code for reproducing the results shown in "On the Expressivity Role of LayerNorm in Transformers' Attention".

![alt text](images/figure1.png "Figure 1 from the paper")

## Setup

Make sure you have [wandb.ai](wandb.ai]) user and that you are [logged](https://docs.wandb.ai/ref/cli/wandb-login) into your machine.

Install the required python packages:
```
pip install -r requirements.txt 
```

Gurobi is needed for finding unselectable keys, and it requires a license. See in [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Hardware
In general, all experiments can run on either GPU or CPU. 

## Code Structure

1. The subdirectory `majority` contains the needed files to reproduce the results of the Majority task (Figure 1a, 1b, 2).
2. The subdirectory `unselectable` contains the needed files to reproduce the results of the unselectable experiments (Figure 1c, 1d, Table 1, 2, 3, 4).
