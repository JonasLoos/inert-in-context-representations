# Do Language Models Struggle to Use Representations Learned In-Context?

A partial replication and slight extension of the paper [Language Models Struggle to Use Representations Learned In-Context](https://arxiv.org/abs/2602.04212) by Lepori et al. (Google Deepmind).



## Results

Below are the preliminary results for `gemma-3-4b-it` (grid size: 4, walk length: 200, #trials: 50)

```
condition              accuracy  parse rate
-------------------------------------------
instruction (paper)     26%       82%      
prefilled (paper)       96%      100%      
multi-turn              44%      100%      
multi-turn-1-example    64%      100%      
multi-turn-2-examples   72%      100%      
turn-by-turn            74%      100%      
```
