>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, BibTeX entry, link to demos, blog posts and tutorials

todo

## Requirements

To install requirements:
todo
```setup
pip install -r requirements.txt
```

>📋  A licensed gurobi solver installation is required for both data preparation and training. For an educational license see: https://www.gurobi.com/academia/academic-program-and-licenses/
>

## Training and Evaluation

To train the model(s) in the paper, run this command:

```train and evaluate
We integrate the training and evaluation processes into the main_nips.py file
python main_nips.py
```

>📋  The experiments can be reproduced by importing different config file: e.g., import .nips_configs.synthetic.cj, will run the experiment synthetic data with all

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
