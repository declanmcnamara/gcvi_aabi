# Linearized Neural Networks

This repository contains code illustrating the results of fitting a simple neural network compared to fitting its linearization about its initial weights to minimize the expected forward KL objective function. The results are used in the workshop manuscript "Globally Convergent Variational Inference" at the 6th Symposium on Advances in Approximate Bayesian Inference (AABI), Vienna, Austria, 2024. 

To produce the figures, run the following:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd simulation_study
chmod +x runner.sh
./runner.sh
```
