# 0.ppFDetector
## A Generative Adversarial Network-based Fault Detection Ap-proach for Photovoltaic Panel

# 1.Installation
#### conda create -n ppFDetector python=3.7
#### conda activate ppFDetector
#### pip install --user --requirement requirements.txt

# 2.Train and test
#### python train.py                     
####   --dataset <name-of-the-data>    
####   --isize <image-size>            
####   --niter <number-of-epochs>      
####   --display                       ## optional if you want to visualize
  
####  Example: python  train.py --dataset ./gan --display
