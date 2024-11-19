# HSRMNet


## Abstract




## Environment
    python 3.11
    pytorch 2.1.0

## Dataset
There are five datasets (from [CyGNet](https://github.com/CunchaoZ/CyGNet.)): ICEWS18, ICEWS14, GDELT, WIKI, and YAGO. Times of test set should be larger than times of train and valid sets. (Times of valid set also should be larger than times of train set.) Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt'.

## Run the experiment
We first get the historical vocabulary.

    python get_historical_vocabulary.py --dataset DATA_NAME
Then, train the model.

     python train.py --dataset ICEWS18 --entity object --time-stamp 24 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 0 --batch-size 1023 --counts 2 --valid-epoch 5 --his_len 11 
     python train.py --dataset ICEWS18 --entity subject --time-stamp 24 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 1 --batch-size 1023 --counts 2 --valid-epoch 5 --his_len 11 
    
    python train.py --dataset ICEWS14 --entity object --time-stamp 24 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 0 --batch-size 1024 --counts 2 --valid-epoch 5 --his_len 11 
    python train.py --dataset ICEWS14 --entity subject --time-stamp 24 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 1 --batch-size 1024 --counts 2 --valid-epoch 5 --his_len 11 
    
    python train.py --dataset GDELT --entity object --time-stamp 15 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 0 --batch-size 1024 --counts 3 --valid-epoch 5 --his_len 11 
    python train.py --dataset GDELT --entity subject --time-stamp 15 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 1 --batch-size 1024 --counts 3 --valid-epoch 5 --his_len 11 
    
    python train.py --dataset YAGO --entity object --time-stamp 1 --alpha 0.3 --lr 0.001 --n-epoch 50 --hidden-dim 200 --gpu 0 --batch-size 1024 --counts 3 --valid-epoch 5 --his_len 11 
    python train.py --dataset YAGO --entity subject --time-stamp 1 --alpha 0.3 --lr 0.001 --n-epoch 50 --hidden-dim 200 --gpu 1 --batch-size 1024 --counts 3 --valid-epoch 5 --his_len 11
    
    python train.py --dataset WIKI --entity object --time-stamp 1 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 0 --batch-size 1024 --counts 2 --valid-epoch 5 --his_len 15
    python train.py --dataset WIKI --entity subject --time-stamp 1 --alpha 0.3 --lr 0.001 --n-epoch 30 --hidden-dim 200 --gpu 1 --batch-size 1024 --counts 2 --valid-epoch 5 --his_len 15

Finally, test the model.

    python test.py --dataset DATA_NAME

## Reference


​    
