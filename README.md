# Early Stopping for deep learning
Overfitting occurs when a model fits too well on training dataset and fails to predict unseen data. Training deep learning model is susceptible to overfitting as the learning strategy relies upon passing over the training dataset multiple times to obtain optimal weights via gradient descent.

Early stopping helps reduce overfitting in training deep learning models by stopping the fitting procedure early (as its name suggests). The idea is to monitor of some validation quantity while training the model and terminate the training process if the said quantity doesn't improve after some iterations.

This repo provides an implementation of early stopping to train [PyTorch](https://pytorch.org/) model. The code was adapted from [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch). Our contributions are:
- User can select any validation metric (loss, accuracy rate, precision, etc..) of interest. 
- Using multiple validation metrics is also supported. 

If you use other deep learning frameworks, you can easily modify the `save_checkpoint` function to adapt this implementation for your framework.

## Usage
The `EarlyStopping` class allows users to generate object to keep track of validation quantity while training your model. Useful arguments are provided:
- `patience`: Number of iterations with no improvement in validation quantity, after which training will be stopped.  
- `delta`: Minimum percentage change in the monitored quantity to qualify as an improvement.
- `**metric_direction`: dictionary whose keywords are names of metrics used for early stopping. Values are direction in ['low'/'high']. Use 'low' if a small quantity of metric, is desirable for training and vice versa. E.g: loss = 'low', acc = 'high'.

We provide an example of how to use the ```EarlyStopping``` class in training graph neural networks in the [GNNs_early_stopping_example](./early-stopping/GNNs_early_stopping_example.ipynb) notebook. 