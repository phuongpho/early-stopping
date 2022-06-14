import numpy as np
import torch

class EarlyStopping:
    def __init__(self, path, patience=10, verbose=False, delta = 0.0, **metric_direction):
        '''
        Args:
            path (str): Path to save model's checkpoint.
            patience (int): How long to wait after last time validation loss (or acc) improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss (or acc) improvement. 
                            Default: False
            delta (float): Minimum percentage change in the monitored quantity (either validation loss or acc) to qualify as an improvement.
                            Default: 0.0
            **metric_direction (dict): Keywords are names of metrics used for early stopping. Values are direction in ['low'/'high']. Use 'low' if a small quantity of metric,
                            is desirable for training and vice versa. E.g: loss = 'low', acc = 'high'. If not provided, use loss = 'low'
        '''
        
        if metric_direction:
            print(f'Selected metric for early stopping: {metric_direction}')
        else:
            raise ValueError("No metric provided for early stopping")

        # unpacking keys into list of string
        self.metric_name = [*metric_direction.keys()]
        # choose comparison operator w.r.t metric direction: low -> "<"; high -> ">"
        self.metric_operator = [np.less if dir == 'low' else np.greater for dir in metric_direction.values()]
        self.patience = patience
        # assign delta sign to compute reference quantity for early stopping
        self.delta = [-delta if dir == 'low' else delta for dir in metric_direction.values()]
        self.counter = 0
        self.best_score = [None]*len(metric_direction.keys())
        self.best_epoch = None
        self.lowest_loss = None
        self.path = path
        self.verbose = verbose
        self.early_stop = False
          
    def __call__(self, model, epoch, **metric_value):
        '''
        Args:
            metric_value: Keywords are names of metrics used for early stopping. Values are metrics's value obtained during training.
        '''
        if metric_value:
            # Check name of metric
            if set(self.metric_name) != set(metric_value.keys()):
                raise ValueError("Metric name is not matching")
        else:
            raise ValueError("Metric value is missing")
        
        score = [metric_value[key] for key in self.metric_name if key in metric_value]
        
        # if any metric is none, return true
        is_none = any(map(lambda i: i is None,self.best_score))
        
        if is_none:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        else:
            # score condition: if any metric is getting better, save model. 
            # getting better means scr is less(greater) than best_scr*[1-(+)delta/100]
            score_check = any(map(lambda scr,best_scr, op, dlt: op(scr, best_scr*(1+dlt/100)), score, self.best_score, self.metric_operator, self.delta))
            
            if score_check:
                self.best_score = score
                self.best_epoch = epoch
                self.save_checkpoint(model)
            else:
                self.counter += 1
                if self.counter >= 0.8*(self.patience):
                    print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            return self.early_stop

    def save_checkpoint(self, model):
        '''
        Saves model when score condition is met, i.e loss decreases 
        or acc increases
        '''
        if self.verbose:
            message = f'Model saved at epoch {self.best_epoch + 1}.'
            score = self.best_score
            
            if len(self.metric_name) > 1:
                for i,nm in enumerate(self.metric_name):
                    message += f' {nm}={score[i]:.4f}'
                
                print(message)
            else:
                print(f'{message} {self.metric_name[0]}={score[0]:.4f}')
        # Save model state
        torch.save({
            'state_dict':model.state_dict()
        }, self.path)
