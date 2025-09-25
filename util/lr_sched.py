import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.1, patience=3):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.last_loss = float('inf')

    def step(self, loss):
        
        if loss < 1:
            if self.optimizer.param_groups[0]['lr'] > 1e-5:
                for param_group in self.optimizer.param_groups:
                    #param_group['lr'] *= self.factor
                    #param_group['lr'] *= self.factor
                    param_group['lr'] = 1e-5
            #print(f"lr change to : {self.optimizer.param_groups[0]['lr']}")
        
        
        self.last_loss = loss