import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torchviz import make_dot
import numpy as np

REPLACES = [
    [".weight",'.W'],
    ['.bias','.B'],
    ['edge_encoder','EE'],
    ['models.0.',''],
    ['model.','M.'],
    ['pre_nns','R'],
    ['post_nns','O'],
    ['lin','L']
]

def plot_grad_flow(named_parameters,name='plot_grad_flow',save=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and (p.grad is not None):
            for r in REPLACES:
                n = n.replace(r[0],r[1])
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=13,fontsize=8)
    plt.xlim(left=-0.5, right=len(ave_grads)-0.5)
    l = max(1e-3,np.array(max_grads).max().cpu()) # 
    plt.ylim(bottom = -0.01*l, top=1.05*l) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    if save is not None:
        plt.savefig(save + '__temp__'+name+'.png')
        plt.clf()

def plot_computation_graph(output,pl,name='plot_computation_graph'):
    make_dot(output,params=None if pl is None else dict(pl)).render("__temp__"+name, format="png")
