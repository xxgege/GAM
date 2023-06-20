# GAM Optimizer
The official repository for CVPR2023 **highlight** paper "Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization" ([full paper](https://arxiv.org/abs/2303.03108)).

Acknowledgment: This repository is partially based on [https://github.com/davda54/sam](https://github.com/davda54/sam) and [https://github.com/juntang-zhuang/GSAM](https://github.com/juntang-zhuang/GSAM/).

# Introduction
Recently, flat minima are proven to be effective for improving generalization and sharpness-aware minimization (SAM) achieves state-of-the-art performance. Yet the current definition of flatness discussed in SAM and its follow-ups are limited to the zeroth-order flatness (i.e., the worst case loss within a perturbation radius). We show that the zeroth-order flatness can be insufficient to discriminate minima with low generalization error from those with high generalization error both when there is a single minimum or multiple minima within the given perturbation radius. Thus we present first-order flatness, a stronger measure of flatness focusing on the maximal gradient norm within a perturbation radius which bounds both the maximal eigenvalue of Hessian at local minima and the regularization function of SAM. We also present a novel training procedure named Gradient norm Aware Minimization (GAM) to seek minima with uniformly small curvature across all directions. Experimental results show that GAM improves the generalization of models trained with current optimizers such as SGD and AdamW on various datasets and networks. Furthermore, we show that GAM can help SAM find flatter minima and achieve better generalization.

# Further acceleration of GAM
As shown in Appendix D in [full paper](https://arxiv.org/abs/2303.03108), optimizing the gradient of $R^{(1)}_{\rho}(\theta)$ according to Equation 7 and Equation 8 in the paper requires the Hessian vector product operation, which can still introduce considerable extra computation when the model is large. We approximate $\nabla\lVert\nabla \hat{L}(\theta)\rVert$ with first-order gradient as follows. 

$$\nabla\left\lVert\nabla \hat{L}(\theta)\right\rVert \approx \frac{\nabla \hat{L}\left(\theta + \rho' \cdot \frac{\nabla \hat{L}(\theta)}{\lVert\nabla \hat{L}(\theta)\rVert}\right) - \nabla \hat{L}(\theta)}{\rho'} ,$$

where $\rho'$ is a small constant.

Then GAM can be implemented as follows. Please see detailed interpretation and derivation in Appendix D in [full paper](https://arxiv.org/abs/2303.03108). ![GAM algorithm](/images/gam_algorithm.png?raw=true "Title") We find that the accelerated GAM achieves comparable performance compared with the original version of GAM but shows better scalability, so we release the accelerated GAM in this repository. 

# Hyperparameters
As shown in Appendix D in [full paper](https://arxiv.org/abs/2303.03108), the hyperparameters for accelerated GAM are different compared with the original version of GAM. Accelerated GAM has 5 hyperparameters, namely $\rho_t$ (args.grad_norm_rho in code), $\rho'_t$ (args.grad_rho), $\alpha$ (args.grad_beta_1), $\beta$ (args.grad_beta_0), and $\gamma$ (args.grad_gamma). We give the default choice of them (roughly searched) for CIFAR-10 and CIFAR-100 in main_cifar.py. We find that GAM is relatively robust to the choice of hyperparameters, yet carefully tuned hyperparameters can lead to further improvement on various tasks.

# How to use GAM
Basically, GAM can be used as a current PyTorch optimizer with a few extra lines for base optimizer initialization and set_closure. Please see the following code for the usage of GAM. Required additional lines are highlighted with bold font.

```python
# optimizer initialization
from gam import GAM
# initialize a base optimizer, such as SGD, Adam, AdamW ...
base_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# initialize GAM optimizer based on the base optimizer
gam_optimizer = GAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, args=args)

# training
# define the loss function for loss and gradients
def loss_fn(predictions, targets):
    return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()

for inputs, labels in train_loader:
    # get inputs and labels
    inputs = inputs.cuda()
    labels = target.cuda()

    # GAM sets closure and automatically runs predictions = model(inputs), loss = loss_fn(predictions, targets), loss.backward() in it 
    gam_optimizer.set_closure(loss_fn, inputs, labels)

    # update model parameters
    predictions, loss = gam_optimizer.step()

```


# Citing GAM
If you find this repo useful for your research, please consider citing the paper.
```
@inproceedings{zhang2023gradient,
  title={Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization},
  author={Zhang, Xingxuan and Xu, Renzhe and Yu, Han and Zou, Hao and Cui, Peng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20247--20257},
  year={2023}
}
```
