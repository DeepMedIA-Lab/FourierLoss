"""
epoch: the current epoch of training
loss_func: fourier_loss(nn.Module)
loss_func.weights: coefficient of fourier_loss function
warmup_length: the determined warm-up period for training
lr_fourier: the learning rate of the Fourier coefficients
"""
if  epoch >= warmup_length:
    with torch.no_grad():
        loss_func.weights += lr_fourier * loss_func.weights.grad
        loss_func.weights.grad.zero_()