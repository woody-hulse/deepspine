import torch.nn.functional as F

def poisson_nll_loss(output, target):
    loss = F.poisson_nll_loss(output, target)
    return loss
    
def l1_loss(output, target):
    loss = F.l1_loss(output, target)
    return loss

def smooth_l1_loss(output, target):
    loss = F.smooth_l1_loss(output, target)
    return loss

def mse_loss(output, target):
    loss = F.mse_loss(output, target)
    return loss

def binary_cross_entropy(output, target):
    loss = F.binary_cross_entropy(output, target)
    return loss

def masked_l1_loss(output, target, mask):
    loss = F.l1_loss(output, target, reduction='none')
    loss = loss * mask
    loss = loss.sum() / mask.sum()#.item()
    return loss

def masked_mse_loss(output, target, mask):
    loss = F.mse_loss(output, target, reduction='none')
    loss = loss * mask
    loss = loss.sum() / mask.sum()#.item()
    return loss

def masked_pearson_correlation_loss(output, target, mask, return_loss=True):
    output_dim = output.size(1)
    T = output.size(2)

    output = output.view(-1, T)                             # [(batch x dim) x time]
    target = target.view(-1, T)                             # [(batch x dim) x time]
    mask = mask.repeat(1, output_dim, 1).view(-1, T)        # [(batch x dim) x time]
    
    length = mask.sum(1, keepdims=True)                     # [(batch x dim) x 1]
    output_mean = output.sum(1, keepdims=True) / length     # [(batch x dim) x 1]
    target_mean = target.sum(1, keepdims=True) / length     # [(batch x dim) x 1]

    centered_output = output - output_mean
    centered_target = target - target_mean
    output_std = torch.sqrt(torch.sum(centered_output ** 2, dim=1))
    target_std = torch.sqrt(torch.sum(centered_target ** 2, dim=1))

    cov = torch.sum(centered_output * centered_target, dim=1)
    r = cov / (output_std * target_std)                     # [(batch x dim)]
    r = r.mean()

    if return_loss:
        loss = -r
        return loss
    else:
        return r


    # for n in range(input.size(1)):
    #     masked_input = torch.masked_select(input[:,n], mask.byte())
    #     masked_target = torch.masked_select(target[:,n], mask.byte())
    #     # print(n, torch.mean(masked_input).item(), torch.mean(masked_target).item())

    #     mean_input = torch.mean(masked_input)
    #     mean_target = torch.mean(masked_target)

    #     centered_input = masked_input - mean_input
    #     centered_target = masked_target - mean_target

    #     std_input = torch.sqrt(torch.sum(centered_input ** 2))
    #     std_target = torch.sqrt(torch.sum(centered_target ** 2))
    #     cov = torch.sum(centered_input * centered_target)
        
    #     _loss = cov / (std_input * std_target)
    #     loss += _loss

    # loss = -loss / input.size(1)
    # return loss