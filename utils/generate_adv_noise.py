import torch
import torch.nn.functional as F

def generate_adv_img(img,gt,model,epsilon=0.05,iteration=1):
    if epsilon==0:
        return img

    batchsize, _, height, weight = gt.shape
    count = 0

    img_adv = img.clone()
    img_adv.requires_grad = True

    img_min = float(img.min().data.cpu().numpy())
    img_max = float(img.max().data.cpu().numpy())
    mask = (gt > 0)
    gt = gt[mask]

    while count < iteration:
        model.zero_grad()
        img_adv.retain_grad()

        output,_,_,_,_,_ = model(img_adv)

        output = F.upsample(output, size=[ height, weight], mode='bilinear', align_corners=True)
        output = output[mask]
        
        loss = (output - gt.detach()).abs().mean()
        loss.backward(retain_graph=True)

        img_adv.grad.sign_()
        img_adv = img_adv + img_adv.grad
        img_adv = where(img_adv > img + epsilon, img + epsilon, img_adv)
        img_adv = where(img_adv < img - epsilon, img - epsilon, img_adv)
        img_adv = torch.clamp(img_adv, img_min, img_max)

        count += 1
        return img_adv

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond * x) + ((1 - cond) * y)


