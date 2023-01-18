import torch
import torch.nn.functional as F


def compute_exclusion_loss(img1, img2, level=3):
    gradx_loss = []
    grady_loss = []

    for l in range(level):
        gradx1, grady1 = compute_gradient(img1)
        gradx2, grady2 = compute_gradient(img2)
        alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
        alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))

        gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
        grady1_s = (torch.sigmoid(grady1) * 2) - 1
        gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
        grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1

        gradx_loss.append(
            torch.mean(torch.multiply(torch.square(gradx1_s), torch.square(gradx2_s)), reduction_indices=[1, 2,
                                                                                                          3])**0.25)
        grady_loss.append(
            torch.mean(torch.multiply(torch.square(grady1_s), torch.square(grady2_s)), reduction_indices=[1, 2,
                                                                                                          3])**0.25)

        img1 = F.avg_pool2d(img1, 2)
        img2 = F.avg_pool2d(img2, 2)
    import pdb;pdb.set_trace()
    return gradx_loss, grady_loss


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :, ]
    grady = img[..., :, 1:] - img[..., :, :-1, ]
    return gradx, grady
