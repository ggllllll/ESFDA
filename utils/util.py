import torch


@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x * torch.log(x + 1e-30) + (1 - x) * torch.log(1 - x + 1e-30))


def normalize_img_to_0255(img):
    return (img - img.min()) / (img.max() - img.min()) * 255

def sig_label_to_hard(sig_pls, pseudo_label_threshold):
    pseudo_label_s = sig_pls.clone()
    pseudo_label_s[pseudo_label_s > pseudo_label_threshold] = 1.0
    pseudo_label_s[pseudo_label_s <= pseudo_label_threshold] = 0.0

    pseudo_label = pseudo_label_s.float()

    pseudo_labels = torch.zeros([sig_pls.size()[0], sig_pls.size()[2], sig_pls.size()[3]])
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[pseudo_label[:, 0] == 1] = 1
    pseudo_labels[pseudo_label[:, 1] == 1] = 2

    return pseudo_labels.unsqueeze(dim=1)