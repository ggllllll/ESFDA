from torch import nn
import torch
from networks.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F

class EmbeddingHead(nn.Module):
    def __init__(self, dim_in=512, embed_dim=256, embed='convmlp'):
        super(EmbeddingHead, self).__init__()

        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Dropout2d(0.5),
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)
            )
    def forward(self, x):
        return F.normalize(self.embed(x), p=2, dim=1)

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UNet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, is_embed=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.global_features = SaveFeatures(self.rn[-1])
        self.decoder_features = [SaveFeatures(i) for i in [self.up1, self.up2, self.up3, self.up4]]

        self.is_embed = is_embed
        if self.is_embed:
            self.embeddingHead = EmbeddingHead(512, 256)

    def forward(self, x, rfeat=False, mfeat=False):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)
        if self.is_embed:
            feature = self.embeddingHead(self.global_features.features)
        if mfeat:
            # return self.sfs[3].features, self.sfs[2].features, self.sfs[1].features, self.sfs[0].features, self.global_features.features, self.decoder_features[0].features, self.decoder_features[1].features, self.decoder_features[2].features, self.decoder_features[3].features, output
            return output, feature
            # return output, self.global_features.features

        if not rfeat:
            return output
        else:
            return output, fea

    def close(self):
        for sf in self.sfs: sf.remove()
