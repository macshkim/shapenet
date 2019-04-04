import torch.nn as nn
import torchvision.models
from .feature_extractors import Img224x224Kernel7x7SeparatedDims
from ..layer import HomogeneousShapeLayer

class ShapeNet(nn.Module):
    
    def __init__(self, pca, feature_extract=None):
        super(ShapeNet, self).__init__()
        self.shape_layer = HomogeneousShapeLayer(pca, 2)
        self.num_out_params = self.shape_layer.num_params
        in_channels = 1
        norm_class = None
        if feature_extract == 'inception_v3':
            self.feature_extract_layer = torchvision.models.inception_v3(False, num_classes=self.num_out_params, aux_logits=False)
            self.feature_extract_layer.Conv2d_1a_3x3 = \
                torchvision.models.inception.BasicConv2d(in_channels, 32,
                                                         kernel_size=3,
                                                         stride=2)
        elif feature_extract == 'resnet':
            self.feature_extract_layer = torchvision.models.resnet18(False, num_classes=self.num_out_params)
            self.feature_extract_layer.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                                stride=2, padding=3,
                                                bias=False)
        else:
            self.feature_extract_layer = Img224x224Kernel7x7SeparatedDims(in_channels, self.num_out_params, norm_class)        

    def forward(self, imgs):
        """
        Forward input batch through network and shape layer

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            input batch

        Returns
        -------
        :class:`torch.Tensor`
            predicted shapes

        """        
        return self.shape_layer(
                self.feature_extract_layer(imgs)
                    .view(imgs.size(0), self.num_out_params, 1, 1))
