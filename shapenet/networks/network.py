import torch.nn as nn

from .feature_extractors import Img224x224Kernel7x7SeparatedDims
from ..layer import HomogeneousShapeLayer

class ShapeNet(nn.Module):
    
    def __init__(self, pca):
        super(ShapeNet, self).__init__()
        self.shape_layer = HomogeneousShapeLayer(pca, 2)
        self.num_out_params = self.shape_layer.num_params
        in_channels = 1
        norm_class = None
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
