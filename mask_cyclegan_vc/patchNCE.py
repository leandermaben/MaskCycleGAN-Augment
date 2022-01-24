import torch
import torch.nn.functional as F
from mask_cyclegan_vc.model import Generator

def patchNCE(inp,out,generator):
    layers_in = generator.intermediate_outputs(inp)
    layers_out = generator.intermediate_outputs(out)

    for layer_in,layer_out in zip(layers_in,layers_out):
        dot_matrix = torch.bmm(torch.transpose(layer_in,1,2),layer_out)
        positive_diagonal = torch.eye(dot_matrix.size(1),device=dot_matrix.device).unsqueeze(0)*dot_matrix
        positive_scores = torch.sum(positive_diagonal,dim=2)
        


if __name__ == '__main__':
    gen = Generator()
    x = torch.randn(5,80,64)
    y= torch.randn(5,80,64)
    patchNCE(x,y,gen)
