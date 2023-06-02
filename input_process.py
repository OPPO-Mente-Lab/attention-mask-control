import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)



class PositionNet(nn.Module):
    def __init__(self,  in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        self.linears = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, positive_embeddings, masks=None):
        B, N, _ = boxes.shape 
        if masks is None:
            masks = torch.zeros((B,N,1)).to(positive_embeddings.device)
        else:
            masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        positive_null = self.null_positive_feature.view(1,1,-1)
        xyxy_null =  self.null_position_feature.view(1,1,-1)

        # replace padding with learnable null embedding 
        positive_embeddings = positive_embeddings*masks + (1-masks)*positive_null
        xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null

        objs = self.linears(  torch.cat([positive_embeddings, xyxy_embedding], dim=-1)  )
        assert objs.shape == torch.Size([B,N,self.out_dim])        
        return objs


class PositionNetSimple(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.in_dim = in_dim
        # self.out_dim = out_dim 

        # self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        # self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        # self.linears = nn.Sequential(
        #     nn.Linear( self.in_dim + self.position_dim, 512),
        #     nn.SiLU(),
        #     nn.Linear( 512, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, out_dim),
        # )
        
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([in_dim]))
        # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, positive_embeddings, masks=None):
        B, N, _ = positive_embeddings.shape 
        if masks is None:
            masks = torch.zeros((B,N,1)).to(positive_embeddings.device)
        else:
            masks = masks.unsqueeze(-1)

        # # embedding position (it may includes padding as placeholder)
        # xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        positive_null = self.null_positive_feature.view(1,1,-1)
        # xyxy_null =  self.null_position_feature.view(1,1,-1)

        # replace padding with learnable null embedding 
        positive_embeddings = positive_embeddings*masks + (1-masks)*positive_null
        # xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null

        # objs = self.linears(  torch.cat([positive_embeddings, xyxy_embedding], dim=-1)  )
        # assert objs.shape == torch.Size([B,N,self.out_dim])        
        return positive_embeddings
