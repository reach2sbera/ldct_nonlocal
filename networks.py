import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np


def create_mask(neighborhood_size, SIZE):
    mask=torch.zeros([SIZE,SIZE], dtype=torch.bool)
    for j in range(0,SIZE):
        for i in range(0,int(neighborhood_size*np.sqrt(SIZE)),int(np.sqrt(SIZE))):
            for k in range(neighborhood_size):
                mask[j,i+k]=1
    return mask
    
class NonLocal(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation, SIZE):
        super(NonLocal,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 3,padding=1 ),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 3,padding=1 ))            
        #self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        self.SIZE=int(SIZE)
        self.neighborhood_size=3
        #self.n_neighbour=9
        self.mask=create_mask(self.neighborhood_size, self.SIZE)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        #
        mask=self.mask.repeat(m_batchsize,1,1)
    
        energy[~mask]=0
        

        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N


        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
    

        return torch.cat((out,x),dim=1)
      



class UNet(nn.Module):
    
  def __init__(self, image_size):
    super(UNet, self).__init__()   
    self.input_channel=1
    self.inter_channel=64
    self.conv1=nn.Sequential(nn.Conv2d(1,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.layer1=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool1=nn.MaxPool2d(kernel_size=(2, 2))
    self.layer2=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool2=nn.MaxPool2d(kernel_size=(2, 2))
    self.layer3=nn.Sequential(NonLocal(64, 'relu', (image_size/4)*(image_size/4)),
		 	     nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
       			     NonLocal(64, 'relu', image_size/4*(image_size/4)),
                             nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),                
                             NonLocal(64, 'relu', image_size/4*(image_size/4)),
                             nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True), 
                             NonLocal(64, 'relu', image_size/4*(image_size/4)),
                             nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool3=nn.Upsample(scale_factor=2, mode='nearest')
    self.layer4=nn.Sequential(nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool4=nn.Upsample(scale_factor=2, mode='nearest')    
    
    self.layer5=nn.Sequential(nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.conv2=nn.Conv2d(self.inter_channel,1,3,padding=1)

  def forward(self,x):

      x=self.conv1(x)

      x1=self.layer1(x)

      x=self.pool1(x1)

      x2=self.layer2(x)

      x=self.pool2(x2)

      x=self.layer3(x)

      x=self.pool3(x)

      x=self.layer4(torch.cat((x2 , x),1))

      x=self.pool4(x)

      x=self.layer5(torch.cat((x1 , x),1))

      x=self.conv2(x)
      return x



class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out
    

        
class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(1, cnum, 3, 1),
            SNConvWithActivation(cnum, 2*cnum, 3, 2),
            SNConvWithActivation(2*cnum, 4*cnum, 3, 1),
            SNConvWithActivation(4*cnum, 8*cnum, 3, 1),
            SNConvWithActivation(8*cnum, 8*cnum, 3, 2, padding=2),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding=2),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding =2),
            Self_Attn(8*cnum),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding=2),
            Self_Attn(8*cnum),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding=2)
        )
    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        return x