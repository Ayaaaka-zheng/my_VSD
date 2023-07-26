import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import librosa
from sklearn.manifold import TSNE
import random
sys.path.append('./models')
from models.senet import se_resnet_34, se_resnet_18


# trained on high-frequency features
# eval on low-frequency features

def plot_spectrogram(specgram,spect,title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram)-librosa.power_to_db(spect), origin="lower", aspect="auto")
    #im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def change(x):
    size = x.size()
    means = x.mean(dim=(2))
    means = torch.reshape(means, (-1,))
    means=means.view(size[0],size[1],size[3])
    # temp = torch.ones(means.size())
    # temp=temp.to('cuda')
    # temp[means < -1] = 0
    # temp=temp.view(size[0], size[1], 1, size[3])
    # temp = temp.repeat(1, 1, size[2], 1)
    # result = x * temp
    # return result

    mid = torch.zeros((size[0], size[1], size[2],1))
    mid=mid.to('cuda')
    for i in range(0, size[3]):
        if (float(means[0][0][i]) > -0.3):
            mid = torch.cat((mid, x[:, :, : ,i:i + 1]), dim=3)
    mid_size = mid.size()
    mid = mid[:, :, :, 1:]
    mid = torch.cat((mid, torch.zeros(mid_size[0], mid_size[1], size[2],size[3] - mid_size[3] + 1).to('cuda')), dim=3)
    return mid

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.abs().max()
class FB(nn.Module):
    def __init__(self, pretrained=False):
        super(FB, self).__init__()
        self.fft = torchaudio.transforms.Spectrogram(n_fft=1730, win_length=1728, hop_length=130,normalized=True)
        self.input_max_pooling = nn.AdaptiveMaxPool2d((224, 224))
        self.input_avg_pooling = nn.AdaptiveAvgPool2d((224, 224))
        self.relu=nn.ReLU(inplace=True)
        self.leakyRelu=nn.LeakyReLU(0.2)
        self.tanh=nn.Tanh()
        self.senet34 = se_resnet_18(pretrained=pretrained)
        self.bn = nn.BatchNorm1d(1000)
        self.linear = nn.Linear(1000, 2)
        self.fc=nn.Linear(1000,256)
        self.tsne=TSNE(n_components=2, learning_rate='auto', init='random')
    def forward(self, x,x_vocoder):
        x = self.fft(x)
        x = x + 1e-12
        x = x.log()

        x_vocoder = self.fft(x_vocoder)
        x_vocoder=x_vocoder+ 1e-12
        x_vocoder = x_vocoder.log()
        #
        x = x.clamp(min=-12.5)
        x_vocoder = x_vocoder.clamp(min=-12.5)
        x = x - x_vocoder
        #x=change(x)

        x = self.input_avg_pooling(x)
        x = x.repeat(1, 3, 1, 1)
        x = self.senet34(x)
        x = F.relu(x)
        #x=self.tanh(x)
        x = self.bn(x)
        feats=self.fc(x)
        weight_emb = self.tsne.fit_transform(feats.clone().detach().cpu().numpy())
        plt.scatter(weight_emb[:, 0], weight_emb[:, 1])
        plt.show()
        x = self.linear(x)
        return feats,x
        #return x

    # def forward(self, x):
    #     x = self.fft(x)+1e-6
    #     x=x.log()
    #     x = self.input_avg_pooling(x)
    #
    #     x = x.repeat(1, 3, 1, 1)
    #     x = self.senet34(x)
    #     x = F.relu(x)
    #     x = self.bn(x)
    #     x = self.linear(x)
    #     return x


if __name__ == '__main__':
    x = torch.randn(16, 1, 16000 * 7)
    model = FB()
    out = model(x)
    print(out.size())
