import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Subset
from torchvision import transforms
import numpy as np 
from constants import config
from prettytable import PrettyTable
from collections import OrderedDict, defaultdict
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def layers_summary(self, input_size, batch_size=-1, device="cuda", recurse=True):
        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        x_input = [torch.rand(2, *input_size).to(device)]
        setattr(self, '_tmp_summary', OrderedDict())
        setattr(self, '_tmp_hook_summary', [])
        
        def make_hook(name=None):
            def hook(module:nn.Module, input, output, name=name):
                    summary = self._tmp_summary
                    class_name = str(module.__class__).split(".")[-1].split("'")[0]
                    module_idx = len(summary)
                    m_key = f"{class_name}-{module_idx + 1}" if name is None else name

                    summary[m_key] = OrderedDict()
                    summary[m_key]["input_shape"] = list(input[0].size())
                    summary[m_key]["input_shape"][0] = batch_size
                    
                    if isinstance(output, (list, tuple)):
                        summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                    else:
                        summary[m_key]["output_shape"] = list(output.size())
                        summary[m_key]["output_shape"][0] = batch_size

                    params = 0
                    for name, param in module.named_parameters():
                        params += torch.prod(torch.LongTensor(list(param.size())))
                    summary[m_key]["nb_params"] = params
            return hook
            

        def register_hook(module):
            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == self)
            ):
                self._tmp_hook_summary.append(module.register_forward_hook(make_hook()))

        if recurse==True:
            self.apply(register_hook)
        else:
            for name, module in self.named_children():
                 self._tmp_hook_summary.append(module.register_forward_hook(make_hook(name)))

        self(*x_input)
        for h in self._tmp_hook_summary:
            h.remove()


        table = PrettyTable()
        table.title = str(self.__class__).split(".")[-1].split("'")[0]
        table.field_names = ["Layer (type)", "Output Shape", "Param #"]
        total_params = sum(torch.prod(torch.LongTensor(list(param.size()))) for param in self.parameters())
        trainable_params =  sum(torch.prod(torch.LongTensor(list(param.size()))) for param in self.parameters() if param.requires_grad)
        
        summary = self._tmp_summary
        for layer in summary:
            table.add_row([layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"])])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
                    
        print(table)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Non-trainable params: {total_params - trainable_params}")
        print("================================================================")
        # delete tmp vars
        delattr(self, '_tmp_hook_summary')
        delattr(self, '_tmp_summary')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-10*x))
        
    
    def enter_training_summary(self, recurse=True, run_summary_dir='../runs/experiment_1/'):
        if recurse == True:
            raise NotImplementedError()
        
        setattr(self, '_tmp_training_summary', defaultdict(dict))
        setattr(self, '_tmp_training_summary_hooks', [])
        setattr(self, '_tmp_training_summary_step', 0)
        setattr(self, '_tmp_training_summary_tensorboard_writer', SummaryWriter(run_summary_dir))
        
        def make_forward_hook(name):
            def forward_hook(module, input, output, name=name):
                output = output[0]
                output_mean, output_min, output_max, output_std = output.mean(), output.min(), output.max(), output.std()
                output_summary = {
                    'mean':output_mean, 
                    'min':output_min, 
                    'max':output_max,
                    'std':output_std,
                }
                if name=='F6' and np.random.random() > 1.9:
                    plt.imshow(self.sigmoid(output.detach().numpy().reshape(12, 7)), cmap='gray')
                    plt.show()
                self._tmp_training_summary['output'][name] = output_summary
            return forward_hook
        
        def make_backward_hook(name):
            def backward_hook(module, grad_input, grad_output, name=name):
                grad_output = grad_output[0]
                grad_output_mean, grad_output_min, grad_output_max, grad_output_std = grad_output.mean(), grad_output.min(), grad_output.max(), grad_output.std()
                grad_output_summary = {
                    'mean':grad_output_mean, 
                    'min':grad_output_min, 
                    'max':grad_output_max,
                    'std':grad_output_std,
                }
                self._tmp_training_summary['grad'][name] = grad_output_summary
            return backward_hook
        
        if recurse == False:
            for name, module in self.named_children():
                self._tmp_training_summary_hooks.append(module.register_backward_hook(make_backward_hook(name)))
                self._tmp_training_summary_hooks.append(module.register_forward_hook(make_forward_hook(name)))
    
    def add_training_summary_step_report(self, step=None):
        if step == None:
            step = self._tmp_training_summary_step 
        self._tmp_training_summary_step += 1
        
        for parameter,metric in product(['output', 'grad'], ['mean', 'max', 'min', 'std']):
            self._tmp_training_summary_tensorboard_writer.add_scalars(f'{parameter}/{metric}', 
                                                                      dict([(name, self._tmp_training_summary[parameter][name][metric]) 
                                                                            for name,_ in self.named_children()]), step)
            

         
    
    def exit_training_summary(self):
        for h in self._tmp_training_summary_hooks:
            h.remove()
        delattr(self, '_tmp_training_summary_hooks')
        delattr(self, '_tmp_training_summary')
        delattr(self, '_tmp_training_summary_step')
        self._tmp_training_summary_tensorboard_writer.close()
        delattr(self, '_tmp_training_summary_tensorboard_writer')
        
        

class LinearSubSample(nn.Module):
    
    def __init__(self, in_channels, kernel_size, stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.a = nn.Parameter(torch.randn((1, in_channels, 1, 1)))
        self.b = nn.Parameter(torch.randn((1, in_channels, 1, 1)))
        self.kernel = torch.ones((in_channels, 1, kernel_size[0], kernel_size[1]))
        
    def forward(self, X):
        # X = F.conv2d(input=X, weight=self.kernel, stride=self.stride, groups=self.in_channels)
        X = F.avg_pool2d(input=X, kernel_size=(self.kernel.shape[-2], self.kernel.shape[-1]), stride=self.stride)
        return X * self.a + self.b
        
class Custom1InChannelSubsetConv2d(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = 16
        self.out_channel_in_mask = torch.tensor([
            [1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
            [1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1],
            [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
            [0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1],
            [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
            [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1],
        ], dtype=torch.bool)
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.randn(1, i.item(), 5, 5))
            for i in self.out_channel_in_mask.sum(dim=0)
        ])
        self.biases = nn.Parameter(torch.randn(self.out_channels)) 
        
        
    def forward(self, X):
        out_channel_in = list(X[:, self.out_channel_in_mask[:, i], :, :] for i in range(self.out_channels))
        assert tuple(t.shape[1] for t in out_channel_in) == tuple(map(lambda x:x.item(), self.out_channel_in_mask.sum(axis=0)))
        out_channels = list(F.conv2d(out_channel_in[i], self.kernels[i])+self.biases[i] for i in range(self.out_channels))
        return torch.cat(out_channels, 1)

class Custome1RBF(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.rbfs = torch.cat([tv.io.decode_image(config.paths.rbf_file_format.format(id=i), mode='GRAY').reshape(1, 1, -1) for i in range(0, 10)], 1)
        assert self.rbfs.min() >= 0 and 1 < self.rbfs.max() <= 255
        self.rbfs = self.rbfs.to(dtype=torch.float64) / 255.0
        assert self.rbfs.min() >= 0 and self.rbfs.max() <= 1
        assert tuple(self.rbfs.shape) == (1, 10, 84) 
        
    def forward(self, X):
        X = X[:, torch.newaxis, :]
        return (self.rbfs - X).pow(2).sum(axis=-1)
        
class ScaledTanh(nn.Module):
    
    def __init__(self, A=1.7159, S=2/3):
        super().__init__()
        self.A = torch.tensor(A) 
        self.S = torch.tensor(S) 
        self.Tanh = nn.Tanh()
        
    def forward(self, X):
        return self.A*self.Tanh(self.S*X) 

class LeNet5(BaseModel):
    def __init__(self, A=1.7159, S=2/3):
        super().__init__()
        self.a = float(A)
        self.s = float(S)
        
        self.C1 = nn.Sequential(nn.Conv2d(1, 6, (5, 5)), ScaledTanh(A=self.a, S=self.s))
        self.S2 = nn.Sequential(LinearSubSample(6, (2,2), 2), ScaledTanh(A=self.a, S=self.s))
        self.C3 = nn.Sequential(Custom1InChannelSubsetConv2d(), ScaledTanh(A=self.a, S=self.s))
        self.S4 = nn.Sequential(LinearSubSample(16, (2,2), 2), ScaledTanh(A=self.a, S=self.s))
        self.C5 = nn.Sequential(nn.Conv2d(16, 120, (5, 5)), ScaledTanh(A=self.a, S=self.s))
        self.F6 = nn.Sequential(nn.Flatten(), nn.Linear(120, 84), ScaledTanh(A=self.a, S=self.s))
        self.OUTPUT = nn.Sequential(Custome1RBF())
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Xavier initialization is better for Tanh/ScaledTanh
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LinearSubSample):
                nn.init.xavier_uniform_(m.a)
                if m.b is not None:
                    nn.init.constant_(m.b, 0)
        
        
    def forward(self, X):
        batch_size = X.shape[0]
        X = self.C1(X)
        assert tuple(X.shape) == (batch_size, 6, 28, 28)
        X = self.S2(X)
        assert tuple(X.shape) == (batch_size, 6, 14, 14)
        X = self.C3(X)
        assert tuple(X.shape) == (batch_size, 16, 10, 10)
        X = self.S4(X)
        assert tuple(X.shape) == (batch_size, 16, 5, 5)
        X = self.C5(X)
        assert tuple(X.shape) == (batch_size, 120, 1, 1)
        X = self.F6(X)
        assert tuple(X.shape) == (batch_size, 84)
        y = self.OUTPUT(X)
        assert tuple(y.shape) == (batch_size, 10)
        return y
        
    def test_layer_shapes(self):
        batch_size = 3
        X = torch.ones(batch_size, 1, 32, 32)
        
        X = self.C1(X)
        assert tuple(X.shape) == (batch_size, 6, 28, 28)
        assert sum(p.numel() for p in self.C1.parameters()) == 156
        
        X = self.S2(X)
        assert tuple(X.shape) == (batch_size, 6, 14, 14)
        assert sum(p.numel() for p in self.S2.parameters()) == 12
        
        X = self.C3(X)
        assert tuple(X.shape) == (batch_size, 16, 10, 10)
        assert sum(p.numel() for p in self.C3.parameters()) == 1516

        
        X = self.S4(X)
        assert tuple(X.shape) == (batch_size, 16, 5, 5)
        assert sum(p.numel() for p in self.S4.parameters()) == 32
        
        X = self.C5(X)
        assert tuple(X.shape) == (batch_size, 120, 1, 1)
        assert sum(p.numel() for p in self.C5.parameters()) == 48120
        
        X = self.F6(X)
        assert tuple(X.shape) == (batch_size, 84)
        assert sum(p.numel() for p in self.F6.parameters()) == 10164
        
        X = self.OUTPUT(X)
        assert tuple(X.shape) == (batch_size, 10)
        assert sum(p.numel() for p in self.OUTPUT.parameters()) == 0
        
    
def criterion(y_predict, y_actual, j=0.1):
    correct_class_energy = y_predict.gather(1, y_actual.view(-1, 1)).squeeze()
    exp_neg_energy = torch.exp(-y_predict)
    exp_correct = torch.exp(-correct_class_energy)
    sum_incorrect = exp_neg_energy.sum(dim=1) - exp_correct
    indirect_term = torch.log(torch.exp(torch.tensor(-j)) + sum_incorrect)
    loss = (correct_class_energy + indirect_term).mean()
    return loss


def train():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
    mnist_train = tv.datasets.MNIST(config.paths.dataset_dir, download=True, train=True, transform=transform)
    mnist_test = tv.datasets.MNIST(config.paths.dataset_dir, transform=transform)

    mnist_train = Subset(mnist_train, np.random.choice(len(mnist_train), config.training.training_size, replace=False))
    
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=config.training.batch_size, shuffle=config.training.shuffle, num_workers=config.training.num_workers)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=config.training.batch_size, num_workers=config.training.num_workers)
    
    model = LeNet5(A=config.model.A, S=config.model.S)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    
    
    model.train()
    model.enter_training_summary(recurse=False, run_summary_dir=config.paths.run_summary_dir)
    model.layers_summary(input_size=(1, 32, 32), device='cpu', recurse=False)
    step = 0
    for epoch in range(config.training.num_epochs):
        for X, y in mnist_train_loader:
            y_pred = model(X)
            loss = criterion(y_pred, y)
            step+=1
            print(f'step:{step} -> loss={loss}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.add_training_summary_step_report()
    model.exit_training_summary()
    
    y_test, y_test_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in mnist_test_loader:
            y_pred = model(X)
            y_pred:torch.Tensor = y_pred.argmin(dim=-1)
            y_test.extend(y)
            y_test_pred.extend(y_pred)
    
    print(classification_report(y_test, y_test_pred))
    


if __name__=='__main__':
    train()
    