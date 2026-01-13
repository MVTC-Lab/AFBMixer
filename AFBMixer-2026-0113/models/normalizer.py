import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ 1. 基础组件保留（修复设备兼容性） ------------------------------
class RevIN(nn.Module):
    """可逆实例归一化（RevIN）- 修复设备不匹配问题"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        # 可训练参数（自动迁移设备）
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.affine_weight = None
            self.affine_bias = None
        
        # 统计量缓存（自动迁移设备）
        self.register_buffer('mean', None)
        self.register_buffer('stdev', None)
    
    def forward(self, x, mode:str):
        if mode == 'norm':
            return self.normalize(x)
        elif mode == 'denorm':
            return self.denormalize(x)
        else:
            raise ValueError(f"未知模式: {mode}")
    
    def normalize(self, x):
        device = x.device
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        
        normalized = (x - self.mean) / self.stdev
        if self.affine:
            normalized = normalized * self.affine_weight.to(device) + self.affine_bias.to(device)
        return normalized
    
    def denormalize(self, x):
        if self.mean is None or self.stdev is None:
            raise RuntimeError("必须先调用normalize方法进行归一化")
        device = x.device
        
        if self.affine:
            x = (x - self.affine_bias.to(device)) / (self.affine_weight.to(device) + self.eps)
        return x * self.stdev.to(device) + self.mean.to(device)

class NormalizationMethods:
    """时间序列归一化方法集合"""
    @staticmethod
    def zscore_normalize(x, dim=1, eps=1e-8):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True, unbiased=False)
        return (x - mean) / (std + eps), {"mean": mean, "std": std}
    
    @staticmethod
    def zscore_denormalize(x, stats):
        return x * stats["std"] + stats["mean"]

class Normalizer(nn.Module):
    """通用归一化模块(支持zscore/RevIN)"""
    def __init__(self, method='revin', eps=1e-8, affine=True, num_features=None):
        super().__init__()
        self.method = method
        self.eps = eps
        self.affine = affine
        self.stats = None
        
        # 初始化RevIN（若启用）
        if method == 'revin':
            if num_features is None:
                raise ValueError("RevIN需要指定num_features")
            self.revin_module = RevIN(num_features, eps, affine)
        else:
            self.revin_module = None
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            return self.normalize(x)
        elif mode == 'denorm':
            return self.denormalize(x)
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")
    
    def normalize(self, x):
        if self.method == 'revin':
            return self.revin_module(x, mode='norm')
        
        normalized, self.stats = NormalizationMethods.zscore_normalize(x, eps=self.eps)
        return normalized
    
    def denormalize(self, x):
        if self.method == 'revin':
            return self.revin_module(x, mode='denorm')
        if self.stats is None:
            raise RuntimeError("Must call normalize first")
        return NormalizationMethods.zscore_denormalize(x, self.stats)
    
    def to(self, device=None, dtype=None):
        """重写to方法确保RevIN正确迁移设备"""
        super().to(device=device, dtype=dtype)
        if self.revin_module:
            self.revin_module.to(device=device, dtype=dtype)
        return self