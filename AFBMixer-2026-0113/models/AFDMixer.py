import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft
from models.normalizer import Normalizer
import numpy as np

# ------------------------------ 优化版FFT特征提取器 ------------------------------
class DynamicFreqFeatureExtractor(nn.Module):
    def __init__(self, input_dim, eps=1e-8, num_freq_bands=4, 
                 freq_embed_dim=32, dropout=0.1):
        super().__init__()
        self.eps = eps
        self.num_freq_bands = num_freq_bands
        self.freq_embed_dim = freq_embed_dim
        
        # 自适应频带划分层
        self.band_boundaries = nn.Parameter(torch.linspace(0, 1, num_freq_bands+1)[1:-1], requires_grad=True)
        # 多尺度特征提取
        self.feat_projector = nn.Sequential(
            nn.Linear(num_freq_bands + 4, freq_embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(freq_embed_dim*2, freq_embed_dim)
        )
        
        # 简单特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(freq_embed_dim, freq_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(freq_embed_dim, freq_embed_dim)
        )
        
        # 动态门控机制 - 修复输入维度问题
        self.gate_network = nn.Sequential(
            nn.Linear(freq_embed_dim + 1, 32),  # 输入维度改为 freq_embed_dim + 1
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
        
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def _adaptive_band_decomposition(self, freq_energy, total_energy, freq_bins):
        batch, dim, _ = freq_energy.shape
        device = freq_energy.device
        
        # 1. Sigmoid 归一化并排序，防止边界交叉
        boundaries_norm = torch.sigmoid(self.band_boundaries)
        sorted_boundaries, _ = torch.sort(boundaries_norm)
        
        # 2. 构建完整边界 [0, b1, b2, ..., 1]
        zeros = torch.zeros(1, device=device)
        ones = torch.ones(1, device=device)
        bounds = torch.cat([zeros, sorted_boundaries, ones])
        bounds_scaled = bounds * freq_bins 
        
        # 3. 构建频率网格
        freq_grid = torch.arange(freq_bins, device=device).float().view(1, 1, -1)
        
        band_ratios = []
        # 【关键参数】温度系数 tau
        # 建议设置在 1.0 到 5.0 之间。太大会导致梯度消失，太小会导致频带模糊。
        tau = 2 #2-PSGH-PV/2.5-AS-PV/2.5-Yulara-PV/3.5-SDTG/4-CC-PV   336pred-ccpv-tau=3
        
        for i in range(self.num_freq_bands):
            b_left = bounds_scaled[i]
            b_right = bounds_scaled[i+1]
            
            # 4. Soft Masking (核心可导部分)
            # 逻辑：sigmoid(f - left) * sigmoid(right - f)
            mask = torch.sigmoid(tau * (freq_grid - b_left)) * torch.sigmoid(tau * (b_right - freq_grid))
            
            # 5. 加权能量
            band_energy = (freq_energy * mask).sum(dim=-1)
            band_ratio = band_energy / (total_energy.squeeze(-1) + self.eps)
            band_ratios.append(band_ratio)
        
        return torch.stack(band_ratios, dim=-1)

    def forward(self, x):
        batch, seq_len, input_dim = x.shape
        device = x.device
        
        # 时域->频域转换
        x_perm = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x_fft = rfft(x_perm, dim=-1)
        freq_magnitude = torch.abs(x_fft)
        freq_bins = freq_magnitude.shape[-1]
        
        # 1. 基础频谱特征
        freq_energy = freq_magnitude ** 2
        total_energy = freq_energy.sum(dim=-1, keepdim=True)
        
        # 2. 自适应频带能量分布
        band_ratios = self._adaptive_band_decomposition(freq_energy, total_energy, freq_bins)
        #print(f"band_ratios.shape={band_ratios.shape}")
        
        # 3. 频谱熵特征 (改进稳定性)
        freq_prob = freq_energy / (total_energy + self.eps)
        clamped_prob = torch.clamp(freq_prob, min=1e-10, max=1.0)  # 避免log(0)
        spectral_entropy = -torch.sum(clamped_prob * torch.log(clamped_prob), dim=-1)
        norm_entropy = spectral_entropy / np.log(freq_bins)
        #print(f"norm_entropy.shape={norm_entropy.shape}")
        
        # 4. 相位一致性特征 (增强周期性检测)
        real_part = x_fft.real
        imag_part = x_fft.imag
        phase_consistency = torch.sqrt(real_part**2 + imag_part**2) / (freq_magnitude + self.eps)
        phase_consistency = phase_consistency.mean(dim=-1)
        
        # 5. 频谱波动特征 (保持与原始版本兼容)
        energy_diff = torch.diff(freq_energy, dim=-1).abs().mean(dim=-1)
        dynamic_fluct = energy_diff / (freq_energy.mean(dim=-1) + self.eps)
        #print(f"dynamic_fluct.shape={dynamic_fluct.shape}")
        
        # 6. 主频特征 (新增)
        _, dominant_idx = torch.max(freq_magnitude, dim=-1)
        dominant_freq = dominant_idx.float() / freq_bins
        #print(f"dominant_freq.shape={dominant_freq.shape}")
        
        # 特征聚合 (保持原始维度)
        dim_features = torch.cat([
            band_ratios,
            norm_entropy.unsqueeze(-1),
            dynamic_fluct.unsqueeze(-1),
            phase_consistency.unsqueeze(-1),
            dominant_freq.unsqueeze(-1)
        ], dim=-1)  # [batch, input_dim, num_freq_bands+4]
        
        # 多维特征投影
        projected = self.feat_projector(dim_features)  # [batch, input_dim, freq_embed_dim]
        
        # 替代多头注意力的特征融合
        # 首先对通道维度取平均
        channel_avg = torch.mean(projected, dim=1)  # [batch, freq_embed_dim]
        
        # 然后通过融合层
        freq_embed = self.feature_fusion(channel_avg)  # [batch, freq_embed_dim]
        
        # 动态特征增强 - 修复维度问题
        time_var = x_perm.var(dim=-1).mean(dim=-1, keepdim=True)  # [batch, 1]
        gate_input = torch.cat([freq_embed, time_var], dim=-1)    # [batch, freq_embed_dim + 1]
        
        fusion_gate = self.gate_network(gate_input)  # 现在输入维度匹配
        freq_embed_proj = torch.einsum('bd,btd->btd', fusion_gate, x)  # 门控增强
        
        # 特征增强输出 (保持原始结构)
        x_enhanced = x + freq_embed_proj
        
        # 保持与原始版本兼容的输出结构
        return {
            "freq_embed": freq_embed,
            "x_freq_enhanced": x_enhanced,
            "dim_freq_feat": dim_features,  # 保持名称但内容已增强
            "fusion_gate": fusion_gate,
            "dynamic_fluct": dynamic_fluct
        }

# ------------------------------ 动态时间混合器（模式兼容，严格控维） ------------------------------
class DynamicTemporalMixer(nn.Module):
    def __init__(self, input_dim,
                 in_seq_len, out_seq_len,
                 dtm_hidden=64, dtm_dropout_base=0.1, dtm_dropout_max=0.5,
                 dtm_linear_levels=[1, 3, 5], dtm_activation='selu', expansion=1,
                 num_freq_bands=3, freq_embed_dim=16, complexity_type='fft'):
        super().__init__()
        self.input_dim = input_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.dtm_hidden = dtm_hidden
        self.dtm_dropout_base = dtm_dropout_base
        self.dtm_dropout_max = dtm_dropout_max
        self.expansion = expansion
        self.num_freq_bands = num_freq_bands
        self.freq_embed_dim = freq_embed_dim
        self.complexity_type = complexity_type
        self.eps = 1e-8

        activation_map = {'selu': nn.SELU, 'elu': nn.ELU, 'tanh': nn.Tanh}
        self.activation = activation_map[dtm_activation]()

        self.branches = nn.ModuleList()
        self.dtm_linear_levels = dtm_linear_levels
        for levels in dtm_linear_levels:
            self.branches.append(self._build_mlp_branch(levels))
        self.num_branches = len(dtm_linear_levels)

        self.estimator_input_dim = 7 

        self.weight_estimator = nn.Sequential(
            nn.Linear(self.estimator_input_dim, 16),
            nn.SELU(),
            nn.Linear(16, self.num_branches),
            nn.Softmax(dim=-1)
        )

        self.dropout_estimator = nn.Sequential(
            nn.Linear(self.estimator_input_dim, 8),
            nn.SELU(),
            nn.Linear(8, self.num_branches),
            nn.Sigmoid()
        )

        self.ablate_fluct = False
        self.ablate_dim_feat = False

    def _build_mlp_branch(self, levels):
        layers = []
        input_dim = self.in_seq_len
        current_dim = input_dim
        for _ in range(levels - 1):
            layers.append(nn.Linear(current_dim, self.expansion * self.dtm_hidden))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dtm_dropout_base))
            current_dim = self.expansion * self.dtm_hidden
        layers.append(nn.Linear(current_dim, self.out_seq_len))
        return nn.Sequential(*layers)

    def _extract_none_features(self, x):
        batch, seq_len, input_dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # 为none模式直接返回零张量
        if self.complexity_type == 'none':
            dim_temporal_feat = torch.zeros(batch, input_dim, 7, device=device, dtype=dtype)
            return dim_temporal_feat, None

    def forward(self, x_input, fft_output=None):
        batch, seq_len, input_dim_total = x_input.shape
        device = x_input.device
        original_input_dim = self.input_dim

        # 特征提取逻辑严格对齐原始代码
        if self.complexity_type == 'fft' and fft_output is not None:
            dim_feat = fft_output["dim_freq_feat"]
            dynamic_fluct = fft_output["dynamic_fluct"]
        else:
            dim_feat, dynamic_fluct = self._extract_none_features(x_input)

        # 样本级特征计算（保持维度一致性）
        if self.complexity_type == 'fft' and self.ablate_dim_feat:
            avg_dim_feat = torch.zeros_like(dim_feat.mean(dim=1))
        else:
            avg_dim_feat = dim_feat.mean(dim=1)

        # 分支权重与dropout计算（复用原始逻辑）
        branch_weights = self.weight_estimator(avg_dim_feat)
        #print(f"branch_weights.shape={branch_weights.shape}")

        #--------------------------#
        # 分支权重固定消融组
        # branch_weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device).unsqueeze(0).repeat(batch, 1)
        #--------------------------#

        if dynamic_fluct is not None:
            fluct_mean = dynamic_fluct.mean(dim=1).unsqueeze(-1)

        estimator_dropout = self.dropout_estimator(avg_dim_feat)

        if self.complexity_type == 'none':
            branch_dropout = self.dtm_dropout_base + (self.dtm_dropout_max - self.dtm_dropout_base) * estimator_dropout
        else:
            if self.complexity_type == 'fft' and self.ablate_fluct:
                branch_dropout = self.dtm_dropout_base + (self.dtm_dropout_max - self.dtm_dropout_base) * estimator_dropout
            else:
                branch_dropout = self.dtm_dropout_base + (self.dtm_dropout_max - self.dtm_dropout_base) * estimator_dropout * fluct_mean

        # 多分支计算（严格保持原始维度处理）
        branch_outputs = []
        for idx in range(self.num_branches):
            branch = self.branches[idx]
            x_perm = x_input.permute(0, 2, 1)
            x_branch = x_perm.reshape(batch * original_input_dim, seq_len)

            for layer in branch:
                if isinstance(layer, nn.Dropout):
                    dropout_p = min(0.9, max(0.01, branch_dropout[:, idx].mean().item()))
                    x_branch = F.dropout(x_branch, p=dropout_p, training=self.training)
                else:
                    x_branch = layer(x_branch)

            x_branch = x_branch.reshape(batch, original_input_dim, self.out_seq_len).permute(0, 2, 1)
            branch_outputs.append(x_branch.unsqueeze(1))
        
        branch_outputs = torch.cat(branch_outputs, dim=1)
        branch_weights = branch_weights.unsqueeze(-1).unsqueeze(-1)
        x_out = torch.sum(branch_weights * branch_outputs, dim=1)

        return x_out

# ------------------------------ 输出处理层构建函数 ------------------------------
def build_output_Mixer(input_dim, pred_len,
                       bom_conv_levels=0, bom_conv_kernel=3, bom_linear_levels=0, bom_hidden=32,
                       bom_dropout_base=0.1, bom_activation='selu'):
    layers = []
    activation_map = {'selu': nn.SELU, 'elu': nn.ELU, 'tanh': nn.Tanh}
    activation = activation_map[bom_activation]()
    
    for _ in range(bom_conv_levels):
        layers.append(nn.Conv1d(input_dim, input_dim, kernel_size=bom_conv_kernel, padding=bom_conv_kernel//2))
    
    if bom_linear_levels > 0:
        dim_sequence = [pred_len] + [bom_hidden] * (bom_linear_levels - 1) + [pred_len]
        for i in range(len(dim_sequence)-1):
            layers.append(nn.Linear(dim_sequence[i], dim_sequence[i+1]))
            if i < len(dim_sequence)-2:
                layers.append(activation)
                layers.append(nn.Dropout(bom_dropout_base))
    
    return nn.Sequential(*layers)

# ------------------------------ 核心模型AFBMixer ------------------------------
class AFDMixer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, pred_len,
                 num_freq_bands=4, freq_embed_dim=32,  # 使用优化后的默认参数
                 dtm_hidden=64, dtm_dropout_base=0.1, dtm_dropout_max=0.5,
                 dtm_linear_levels=[1,3,5], dtm_activation='selu',
                 bom_conv_levels=0, bom_conv_kernel=3, bom_linear_levels=0, bom_hidden=32,
                 bom_dropout_base=0.1, bom_activation='selu',
                 complexity_type='fft',
                 norm_method='zscore', expansion=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.complexity_type = complexity_type
        self.num_freq_bands = num_freq_bands
        self.freq_embed_dim = freq_embed_dim

        assert complexity_type in ['fft', 'none'], f"不支持的complexity_type: {complexity_type}"

        self.normalizer = Normalizer(
            method=norm_method, eps=1e-5, affine=True, num_features=input_dim
        )
        
        # 使用优化后的FFT特征提取器
        self.fft_extractor = DynamicFreqFeatureExtractor(
            input_dim=input_dim,
            num_freq_bands=num_freq_bands,
            freq_embed_dim=freq_embed_dim
        ) if complexity_type == 'fft' else None
        
        self.time_proj = DynamicTemporalMixer(
            input_dim=input_dim,
            in_seq_len=seq_len,
            out_seq_len=pred_len,
            dtm_hidden=dtm_hidden,
            dtm_dropout_base=dtm_dropout_base,
            dtm_dropout_max=dtm_dropout_max,
            dtm_linear_levels=dtm_linear_levels,
            dtm_activation=dtm_activation,
            expansion=expansion,
            num_freq_bands=num_freq_bands,
            freq_embed_dim=freq_embed_dim,
            complexity_type=complexity_type
        )
        
        self.bom_deal = build_output_Mixer(
            input_dim=output_dim,  # 使用output_dim而非input_dim
            pred_len=pred_len,
            bom_conv_levels=bom_conv_levels,
            bom_conv_kernel=bom_conv_kernel,
            bom_linear_levels=bom_linear_levels,
            bom_hidden=bom_hidden,
            bom_dropout_base=bom_dropout_base,
            bom_activation=bom_activation
        )
    
    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.normalizer.to(device=device, dtype=dtype)
        if self.fft_extractor is not None:
            self.fft_extractor.to(device=device, dtype=dtype)
        self.time_proj.to(device=device, dtype=dtype)
        self.bom_deal.to(device=device, dtype=dtype)
        return self
    
    def _build_empty_fft_output(self, batch, input_dim, device, dtype):
        return {
            "dim_freq_feat": torch.zeros(batch, input_dim, self.num_freq_bands + 4, device=device, dtype=dtype),
            "dynamic_fluct": torch.zeros(batch, input_dim, device=device, dtype=dtype),
            "freq_embed": torch.zeros(batch, self.freq_embed_dim, device=device, dtype=dtype),
            "fusion_gate": torch.zeros(batch, 1, device=device, dtype=dtype),
            "x_freq_enhanced": torch.zeros(batch, self.seq_len, input_dim, device=device, dtype=dtype)
        }
    
    def forward(self, x):
        batch, seq_len, input_dim = x.shape
        assert seq_len == self.seq_len and input_dim == self.input_dim, \
            f"输入维度错误：预期(any, {self.seq_len}, {self.input_dim})，实际({batch}, {seq_len}, {input_dim})"
        
        device = x.device
        target_dtype = next(self.parameters()).dtype

        x_normalized = self.normalizer(x, mode='norm').type(target_dtype)

        fft_output = None
        x_processed = x_normalized
        if self.complexity_type == 'fft' and self.fft_extractor is not None:
            fft_output = self.fft_extractor(x_normalized)
            x_processed = fft_output["x_freq_enhanced"].type(target_dtype)
            assert x_processed.shape == (batch, seq_len, input_dim), \
                f"FFT增强后维度错误：预期{(batch, seq_len, input_dim)}，实际{x_processed.shape}"
        else:
            fft_output = self._build_empty_fft_output(batch, input_dim, device, target_dtype)

        x_proj = self.time_proj(x_processed, fft_output)
        
        x_perm = x_proj.permute(0, 2, 1)
        
        x_transformed = x_perm

        # 输出处理
        x_output_perm = self.bom_deal(x_transformed)
        x_output = x_output_perm.permute(0, 2, 1)
        x_output = self.normalizer(x_output, mode='denorm')
        
        if self.training:
            if self.complexity_type == 'fft' and self.fft_extractor is not None and fft_output is not None:
                avg_dim_freq = fft_output["dim_freq_feat"].mean(dim=1)
                branch_weights = self.time_proj.weight_estimator(avg_dim_freq)
                return x_output, fft_output["freq_embed"], branch_weights
            else:
                dim_temporal_feat, _ = self.time_proj._extract_none_features(x_processed)
                avg_dim_feat = dim_temporal_feat.mean(dim=1)
                branch_weights = self.time_proj.weight_estimator(avg_dim_feat)
                return x_output, torch.zeros(batch, self.freq_embed_dim, device=device), branch_weights
        else:
            return x_output
