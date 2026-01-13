import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
from data_process.etth_data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Daily, Dataset_Weekly, Dataset_CC_PV
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from models.AFDMixer import AFDMixer
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator  # 用于控制纵轴刻度间隔
from sklearn.metrics import r2_score

sns.set_style("whitegrid")

class Exp_data(Exp_Basic):
    def __init__(self, args):
        super(Exp_data, self).__init__(args)
        self.c_out = None
    
    def _build_model(self):

        if self.args.features == 'S':
            in_dim = 1
            self.c_out = 1
        elif self.args.features == 'M':
            in_dim = 7  # 数据的特征维度大小
            self.c_out = 7
        else:
            in_dim = 7
            self.c_out = 1  # 确保c_out有默认值

        
        # 打印调试信息
        print(f"Building model: {self.args.model}")
        print(f"Input dim: {in_dim}, Output dim: {self.c_out}")
        print(f"Seq len: {self.args.seq_len}, Pred len: {self.args.pred_len}")
        
        model = None

        # 根据模型名称选择构建不同的模型
        if self.args.model == 'AFDMixer':
            try:
                model = AFDMixer(
                    input_dim=in_dim,
                    output_dim=self.c_out,
                    seq_len=self.args.seq_len,
                    pred_len=self.args.pred_len,

                    num_freq_bands=self.args.num_freq_bands, 
                    freq_embed_dim=self.args.freq_embed_dim,

                    dtm_hidden=self.args.dtm_hidden,
                    dtm_dropout_base=self.args.dtm_dropout_base,
                    dtm_dropout_max=self.args.dtm_dropout_max,
                    dtm_linear_levels=self.args.dtm_linear_levels,
                    dtm_activation=self.args.dtm_activation,

                    bom_conv_levels=self.args.bom_conv_levels,
                    bom_conv_kernel=self.args.bom_conv_kernel,
                    bom_linear_levels=self.args.bom_linear_levels,
                    bom_hidden=self.args.bom_hidden,
                    bom_dropout_base=self.args.bom_dropout_base,
                    bom_activation=self.args.bom_activation,

                    complexity_type=self.args.complexity_type,
                    norm_method=self.args.norm_method,
                    expansion=self.args.expansion  
                )
                print(model)
                print("AFDMixer model created successfully")

                # 添加参数量和FLOPs计算
                from thop import profile
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                # 添加前向测试
                test_input = torch.randn(1, self.args.seq_len, in_dim).float()

                flops, _ = profile(model, inputs=(test_input,))
                self.gflops = flops / 1e9

                # 修复输出解包问题
                if model.training:
                    outputs = model(test_input)
                    test_output = outputs[0] if isinstance(outputs, tuple) else outputs 
                else:
                    test_output = model(test_input)

                print(f"Test input shape: {test_input.shape}")
                print(f"Test output shape: {test_output.shape}")

                print(f"Total parameters: {total_params / 1e6:.4f}M")
                print(f"Trainable parameters: {trainable_params / 1e6:.4f}M")
                print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")  # 转换为GFLOPs
                
                # 修正断言：检查序列长度和batch size，特征维度需等于输入维度
                expected_shape = (1, self.args.pred_len, in_dim)
                assert test_output.shape == expected_shape, \
                    f"Output shape {test_output.shape} != expected {expected_shape}"
            except Exception as e:
                print(f"Error creating AFDMixer: {e}")
                raise

            if model is None:
                raise ValueError("Model creation failed - returned None")
            
        return model.float()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,

            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,

            'weather':Dataset_ETT_minute,
            'traffic':Dataset_ETT_hour,
            'exchange_rate':Dataset_Daily,
            'national_illness':Dataset_Weekly,
            'electricity':Dataset_ETT_hour,

            'yahoo_stock':Dataset_Daily,
            'hsi_reverse':Dataset_Daily,
            'fchi_reverse':Dataset_Daily,

            'CC-PV':Dataset_CC_PV,
            'SDTG-PV':Dataset_CC_PV,
            'Yulara-PV':Dataset_CC_PV,
            'AS-PV':Dataset_CC_PV,
            'PSGH-PV':Dataset_CC_PV,
            'PVGIS-India-PV':Dataset_CC_PV

        }
        
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def calculate_r2(self, preds, trues):
        """
        计算R-squared指标
        Args:
            preds: 预测值数组 (num_samples, pred_len, num_features)
            trues: 真实值数组 (num_samples, pred_len, num_features)
        Returns:
            每个特征的R-squared值的平均值
        """
        # 展平数据以便计算
        flat_preds = preds.reshape(-1, preds.shape[-1])
        flat_trues = trues.reshape(-1, trues.shape[-1])
        
        # 对每个特征计算R2
        r2_scores = []
        for i in range(flat_preds.shape[1]):
            r2 = r2_score(flat_trues[:, i], flat_preds[:, i])
            # 处理特殊值（如所有值相同的情况）
            if np.isnan(r2):
                r2 = 0.0
            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, batch_x, batch_y)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

            total_loss.append(loss)

        total_loss = np.average(total_loss)

        preds = np.array(preds)
        trues = np.array(trues)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        if self.args.features != 'MS':
            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            maes, mses, rmses, mapes, mspes = metric(pred_scales, true_scales)
            
            # 添加R方计算
            r2 = self.calculate_r2(preds, trues)
            r2_denorm = self.calculate_r2(pred_scales, true_scales)

            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, R2:{:.4f}'.format(mse, mae, rmse, mape, mspe, r2))
            print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, R2:{:.4f}'.format(mses, maes, rmses, mapes, mspes, r2_denorm))
        
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        writer = SummaryWriter('/home/fuqiang/AFDMixer/event/'.format(self.args.model_name))

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()

            epoch_time = time.time()

            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y)

                loss = criterion(pred, true)
                
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    print('use amp')    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()

                    # if self.model.fft_extractor.band_boundaries.grad is not None:
                    #     print(f"Update Grad: {self.model.fft_extractor.band_boundaries.grad}")
                    # else:
                    #     print("Update Grad: None (Check if requires_grad=True)")

                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('test_loss', test_loss, global_step=epoch)

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)
            
        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))  #<-----

        # ===== 结果保存功能 =====
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(self.args.checkpoints, 'results_log')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'results.txt')

        # 获取最终指标
        if self.args.features == 'MS':
            mae, maes, mse, mses, r2, r2_denorm = self.test(setting, evaluate=True)
        else:
            mae, maes, mse, mses = self.test(setting, evaluate=True)

        if self.args.model == 'AFDMixer' and self.args.features != 'MS':
            log_content = (
                f"\n[{timestamp}] Experiment: {setting}\n"
                f"Model Config: data={self.args.data}, features={self.args.features}, seq_len={self.args.seq_len}, pred_len={self.args.pred_len},\n" 
                f"batch_size={self.args.batch_size}, patience={self.args.patience}, lr={self.args.lr}, lradj={self.args.lradj},\n"
                
                f"dtm_hidden={self.args.dtm_hidden}, dtm_dropout_base={self.args.dtm_dropout_base}, dtm_dropout_max={self.args.dtm_dropout_max},\n" 
                f"dtm_linear_levels={self.args.dtm_linear_levels}, dtm_activation={self.args.dtm_activation},\n"

                f"bom_hidden={self.args.bom_hidden}, bom_linear_levels={self.args.bom_linear_levels}, bom_dropout_base={self.args.bom_dropout_base},\n"
                f"bom_activation={self.args.bom_activation}, bom_conv_levels={self.args.bom_conv_levels}, bom_conv_kernel={self.args.bom_conv_kernel},\n"

                f"complexity_type={self.args.complexity_type},\n"

                f"norm_method={self.args.norm_method}, expansion={self.args.expansion}.\n"
            
                "Normalized Metrics:\n"
                f"MSE: {mse:.4f}, MAE: {mae:.4f}\n"
                "Denormalized Metrics:\n"
                f"MSE: {mses:.4f}, MAE: {maes:.4f}\n"
                f"Training Epochs: {epoch+1}/{self.args.train_epochs}\n"
                "----------------------------------------"
            )
        elif self.args.features == 'MS':
            log_content = (
                f"\n[{timestamp}] Experiment: {setting}\n"
                f"Model Config: data={self.args.data}, features={self.args.features}, seq_len={self.args.seq_len}, pred_len={self.args.pred_len},\n" 
                f"batch_size={self.args.batch_size}, patience={self.args.patience}, lr={self.args.lr}, lradj={self.args.lradj},\n"
                
                f"dtm_hidden={self.args.dtm_hidden}, dtm_dropout_base={self.args.dtm_dropout_base}, dtm_dropout_max={self.args.dtm_dropout_max},\n" 
                f"dtm_linear_levels={self.args.dtm_linear_levels}, dtm_activation={self.args.dtm_activation},\n"

                f"bom_hidden={self.args.bom_hidden}, bom_linear_levels={self.args.bom_linear_levels}, bom_dropout_base={self.args.bom_dropout_base},\n"
                f"bom_activation={self.args.bom_activation}, bom_conv_levels={self.args.bom_conv_levels}, bom_conv_kernel={self.args.bom_conv_kernel},\n"

                f"complexity_type={self.args.complexity_type}, norm_method={self.args.norm_method}, expansion={self.args.expansion}.\n"
            
                "Normalized Metrics:\n"
                f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}\n"
                "Denormalized Metrics:\n"
                f"MSE: {mses:.4f}, MAE: {maes:.4f}, R2: {r2_denorm:.4f}\n"
                f"Training Epochs: {epoch+1}/{self.args.train_epochs}\n"
                "----------------------------------------"
            )   

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_content)
                print(f"Results saved to {log_file}")
        except Exception as e:
            print(f"Failed to save results: {str(e)}")

        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        pred_scales = []
        true_scales = []
        
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, batch_x, batch_y)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        if self.args.features != 'MS':
            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            print('TTTT denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            maes, mses, rmses, mapes, mspes = metric(pred_scales, true_scales)
            
            # 添加R方计算
            r2 = self.calculate_r2(preds, trues)
            r2_denorm = self.calculate_r2(pred_scales, true_scales)

            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, R2:{:.4f}'.format(mse, mae, rmse, mape, mspe, r2))
            print('TTTT denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, R2:{:.4f}'.format(mses, maes, rmses, mapes, mspes, r2_denorm))


        # result save
        if self.args.save:
            folder_path = f'/home/fuqiang/AFDMixer/exp/{self.args.data}_several_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # mae, mse, rmse, mape, mspe, corr = metric(preds, trues)    # M, S
            mae, mse, rmse, mape, mspe = metric(preds, trues)   # MS
            print('Test:mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mse, mae, rmse, mape, mspe))

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'pred_scales.npy', pred_scales)
            np.save(folder_path + 'true_scales.npy', true_scales)
            
        if self.args.features == 'MS':
            return mae, maes, mse, mses, r2, r2_denorm

        return mae, maes, mse, mses

    
    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        #----------ori----------------#
        # batch_x = batch_x.float().cuda()
        # batch_y = batch_y.float()
        #-----------------------------#

        # ------------------------------ 🔍 新增------------------------------
        # 获取模型所在设备（从第一个参数提取）
        device = next(self.model.parameters()).device
        
        # 获取模型参数的数据类型（从第一个参数提取）
        target_dtype = next(self.model.parameters()).dtype

        # 统一批次数据类型和设备
        batch_x = batch_x.to(dtype=target_dtype, device=device)
        batch_y = batch_y.to(dtype=target_dtype, device=device)

        #------------------#
        # 特征维度处理
        if self.args.features == 'MS':
            # MS模式：多变量输入，单变量输出
            # 提取目标特征作为y
            f_dim = -1
            target_idx = slice(-1, None)  # 只取最后一个特征（目标特征）
        else:
            # S/M模式：单变量或多变量输出
            f_dim = 0
            target_idx = slice(f_dim, None)
        #------------------#

        #_____________________________________________________________________

        if self.args.model == 'AFDMixer':
            # 输入模型的是完整特征
            model_output = self.model(batch_x)
            outputs = model_output[0] if isinstance(model_output, tuple) else model_output
            
            # 只取目标特征对应的输出
            outputs = outputs[:, :, target_idx]
            # print(f"outputs.shape:{outputs.shape}")
            outputs_scaled = dataset_object.inverse_transform(outputs.cpu())
            
            # 标签只包含目标特征
            batch_y = batch_y[:, -self.args.pred_len:, target_idx]
            batch_y_scaled = dataset_object.inverse_transform(batch_y.cpu())
            
            return outputs, outputs_scaled, None, None, batch_y, batch_y_scaled
        else:
            if self.args.stacks == 1:
                outputs = self.model(batch_x)
            elif self.args.stacks == 2:
                outputs, mid = self.model(batch_x)
            else:
                print('Error!')

            #if self.args.inverse:
            outputs_scaled = dataset_object.inverse_transform(outputs)
            if self.args.stacks == 2:
                mid_scaled = dataset_object.inverse_transform(mid)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].cuda()
            batch_y_scaled = dataset_object.inverse_transform(batch_y)

            if self.args.stacks == 1:
                return outputs, outputs_scaled, 0,0, batch_y, batch_y_scaled
            elif self.args.stacks == 2:
                return outputs, outputs_scaled, mid, mid_scaled, batch_y, batch_y_scaled
            else:
                print('Error!')