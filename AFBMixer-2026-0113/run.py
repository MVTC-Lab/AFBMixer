import argparse
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from experiments.exp_ETTh import Exp_ETTh
from experiments.exp_data import Exp_data

parser = argparse.ArgumentParser(description='AFDMixer on several datasets')

parser.add_argument('--model', type=str, required=False, default='AFDMixer', help='model of the experiment')
### -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='PSGH-PV', choices=['CC-PV', 'AS-PV', 'PSGH-PV', 'YLR-PV', 'SDTG-PV'], help='name of dataset')
parser.add_argument('--root_path', type=str, default='/home/fuqiang/AFDMixer/datasets/Energy/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='PSGH-PV.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='MS', choices=['MS', 'M', 'S'], help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature:[OT/Active_Power]')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='/home/fuqiang/AFDMixer/AFD-Test-Result/AFDMixer-adaptive-revise-0106-check/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')


### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')
                                                                                  
### -------  input/output length settings --------------                                                                            
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of SCINet encoder, look back window')  # 270-192/96 336-336 648-720   | 96
parser.add_argument('--label_len', type=int, default=0, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length, horizon')
parser.add_argument('--concat_len', type=int, default=0)
parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0)
                                                              
### -------  training settings --------------  
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')  # 128  
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')  # 5 
parser.add_argument('--lr', type=float, default=0.003, help='optimizer learning rate') 
parser.add_argument('--loss', type=str, default='mae', help='loss function')
parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--save', type=bool, default =True, help='save the output results')
parser.add_argument('--model_name', type=str, default='AFDMixer')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)

# AFDMixer模型所需参数 
parser.add_argument('--num_freq_bands', type=int, default=3)  # 3
parser.add_argument('--freq_embed_dim', type=int, default=16)  # 16

parser.add_argument('--dtm_hidden', type=int, default=168)
parser.add_argument('--dtm_dropout_base', type=float, default=0.1) 
parser.add_argument('--dtm_dropout_max', type=float, default=0.1) 
parser.add_argument('--dtm_linear_levels', type=list, default=[1,3,5,7])
parser.add_argument('--dtm_activation', type=str, default='selu',help="selu/elu/tanh")


parser.add_argument('--bom_hidden', type=int, default=32)
parser.add_argument('--bom_linear_levels', type=int, default=2) 
parser.add_argument('--bom_dropout_base', type=float, default=0.1) 
parser.add_argument('--bom_activation', type=str, default='selu', help='selu/elu/tanh') 
parser.add_argument('--bom_conv_levels', type=int, default=0)  
parser.add_argument('--bom_conv_kernel', type=int, default=3, help="convolution kernel size")

parser.add_argument('--complexity_type', type=str, default='fft', help='fft/time/none') 
parser.add_argument('--norm_method', type=str, default='zscore', help='zscore/revin') 
parser.add_argument('--expansion', type=int, default=1)


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    
    'weather': {'data': 'weather.csv', 'T': 'OT', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1]},
    'traffic': {'data': 'traffic.csv', 'T': 'OT', 'M': [862, 862, 862], 'S': [1, 1, 1], 'MS': [862, 862, 1]},
    'exchange_rate': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
    'national_illness': {'data': 'national_illness.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'electricity': {'data': 'electricity.csv', 'T': 'electricPowerLoad', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
      
    'Yulara-PV': {'data': 'Yulara-PV.csv', 'T': 'Active_Power', 'M': [36, 36, 36], 'S': [1, 1, 1], 'MS': [36, 36, 1]},
    'SDTG-PV': {'data': 'SDTG-PV.csv', 'T': 'Active_Power', 'M': [11, 11, 11], 'S': [1, 1, 1], 'MS': [11, 11, 1]},
    'AS-PV': {'data': 'AS-PV.csv', 'T': 'Active_Power', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'CC-PV': {'data': 'CC-PV.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
    'PSGH-PV': {'data': 'PVGIS-SARAH3-Germany-Hainich.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'PSA-PV': {'data': 'PVGIS-EAR5-South-Africa.csv', 'T': 'OT', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
    'GBN-PV': {'data': 'GBN-PV.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp_data
# Exp = Exp_ETTh

mae_ = []
maes_ = []
mse_ = []
mses_ = []

if args.evaluate:
    setting = None

    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_dh{}_ddb{}_ddm{}_dll{}_da{}_bh{}_bll{}_bdb{}_ba{}_bcl{}_bck{}_nm_{}_itr0'.format(
        args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.lr, args.batch_size, 
        args.dtm_hidden, args.dtm_dropout_base, args.dtm_dropout_max, args.dtm_linear_levels, args.dtm_activation,
        args.bom_hidden, args.bom_linear_levels, args.bom_dropout_base, args.bom_activation, args.bom_conv_levels, args.bom_conv_kernel,
        args.norm_method
    ) if args.model == 'AFDMixer' else setting  # 保留原有设置

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting, evaluate=True)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
else:
    if args.itr:
        for ii in range(args.itr):
            # setting record of experiments
            setting = None
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_dh{}_ddb{}_ddm{}_dll{}_da{}_bh{}_bll{}_bdb{}_ba{}_bcl{}_bck{}_nm_{}_itr{}'.format(
                args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.lr, args.batch_size, 
                args.dtm_hidden, args.dtm_dropout_base, args.dtm_dropout_max, args.dtm_linear_levels, args.dtm_activation,
                args.bom_hidden, args.bom_linear_levels, args.bom_dropout_base, args.bom_activation, args.bom_conv_levels, args.bom_conv_kernel,
                args.norm_method, ii
            ) if args.model == 'AFDMixer' else setting  # 保留原有设置

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, maes, mse, mses = exp.test(setting)
            mae_.append(mae)
            mse_.append(mse)
            maes_.append(maes)
            mses_.append(mses)

            torch.cuda.empty_cache()
        print('Final mean normed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mse_), np.std(mse_), np.mean(mae_),np.std(mae_)))
        print('Final mean denormed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(np.mean(mses_),np.std(mses_), np.mean(maes_), np.std(maes_)))
        print('Final min normed mse:{:.4f}, mae:{:.4f}'.format(min(mse_), min(mae_)))
        print('Final min denormed mse:{:.4f}, mae:{:.4f}'.format(min(mses_), min(maes_)))
    else:
        setting = None
        
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_dh{}_ddb{}_ddm{}_dll{}_da{}_bh{}_bll{}_bdb{}_ba{}_bcl{}_bck{}_nm_{}_itr{}'.format(
                args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.lr, args.batch_size, 
                args.dtm_hidden, args.dtm_dropout_base, args.dtm_dropout_max, args.dtm_linear_levels, args.dtm_activation,
                args.bom_hidden, args.bom_linear_levels, args.bom_dropout_base, args.bom_activation, args.bom_conv_levels, args.bom_conv_kernel,
                args.norm_method, 0
            ) if args.model == 'AFDMixer' else setting  # 保留原有设置


        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # if args.features == 'MS':
        #     mae, maes, mse, mses, r2, r2_denorm= exp.test(setting)
        #     print('Final mean normed mse:{:.4f},mae:{:.4f},r2:{:.4f},denormed mse:{:.4f},mae:{:.4f},r2:{:.4f}'.format(mse, mae, r2, mses, maes, r2_denorm))
        # else:
        #     mae, maes, mse, mses = exp.test(setting)
        #     print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))



