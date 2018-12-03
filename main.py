from predict import predict_func
from train import func_to_optimize
import time

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parse = argparse.ArgumentParser(description='hyperparams')

parse.add_argument('-train', action='store', dest='train',help='Bool, True if u want to train',
                   type=str2bool, nargs='?',const=True,default=True)

parse.add_argument('-predict', action='store', dest='predict',help='Bool, True if u want to predict',
                   type=str2bool, nargs='?',const=True,default=False)

parse.add_argument('-epochs', action='store',dest='epochs', default=6)

parse.add_argument('-dropout', action='store', dest='dropout',help='Bool, True if u want dropout',
                   type=str2bool, nargs='?',const=False)

parse.add_argument('-batchnorm', action='store', dest='batchnorm',help='Bool, True if u want use batchnorm',
                   type=str2bool, nargs='?',const=False)

parse.add_argument('-model', action='store',dest='deep_model',default='resnet18')

parse.add_argument('-loss', action='store',dest='lossFunction',default='NLLLoss')

parse.add_argument('-activation', action='store',dest='activation',default='LogSoftmax')


results = parse.parse_args()


deep_model = results.deep_model

dataset_dir = '../../jpg/'

predict_data_path = '../../jpg_test/'

# time_stamp = str(time.time()) #id_here
time_stamp = '1532091844.56'

if results.train:
    print("===========================You Choose to Train=============================")

    # time_stamp = time.time()


    func_to_optimize({'deep_model': {'number_of_nodes': 500,
    'drop_out': {'dropping_factor': 0.33, 'drop_out': results.dropout},
    'numberNodesLastHiddenLayer': 100,
    'deep_model': deep_model,
    'batch_norm': {'batch_norm': results.batchnorm},
    'batch_size': 9,
    'epochs': int(results.epochs),
    'optimizerAlgo': {'momentum': 0.01, 'nesterov':False, 'weight_decay': 0, 'lr': 0.01, 'Algo': 'SGD'},
    'lossFunction': {'func': results.lossFunction, 'size_average': False},
    'classifier_hidden_layers': 2,
    'activationFunctionHiddenLayer': {'activation': 'leakyRelu', 'negative_slope': 0.0001},
    'activationOutputLayer': results.activation},
    'type': 'freeze_train',
    'dataset_dir':dataset_dir,
    'time_stamp':time_stamp})


if results.predict:
    print("==========================You Choose to Predict============================")

    predict_func('freeze_train',deep_model,time_stamp,dataset_dir,predict_data_path)
