from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import torchvision.models as models
# from hyperopt import tpe, fmin
# from hyperoptSpace import space
import pickle
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import pickle
import  shutil

def func_to_optimize(args):

    dataset_dir = args['dataset_dir']

    time_stamp = args['time_stamp']

    try:
        os.mkdir(str(time_stamp))
    except:
        pass

    time_stamp = str(time_stamp)

    args['training_id'] = str(time_stamp)
    print("Training Stamp:", time_stamp)

    print(args)
    #
    # return 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transfer_learning_type = args['type']

    deep_model = args['deep_model']['deep_model']

    classifier_hidden_layers = args['deep_model']['classifier_hidden_layers']

    numberNodesHiddenLayer = args['deep_model']['number_of_nodes']

    numberNodesLastHiddenLayer = args['deep_model']['numberNodesLastHiddenLayer']

    num_epochs = args['deep_model']['epochs']

    batch_size = args['deep_model']['batch_size']


    #----------------------------Batch Norm---------------------
    batch_norm = args['deep_model']['batch_norm']['batch_norm']

    if batch_norm:
        eps_batchNorm = None#args['deep_model']['batch_norm']['eps']
        momentum_batchNorm = None#args['deep_model']['batch_norm']['momentum']
    else:
        eps_batchNorm = None
        momentum_batchNorm = None
    #-----------------------------------------------------------

    #---------------------------dropout-------------------------
    drop_out = args['deep_model']['drop_out']['drop_out']

    if drop_out:
        p = args['deep_model']['drop_out']['dropping_factor']
    else:
        p = None
    #-----------------------------------------------------------


    #------------------ActivationHiddenLayer--------------------
    activationFunctionHiddenLayer = args['deep_model']['activationFunctionHiddenLayer']['activation']

    if activationFunctionHiddenLayer == 'leakyRelu':
        negative_slope = args['deep_model']['activationFunctionHiddenLayer']['negative_slope']
    else:
        negative_slope=None
    #-----------------------------------------------------------

    #-------------------------------Load Data----------------------------------------

    def load_data(batch_size):

        data_dir = dataset_dir
        model_name = deep_model

        if model_name == 'inception_v3':
            data_transforms = {
            'train': transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            #transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            'val': transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            }

        else:
            data_transforms = {
            'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            'val': transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            }


        # Image dataset reads the images by accessing the path of directory,
        # converts the images to tensors and normalizes the tensors.
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}

        if transfer_learning_type == 'fine_tune':
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                         batch_size=batch_size,shuffle=True, num_workers=5)
                          for x in ['train', 'val']}

        elif transfer_learning_type == 'freeze_train':
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                         batch_size=1,shuffle=False, num_workers=5)
                          for x in ['train', 'val']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        # stores the names of the classes in the order present in file
        class_names = image_datasets['train'].classes

        print("Completed the loading of data to begin training or testing")
        dataloaders = dataloaders
        dataset_sizes = dataset_sizes
        class_names = class_names
        num_classes = len(class_names)

        return dataloaders, dataset_sizes, class_names, num_classes

    dataloaders, dataset_sizes, class_names, num_classes = load_data(batch_size)

    #--------------------------------------------------------------------------------


    #-----------------------------------Activation Final Layer-------------------------
    activationOutputLayer = args['deep_model']['activationOutputLayer']
    if activationOutputLayer=='sigmoid':
        activationFnOutputLayer = nn.Sigmoid()
    elif activationOutputLayer == 'Softmax':
        activationFnOutputLayer = nn.Softmax()
    elif activationOutputLayer == 'LogSoftmax':
        activationFnOutputLayer = nn.LogSoftmax()


    #----------------------------------------------------------------------------------

    #------------------------------Building Hidden Layers----------------------------
    def build_hidden_layers(num_fltrs,
                            activationFunctionHiddenLayer,
                            negative_slope,
                            classifier_hidden_layers,
                            numberNodesHiddenLayer,
                            numberNodesLastHiddenLayer,
                            batch_norm,
                            eps_batchNorm,
                            momentum_batchNorm,
                            drop_out,
                            p):
        modules = []

        for i in range(1,classifier_hidden_layers+1):

            #Dense Layer
            if i==1 and classifier_hidden_layers!=1:
                modules.append(nn.Linear(num_fltrs,numberNodesHiddenLayer))
            elif i==1 and i==classifier_hidden_layers:
                modules.append(nn.Linear(num_fltrs,numberNodesLastHiddenLayer))
            elif i==classifier_hidden_layers:
                modules.append(nn.Linear(numberNodesHiddenLayer,numberNodesLastHiddenLayer))
            else:
                modules.append(nn.Linear(numberNodesHiddenLayer,numberNodesHiddenLayer))

            #Batch Norm
            if batch_norm and i!=classifier_hidden_layers:
                modules.append(nn.BatchNorm1d(numberNodesHiddenLayer))
                                                # eps = eps_batchNorm,
                                                # momentum = momentum_batchNorm))
            elif batch_norm and i==classifier_hidden_layers:
                modules.append(nn.BatchNorm1d(numberNodesLastHiddenLayer))
                                                # eps = eps_batchNorm,
                                                # momentum = momentum_batchNorm))

            #Activation
            if activationFunctionHiddenLayer=='relu':
                modules.append(nn.ReLU())
            elif activationFunctionHiddenLayer=='leakyRelu':
                modules.append(nn.LeakyReLU(negative_slope=negative_slope))

            #Dropout
            if drop_out:
                modules.append(nn.Dropout(p,inplace=True))

        #Output Layer
        modules.append(nn.Linear(numberNodesLastHiddenLayer, num_classes))
        if activationOutputLayer:
            modules.append(activationFnOutputLayer)

        classifier_container = nn.Sequential(*modules)

        return classifier_container
    #----------------------------------------------------------------------------------





    #-----------------------lossFunction------------------------
    lossFunction = args['deep_model']['lossFunction']
    #-----------------------------------------------------------





    #Fine Tune
    if transfer_learning_type == 'fine_tune':
        model_name = deep_model

        if model_name=='resnet18':
            model_ft = models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            del model_ft.fc
            model_ft.fc = build_hidden_layers(num_ftrs,
                                                        activationFunctionHiddenLayer,
                                                        negative_slope,
                                                        classifier_hidden_layers,
                                                        numberNodesHiddenLayer,
                                                        numberNodesLastHiddenLayer,
                                                        batch_norm,
                                                        eps_batchNorm,
                                                        momentum_batchNorm,
                                                        drop_out,
                                                        p)

            model_ft = model_ft.to(device)



        elif model_name=='alexnet':
            model_ft = models.alexnet(pretrained=True)
            num_ftrs = model_ft.classifier[1].in_features
            del model_ft.classifier
            model_ft.classifier = build_hidden_layers(num_ftrs,
                                                        activationFunctionHiddenLayer,
                                                        negative_slope,
                                                        classifier_hidden_layers,
                                                        numberNodesHiddenLayer,
                                                        numberNodesLastHiddenLayer,
                                                        batch_norm,
                                                        eps_batchNorm,
                                                        momentum_batchNorm,
                                                        drop_out,
                                                        p)
            model_ft = model_ft.to(device)



        elif model_name=='vgg16':
            model_ft = models.vgg16(pretrained=True)
            num_ftrs = model_ft.classifier[0].in_features
            del model_ft.classifier
            model_ft.classifier = build_hidden_layers(num_ftrs,
                                                        activationFunctionHiddenLayer,
                                                        negative_slope,
                                                        classifier_hidden_layers,
                                                        numberNodesHiddenLayer,
                                                        numberNodesLastHiddenLayer,
                                                        batch_norm,
                                                        eps_batchNorm,
                                                        momentum_batchNorm,
                                                        drop_out,
                                                        p)
            model_ft = model_ft.to(device)



        elif model_name=='densenet161':
            model_ft = models.densenet161(pretrained=True)
            # num_ftrs = model_ft.classifier.in_features
            num_ftrs = 108192
            del model_ft.classifier
            model_ft.classifier = build_hidden_layers(num_ftrs,
                                                        activationFunctionHiddenLayer,
                                                        negative_slope,
                                                        classifier_hidden_layers,
                                                        numberNodesHiddenLayer,
                                                        numberNodesLastHiddenLayer,
                                                        batch_norm,
                                                        eps_batchNorm,
                                                        momentum_batchNorm,
                                                        drop_out,
                                                        p)
            model_ft = model_ft.to(device)



        elif model_name=='inception_v3':
            model_ft = models.inception_v3(pretrained=True)
            # num_ftrs = model_ft.AuxLogits.fc.in_features
            # model_ft.AuxLogits.fc = nn.Linear(num_ftrs,num_classes)

            model_ft.aux_logits = False
            num_ftrs = model_ft.fc.in_features
            del model_ft.fc
            model_ft.fc = build_hidden_layers(num_ftrs,
                                            activationFunctionHiddenLayer,
                                            negative_slope,
                                            classifier_hidden_layers,
                                            numberNodesHiddenLayer,
                                            numberNodesLastHiddenLayer,
                                            batch_norm,
                                            eps_batchNorm,
                                            momentum_batchNorm,
                                            drop_out,
                                            p)
            model_ft = model_ft.to(device)



        param_model_ft = model_ft.parameters()
        model = model_ft


    #Freeze Train
    elif transfer_learning_type == 'freeze_train':

        model_name = deep_model

        if model_name=='resnet18':
            model_conv = models.resnet18(pretrained=True)
            for param in model_conv.parameters():
                    param.requires_grad = False

            num_ftrs = model_conv.fc.in_features
            del model_conv.fc



        elif model_name=='alexnet':
            model_conv = models.alexnet(pretrained=True)
            for param in model_conv.parameters():
                    param.requires_grad = False

            num_ftrs = model_conv.classifier[1].in_features
            del model_conv.classifier



        elif model_name=='vgg16':
            model_conv = models.vgg16(pretrained=True)
            for param in model_conv.parameters():
                    param.requires_grad = False

            num_ftrs = model_conv.classifier[0].in_features
            del model_conv.classifier



        elif model_name=='densenet161':
            model_conv = models.densenet161(pretrained=True)
            for param in model_conv.parameters():
                    param.requires_grad = False

            # num_ftrs = model_conv.classifier.in_features
            num_ftrs = 108192
            del model_conv.classifier



        elif model_name=='inception_v3':
            model_urls = {
                # Inception v3 ported from TensorFlow
                'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
            }

            def inception_v3(pretrained=False, **kwargs):
                if pretrained:
            #         if 'transform_input' not in kwargs:
            #             kwargs['transform_input'] = True
                    model = Inception3_custom(**kwargs)
                    model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
                    return model

                return Inception3_custom(**kwargs)

            class Inception3_custom(models.Inception3):

                def __init__(self):
                    super(Inception3_custom, self).__init__()
                    self.aux_logits = False

                def forward(self,x):

                    if self.transform_input:
                        x = x.clone()
                        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
                        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
                        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
                    # 299 x 299 x 3
                    x = self.Conv2d_1a_3x3(x)
                    # 149 x 149 x 32
                    x = self.Conv2d_2a_3x3(x)
                    # 147 x 147 x 32
                    x = self.Conv2d_2b_3x3(x)
                    # 147 x 147 x 64
                    x = F.max_pool2d(x, kernel_size=3, stride=2)
                    # 73 x 73 x 64
                    x = self.Conv2d_3b_1x1(x)
                    # 73 x 73 x 80
                    x = self.Conv2d_4a_3x3(x)
                    # 71 x 71 x 192
                    x = F.max_pool2d(x, kernel_size=3, stride=2)
                    # 35 x 35 x 192
                    x = self.Mixed_5b(x)
                    # 35 x 35 x 256
                    x = self.Mixed_5c(x)
                    # 35 x 35 x 288
                    x = self.Mixed_5d(x)
                    # 35 x 35 x 288
                    x = self.Mixed_6a(x)
                    # 17 x 17 x 768
                    x = self.Mixed_6b(x)
                    # 17 x 17 x 768
                    x = self.Mixed_6c(x)
                    # 17 x 17 x 768
                    x = self.Mixed_6d(x)
                    # 17 x 17 x 768
                    x = self.Mixed_6e(x)
                    # 17 x 17 x 768
                    if self.training and self.aux_logits:
                        aux = self.AuxLogits(x)
                    # 17 x 17 x 768
                    x = self.Mixed_7a(x)
                    # 8 x 8 x 1280
                    x = self.Mixed_7b(x)
                    # 8 x 8 x 2048
                    x = self.Mixed_7c(x)
                    # 8 x 8 x 2048
                    x = F.avg_pool2d(x, kernel_size=8)
                    # 1 x 1 x 2048
                    x = F.dropout(x, training=self.training)
                    # 1 x 1 x 2048
                    x = x.view(x.size(0), -1)
                    # 2048
                    # x = self.fc(x)
                    # 1000 (num_classes)
                    if self.training and self.aux_logits:
                        return x, aux
                    return x


            model_conv = inception_v3(pretrained=True)
            del model_conv.fc
            for param in model_conv.parameters():
                    param.requires_grad = False
            num_ftrs = 2048


            #Aux fc Layer
            # num_ftrs = model_conv.AuxLogits.fc.in_features
            # model_conv.AuxLogits.fc = nn.Linear(num_ftrs,num_classes)

            # model_conv.aux_logits = False



        if model_name!='inception_v3':
            model_conv = nn.Sequential(*list(model_conv.children()))

        # print(model_conv)
        model_conv = model_conv.to(device)

        path_pickle_val = './'+ time_stamp + '/outfile_val_'+model_name + '.pkl'
        path_pickle_train = './'+ time_stamp + '/outfile_train_'+model_name + '.pkl'

        model_conv.train(False)

        if os.path.exists(path_pickle_val):
            print('pickle file already created, will load it soon')
        else:
            i=0
            print("Processing Validation Dataset and saving it to pickle...")
            for inputs, labels in dataloaders['val']:
                # print(inputs.shape)
                outputs = model_conv(inputs)
                dum = (outputs,labels)
                i+=1
                with open(path_pickle_val, 'a') as fp:
                    pickle.dump(dum, fp)
                if i%100==0:
                    print(i)

        if os.path.exists(path_pickle_train):
            print('pickle file already created, will load it soon')
        else:
            i=0
            print("Processing Training Dataset and saving it to pickle...")
            for inputs, labels in dataloaders['train']:
                outputs = model_conv(inputs)
                dum = (outputs,labels)
                i+=1
                with open(path_pickle_train, 'a') as fp:
                    pickle.dump(dum, fp)
                if i%100==0:
                    print(i)


        itemlist = {'train':[],
                    'val':[]}
        with open (path_pickle_train, 'rb') as fp_t:
            for i in range(dataset_sizes['train']):
                itemlist['train'].append(pickle.load(fp_t))

        with open (path_pickle_val, 'rb') as fp_v:
            for i in range(dataset_sizes['val']):
                itemlist['val'].append(pickle.load(fp_v))

        dataloaders = {}
        dataloaders['val'] = torch.utils.data.DataLoader(itemlist['val'],batch_size=batch_size,shuffle=True)
        dataloaders['train'] = torch.utils.data.DataLoader(itemlist['train'],batch_size=batch_size,shuffle=True)
        print("================",batch_size)

        class freeze_retrain(nn.Module):

            def __init__(self):
                super(freeze_retrain, self).__init__()

                self.classifier = build_hidden_layers(num_ftrs,
                                                        activationFunctionHiddenLayer,
                                                        negative_slope,
                                                        classifier_hidden_layers,
                                                        numberNodesHiddenLayer,
                                                        numberNodesLastHiddenLayer,
                                                        batch_norm,
                                                        eps_batchNorm,
                                                        momentum_batchNorm,
                                                        drop_out,
                                                        p)


            def forward(self,x):
                return self.classifier.forward(x)


        model = freeze_retrain()
        model.train(True)
        print(model)
        model = model.to(device)
        param_model_ft = model.parameters()




        # model = build_hidden_layers(num_ftrs,
        #                             activationFunctionHiddenLayer,
        #                             negative_slope,
        #                             classifier_hidden_layers,
        #                             numberNodesHiddenLayer,
        #                             numberNodesLastHiddenLayer,
        #                             batch_norm,
        #                             eps_batchNorm,
        #                             momentum_batchNorm,
        #                             drop_out,
        #                             p)
        # model = model.to(device)


    #------------------OptimizationAlgorithm-------------------
    optimizerAlgo = args['deep_model']['optimizerAlgo']['Algo']

    #optimization_params, dict to unpack **kwargs

    #Adadelta
    if optimizerAlgo == 'Adadelta':
        lr = args['deep_model']['optimizerAlgo']['lr']
        rho = args['deep_model']['optimizerAlgo']['rho']
        eps = args['deep_model']['optimizerAlgo']['eps']
        weight_decay = args['deep_model']['optimizerAlgo']['weight_decay']
        optimization_params = {'lr':lr,
                                'rho':rho,
                                'eps':eps,
                                'weight_decay':weight_decay
                                }
        optimizer = optim.Adadelta(params = param_model_ft, **optimization_params)

    #Adagrad
    elif optimizerAlgo == 'Adagrad':
        lr = args['deep_model']['optimizerAlgo']['lr']
        lr_decay = args['deep_model']['optimizerAlgo']['lr_decay']
        weight_decay = args['deep_model']['optimizerAlgo']['weight_decay']
        optimization_params = {'lr':lr,
                                'lr_decay':lr_decay,
                                'weight_decay':weight_decay
                                }
        optimizer = optim.Adagrad(params = param_model_ft, **optimization_params)


    #Adam
    elif optimizerAlgo == 'Adam':
        lr = args['deep_model']['optimizerAlgo']['lr']
        eps = args['deep_model']['optimizerAlgo']['eps']
        weight_decay = args['deep_model']['optimizerAlgo']['weight_decay']
        amsgrad = args['deep_model']['optimizerAlgo']['amsgrad']
        optimization_params = {'lr':lr,
                                'eps':eps,
                                'weight_decay':weight_decay,
                                'amsgrad':amsgrad
                                }
        optimizer = optim.Adam(params = param_model_ft, **optimization_params)


    # #SparseAdam
    # elif optimizerAlgo == 'SparseAdam':
    #     lr = args['deep_model']['optimizerAlgo']['lr']
    #     eps = args['deep_model']['optimizerAlgo']['eps']
    #     optimization_params = {'lr':lr,
    #                             'eps':eps
    #                             }
    #     optimizer = optim.SparseAdam(params = param_model_ft, **optimization_params)

    #Adamax
    elif optimizerAlgo == 'Adamax':
        lr = args['deep_model']['optimizerAlgo']['lr']
        eps = args['deep_model']['optimizerAlgo']['eps']
        weight_decay = args['deep_model']['optimizerAlgo']['weight_decay']
        optimization_params = {'lr':lr,
                                'eps':eps,
                                'weight_decay':weight_decay
                                }
        optimizer = optim.Adamax(params = param_model_ft, **optimization_params)


    #ASGD
    elif optimizerAlgo == 'ASGD':
        lr = args['deep_model']['optimizerAlgo']['lr']
        lambda_asgd = args['deep_model']['optimizerAlgo']['lambda']
        alpha = args['deep_model']['optimizerAlgo']['alpha']
        weight_decay = args['deep_model']['optimizerAlgo']['weight_decay']
        optimization_params = {'lr':lr,
                                'lambd':lambda_asgd,
                                'alpha':alpha,
                                'weight_decay':weight_decay
                                }
        optimizer = optim.ASGD(params = param_model_ft, **optimization_params)


    # #LBFGS
    # elif optimizerAlgo == 'LBFGS':
    #     lr = args['deep_model']['optimizerAlgo']['lr']
    #     optimization_params = {'lr':lr}
    #     optimizer = optim.LBFGS(params = param_model_ft, **optimization_params)

    #RMSprop
    elif optimizerAlgo == 'RMSprop':
        lr = args['deep_model']['optimizerAlgo']['lr']
        eps = args['deep_model']['optimizerAlgo']['eps']
        weight_decay = args['deep_model']['optimizerAlgo']['weight_decay']
        alpha = args['deep_model']['optimizerAlgo']['alpha']
        momentum = args['deep_model']['optimizerAlgo']['momentum']
        optimization_params = {'lr':lr,
                                'eps':eps,
                                'weight_decay':weight_decay,
                                'alpha':alpha,
                                'momentum':momentum
                                }
        optimizer = optim.RMSprop(params = param_model_ft, **optimization_params)


    #Rprop
    elif optimizerAlgo == 'Rprop':
        lr = args['deep_model']['optimizerAlgo']['lr']
        optimization_params = {'lr':lr}
        optimizer = optim.Rprop(params = param_model_ft, **optimization_params)

    #SGD
    elif optimizerAlgo == 'SGD':
        lr = args['deep_model']['optimizerAlgo']['lr']
        momentum = args['deep_model']['optimizerAlgo']['momentum']
        weight_decay = args['deep_model']['optimizerAlgo']['weight_decay']
        nesterov = args['deep_model']['optimizerAlgo']['nesterov']
        optimization_params = {'lr':lr,
                                'nesterov':nesterov,
                                'weight_decay':weight_decay,
                                'momentum':momentum
                                }
        optimizer = optim.SGD(params = param_model_ft, **optimization_params)
    #--------------------------------------------------------------------------------


    #-------------------------------lr_scheduler-------------------------------------

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #--------------------------------------------------------------------------------


    #--------------------------------Loss Function-----------------------------------

    criterion_func = args['deep_model']['lossFunction']['func']

    if criterion_func == 'L1Loss':
        size_average = args['deep_model']['lossFunction']['size_average']
        criterion = nn.L1Loss(size_average=size_average)

    elif criterion_func == 'MSELoss':
        size_average = args['deep_model']['lossFunction']['size_average']
        criterion = nn.MSELoss(size_average=size_average)

    elif criterion_func == 'CrossEntropyLoss':
        size_average = args['deep_model']['lossFunction']['size_average']
        criterion = nn.CrossEntropyLoss(size_average=size_average)

    elif criterion_func == 'PoissonNLLLoss':
        log_input = args['deep_model']['lossFunction']['log_input']
        full = args['deep_model']['lossFunction']['full']
        size_average = args['deep_model']['lossFunction']['size_average']
        criterion = nn.PoissonNLLLoss(size_average=size_average, full=full, log_input=log_input)

    elif criterion_func == 'MultiLabelMarginLoss':
        size_average = args['deep_model']['lossFunction']['size_average']
        criterion = nn.MultiLabelMarginLoss(size_average=size_average)

    elif criterion_func == 'SmoothL1Loss':
        size_average = args['deep_model']['lossFunction']['size_average']
        criterion = nn.SmoothL1Loss(size_average=size_average)

    elif criterion_func == 'MultiMarginLoss':
        p = args['deep_model']['lossFunction']['p']
        size_average = args['deep_model']['lossFunction']['size_average']
        criterion = nn.MultiMarginLoss(size_average=size_average)

    elif criterion_func == 'NLLLoss':
        criterion = nn.NLLLoss()

    #--------------------------------------------------------------------------------

    args['num_classes'] = num_classes
    if transfer_learning_type == 'fine_tune':
        #-------------------------------Training-----------------------------------------
        # print(model)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        loss_history = {'train':[],'val':[]}
        acc_history = {'train':[],'val':[]}

        since = time.time()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward prop
                    # track history if only in train
                    # No requirement to track in evaluation as there is no
                    # weights updation in it.
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        # if criterion_func == 'CrossEntropyLoss' or criterion_func == 'MultiMarginLoss':
                        #     pass
                        # else:
                        #     y_onehot = torch.FloatTensor(batch_size, num_classes)
                        #     y_onehot.zero_()
                        #     y_onehot.scatter_(1, labels, 1)
                        #     labels = y_onehot



                        # if criterion_func == 'MultiMarginLoss' or criterion_func == 'CrossEntropyLoss':
                        loss = criterion(outputs, labels)
                        print(loss)
                        # else:
                        #     loss = criterion(outputs, labels.float())

                        # loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                #append history to keep track of stats
                loss_history[phase].append(epoch_loss)
                acc_history[phase].append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        loss = loss_history
        accuracy = acc_history

        best_loss = best_loss
        best_acc = best_acc

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        model_saving_path = './'+time_stamp+'/jpg_'+transfer_learning_type+'_'+model_name+'_'+dataset_dir[2:]
        weight_saving_path = './'+time_stamp+'/jpg_'+transfer_learning_type+'_' + model_name + '_weights'+'_'+dataset_dir[2:]


        # torch.save(model, model_saving_path)
        # print("The model has been saved to: ", model_saving_path)

        #----saving model info to create a new one-----------
        with open(model_saving_path, 'w') as f:
            pickle.dump(args,f)


        torch.save(model.state_dict(), weight_saving_path)
        print('Saving Weights to: ', weight_saving_path)



    elif transfer_learning_type == 'freeze_train':

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        loss_history = {'train':[],'val':[]}
        acc_history = {'train':[],'val':[]}
        model.train(True)
        since = time.time()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # print(input)
                    if batch_size!=inputs.shape[0]:
                        batch_size=inputs.shape[0]
                    inputs = inputs.reshape(batch_size,num_ftrs)
                    labels = labels.reshape(batch_size)

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward prop
                    # track history if only in train
                    # No requirement to track in evaluation as there is no
                    # weights updation in it.
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        # print(outputs)
                        # if criterion_func == 'CrossEntropyLoss' or criterion_func == 'MultiMarginLoss':
                        #     pass
                        # else:
                        #     y_onehot = torch.FloatTensor(batch_size, num_classes)
                        #     y_onehot.zero_()
                        #     y_onehot.scatter_(1, labels, 1)
                        #     labels = y_onehot

                        # print(labels)

                        # if criterion_func == 'MultiMarginLoss' or criterion_func == 'CrossEntropyLoss':
                        loss = criterion(outputs, labels)
                        # else:
                        #     loss = criterion(outputs, labels.float())

                        # loss = criterion(outputs, labels)

                        # backward & optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                #append history to keep track of stats
                loss_history[phase].append(epoch_loss)
                acc_history[phase].append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        loss = loss_history
        accuracy = acc_history

        best_loss = best_loss
        best_acc = best_acc

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        model_saving_path = './'+time_stamp+'/jpg_'+transfer_learning_type+'_'+model_name
        weight_saving_path = './'+time_stamp+'/jpg_'+transfer_learning_type+'_' + model_name + '_weights'



        #----saving model info to create a new one-----------
        with open(model_saving_path, 'w') as f:
            pickle.dump(args,f)
        print('Saving model args: ',model_saving_path)

        torch.save(model.state_dict(), weight_saving_path)
        print('Saving Weights to: ', weight_saving_path)



    return best_loss

# space = space()
#
# best = fmin(func_to_optimize, space, algo=tpe.suggest, max_evals = 1)

# func_to_optimize({'deep_model': {'number_of_nodes': 500,
# 'drop_out': {'dropping_factor': 0.33, 'drop_out': False},
# 'numberNodesLastHiddenLayer': 100,
# 'deep_model': 'resnet18',
# 'batch_norm': {'batch_norm': False},
# 'batch_size': 9,
# 'epochs': 5,
# 'optimizerAlgo': {'momentum': 0.01, 'nesterov':False, 'weight_decay': 0, 'lr': 0.01, 'Algo': 'SGD'},
# 'lossFunction': {'func': 'NLLLoss', 'size_average': False},
# 'classifier_hidden_layers': 2,
# 'activationFunctionHiddenLayer': {'activation': 'leakyRelu', 'negative_slope': 0.0001},
# 'activationOutputLayer': 'LogSoftmax'},
# 'type': 'freeze_train'})

# Note: Fine tuning with densenet was not tested due to deficiency of GPU resource
#       Fine tuning with Inception v3 loss not decreasing
#       Rest everything tested and working fine
