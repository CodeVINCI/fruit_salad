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



def predict_func(transfer_learning_type,model_name,time_stamp,dataset_dir,predict_data_path):

    time_stamp = str(time_stamp)

    dataset_dir = dataset_dir

    model_saving_path = './' + time_stamp + '/jpg_' + transfer_learning_type + '_' + model_name
    weight_saving_path = './' + time_stamp + '/jpg_' + transfer_learning_type + '_' + model_name + '_weights'

    with open(model_saving_path, 'r') as f:
        args = pickle.load(f)


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

    num_classes = args['num_classes']

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


    #-----------------------------------Activation Final Layer-------------------------
    activationOutputLayer = args['deep_model']['activationOutputLayer']
    if activationOutputLayer=='sigmoid':
        activationFnOutputLayer = nn.Sigmoid()
    elif activationOutputLayer == 'Softmax':
        activationFnOutputLayer = nn.Softmax()
    elif activationOutputLayer == 'LogSoftmax':
        activationFnOutputLayer = nn.LogSoftmax()


    #----------------------------------------------------------------------------------


    def load_test_data(batch_size):

        data_dir = predict_data_path
        model_name = deep_model

        if model_name == 'inception_v3':
            data_transforms = {

            'test': transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            }

        else:
            data_transforms = {

            'test': transforms.Compose([
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
                          for x in ['test']}

        if transfer_learning_type == 'fine_tune':
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                         batch_size=batch_size,shuffle=True, num_workers=5)
                          for x in ['test']}

        elif transfer_learning_type == 'freeze_train':
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                         batch_size=1,shuffle=False, num_workers=5)
                          for x in ['test']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

        # stores the names of the classes in the order present in file
        # class_names = image_datasets['train'].classes

        print("Completed the loading of data to begin testing")
        # dataloaders = dataloaders
        # dataset_sizes = dataset_sizes
        # class_names = class_names
        # num_classes = len(class_names)

        return dataloaders, dataset_sizes #class_names, num_classes

    test_dataloaders, test_dataset_sizes = load_test_data(1)

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
        model.train(False)


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
                    model = Inception3_tanay(**kwargs)
                    model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
                    return model

                return Inception3_tanay(**kwargs)

            class Inception3_tanay(models.Inception3):

                def __init__(self):
                    super(Inception3_tanay, self).__init__()
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


        if model_name!='inception_v3':
            model_conv = nn.Sequential(*list(model_conv.children()))

        model_conv = model_conv.to(device)

        try:
            os.mkdir('./'+time_stamp+'/temp')

        except:
            pass

        path_pickle_test = './' + time_stamp + '/temp' + '/outfile_test_' + model_name + '.pkl'

        model_conv.train(False)

        if os.path.exists(path_pickle_test):
            print('pickle file already created, will load it soon')
        else:
            i=0
            print("Processing Validation Dataset and saving it to pickle...")
            for inputs, labels in test_dataloaders['test']:
                print(inputs.shape)
                outputs = model_conv(inputs)
                dum = (outputs,labels)
                i+=1
                with open(path_pickle_test, 'a') as fp:
                    pickle.dump(dum, fp)
                if i%100==0:
                    print(i)


        test_itemlist = {'test':[]}
        with open (path_pickle_test, 'rb') as fp_t:
            for i in range(test_dataset_sizes['test']):
                test_itemlist['test'].append(pickle.load(fp_t))

        test_dataloaders = {}
        test_dataloaders['test'] = torch.utils.data.DataLoader(test_itemlist['test'],batch_size=1,shuffle=False)


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
        model.train(False)
        # print(model)
        model = model.to(device)
        param_model_ft = model.parameters()


    #-----------prediction--------------------

    if transfer_learning_type=='freeze_train':

        model_saving_path = './'+time_stamp+'/jpg_'+transfer_learning_type+'_'+model_name
        weight_saving_path = './'+time_stamp+'/jpg_'+transfer_learning_type+'_' + model_name + '_weights'


        # try:
        #     # load_model = torch.load(model_saving_path)
        #     # print(load_model)
        #     # flag_loaded_model=True



        weights = torch.load(weight_saving_path)
        model.load_state_dict(weights)
        # print(weights)

        # inp = torch.rand(1,1,2048)
        # model.train(False)
        #
        # ot = model(inp)
        # print(ot)

        # except:
        #     raise NameError("stupid file name!")


        # model = model
        # model.eval()

        # class_names = class_names
        # num_images = 4
        # images_so_far = 0
        # with torch.no_grad():
        #      for i, (inputs, labels) in enumerate(iter(dataloader['predict'])):
        #             inputs = inputs.to(device)
        #             labels = labels.to(device)
        #             outputs = model(inputs)
        #             _, preds = torch.max(outputs, 1)
        #             print(preds)
        #
        #             for j in range(inputs.size()[0]):
        #                 images_so_far += 1
        #                 ax = plt.subplot(num_images//2, 2, images_so_far)
        #                 ax.axis('off')
        #                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
        #                 imshow(inputs.cpu().data[j])

        # print(model)
        model.train(False)
        for inputs,label in test_dataloaders['test']:
            print(inputs.shape)
            inputs = inputs.reshape(1,num_ftrs)
            out = model(inputs)
            print(out)
            print(out.argmax())


# transfer_learning_type = 'freeze_train'
# model_name = 'resnet18'
# predict_func(transfer_learning_type,model_name)
