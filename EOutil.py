import numpy as np
from skimage import io
import rasterio as rio
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
from os import path
import os
import shutil
# Matplotlib
import matplotlib.pyplot as plt
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import visdom
from skimage import exposure


class EOtorch:
    WINDOW_SIZE = (256, 256)  # Patch size
    STRIDE = 128  # Stride for testing
    IN_CHANNELS = 7  # Number of input channels (e.g. RGB)
    BATCH_SIZE = 8  # Number of samples in a mini-batch
    CACHE = True  # Store the dataset in-memory
    save_epoch = 5
    train_epoch=50
    base_lr=0.01
    MAIN_FOLDER = None
    DATASET = None
    GPU_TRAIN=False
    MODEL_DIR = None
    MODEL_BASENAME=None
    MODEL_TEST=None
    MODEL_LOAD=None
    is_Load_vgg=False
    TRAINED = False
    im_min=None
    im_max=None

    data_scale=255

    def __init__(self,channels,LABELS,resume=None,weight=None,array_parse=None):
        self.IN_CHANNELS=channels
        self.LABELS = LABELS  # Label names
        self.N_CLASSES = len(self.LABELS)  # Number of classes
        self.net = SegNet(in_channels=self.IN_CHANNELS, out_channels=self.N_CLASSES)
        self.start_epoch = 0
        self.resume = resume
        if weight:
            self.WEIGHTS = torch.Tensor(weight)
        else:
            self.WEIGHTS = torch.ones(self.N_CLASSES)  # Weights for class balancing
        self.im_min=np.zeros(channels)
        self.array_parse=array_parse


    def test(self,test_files, label_files, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
        print('this function has been discard')
        return
        test_images = (1 / self.data_scale * np.asarray(io.imread(file), dtype='float32') for file in test_files)
        test_labels = (np.asarray(io.imread(file), dtype='uint8') for file in label_files)
        #     eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
        all_preds = []
        all_gts = []

        # Switch the network to inference mode
        self.net.eval()

        for img, gt in tqdm(zip(test_images, test_labels), total=len(label_files), leave=False):
            pred = np.zeros(img.shape[:2] + (self.N_CLASSES,))
            gt[img[:, :, 0] == 0] = 0
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         leave=False)):
                # Display in progress results
                #             if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                #                     _pred = np.argmax(pred, axis=-1)
                #                     fig = plt.figure()
                #                     fig.add_subplot(1,3,1)
                #                     plt.imshow(np.asarray(255 * img, dtype='uint8'))
                #                     fig.add_subplot(1,3,2)
                #                     plt.imshow(convert_to_color(_pred))
                #                     fig.add_subplot(1,3,3)
                #                     plt.imshow(gt)
                #                     clear_output()
                #                     plt.show()

                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]

                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)

            # Display the result
            #         clear_output()
            #         fig = plt.figure()
            #         fig.add_subplot(1,3,1)
            #         plt.imshow(np.asarray(255 * img, dtype='uint8'))
            #         fig.add_subplot(1,3,2)
            #         plt.imshow(convert_to_color(pred))
            #         fig.add_subplot(1,3,3)
            #         plt.imshow(gt)
            #         plt.show()
            gt = convert_from_color(gt)
            all_preds.append(pred)
            all_gts.append(gt)

            #         clear_output()
            # Compute some metrics
            self.metrics(pred.ravel(), gt.ravel())
        accuracy = self.metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
        if all:
            return accuracy, all_preds  # , all_gts


    def train(self):
        if not path.exists(self.MODEL_DIR):
            os.mkdir(self.MODEL_DIR)
        self.Validate_para()
        self.Set_Dataset()
        if self.is_Load_vgg:
            self.load_vgg()
        if self.GPU_TRAIN:
            self.net.cuda()
        if self.MODEL_LOAD is not None:
            self.net.load_state_dict(torch.load(self.MODEL_LOAD))
        base_lr = self.base_lr
        params_dict = dict(self.net.named_parameters())
        params = []
        for key, value in params_dict.items():
            if '_D' in key:
                # Decoder weights are trained at the nominal learning rate
                params += [{'params': [value], 'lr': base_lr}]
            else:
                # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
                params += [{'params': [value], 'lr': base_lr / 2}]

        optimizer = optim.SGD(self.net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
        # We define the scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
        if self.resume is not None:
            if os.path.isfile(self.resume):
                print("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                # best_prec1 = checkpoint['best_prec1']
                self.net.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))
        self._train_work(self.net, optimizer, self.train_epoch, scheduler, save_epoch=self.save_epoch)
        self.TRAINED=True


    def Set_Dataset(self):
        if self.DATASET == 'Potsdam':
            self.DATA_FOLDER = self.MAIN_FOLDER + 'Y_Ortho_IRRG/top_potsdam_{}_IRRG.tif'
            self.LABEL_FOLDER = self.MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
            self.ERODED_FOLDER = self.MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
        elif self.DATASET == 'Vaihingen':
            self.DATA_FOLDER = self.MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
            self.LABEL_FOLDER = self.MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
            self.ERODED_FOLDER = self.MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
        elif self.DATASET == 'Landsat':
            self.DATA_FOLDER = path.join(self.MAIN_FOLDER,'src')
            self.LABEL_FOLDER = path.join(self.MAIN_FOLDER ,'GTS')
            self.palette = {1: (255, 0, 0),  # Impervious surfaces (white)
                            0: (0, 0, 0)}  # Undefined (black)
            self.invert_palette = {v: k for k, v in self.palette.items()}
        elif self.DATASET == 'Landsat_meilan':
            self.DATA_FOLDER = path.join(self.MAIN_FOLDER,'src')
            self.LABEL_FOLDER = path.join(self.MAIN_FOLDER ,'GTS')
            self.palette = {1: (255, 0, 0),
                            2: (0,0,255),
                            3: (255,255,255),
                            0: (0, 0, 0),
                            4: (254,254,254)}  # Undefined (black)

            self.invert_palette = {v: k for k, v in self.palette.items()}
        elif self.DATASET == 'road_xiangbo':
            self.DATA_FOLDER = path.join(self.MAIN_FOLDER, 'src')
            self.LABEL_FOLDER = path.join(self.MAIN_FOLDER, 'GTS')
            self.palette = {1: (255, 255, 0),# highway
                            2: (151, 255, 255),# country
                            3: (255, 0, 0),# province
                            0: (255, 255, 255)}  # Undefined (black)

            self.invert_palette = {v: k for k, v in self.palette.items()}
        self.init_data()


    def init_data(self):
        if self.DATASET == 'Potsdam':
            all_files = sorted(glob(self.LABEL_FOLDER.replace('{}', '*')))
            all_ids = ["_".join(f.split('_')[3:5]) for f in all_files]
        elif self.DATASET == 'Vaihingen':
            # all_ids =
            all_files = sorted(glob(self.LABEL_FOLDER.replace('{}', '*')))
            all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
        elif self.DATASET == 'Landsat':
            self.all_files, self.label_files = src_gts_list(self.DATA_FOLDER, self.LABEL_FOLDER)
        elif self.DATASET == 'Landsat_meilan':
            self.all_files, self.label_files = src_gts_list(self.DATA_FOLDER, self.LABEL_FOLDER)
            self.data_scale=10000
            if self.im_max is None:
                self.im_max=10000*np.ones(self.IN_CHANNELS)
            self.train_set = EO_Dataset(self.all_files, self.label_files, cache=self.CACHE,array_parse=self.array_parse)
            # self.train_set.scale=self.data_scale
            self.train_set.im_min=self.im_min
            self.train_set.im_max=self.im_max
            self.train_set.palette=self.palette
            self.train_set.invert_palette=self.invert_palette
            self.train_set.WINDOW_SIZE=self.WINDOW_SIZE
            self.train_set.ignore.append(3)
            self.train_set.fillmarker={'marker':4, 'fill':3}
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.BATCH_SIZE)
        elif self.DATASET == 'road_xiangbo':
            self.all_files, self.label_files = src_gts_list(self.DATA_FOLDER, self.LABEL_FOLDER)
            self.data_scale = 255
            if self.im_max is None:
                self.im_max = 255 * np.ones(self.IN_CHANNELS)
            self.train_set = EO_Dataset(self.all_files, self.label_files, cache=self.CACHE)
            # self.train_set.scale=self.data_scale
            self.train_set.im_min = self.im_min
            self.train_set.im_max = self.im_max
            self.train_set.palette = self.palette
            self.train_set.invert_palette = self.invert_palette
            self.train_set.WINDOW_SIZE = self.WINDOW_SIZE
            self.train_set.ignore.append(0)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.BATCH_SIZE)


    def _train_work(self, net, optimizer, epochs, scheduler=None,  save_epoch=5):
        self.vis = visdom.Visdom()
        losses = []
        mean_losses = []
        weights = self.WEIGHTS
        weights = weights.cuda()

        # criterion = nn.NLLLoss2d(weight=weights)
        iter_ = 0
        # first_draw=True
        for e in range(self.start_epoch, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            net.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = Variable(data.cuda()), Variable(target.cuda())
                optimizer.zero_grad()
                output = net(data)
                loss = CrossEntropy2d(output, target, weight=weights)
                loss.backward()
                optimizer.step()

                # losses[iter_] = loss.data[0]
                # mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])
                # losses.append(loss.data[0])
                losses.append(loss.item())
                mean_losses.append(np.mean(losses[max(0, iter_ - 50):iter_]))
                if iter_ % 50 == 0:
                    # rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')[:, :,
                    #       4:1:-1]
                    rgb = np.asarray(255 * data.data.cpu().numpy()[0], dtype='uint8')[4:1:-1,:,:]
                    pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                    gt = target.data.cpu().numpy()[0]
                    # if first_draw:
                    self.vis.close()
                    loss_vis=self.vis.line(np.array(mean_losses))
                    loss_text=self.vis.text('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                        e, epochs, batch_idx, len(self.train_loader),
                        100. * batch_idx / len(self.train_loader), loss.data.item(), accuracy(pred, gt)))
                    RGB_win=self.vis.image(rgb,opts=dict(title='RGB', caption='RGB'))
                    GT_win=self.vis.image(convert_to_color(gt,self.palette),opts=dict(title='Ground truth', caption='Ground truth'))
                    Pred_win=self.vis.image(convert_to_color(pred,self.palette),opts=dict(title='Prediction', caption='Prediction'))
                        # first_draw=False
                    # else:
                    #     self.vis.line(np.array(mean_losses),win=loss_vis)
                    #     self.vis.text('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    #         e, epochs, batch_idx, len(self.train_loader),
                    #         100. * batch_idx / len(self.train_loader), loss.data.item(), accuracy(pred, gt)),win=loss_text)
                    #     self.vis.image(rgb,win=RGB_win,opts=dict(title='RGB', caption='RGB'))
                    #     self.vis.image(convert_to_color(gt,self.palette),win=GT_win,opts=dict(title='Ground truth', caption='Ground truth'))
                    #     self.vis.image(convert_to_color(pred,self.palette),win=Pred_win,opts=dict(title='Prediction', caption='Prediction'))
                    # plt.plot(mean_losses[:iter_]) and plt.show()
                    # fig = plt.figure()
                    # fig.add_subplot(131)
                    # plt.imshow(rgb)
                    # plt.title('RGB')
                    # fig.add_subplot(132)
                    # plt.imshow(convert_to_color(gt))
                    # plt.title('Ground truth')
                    # fig.add_subplot(133)
                    # plt.title('Prediction')
                    # plt.imshow(convert_to_color(pred))
                    # plt.show()
                iter_ += 1

                del (data, target, loss)

            if e % save_epoch == 0:
                # We validate with the largest possible stride for faster computing
                #             acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
                # torch.save(net, path.join(self.MODEL_DIR, self.MODEL_BASENAME+'_epoch{}'.format(e)))
                self.save_checkpoint({
                    'epoch': e + 1,
                    # 'arch': args.arch,
                    'state_dict': self.net.state_dict(),
                    # 'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },filename=path.join(self.MODEL_DIR, self.MODEL_BASENAME+'_epoch{}'.format(e)))
        # torch.save(net.state_dict(), path.join(self.MODEL_DIR, self.MODEL_BASENAME+'_final'))
        self.save_checkpoint({
            'epoch': epochs + 1,
            # 'arch': args.arch,
            'state_dict': self.net.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },filename=self.MODEL_BASENAME+'_final')


    def save_checkpoint(self,state, is_best=False, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')


    def metrics(self,predictions, gts):
        label_values = self.LABELS
        cm = confusion_matrix(
            gts,
            predictions,
            range(len(label_values)))

        print("Confusion matrix :")
        print(cm)

        print("---")

        # Compute global accuracy
        total = sum(sum(cm))
        accuracy = sum([cm[x][x] for x in range(len(cm))])
        accuracy *= 100 / float(total)
        print("{} pixels processed".format(total))
        print("Total accuracy : {}%".format(accuracy))

        print("---")

        # Compute F1 score
        F1Score = np.zeros(len(label_values))
        for i in range(len(label_values)):
            try:
                F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
            except:
                # Ignore exception if there is no element in class i for test set
                pass
        print("F1Score :")
        for l_id, score in enumerate(F1Score):
            print("{}: {}".format(label_values[l_id], score))

        print("---")

        # Compute kappa coefficient
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (pa - pe) / (1 - pe);
        print("Kappa: " + str(kappa))
        return accuracy


    def load_vgg(self):
        try:
            from urllib.request import URLopener
        except ImportError:
            from urllib import URLopener
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}
        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.net.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))

        try:
            self.net.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


    def train_set_list(self,src_folder,label_folder):
        srclist=glob(os.path.join(src_folder,'*.tif'))
        labellist=[os.path.join(label_folder,os.path.split(i)[-1].replace('src','label')) for i in srclist]
        return srclist,labellist


    def Validate_para(self):
        pass


    def rescale(self,img):
        # accept img=[band_num,height,width]
        img=np.asarray(img,'float32')
        for band in range(img.shape[0]):
            img[band] = exposure.rescale_intensity(img[band], in_range=(self.im_min[band], self.im_max[band]))
        return img


    def Test(self,src_folder, label_folder):
        test_files,label_files=src_gts_list(src_folder, label_folder)
        if not self.TRAINED:
            self.net.load_state_dict(torch.load(self.MODEL_TEST)['state_dict'])
        all_preds = self._test_work(test_files, label_files, all=True, stride=200)
        for src_f, pred_array in zip(test_files, all_preds):
            save_pred(src_f, pred_array)


    def _test_work(self,test_files, label_files,all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
        net=self.net
        # test_images = (1 / self.data_scale * np.asarray(rio.open(file).read().transpose([1,2,0]), dtype='float32') for file in test_files)
        test_images = (self.rescale(rio.open(file).read()).transpose([1,2,0]) for file in test_files)
        test_labels = (np.asarray(io.imread(file), dtype='uint8') for file in label_files)
        #     eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
        all_preds = []
        all_gts = []

        # Switch the network to inference mode
        net.eval()

        for img, gt in tqdm(zip(test_images, test_labels), total=len(label_files), leave=False):
            pred = np.zeros(img.shape[:2] + (self.N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         leave=False)):
                # Display in progress results
                #             if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                #                     _pred = np.argmax(pred, axis=-1)
                #                     fig = plt.figure()
                #                     fig.add_subplot(1,3,1)
                #                     plt.imshow(np.asarray(255 * img, dtype='uint8'))
                #                     fig.add_subplot(1,3,2)
                #                     plt.imshow(convert_to_color(_pred))
                #                     fig.add_subplot(1,3,3)
                #                     plt.imshow(gt)
                #                     clear_output()
                #                     plt.show()

                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]

                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)

            # Display the result
            # clear_output()
            # fig = plt.figure()
            # fig.add_subplot(1, 3, 1)
            # plt.imshow(np.asarray(255 * img, dtype='uint8'))
            # fig.add_subplot(1, 3, 2)
            # plt.imshow(convert_to_color(pred))
            # fig.add_subplot(1, 3, 3)
            # plt.imshow(gt)
            # plt.show()
            gt = convert_from_color(gt)
            all_preds.append(pred)
            all_gts.append(gt)

            # clear_output()
            # Compute some metrics
            self.metrics(pred.ravel(), gt.ravel())
        #         accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
        if all:
            return all_preds  # , all_gts
    #     else:
    #         return accuracy


    def Predict(self,src_list,result_dir):
        self.result_dir = result_dir
        if not path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        if not self.TRAINED:
            self.net.load_state_dict(torch.load(self.MODEL_TEST)['state_dict'])
        self._predict_work(src_list)
        # for src_f, pred_array in zip(src_list, all_preds):
        #     save_pred(src_f, pred_array)


    def _predict_work(self,test_files,window_size=WINDOW_SIZE):
        batch_size = self.BATCH_SIZE
        net=self.net
        # stride = WINDOW_SIZE[0]
        stride = self.STRIDE
        # test_images = (1 / self.data_scale * np.asarray(rio.open(file).read().transpose([1,2,0]), dtype='float32') for file in test_files)
        test_images = (self.rescale(rio.open(file).read()).transpose([1,2,0]) for file in test_files)
        # Switch the network to inference mode
        net.eval()
        for img,src_f in tqdm(zip(test_images,test_files), total=len(test_files), leave=False):
            pred = np.zeros(img.shape[:2] + (self.N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         leave=False)):
                # Display in progress results
                #             if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                #                     _pred = np.argmax(pred, axis=-1)
                #                     fig = plt.figure()
                #                     fig.add_subplot(1,3,1)
                #                     plt.imshow(np.asarray(255 * img, dtype='uint8'))
                #                     fig.add_subplot(1,3,2)
                #                     plt.imshow(convert_to_color(_pred))
                #                     fig.add_subplot(1,3,3)
                #                     plt.imshow(gt)
                #                     clear_output()
                #                     plt.show()

                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)
            pred = np.argmax(pred, axis=-1)
            save_pred(src_f, convert_to_color(pred,self.palette) ,self.result_dir)
            # all_preds.append(pred)
        # if all:
        #     return all_preds  # , all_gts


class EO_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_files, label_files,
                 cache=False, augmentation=True,array_parse=None):
        super(EO_Dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = data_files
        self.label_files = label_files

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.scale = 255
        self.data_cache_ = {}
        self.label_cache_ = {}
        self.invert_palette = None
        self.palette = None
        self.WINDOW_SIZE = None
        self.len = 10000
        self.nodata = None
        self.ignore = []
        self.im_min = None
        self.im_max = None
        self.array_parse = array_parse
        self.fillmarker = None


    def __len__(self):
        return self.len


    @classmethod
    # 上下翻转和左右反转
    def data_augmentation(self, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        # if self.array_parse!=None:

        return tuple(results)


    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        # print(i,' ',self.data_files[random_idx])
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1] data_cach为字典
            data = rio.open(self.data_files[random_idx]).read()
            # data = io.imread(self.data_files[random_idx]).transpose((2, 0, 1))
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            # label = convert_from_color(io.imread(self.label_files[random_idx]),self.invert_palette)
            label = convert_from_color(np.transpose(rio.open(self.label_files[random_idx]).read(),(1,2,0)),self.invert_palette)

            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        # todo skip nodate, by new function
        #         x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        #         data_p = 1/255 *data[:, x1:x2,y1:y2].astype('float32')
        #         label_p = label[x1:x2,y1:y2].astytpe('int64')
        data_p, label_p = self.get_random_sample(data, label, self.WINDOW_SIZE)
        if self.fillmarker is not None:
            label_p[label_p==self.fillmarker['marker']] = self.fillmarker['fill']
        if self.array_parse!=None:
            data_p=self.array_parse(data_p)
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


    def get_random_sample(self, data, label, window_shape):
        pixels = window_shape[0] * window_shape[1]
        while (True):
            skip=False
            x1, x2, y1, y2 = get_random_pos(data, self.WINDOW_SIZE)
            data_p = data[:, x1:x2, y1:y2]
            if np.count_nonzero(data_p[0, :, :] == 0) > pixels * 0.4:
                continue
            label_p = np.asarray(label[x1:x2, y1:y2], 'int64')
            for item in self.ignore:
                if np.count_nonzero(label_p == item) > pixels * 0.999:
                    skip=True
            if skip:
                continue
            data_p = np.asarray(data_p, 'float32')
            # data_p = 1 / self.scale * np.asarray(data_p, 'float32')

            # if np.unique(c, return_counts=True)[1].max()
            # if np.count_nonzero(label_p == 1) < pixels * 0.07:
            #     continue
            # label_p[data_p[0, :, :] == 0] = 0
            # print(data_p.shape)
            for band in range(data_p.shape[0]):
                data_p[band]=exposure.rescale_intensity(data_p[band],in_range=(self.im_min[band],self.im_max[band]))
            return data_p, label_p


def src_gts_list(src_folder, label_folder):
    # srclist=glob(os.path.join(src_folder,'*.tif'))
    labellist=glob(os.path.join(label_folder,'*.tif'))
    # labellist=[os.path.join(label_folder,os.path.split(i)[-1].replace('src','label')) for i in srclist]
    srclist=[os.path.join(src_folder,os.path.split(i)[-1].replace('label','src')) for i in labellist]
    return srclist,labellist


def save_pred(src_f,pred_array,pred_dir=r'.\refine_work\predict'):
    src = rio.open(src_f)
    profile = src.profile
    profile['dtype']='uint8'
    profile['count']=3
    pred_array=np.asarray(pred_array,dtype='uint8')
    PRED_fname=path.join(pred_dir,'pred_'+path.split(src_f)[-1])
    with rio.open(PRED_fname, 'w', **profile) as dst:
        dst.write(pred_array)


def convert_to_color(arr_2d,palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return np.transpose(arr_3d,[2,0,1])


def convert_from_color(arr_3d,palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            #             if np.count_nonzero(top[x:x+window_size[0], y:y+window_size[1]])<top.shape[0]*top.shape[1]*top.shape[2]*0.05:
            #                 continue
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            #             if np.count_nonzero(top[x:x+window_size[0], y:y+window_size[1]])<top.shape[0]*top.shape[1]*top.shape[2]*0.05:
            #                 continue
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        # x = F.log_softmax(self.conv1_1_D(x),dim = 0)
        x = self.conv1_1_D(x)
        return x

if __name__=='__main__':
    landsat_torch = EOtorch()
    landsat_torch.GPU_TRAIN = True
    landsat_torch.save_epoch = 5
    landsat_torch.base_lr=0.01
    landsat_torch.DATASET = 'Landsat'
    landsat_torch.MAIN_FOLDER = 'C:\\Users\\cngs\\Desktop\\jupyter\\landsat\\shandong_landsat_band7'
    landsat_torch.MODEL_DIR = r''
    landsat_torch.MODEL_BASENAME=''
    landsat_torch.train()

    


