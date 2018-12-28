import EOutil
from glob import glob
landsat_torch = EOutil.EOtorch(16,['nodata','builtup','water','nobuilt'])#,weight=[1, 1, 1, 0.9])
landsat_torch.MODEL_TEST = r"Z:\yinry\lanmei_watershed\model_segnet\lanmei_final"
srclist = glob(r'Z:\yinry\lanmei_watershed\feature_merge\*.tif')
landsat_torch.im_min = [0,0,0,0,0,0,-2000,-4000,-10000,-6000,-6000,-10000,-7000,-8000,-10000,0]
landsat_torch.im_max = [1000,1500,2000,4000,4000,4000,10000,10000,10000,5000,4000,1000,10000,10000,7000,7000]
landsat_torch.net.cuda()
landsat_torch.BATCH_SIZE = 60
landsat_torch.STRIDE = 150
landsat_torch.palette = {1: (255, 0, 0),
                         2: (0, 0, 255),
                         3: (255, 255, 255),
                         0: (0, 0, 0),
                         4: (254, 254, 254)}  # Undefined (black)
landsat_torch.Predict(srclist, r'Z:\yinry\lanmei_watershed\model_segnet\pred_final')