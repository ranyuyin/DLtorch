import EOutil
landsat_torch = EOutil.EOtorch(16,['nodata','builtup','water','nobuilt'])#,weight=[1, 1, 1, 0.9])
landsat_torch.is_Load_vgg=False
landsat_torch.GPU_TRAIN = True
landsat_torch.save_epoch = 5
landsat_torch.train_epoch=50
landsat_torch.base_lr=0.01
landsat_torch.BATCH_SIZE=10
landsat_torch.DATASET = 'Landsat_meilan'
landsat_torch.MAIN_FOLDER = r'./data223/0919/sample'
landsat_torch.MODEL_DIR = r'./model_lanmei'
# landsat_torch.MODEL_LOAD = r"C:\Users\cngs\Desktop\jupyter\landsat\slim_work\band6_epoch20"
landsat_torch.MODEL_BASENAME='lanmei'
landsat_torch.LABELS=['nodata','no-built','water','builtup']
landsat_torch.im_min=[0,0,0,0,0,0,-2000,-4000,-10000,-6000,-6000,-10000,-7000,-8000,-10000,0]
landsat_torch.im_max=[1000,1500,2000,4000,4000,4000,10000,10000,10000,5000,4000,1000,10000,10000,7000,7000]
landsat_torch.train()