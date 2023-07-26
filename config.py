from easydict import EasyDict as edict

config = edict()
config.is_train =False
config.save_path = './home/'
config.tag = 'Baseline_FB'

config.model = edict()
config.model.model_path = './home/Baseline_FB/model_epoch39_validAcc1.1207.pth'
#config.model.model_path = './home/Baseline_FB/model_epoch50_validAcc92.727.pth'
#config.model.model_path = None

#config.criterion_path=None
config.criterion_path='./home/loss/loss_epoch39_validAcc1.1207.pth'

config.dataset = edict()
config.dataset.train_file_list = './ASVspoof2019.PA.cm.train.trn.txt'
config.dataset.train_dir = './ASVspoof2019_PA_train/'
config.dataset.dev_file_list = './ASVspoof2019.PA.cm.dev.trl.txt'
config.dataset.dev_dir = './ASVspoof2019_PA_dev/'
config.dataset.eval_file_list = './ASVspoof2019.PA.cm.eval.trl.txt'
config.dataset.eval_dir = './ASVspoof2019_PA_eval/'
config.dataset.progress_file_list = './ASVspoof2021.PA.cm.eval.progress.trl.txt'
config.dataset.progress_dir = './ASVspoof2021_PA_eval_progress/'

config.train = edict()
config.train.batch_size = 128
#config.train.epoch = 100
config.train.epoch = 60
config.train.num_works = 8
config.train.optim = edict()
config.train.optim.lr = 0.001
config.train.optim.weight_decay = 0.0001
config.train.optim.betas = (0.9, 0.99)
config.train.optim.eps = 1e-9

config.eval = edict()
config.eval.use_progress_set = True
