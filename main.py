from models.FB import FB
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import genSpoof_list, Dataset_ASVspoof2019, Dataset_ASVspoof2021_eval
from config import config
from loss import *

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_v, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_v = batch_v.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            feats,batch_out = model(batch_x,batch_v)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

    # with torch.no_grad():
    #     for batch_x, batch_y in dev_loader:
    #         batch_size = batch_x.size(0)
    #         num_total += batch_size
    #         batch_x = batch_x.to(device)
    #         batch_y = batch_y.view(-1).type(torch.int64).to(device)
    #         # batch_y = batch_y.to(device)
    #         batch_out = model(batch_x)
    #         _, batch_pred = batch_out.max(dim=1)
    #         num_correct += (batch_pred == batch_y).sum(dim=0).item()

    return 100 * (num_correct / num_total)


def evaluate_eer():
    pass


def produce_evaluation_file(dataset, model, device, save_path,oc):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    model.eval()
    oc.eval()
    ii = 0
    for batch_x, batch_v,utt_id in data_loader:
        fname_list = []
        score_list = []
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_v=batch_v.to(device)
        feats,batch_out = model(batch_x,batch_v)
        batch_loss,batch_score=oc(feats,0)
        #batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        ii += 1
        if ii % 100 == 0:
            print('Eval process:{}/{}'.format(ii, data_loader.__len__()))

        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()




    # for batch_x, utt_id in data_loader:
    #     fname_list = []
    #     score_list = []
    #     batch_size = batch_x.size(0)
    #     batch_x = batch_x.to(device)
    #     batch_out = model(batch_x)
    #     batch_score=batch_out.argmax(dim=1)
    #     #batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
    #     # add outputs
    #     fname_list.extend(utt_id)
    #     score_list.extend(batch_score.tolist())
    #     ii += 1
    #     if ii % 100 == 0:
    #         print('Eval process:{}/{}'.format(ii, data_loader.__len__()))
    #
    #     with open(save_path, 'a+') as fh:
    #         for f, cm in zip(fname_list, score_list):
    #             fh.write('{} {}\n'.format(f, cm))
    #     fh.close()

    print('Scores saved to {}'.format(save_path), flush=True)
def train_epoch(train_loader, model, optim, device,criterion,oc,oc_optim,lr_scheduler=None):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    oc.train()
    #oc.eval()
    for batch_x, batch_v, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_v = batch_v.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        feats,batch_out = model(batch_x,batch_v)
        #batch_loss,batch_score= oc(feats, batch_y)
        batch_loss=criterion(batch_out,batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        #if ii % 100 == 0:
        #   print('\r \t {:.2f}'.format((num_correct / num_total) * 100))
         #   print('{}/{} :{:.2f}'.format(ii, train_loader.__len__(), (num_correct / num_total) * 100))
        optim.zero_grad()
        oc_optim.zero_grad()
        batch_loss.backward()
        optim.step()
        oc_optim.step()

    # for batch_x, batch_y in train_loader:
    #     batch_size = batch_x.size(0)
    #     num_total += batch_size
    #     ii += 1
    #     batch_x = batch_x.to(device)
    #     batch_y = batch_y.view(-1).type(torch.int64).to(device)
    #     batch_out = model(batch_x)
    #     batch_loss = criterion(batch_out, batch_y)
    #     _, batch_pred = batch_out.max(dim=1)
    #     num_correct += (batch_pred == batch_y).sum(dim=0).item()
    #     running_loss += (batch_loss.item() * batch_size)
    #     #if ii % 100 == 0:
    #     #   print('\r \t {:.2f}'.format((num_correct / num_total) * 100))
    #      #   print('{}/{} :{:.2f}'.format(ii, train_loader.__len__(), (num_correct / num_total) * 100))
    #     optim.zero_grad()
    #     batch_loss.backward()
    #     optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100

    return running_loss, train_accuracy


if __name__ == '__main__':
    # config
    batch_size = config.train.batch_size
    lr = config.train.optim.lr
    weight_decay = config.train.optim.weight_decay
    betas = config.train.optim.betas
    eps = config.train.optim.eps
    model_path = config.model.model_path
    is_train = config.is_train
    num_epochs = config.train.epoch
    save_path = config.save_path
    tag = config.tag
    num_workers = config.train.num_works
    criterion_path=config.criterion_path
    # save path
    save_path = os.path.join(save_path, tag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device), flush=True)

    # model
    pretrained = False
    model = FB(pretrained=pretrained)
    print('pretrained: {}'.format(pretrained))
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print('Model Params: {}'.format(nb_params))

    #load my criterion

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('Model loaded : {}'.format(model_path))

    # train data
    d_label_trn, file_train = genSpoof_list(
        dir_meta=config.dataset.train_file_list,
        is_train=True,
        is_eval=False
    )
    print('no. of training trials', len(file_train), flush=True)

    train_set = Dataset_ASVspoof2019(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=config.dataset.train_dir
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    del train_set, d_label_trn

    # valid data
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=config.dataset.dev_file_list,
        is_train=False,
        is_eval=False
    )
    print('no. of validation trials', len(file_dev), flush=True)

    dev_set = Dataset_ASVspoof2019(
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=config.dataset.dev_dir
    )
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    del dev_set, d_label_dev

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    ocsoftmax = OCSoftmax(feat_dim=256,r_fake=-0.5,r_real=0.9).to(device)
    oc_optim=torch.optim.SGD(ocsoftmax.parameters(), lr=lr,weight_decay=1.0)
    if criterion_path:
        ocsoftmax.load_state_dict(torch.load(criterion_path,map_location=device))
        print('criterion loaded : {}'.format(model_path))
    ocsoftmax.r_fake=-0.9
    ocsoftmax.r_real=0.5
    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2,
                                                           verbose=False, factor=0.5)

    # evaluation
    if not is_train:
        if config.eval.use_progress_set:
            file_progress = genSpoof_list(
                dir_meta=config.dataset.progress_file_list,
                is_train=False,
                is_eval=True
            )
            print('no. of progress trials', len(file_progress), flush=True)
            progress_set = Dataset_ASVspoof2021_eval(
                list_IDs=file_progress,
                base_dir=config.dataset.progress_dir
            )
            produce_evaluation_file(progress_set, model, device, os.path.join(save_path, 'score_progress.txt'),ocsoftmax)
            exit(1)

        else:
            label_eval,file_eval = genSpoof_list(
                dir_meta=config.dataset.eval_file_list,
                is_train=True,
                is_eval=False
            )
            print('no. of eval trials', len(file_eval), flush=True)
            eval_set = Dataset_ASVspoof2021_eval(
                list_IDs=file_eval,
                base_dir=config.dataset.eval_dir
            )
            produce_evaluation_file(eval_set, model, device, os.path.join(save_path, 'score_eval.txt'))
            # produce_evaluation_file(eval_set, model, device,
            #                         '/home/wangli/VoiceSpoofingDetection/exps/Baseline_FB/epoch87.txt')
            exit(1)

    # loss
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # train and valid
    #best_acc = 0
    best_acc=10

    print('Number of epochs: ',num_epochs)
    with open('./train_21.txt','w') as f:
        for epoch in range(num_epochs):
            #running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, device, criterion)
            running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, device, criterion,ocsoftmax,oc_optim)
            print('lr: {:.7f}'.format(optimizer.param_groups[0]['lr']))
            #valid_accuracy = evaluate_accuracy(dev_loader, model, device)
            valid_accuracy=0.0
            print('{} - {} - {:.2f} - {:.2f}'.format(epoch, running_loss, train_accuracy, valid_accuracy), flush=True)
            f.write('{} - {} - {:.2f} - {:.2f}\n'.format(epoch, running_loss, train_accuracy, valid_accuracy))

            # scheduler.step(valid_accuracy)
            # if valid_accuracy > best_acc:
            #     print('best model find at epoch', epoch, flush=True)
            # best_acc = max(valid_accuracy, best_acc)

            scheduler.step(-running_loss)
            if running_loss < best_acc:
                print('best model find at epoch', epoch, flush=True)
            best_acc=min(running_loss,best_acc)
            torch.save(model.state_dict(),
                       os.path.join(save_path, 'model_epoch{}_validAcc{}.pth'.format(epoch, str(running_loss)[:6])))
            torch.save(ocsoftmax.state_dict(),os.path.join('./home/loss/','loss_epoch{}_validAcc{}.pth'.format(epoch,str(running_loss)[:6])))
            print()
