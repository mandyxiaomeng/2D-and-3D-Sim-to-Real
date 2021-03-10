import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pointnet_plus import PointNet_plus
from dataloader import Cadnet_data, Camnet_data
from sklearn.metrics import classification_report
import csv
import time
import os
import argparse
import pandas as pd
import csv
import numpy as np

from tensorboardX import SummaryWriter

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='cadnet')
parser.add_argument('-target', '-t', type=str, help='target dataset', default='camnet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=10)
#parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0,1,2')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=1)
parser.add_argument('-lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('-scaler',type=float, help='scaler of learning rate', default=1.)
parser.add_argument('-datadir', type=str, help='directory of data', default='C:/Mandy/PhD/3D_DA/Program1/data/Industrypointdan/pointdan/')
#parser.add_argument('-datadir', type=str, help='directory of data', default='./data/')
parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./logs/src_m_s_ss')
args = parser.parse_args()

if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
    from tensorboardX import SummaryWriter
    os.makedirs(os.path.join(os.getcwd(), args.tb_log_dir))
writer = SummaryWriter(log_dir=args.tb_log_dir)

device = torch.device('cuda:0')
#device = 'cuda'
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = args.batchsize
#BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
num_class = 4
dir_root = os.path.join(args.datadir, 'PointDA_data/')
epo = []
o_acc = []
acc_los = []
losss = []



def classification_report_csv(report,best_target_class_acc):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:6]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        clas = int(row_data[0][-1])-1
        row['Accuracy'] = float(best_target_class_acc[clas][clas])   
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe
    # dataframe.to_csv('classification_report.csv', index = False)



def main():
    print ('Start Training\nInitiliazing\n')
    print('src:', args.source)
    print('tar:', args.target)

    data_func={'camnet': Camnet_data, 'cadnet': Cadnet_data}

    source_train_dataset = data_func[args.source](pc_input_num=1024, status='train', aug=True, pc_root = dir_root + args.source)
    source_test_dataset = data_func[args.source](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.source)
    target_test_dataset = data_func[args.target](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.target)


    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_test = len(target_test_dataset)


    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    target_test_dataloader = DataLoader(target_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)

    print('num_source_train: {:d}, num_source_test: {:d}, num_target_test: {:d}'.format(
        num_source_train, num_source_test, num_target_test))
    print('batch_size:', BATCH_SIZE)

    # Model

    model = PointNet_plus()
    model.to(device)
    #model = nn.DataParallel(model, device_ids=[1,0]) ###########for multi GPU
    #model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)


    # Optimizer
    remain_epoch=50

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs+remain_epoch)
    # lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)


    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by half by every 5 or 10 epochs"""
        if epoch > 0:
            if epoch <= 30:
                lr = args.lr * args.scaler * (0.5 ** (epoch // 5))
            else:
                lr = args.lr * args.scaler * (0.5 ** (epoch // 10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            writer.add_scalar('lr_dis', lr, epoch)

    def discrepancy(out1, out2):
        """discrepancy loss"""
        out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
        return out

    def make_variable(tensor, volatile=False):
        """Convert Tensor to Variable."""
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return Variable(tensor, volatile=volatile)



    best_source_test_acc = 0
    best_target_test_acc = 0
    best_target_class_acc = torch.zeros(4,4)
    rd = {'class':['0'], 'precision': [0.0], 'recall': [0.0], 'f1_score': [0.0], 'support': [0.0],'Accuracy':[0.0]}
    dataframe_full = pd.DataFrame.from_dict(rd)


    for epoch in range(max_epoch):
        since_e = time.time()

        lr_schedule.step(epoch=epoch)
        adjust_learning_rate(optimizer, epoch)
        print(lr_schedule.get_lr())

        writer.add_scalar('lr', lr_schedule.get_lr(), epoch)

        model.train()
        loss_total = 0
        data_total = 0

        for batch_idx, batch_s in enumerate(source_train_dataloader):

            data, label = batch_s
            data = torch.squeeze(data)
            ab,bc,cd = data.shape
            data = torch.reshape(data, (ab, cd, bc))
            data = data.to(device=device)
            label = label.to(device=device).long()

            output_s = model(data)

            loss_s = criterion(output_s, label)

            loss_s.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss_s.item() * data.size(0)
            data_total += data.size(0)

            if (batch_idx + 1) % 50 == 0:
                print('Train:{} [{} /{}  loss: {:.4f} \t]'.format(
                epoch, data_total, num_source_train, loss_total/data_total))
                losss.append(loss_total/data_total)

        torch.save(model, './test.h5')
        #print('history dict:', history.history)
        #print('history:',history.history.keys())


        with torch.no_grad():
            #torch.save("./pointnet++200.h5", model)
            model.eval()

            # ------------Source------------
            loss_total = 0
            correct_total = 0
            data_total = 0
            acc_class = torch.zeros(4, 1)
            acc_to_class = torch.zeros(4, 1)
            acc_to_all_class = torch.zeros(4, 4)


            for batch_idx, (data, label) in enumerate(source_test_dataloader):

                data = torch.squeeze(data)
                ab,bc,cd = data.shape
                data = torch.reshape(data, (ab, cd, bc))
                data = data.to(device=device)
                label = label.to(device=device).long()
                output = model(data)
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)

                acc = pred == label

                for j in range(0, 4):
                    label_j_list = (label == j)
                    acc_class[j] += (pred[acc] == j).sum().cpu().float()
                    acc_to_class[j] += label_j_list.sum().cpu().float()
                    for k in range(0, 4):
                        acc_to_all_class[j, k] += (pred[label_j_list] == k).sum().cpu().float()

                loss_total += loss.item() * data.size(0)
                correct_total += torch.sum(pred == label)
                data_total += data.size(0)

            pred_loss = loss_total / data_total
            pred_acc = correct_total.double() / data_total

            if pred_acc > best_source_test_acc:
                best_source_test_acc = pred_acc
            # for j in range(0, 4):
            #     for k in range(0, 4):
            #         acc_to_all_class[j, k] = acc_to_all_class[j, k] / acc_to_class[j]
            print('Source Test:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Source Test Acc: {:.4f}]'.format(
                epoch, pred_acc, pred_loss, best_source_test_acc
            ))
            writer.add_scalar('accs/source_test_acc', pred_acc, epoch)

            # ------------Target------------
            loss_total = 0
            correct_total = 0
            data_total = 0
            acc_class = torch.zeros(4,1)
            acc_to_class = torch.zeros(4,1)
            acc_to_all_class = torch.zeros(4,4)
            total_pred = np.empty(0)
            total_label = np.empty(0)


            for batch_idx, (data,label) in enumerate(target_test_dataloader):

                data = torch.squeeze(data)
                ab,bc,cd = data.shape
                data = torch.reshape(data, (ab, cd, bc))
                data = data.to(device=device)
                label = label.to(device=device).long()
                output = model(data)
                loss = criterion(output, label)
                _, pred = torch.max(output, 1)

                acc = pred == label

                for j in range(0, 4):
                    label_j_list = (label == j)
                    acc_class[j] += (pred[acc] == j).sum().cpu().float()
                    acc_to_class[j] += label_j_list.sum().cpu().float()
                    for k in range(0, 4):
                        acc_to_all_class[j, k] += (pred[label_j_list] == k).sum().cpu().float()

                loss_total += loss.item() * data.size(0)
                correct_total += torch.sum(pred == label)
                total_pred = np.concatenate([total_pred,pred.cpu().numpy()])
                total_label = np.concatenate([total_label,label.cpu().numpy()])
                data_total += data.size(0)

            pred_class_acc = acc_class/acc_to_class
            pred_loss = loss_total/data_total
            pred_acc = correct_total.double()/data_total
            
            for j in range(0, 4):
                for k in range(0, 4):
                    acc_to_all_class[j, k] = acc_to_all_class[j, k]/acc_to_class[j]
            
            target_names = ['class 1', 'class 2', 'class 3', 'class 4']
            report = classification_report(total_label, total_pred, labels = [0,1,2,3],target_names=target_names)
            dataframe = classification_report_csv(report,best_target_class_acc)
            dataframe_full=dataframe_full.append(dataframe)

            if pred_acc > best_target_test_acc:
                best_target_test_acc = pred_acc
                best_target_class_acc = acc_to_all_class
                best_classification = dataframe


            epo.append(epoch)
            o_acc.append(pred_acc.cpu().numpy())
            acc_los.append(pred_loss)


            print ('Target :{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Target Acc: {:.4f}]'.format(
            epoch, pred_acc, pred_loss, best_target_test_acc
            ))
            print('Best Target precision: ')
            print(best_classification)
            print ('Best Target class accuracy : [Class 1: {:.4f} \t Class 2: {:.4f} \t Class 3: {:.4f}  \t Class 4: {:.4f}]'.format(
            best_target_class_acc[0,0], best_target_class_acc[1,1], best_target_class_acc[2,2], best_target_class_acc[3,3]))
            writer.add_scalar('accs/target1_test_acc', pred_acc, epoch)

        time_pass_e = time.time() - since_e
        print('The {} epoch takes {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 60, time_pass_e % 60))
        print(args)
        print(' ')
    return dataframe_full



if __name__ == '__main__':
    since = time.time()
    dataframe_full = main()
    dataframe_full.to_csv('classification_report.csv', index = False)
    time_pass = since - time.time()
    rows = zip(epo,o_acc,acc_los)
    with open('output.csv', "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    
print(losss)


