import argparse
import time, os
import pickle as pickle
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

from transformers import BertModel, BertTokenizer, CLIPModel, CLIPImageProcessor, CLIPProcessor

#import clip
import pytorch_warmup as warmup

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
torch.cuda.set_device(0)

GT_size = 224
word_token_length = 197 # identical to size of MAE
image_token_length = 197

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
token_chinese = BertTokenizer.from_pretrained('bert-base-chinese',  )
token_english = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True,  
     
)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()

from utils import Progbar, create_dir, stitch_images, imsave
stateful_metrics = ['L-RealTime','lr','APEXGT','empty','exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
                                'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']


def main(args):
    print(args)

    seed = 25
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    setting = {}
    setting['train_dataname'] = args.train_dataset
    setting['val_dataname'] = args.test_dataset
    setting['val'] = args.val

    custom_batch_size = args.batch_size
    custom_num_epochs = args.epochs
    train_dataset, validate_dataset, train_loader, validate_loader = None,None,None,None
    shuffle, num_workers = True, 4
    train_sampler = None


    ########## train dataset ####################
    if setting['train_dataname']=='weibo':
        print("Using weibo as training")
        from data.dataset_local import H5FeatureDataset
        train_dataset = H5FeatureDataset("/home/shunlizhang/zwk/CMFQN/data/weibo_dataset_local/features_dif.h5")
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=shuffle,
                                num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                pin_memory=True)
            
        setting['thresh'] = 0.5
        print(f"thresh:{setting['thresh']}")

    elif setting['train_dataname']=='twitter':
        print("Using twitter as training")
        from data.dataset_local import H5FeatureDataset
        train_dataset = H5FeatureDataset("/home/shunlizhang/zwk/CMFQN/data/twitter_dataset_local/features_dif.h5")
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=shuffle,
                                num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                pin_memory=True)
        setting['thresh'] = 0.5
        print(f"thresh:{setting['thresh']}")

    elif setting['train_dataname']=='ikcest':
        print("Using ikcest as training")
        from data.dataset_local import H5FeatureDataset
        train_dataset = H5FeatureDataset("/home/shunlizhang/zwk/CMFQN/data/ikcest_dataset_local/features_dif.h5")
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=shuffle,
                                num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                pin_memory=True)
        setting['thresh'] = 0.5
        print(f"thresh:{setting['thresh']}")

    ########## validate dataset ####################
    if setting['val_dataname']=='weibo':
        print("Using weibo as inference")
        from data.dataset_local import H5FeatureDataset
        validate_dataset = H5FeatureDataset("/home/shunlizhang/zwk/CMFQN/data/weibo_dataset_local/val_features_dif.h5")
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                    num_workers=4, sampler=None, drop_last=False,
                                    pin_memory=True)

    elif setting['val_dataname']=='twitter':
        from data.dataset_local import H5FeatureDataset
        validate_dataset = H5FeatureDataset("/home/shunlizhang/zwk/CMFQN/data/twitter_dataset_local/val_features_dif.h5")
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                    num_workers=4, sampler=None, drop_last=False,
                                    pin_memory=True)

    elif setting['val_dataname']=='ikcest':
        from data.dataset_local import H5FeatureDataset
        validate_dataset = H5FeatureDataset("/home/shunlizhang/zwk/CMFQN/data/ikcest_dataset_local/test_features_dif.h5")
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                    num_workers=4, sampler=None, drop_last=False,
                                    pin_memory=True)

    print('building model')
    from models.Net import Net
    model = Net()

    start_epoch = 0
    best_validate_acc = 0.000
    global_step = 0
    
    model = model.cuda()
    model.train()

    ############################################################
    ##################### Loss and Optimizer ###################
    loss_bce = nn.BCEWithLogitsLoss().cuda()
    criterion = loss_bce 
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01)
    num_steps = int(len(train_loader) * custom_num_epochs * 1.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    print("Using CosineAnnealingLR+UntunedLinearWarmup")
    #############################################################

    print("loader size " + str(len(train_loader)))
    best_acc_so_far = 0.000
    best_epoch_record = 0
    print('training model')

    if args.checkpoint_pth:
        if os.path.exists(args.checkpoint_pth):
            checkpoint = torch.load(args.checkpoint_pth)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_validate_acc = checkpoint['best_validate_acc']
            global_step = checkpoint['global_step']

            print(f"Loaded checkpoint from epoch {start_epoch} with best acc {best_validate_acc:.4f}")

    if setting['val']!=0:
        with torch.no_grad():
            total = len(validate_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)

            model.eval()
            print("begin inference...")
            print(f"[Image Masker] Epoch: {model.co_attention.img_masker.current_epoch.item()}, "
                f"Temperature: {model.co_attention.img_masker.temperature.item()}")
            print(f"[Text Masker] Epoch: {model.co_attention.txt_masker.current_epoch.item()}, "
                f"Temperature: {model.co_attention.txt_masker.temperature.item()}")
        
            validate_acc_list, validate_real_items, validate_fake_items, val_loss, img2text_mask, text2img_mask = evaluate(validate_loader, model, criterion, progbar=progbar, setting=setting)

            validate_acc = max(validate_acc_list)
            val_thresh = validate_acc_list.index(validate_acc)

            validate_real_precision, validate_real_recall, validate_real_accuracy, validate_real_F1 = validate_real_items
            validate_fake_precision, validate_fake_recall, validate_fake_accuracy, validate_fake_F1 = validate_fake_items
            
            print('Val_Acc: %.4f. at thresh %.4f.'
                % (
                    validate_acc, val_thresh,
                ))
            print("------Real News -----------")
            print("Precision: {:.2%}".format(validate_real_precision[val_thresh]))
            print("Recall: {:.2%}".format(validate_real_recall[val_thresh]))
            print("Accuracy: {:.2%}".format(validate_real_accuracy[val_thresh]))
            print("F1: {:.2%}".format(validate_real_F1[val_thresh]))
            print("------Fake News -----------")
            print("Precision: {:.2%}".format(validate_fake_precision[val_thresh]))
            print("Recall: {:.2%}".format(validate_fake_recall[val_thresh]))
            print("Accuracy: {:.2%}".format(validate_fake_accuracy[val_thresh]))
            print("F1: {:.2%}".format(validate_fake_F1[val_thresh]))
            print("---------------------------")
            print("end evaluate...")
            print("validate_acc:", validate_acc)
            print("best_validate_acc:", best_validate_acc)

            print(f"[Image Masker] Epoch: {model.co_attention.img_masker.current_epoch.item()}, "
                f"Temperature: {model.co_attention.img_masker.temperature.item()}")
            print(f"[Text Masker] Epoch: {model.co_attention.txt_masker.current_epoch.item()}, "
                f"Temperature: {model.co_attention.txt_masker.temperature.item()}")
        
            return


    if args.que != 0:
        from collections import deque
        class DynamicMemoryQueue:
            def __init__(self, max_size=12000):
                self.max_size = max_size
                self.queue = {
                    'contradiction': deque(maxlen=max_size),
                    'label': deque(maxlen=max_size)
                }
            
            def update(self, contradiction, labels):
                labels = labels.squeeze() 
                self.queue['contradiction'].extend(contradiction.detach().cpu().numpy())
                self.queue['label'].extend(labels.detach().cpu().numpy())
            
            def get_distribution(self):
                cont = np.array(self.queue['contradiction'])
                labels = np.array(self.queue['label'])
                
                real_stats = {
                    'mean': np.mean(cont[labels==0]),
                    'std': np.std(cont[labels==0]),
                    'quantiles': np.quantile(cont[labels==0], [0.05, 0.5, 0.95])
                }
                
                fake_stats = {
                    'mean': np.mean(cont[labels==1]),
                    'std': np.std(cont[labels==1]),
                    'quantiles': np.quantile(cont[labels==1], [0.05, 0.5, 0.95])
                }
                
                return {'real': real_stats, 'fake': fake_stats}
            
        class EnhancedTripletLoss(nn.Module):
            def __init__(self, alpha=0.5):
                super().__init__()
                
            def forward(self, embeddings, labels, contradiction, queue_stats):
                labels = labels.squeeze() 

                real_mask = (labels == 0)
                upper_bound = queue_stats['fake']['quantiles'][2]
                loss_real = F.relu(contradiction[real_mask] - upper_bound).mean()

                fake_mask = (labels == 1)
                lower_bound = queue_stats['real']['quantiles'][0]
                loss_fake = F.relu(lower_bound - contradiction[fake_mask]).mean()
                return loss_fake+loss_real
            
        queue = DynamicMemoryQueue()  
        loss_fn = EnhancedTripletLoss()  


    if args.mask != 0:
        class MaskLoss(nn.Module):
            def __init__(self): 
                super().__init__()
                
            def forward(self, cos_sim, text2img, img2text):
                mask = (text2img + img2text) / 2
                sparsity_reg = torch.abs(cos_sim - mask).mean()
                return sparsity_reg
        mask_loss_fn = MaskLoss()

    for epoch in range(start_epoch, custom_num_epochs):
        cost_vector = []
        acc_vector = []
        if setting['val']==0:
            total = len(train_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            for i, items in enumerate(train_loader):
                with torch.enable_grad():
                    logs = []
                    model.train()
                    
                    """
                    (input_ids, attention_mask, token_type_ids), (image, labels, category, sents)
                    """

                    model.co_attention.img_masker.update_temperature(epoch, custom_num_epochs)
                    model.co_attention.txt_masker.update_temperature(epoch, custom_num_epochs)

                    labels = to_var(items["labels"].float().unsqueeze(1))
                    mae = to_var(items["mae"])
                    bert = to_var(items["bert"])
                    clip_image = to_var(items["clip_image"])
                    clip_text = to_var(items["clip_text"])
                    output, img2text_mask, text2img_mask, inco_output, co_output, cos_sim, contradiction, conLoss, feats = model(mae_features=mae,
                                        text_outputs=bert,
                                        clip_image_features=clip_image,
                                        clip_text_features=clip_text,
                                        idx=0,
                                        labels = labels,
                                        isTrain = True
                                    )

                    inco_loss = criterion(inco_output, labels)
                    co_loss = criterion(co_output, labels)
                    ## CROSS ENTROPY LOSS
                    if args.mask != 0:
                        mask_loss = mask_loss_fn(cos_sim, text2img_mask, img2text_mask)
                        loss = criterion(output, labels)  + 1.0 * inco_loss + 2.0 * co_loss + args.mask * mask_loss
                    else:
                        loss = criterion(output, labels)  + 1.0 * inco_loss + 2.0 * co_loss 

                    if args.que != 0:
                        queue.update(contradiction, labels)  
                        
                        if len(queue.queue['contradiction']) >= total // 2:
                            stats = queue.get_distribution()
                        else:
                            stats = None  
                        if epoch > 10:
                            if stats is not None:
                                triplet_loss = loss_fn(output, labels, contradiction, stats)
                                loss = loss + args.que * triplet_loss  
                                loss = loss
                        else:
                            loss = loss

                    global_step += 1

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    with torch.no_grad():
                        model.theta.data.clamp_(min=0.01, max=0.99)

                    optimizer.step()

                    accuracy = (torch.sigmoid(output).round_() == labels.round_()).float().mean()
                    cost_vector.append(loss.item())
                    acc_vector.append(accuracy.item())
                    mean_cost, mean_acc = np.mean(cost_vector), np.mean(acc_vector)
                    logs.append(('CE_loss',loss.item()))
                    logs.append(('mean_acc', mean_acc))
                    logs.append(('img2text_mask', img2text_mask.mean().item())) 
                    logs.append(('text2img_mask', text2img_mask.mean().item())) 
                    logs.append(('lr', optimizer.param_groups[0]['lr']))
                    logs.append(('cos_sim', cos_sim.mean().item()))
                    logs.append(('inco', contradiction.mean().item()))

                    batch_size = labels.size(0)
                    progbar.add(batch_size, values=logs)
                    with warmup_scheduler.dampening():
                        scheduler.step()

            print('Epoch [%d/%d],  Loss: %.4f, Train_Acc: %.4f,  '
                  % (
                      epoch + 1, custom_num_epochs, np.mean(cost_vector), np.mean(acc_vector)))
            print("end training...")

        # test
        with torch.no_grad():
            total = len(validate_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            model.eval()
            print("begin evaluate...")
        
            validate_acc_list, validate_real_items, validate_fake_items, val_loss, img2text_mask, text2img_mask = evaluate(validate_loader, model, criterion, progbar=progbar, setting=setting)

            validate_acc = max(validate_acc_list)
            val_thresh = validate_acc_list.index(validate_acc)

            validate_real_precision, validate_real_recall, validate_real_accuracy, validate_real_F1 = validate_real_items
            validate_fake_precision, validate_fake_recall, validate_fake_accuracy, validate_fake_F1 = validate_fake_items
            
            if validate_acc >= best_acc_so_far:
                best_acc_so_far = validate_acc
                best_epoch_record = epoch+1

            print('Epoch [%d/%d],  Val_Acc: %.4f. at thresh %.4f (so far %.4f in Epoch %d) .'
                % (
                    epoch + 1, custom_num_epochs, validate_acc, val_thresh, best_acc_so_far, best_epoch_record,
                ))
            print("------Real News -----------")
            print("Precision: {:.2%}".format(validate_real_precision[val_thresh]))
            print("Recall: {:.2%}".format(validate_real_recall[val_thresh]))
            print("Accuracy: {:.2%}".format(validate_real_accuracy[val_thresh]))
            print("F1: {:.2%}".format(validate_real_F1[val_thresh]))
            print("------Fake News -----------")
            print("Precision: {:.2%}".format(validate_fake_precision[val_thresh]))
            print("Recall: {:.2%}".format(validate_fake_recall[val_thresh]))
            print("Accuracy: {:.2%}".format(validate_fake_accuracy[val_thresh]))
            print("F1: {:.2%}".format(validate_fake_F1[val_thresh]))
            print("---------------------------")
            print("end evaluate...")
            print("validate_acc:", validate_acc)
            print("best_validate_acc:", best_validate_acc)
            
            if (validate_acc >= best_validate_acc):
                best_validate_acc = validate_acc if validate_acc >= best_validate_acc else best_validate_acc
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_validate_acc': best_validate_acc,
                    'global_step': global_step
                }
                que = args.que
                mask = args.mask
                checkpoint_path = os.path.join(
                    args.output_file,
                    setting['train_dataname'], 
                    f"que{que}_epoch_{epoch+1}_{best_validate_acc}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

                import csv
                log_file = os.path.join(args.output_file, setting['train_dataname'], f"q{que}_training_log.csv")
                header = [
                    "Epoch", "Total Epochs", "Val Acc", "Threshold",
                    "Best Val Acc So Far", "Best Epoch",
                    "Real Precision", "Real Recall", "Real Accuracy", "Real F1",
                    "Fake Precision", "Fake Recall", "Fake Accuracy", "Fake F1"
                ]
                if not os.path.exists(log_file):
                    with open(log_file, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                data_row = [
                    epoch + 1,
                    custom_num_epochs,
                    validate_acc,
                    val_thresh,
                    best_acc_so_far,
                    best_epoch_record,
                    validate_real_precision[val_thresh],
                    validate_real_recall[val_thresh],
                    validate_real_accuracy[val_thresh],
                    validate_real_F1[val_thresh],
                    validate_fake_precision[val_thresh],
                    validate_fake_recall[val_thresh],
                    validate_fake_accuracy[val_thresh],
                    validate_fake_F1[val_thresh]
                ]
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)


from collections import OrderedDict

def evaluate(validate_loader, model, criterion, progbar=None, setting={}):
    model.eval()
    val_loss = 0
    ## THRESH: 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 ##
    threshold = setting['thresh'] # 0.5
    THRESH = [threshold+i/500 for i in range(-20,20)]

    print(f"thresh: {THRESH}")
    realnews_TP, realnews_TN, realnews_FP, realnews_FN = [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH)
    fakenews_TP, fakenews_TN, fakenews_FP, fakenews_FN = [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH), [0]*len(THRESH)
    realnews_sum, fakenews_sum = [0]*len(THRESH), [0]*len(THRESH)
    image_no,results = 0,[]
    dataset_name = setting['val_dataname']

    with torch.no_grad():
        for i, items in enumerate(validate_loader):
            labels = to_var(items["labels"].float().unsqueeze(1))
            mae = to_var(items["mae"])
            bert = to_var(items["bert"])
            clip_image = to_var(items["clip_image"])
            clip_text = to_var(items["clip_text"])
            output, img2text_mask, text2img_mask, inco_output, co_output, cos_sim, contradiction, conLoss, feats = model(mae_features=mae,
                                            text_outputs=bert,
                                            clip_image_features=clip_image,
                                            clip_text_features=clip_text,
                                            idx=0,
                                            labels = labels,
                                            isTrain = False
                                        )

            val_loss = criterion(output, labels)
            if progbar is not None:
                logs = [('loss', val_loss.item())]
                logs.append(('img2text_mask', img2text_mask.mean().item()))
                logs.append(('text2img_mask', text2img_mask.mean().item()))
                logs.append(('cos_sim', cos_sim.mean().item()))
                logs.append(('inco', contradiction.mean().item()))
                batch_size = labels.size(0)
                progbar.add(batch_size, values=logs)

            output = torch.sigmoid(output)

            for thresh_idx, thresh in enumerate(THRESH):
                validate_argmax = torch.where(output<thresh,0,1)
                # y_pred = validate_argmax.squeeze().cpu().numpy()
                # y_GT = labels.int().cpu().numpy()

                y_pred = validate_argmax.cpu().numpy().reshape(-1) 
                y_GT = labels.int().cpu().numpy().reshape(-1)          

                for idx, _ in enumerate(y_pred):
                    if thresh_idx==0:
                        record = {
                            'image_no': image_no, 
                            'y_GT': y_GT[idx], 
                            'y_pred': output[idx].item()
                        }
                        results.append(record)
                        image_no += 1

                    if y_GT[idx]==1: 
                        fakenews_sum[thresh_idx] +=1
                        if y_pred[idx]==0:
                            fakenews_FN[thresh_idx] += 1 
                            realnews_FP[thresh_idx] += 1 
                        else:
                            fakenews_TP[thresh_idx] += 1 
                            realnews_TN[thresh_idx] += 1 
                    else:
                        realnews_sum[thresh_idx] +=1
                        if y_pred[idx]==1:
                            realnews_FN[thresh_idx] +=1 
                            fakenews_FP[thresh_idx] +=1 
                        else:
                            realnews_TP[thresh_idx] += 1 
                            fakenews_TN[thresh_idx] += 1 

    val_accuracy, real_accuracy, fake_accuracy, real_precision, fake_precision = [0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH)
    real_recall, fake_recall, real_F1, fake_F1 = [0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH),[0]*len(THRESH)
    for thresh_idx, _ in enumerate(THRESH):
        val_accuracy[thresh_idx] = (realnews_TP[thresh_idx]+realnews_TN[thresh_idx])/(realnews_TP[thresh_idx]+realnews_TN[thresh_idx]+realnews_FP[thresh_idx]+realnews_FN[thresh_idx])
        real_accuracy[thresh_idx] = (realnews_TP[thresh_idx])/max(1,realnews_sum[thresh_idx])
        fake_accuracy[thresh_idx] = (fakenews_TP[thresh_idx])/max(1,fakenews_sum[thresh_idx])
        real_precision[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FP[thresh_idx]))
        fake_precision[thresh_idx] = fakenews_TP[thresh_idx]/max(1,(fakenews_TP[thresh_idx]+fakenews_FP[thresh_idx]))
        real_recall[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FN[thresh_idx]))
        fake_recall[thresh_idx] = fakenews_TP[thresh_idx]/max(1,(fakenews_TP[thresh_idx]+fakenews_FN[thresh_idx]))
        real_F1[thresh_idx] = 2*(real_recall[thresh_idx]*real_precision[thresh_idx])/max(1,(real_recall[thresh_idx]+real_precision[thresh_idx]))
        fake_F1[thresh_idx] = 2*(fake_recall[thresh_idx]*fake_precision[thresh_idx])/max(1,(fake_recall[thresh_idx]+fake_precision[thresh_idx]))

        import pandas as pd
        df = pd.DataFrame(results)
        pandas_file = f'/home/shunlizhang/zwk/CMFQN/pandas_output/{dataset_name}_experiment.xlsx'
        df.to_excel(pandas_file)
        print(f"Excel Saved at {pandas_file}")
        return val_accuracy, (real_precision, real_recall, real_accuracy, real_F1),\
            (fake_precision, fake_recall, fake_accuracy, fake_F1), val_loss, img2text_mask, text2img_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_file', type=str, default='/home/shunlizhang/zwk/CMFQN/output', help='')
    parser.add_argument('-train_dataset', type=str, default='weibo', help='')
    parser.add_argument('-test_dataset', type=str, default='weibo', help='')
    parser.add_argument('-val', type=int, default=1, help='')
    parser.add_argument('-batch_size', type=int, default=16, help='')
    parser.add_argument('-epochs', type=int, default=100, help='')
    parser.add_argument('-checkpoint_pth', type=str, default='', help='')
    parser.add_argument('-mask', type=float, default=0.0, help='')
    parser.add_argument('-que', type=float, default=0.0, help='')

    parser.add_argument('-test', type=int, default=2, help='')

    args = parser.parse_args()

    main(args)