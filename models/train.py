from tqdm import tqdm
import argparse
import shutil
import glob
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
import cv2

from dataloader import TrainSet
from models import resfcn18, resnet18

from tensorboardX import SummaryWriter


def get_parser():
    parser = argparse.ArgumentParser(description='Parameters to Cifar100 Classification')
    ## Hyper Paramters
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument('--train_dir', default="./001_dataset/images/images", help='train dir path')
    parser.add_argument('--test_dir', default="./001_dataset/images/images", help='test dir path')
    parser.add_argument('--batch_size', type=int, default=512, help='training batch size.')
    parser.add_argument('--lr', default=2e-4, help='base value of learning rate.')
    parser.add_argument('--wd', default=1e-7, help='base value of weight decay.')
    parser.add_argument('--img_size', default=224, help='input image size.')
    ## Network Parameters
    parser.add_argument('--arch', default='resnet18', help='select model architecture (default:resnet18): resnet18, resfcn18')
    parser.add_argument('--pretrained', default='', help='pretrained base model')
    ## Resource Setting
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', action='store_true', help='Use cuda to train model')
    parser.add_argument('--device', default="0", help='cuda device or cpu')
    ## Save & Load
    parser.add_argument('--save_path', default='result', help='Location to save checkpoint models')
    args = parser.parse_args()
    return args


def get_dataset(train_path, test_path, train_preproc, test_preproc, split_ratio=0.7):
    ## 데이터 경로 불러오기
    train_img_paths = [path for path in glob.glob(train_path + "/**/*", recursive=True) if os.path.isfile(path)]
    ## 데이터 경로에서 Label 추출
    train_labs = [img_path.split("/")[-2].split("-")[-1] for img_path in train_img_paths]
    ## Label 을 Index 로 변환
    lab2idx = {l: i for i, l in enumerate(set(train_labs))}
    lab2idx[len(lab2idx)] = "unknown"

    if train_path == test_path: ## Test 경로를 따로 지정해주지 않은 경우, Train 데이터 중 일부를 Test 로 사용
        indices = list(range(len(train_img_paths)))
        random.shuffle(indices)
        train_idxs = indices[:int(len(indices) * split_ratio)]
        test_idxs = indices[int(len(indices) * split_ratio):]
        train_dataset = TrainSet(train_img_paths, train_labs, train_preproc, lab2idx, train_idxs)
        test_dataset = TrainSet(train_img_paths, train_labs, test_preproc, lab2idx, test_idxs)
    else:   ## Test 경로를 따로 지정해 준 경우, Test 데이터 불러오기
        test_img_paths = [path for path in glob.glob(test_path + "/**/*", recursive=True) if os.path.isfile(path)]
        test_labs = [img_path.split("/")[-2].split("-")[-1] for img_path in test_img_paths]
        train_dataset = TrainSet(train_img_paths, train_labs, train_preproc, lab2idx)
        test_dataset = TrainSet(test_img_paths, test_labs, test_preproc, lab2idx)
    return train_dataset, test_dataset
    

def train(args):
    ## 학습 결과를 저장할 폴더 생성
    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    
    ## 데이터 전처리 정의
    train_preproc = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomRotation(15),
                                        transforms.Resize((args.img_size, args.img_size)),
                                        transforms.ToTensor(),
                                        ])
    test_preproc = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                        transforms.ToTensor(),
                                        ])
    ## 데이터셋 정의
    train_dataset, test_dataset = get_dataset(args.train_dir, args.test_dir, train_preproc, test_preproc)
    
    ## 모델 불러오기
    model = eval(args.arch)(len(train_dataset.lab2idx), img_size=args.img_size)
    
    ## 데이터 로더 정의
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              )
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size * 2,
                            shuffle=False,
                            num_workers=args.num_workers,
                            )
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.wd)
    
    ## 손실 함수 정의
    criterion = torch.nn.CrossEntropyLoss()
    
    ## 학습 경과 기록용 텐서보드
    writer = SummaryWriter(f"{args.save_path}/logs/")
    
    ## GPU 사용할 경우 GPU에 모델 업로드
    if args.cuda:
        model = model.to(f"cuda:{args.device}")
    for epoch in range(args.n_epochs):
        pbar = tqdm(enumerate(train_loader), 
                    desc=f"Train {epoch+1} / {args.n_epochs}", 
                    total=len(train_loader),
        )
        total_iter = 1
        total_acc, total_loss = 0, 0
        model.train()
        for idx, batch in pbar:
            images, labels = batch
            if args.cuda:
                images = images.to(f"cuda:{args.device}")
                labels = labels.to(f"cuda:{args.device}")
            
            ## 기울기 초기화
            optimizer.zero_grad()
            logits = model(images)
            ## 손실율 계산
            loss = criterion(logits, labels)
            ## 역전파
            loss.backward()
            ## 모델 학습
            optimizer.step()
            
            ## 학습 경과 기록
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                accuracy = correct / len(preds)
            writer.add_scalar("Train_Iter/Loss", loss.item(), len(train_loader) * epoch + idx)
            writer.add_scalar("Train_Iter/Accuracy", accuracy, len(train_loader) * epoch + idx)
            pbar.set_postfix(loss=total_loss / total_iter, accuracy=total_acc / total_iter)
            
            total_acc += accuracy
            total_loss += loss.item()
            total_iter += 1
        
        writer.add_scalar("Train_Epoch/Loss", total_loss / total_iter, epoch+1)
        writer.add_scalar("Train_Epoch/Accuracy", total_acc / total_iter, epoch+1)
        ## 한 에폭 학습이 끝난 후 테스트 수행
        acc = test(model, test_loader, criterion, writer, epoch, args)
        writer.flush()
        ## 에폭마다 모델 저장
        torch.save({"state_dict": model.state_dict(),
                    "lab2idx": train_dataset.lab2idx}, 
                   f"{args.save_path}/saved_{epoch+1}_epoch_model.pth")
    
    
def test(model, loader, criterion, writer, epoch, args):
    pbar = tqdm(enumerate(loader), 
                desc=f"Test {epoch+1} / {args.n_epochs}", 
                total=len(loader),
            )
    total_iter = 1
    total_acc, total_loss = 0, 0
    model.eval()
    with torch.no_grad():
        for idx, batch in pbar:
            images, labels = batch
            if args.cuda:
                images = images.to(f"cuda:{args.device}")
                labels = labels.to(f"cuda:{args.device}")
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            accuracy = correct / len(preds)
            
            pbar.set_postfix(loss=total_loss / total_iter, accuracy=total_acc / total_iter)
            
            total_acc += accuracy
            total_loss += loss.item()
            total_iter += 1
    
        writer.add_scalar("Test_Epoch/Loss", total_loss / total_iter, epoch+1)
        writer.add_scalar("Test_Epoch/Accuracy", total_acc / total_iter, epoch+1)
        writer.flush()
        
    return total_acc / total_iter


if __name__ == "__main__":
    args = get_parser()
    train(args)
