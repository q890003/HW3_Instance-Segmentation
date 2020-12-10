import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import os
import copy
import time
import json
import numpy as np

import dataset
import config
from model import get_model_instance_segmentation


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc, best_loss = 0.0, 9999
    gt_targets, pred_targets = [], []
    for epoch in range(num_epochs):
        epoch_bgin = time.time()

        print("Epoch %d/%d\n" % (epoch, num_epochs - 1), "-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train" or phase == "val":
                model.train()  # Set model to training mode

            running_loss, val_loss = 0.0, 0
            current_num_data, all_data = 0, len(dataloaders[phase].dataset)
            loss_classifier, loss_box_reg, loss_mask, loss_rpn_box_reg = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            # Iterate over data.
            loss_weights = np.array([1, 1, 1, 1])
            for i, (imgs, _targets) in enumerate(dataloaders[phase]):
                if current_num_data % (32) == 0:
                    batch_bgin = time.time()

                imgs = [img.to(device) for img in imgs]
                targets = [
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                        "masks": target["masks"].to(device),
                    }
                    for target in _targets
                ]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                pred = model(imgs, targets)
                loss = sum([l for l in pred.values()])

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # statistics
                if phase == "train":
                    current_num_data += len(imgs)
                    running_loss += loss.item() * len(imgs)
                    time_batch = time.time() - batch_bgin
                    minute_batch, second_batch = time_batch // 60, time_batch % 60
                    if current_num_data % (32) == 0:
                        print(
                            "epoch:{epoch}/{total_epoch} [{learned_data}/{data}], "
                            "time: {minute:.0f}m {sec:.0f}s, Batch Loss: {loss:.4f}".format(
                                epoch=epoch,
                                total_epoch=(num_epochs - 1),
                                learned_data=current_num_data,
                                data=all_data,
                                minute=minute_batch,
                                sec=second_batch,
                                loss=loss.item(),
                            )
                        )
                # record validation prediction and the corresponding GT
                elif phase == "val":
                    with torch.no_grad():
                        val_loss += loss.item() * len(imgs)

            if phase == "train":
                scheduler.step()
                print("Train loss:{loss:.4f}".format(loss=running_loss / all_data))

            elif phase == "val":

                val_loss_avg = val_loss / all_data
                time_epoch = time.time() - epoch_bgin
                print(
                    "Val Loss: {loss:.4f}".format(
                        loss=val_loss_avg,
                    )
                )

                print(
                    "\nepoch time:{minute:.0f}m {sec:.0f}s".format(
                        minute=(time_epoch / 60),
                        sec=(time_epoch % 60),
                    )
                )
                # save checkpoints
                if val_loss_avg < best_loss:
                    best_loss = val_loss_avg
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(
                        best_model_wts,
                        config.ckpt_dir
                        + config.model_name
                        + "alldataaug__SGD_LrStep_resnest50_val_loss_no_setGrad"
                        "_epoch{epoch}_loss{loss}".format(
                            epoch=epoch, loss=val_loss_avg
                        ),
                    )

        # End of one epoch over phase [train, val]
    # End of epochs

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s, "
        "Best val Loss: {:4f}".format(time_elapsed // 60, time_elapsed % 60, best_loss)
    )

    # load best model weights
    model.load_state_dict(best_model_wts)

    state = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sheduler": scheduler.state_dict(),
    }
    return model, state


if __name__ == "__main__":

    checkpoint = None

    # define training and validation data loaders
    dataset = dataset.PASCAL_VOC_Dataset(
        folder_path=config.train_folder, trans=dataset.utils.transform123, train=True
    )

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    validation_size = int(len(dataset) * config.split)
    dataset_train = torch.utils.data.Subset(dataset, indices[:-validation_size])
    dataset_test = torch.utils.data.Subset(dataset, indices[-validation_size:])
    dataset_test.transforms = None
    dataset_test.train = False

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=dataset.collate_fn,
    )
    dataloader = {"train": dataloader_train, "val": dataloader_test}
    if checkpoint is None:
        # train on the GPU or on the CPU, if a GPU is not available
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(device)

        with torch.cuda.device(0):
            # load a pre-trained model
            model = get_model_instance_segmentation(21)

        optimizer_ft = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
        )

        # and a learning rate scheduler
        milestones = [10, 23, 33, 43]
        exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_ft, milestones, gamma=0.1, last_epoch=-1
        )
    #         exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft,
    #                                                        step_size=10,
    #                                                        gamma=0.1)
    else:
        checkpoint = torch.load(checkpoint)
        model = checkpoint["model"]
        start_epoch = checkpoint["epoch"]
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
        optimizer = checkpoint["optimizer"]

    with torch.cuda.device(0):
        model = model.to(device)

        # loss function
        criterion = 0

        #         cudnn.benchmark = True
        # training the model
        model_ft, state = train_model(
            model,
            dataloader,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            num_epochs=50,
        )

#         print(state)  # test
# save fine-tuned model
