import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, classification_report
from petws.src import utils



def data_handler(data):

    if data.shape[1] == 2:
        pet, ct = data[:, 0, :, :], data[:, 1, :, :]
        pet, ct = torch.unsqueeze(pet, 1), torch.unsqueeze(ct, 1)
        return pet, ct
    elif data.shape[1] == 1:
        pet = data
        return pet

def create_list_of_data(data, to_append):
    for i in data:
        to_append.append(i[0])


def log_epoch(train_or_val, wandb, dices, epoch):
    
    wandb.log({
        f"Avg {train_or_val} Dice": np.mean(dices),
       "Epoch":epoch
    })



def train_loop(train_loader, optimizer, criterion, net, wandb, device, logger, epoch, pet_only):
    dices = []
    losses = []
    preds = []
    corrects = []

    for data, segmentation in train_loader:
        
        if pet_only:
            pet, ct = data_handler(data)
            data = pet.to(device)
        
        data = data.to(device, non_blocking=True)  
        segmentation = segmentation.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred = net(data)
        loss = criterion(pred, segmentation)
        loss.backward()
        optimizer.step()
        
        preds_threshed = utils.threshold(pred.cpu().detach().numpy())
        create_list_of_data(preds_threshed, preds)
        create_list_of_data(segmentation.cpu().detach().numpy(), corrects)

        for i in range(len(preds)):
            dices.append(np.nan_to_num(utils.dice(preds[i], corrects[i])))
        preds = []
        corrects = []
        losses.append(loss.item())


        logger.info(f"Train Loss: {loss.item()}")
        wandb.log({
            "Train Loss": loss.item(),
        })

    log_epoch("Train", wandb, dices, epoch)
    
    logger.info(f'Avg. Train Loss in epoch {epoch}: {np.mean(losses)}; Avg Dice: {np.mean(dices)}')

def val_loop(test_loader, criterion, net, wandb, device, logger, epoch ,pet_only):
    dices = []
    losses = []
    preds = []
    corrects = []
    example_images = []

    for data, segmentation in test_loader:

        if pet_only:
            pet, ct = data_handler(data)
            data = pet.to(device, non_blocking=True)
        
        data = data.to(device, non_blocking=True)
        segmentation = segmentation.to(device, non_blocking=True)
        
        with torch.no_grad():
            pred = net(data)
            loss = criterion(pred, segmentation)

        preds_threshed = utils.threshold(pred.cpu().detach().numpy())
        create_list_of_data(preds_threshed, preds)
        create_list_of_data(segmentation.cpu().detach().numpy(), corrects)

        for i in range(len(preds)):
            dices.append(np.nan_to_num(utils.dice(preds[i], corrects[i])))

        losses.append(loss.item())

        example_images.append(wandb.Image(
                data[0][0],masks={
                     "predictions": {
                                    "mask_data": preds[0],
                                    "class_labels": {0: "No Tumor", 1: "Tumor"}
                                    },
                    "groud_truth": {
                                    "mask_data": corrects[0],
                                    "class_labels": {0: "No Tumor", 1: "Tumor"}
                                    }}
                                    ))
        
        wandb.log({
            "Val Loss": loss.item(),
        })
        
        preds = []
        corrects = []

    wandb.log({
        "Examples": example_images,
    })
    
    log_epoch("Val", wandb, dices, epoch)


    logger.info(f'Avg. Val Loss in epoch {epoch}: {np.mean(losses)}; Avg Dice: {np.mean(dices)}')


def run(net, train_dataset, test_dataset, train_loader, test_loader,
        criterion, optimizer, epochs, logger, device, wandb, pet_only=False):

    for epoch in range(epochs):
        for phase in ["train", "val"]:
            
            logger.info(f'Phase: {phase}')

            if phase == "train":
                net.train()
                train_loop(train_loader, optimizer, criterion, net, wandb, device, logger, epoch, pet_only)

            else:
                net.eval()
                val_loop(test_loader, criterion, net, wandb, device, logger, epoch, pet_only)

        torch.save(net.state_dict(), f'{epoch}.ckt')


                    





