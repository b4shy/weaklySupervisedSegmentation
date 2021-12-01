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
        to_append.append(i.item())


def log_epoch(train_or_val, wandb, accs, preds, corrects, epoch):
    
    corrects = np.array(corrects, dtype=np.int32)
    preds_threshed = utils.threshold(np.array(preds))
    precision = precision_score(corrects, preds_threshed)
    recall = recall_score(corrects, preds_threshed)
    report = classification_report(corrects, preds_threshed, output_dict=True)
    table = wandb.Table(columns=["Class", "Precision", "Recall", "F1", "Support"])

    table.add_data("0", report["0"]["precision"], report["0"]["recall"],
                    report["0"]["f1-score"], report["0"]["support"] )
    table.add_data("1", report["1"]["precision"], report["1"]["recall"],
                    report["1"]["f1-score"], report["1"]["support"] )

    wandb.log({f"{train_or_val} Classification Report": table})
    wandb.log({
        f"Avg {train_or_val} Acc": np.mean(accs),
        f"{train_or_val} Precision ": precision,
        f"{train_or_val} Recall ": recall,
       "Epoch":epoch
    })





def train_loop(train_loader, optimizer, criterion, net, wandb, device, logger, epoch, pet_only):
    accs = []
    losses = []
    preds = []
    corrects = []
    for data, label in train_loader:
        
        if pet_only:
            pet, ct = data_handler(data)
            data = pet.to(device)
        
        data = data.to(device, non_blocking=True)  
        label = label.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred = net(data)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        
        accs.append(utils.accuracy(pred, label).item())
        losses.append(loss.item())

        create_list_of_data(pred, preds)
        create_list_of_data(label, corrects)

        logger.info(f"Train Loss: {loss.item()}")
        wandb.log({
            "Train Loss": loss.item(),
        })

    log_epoch("Train", wandb, accs, preds, corrects, epoch)
    
    logger.info(f'Avg. Train Loss in epoch {epoch}: {np.mean(losses)}; Avg Acc: {np.mean(accs)}')

def val_loop(test_loader, criterion, net, wandb, device, logger, epoch ,pet_only):
    accs = []
    losses = []
    example_images = []
    preds = []
    corrects = []

    for data, label in test_loader:

        if pet_only:
            pet, ct = data_handler(data)
            data = pet.to(device, non_blocking=True)
        
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        with torch.no_grad():
            pred = net(data)
            loss = criterion(pred, label)

        accs.append(utils.accuracy(pred, label).item())
        losses.append(loss.item())
        example_images.append(wandb.Image(
                data[0][0], caption="Pred: {} Truth: {}".format(utils.threshold(pred[0].item()), label[0].item())))
        
        create_list_of_data(pred, preds)
        create_list_of_data(label, corrects)

        wandb.log({
            "Val Loss": loss.item(),
        })

    wandb.log({
        "Examples": example_images,
    })
    
    log_epoch("Val", wandb, accs, preds, corrects, epoch)


    logger.info(f'Avg. Val Loss in epoch {epoch}: {np.mean(losses)}; Acg Accuracy: {np.mean(accs)}')


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


                    





