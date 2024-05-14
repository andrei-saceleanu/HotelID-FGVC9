import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from tqdm import tqdm
from yaml import safe_load
from tensorboardX import SummaryWriter

from losses import CircleLoss, convert_label_to_similarity
from logger import create_logger
from models import HotelID
from data_pipeline import build_dataloader, TRAIN_TRANSFORM, VAL_TRANSFORM

def init_logging():
    crt_time = datetime.now()
    save_location = os.path.join(
        "weights", "experiments", crt_time.strftime("%d_%m_%Y__%H_%M_%S")
    )
    logger, tb_log_dir = create_logger(root_output_dir=save_location)
    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    return logger, writer_dict, save_location

def save_checkpoint(
    states: dict,
    is_best: bool,
    output_dir: str,
    filename: str = "checkpoint.pth",
) -> None:

    torch.save(states, os.path.join(output_dir, filename))
    if is_best and "state_dict" in states.keys():
        torch.save(states, os.path.join(output_dir, "model_best.pth"))

def map_at_k(preds, labels, k=5):

    sorted_preds = torch.argsort(preds, dim=-1, descending=True)
    rank = torch.where(sorted_preds == labels[:, None])[1][:, None] + 1
    binary_ind = (sorted_preds[:, :k] == labels[:, None]).any(dim=-1).to(torch.float32)[:, None]

    return (binary_ind / rank).mean()


def parse_args():
    parser = ArgumentParser(description="Training script")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="experiment config file"
    )
    return parser.parse_args()

def get_cfg():

    args = parse_args()
    with open(args.config, "r") as fin:
        cfg = safe_load(fin)
    return cfg

def get_dataloaders(db_path, loader_cfg, val_percent=0.2):

    imgs_path = os.path.join(db_path, "train_images")

    hotel_ids = os.listdir(imgs_path)
    train_paths = []
    val_paths = []
    for hotel_id in hotel_ids:
        curr_imgs = glob(os.path.join(imgs_path, hotel_id, "*.jpg"), recursive=True)
        l = len(curr_imgs)
        if l >= 2:
            train_paths.extend(curr_imgs)
            val_paths.extend(curr_imgs[-1:])
        else:
            train_paths.extend(curr_imgs)
            val_paths.extend(curr_imgs)

    train_loader = build_dataloader(
        img_paths=train_paths,
        ids=hotel_ids,
        labels=True,
        transform=TRAIN_TRANSFORM,
        loader_cfg=loader_cfg["train"]
    )

    val_loader = build_dataloader(
        img_paths=val_paths,
        ids=hotel_ids,
        labels=True,
        transform=VAL_TRANSFORM,
        loader_cfg=loader_cfg["val"]
    )

    return train_loader, val_loader, len(hotel_ids)


def train_epoch(model, loader, optimizer, writer_dict, device):
    model.train()
    running_loss = []
    criterion = nn.CrossEntropyLoss()
    #criterion = CircleLoss(m=0.25, gamma=256)

    for i, batch in enumerate(tqdm(loader)):
        optimizer.zero_grad(set_to_none=True)
        batch = {k:v.to(device) for k, v in batch.items()}
        labels = batch["label"]
        inputs = batch["img"]

        _, preds = model(inputs, labels)
        #embeds = F.normalize(model(inputs))
        #a, b = convert_label_to_similarity(embeds, labels)
        
        loss = criterion(preds, labels)
        #loss = criterion(a, b)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        if i == len(loader)-1:
            print(f"Train: {np.mean(running_loss):.3f}")
            running_loss = []

        writer = writer_dict["writer"]
        global_steps = writer_dict["train_global_steps"]
        writer.add_scalar("train_loss", loss.item(), global_steps)
        writer_dict["train_global_steps"] = global_steps + 1

def validate(model, loader, writer_dict, device):
    model.eval()
    running_loss = []
    criterion = nn.CrossEntropyLoss()
    #criterion = CircleLoss(m=0.25, gamma=256)
    running_map = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = {k:v.to(device) for k, v in batch.items()}
            labels = batch["label"]
            inputs = batch["img"]
            _, preds = model(inputs, labels)
            #embeds = F.normalize(model(inputs))
            #a, b = convert_label_to_similarity(embeds, labels)
            
            loss = criterion(preds, labels)
            #loss = criterion(a, b)
            
            running_loss.append(loss.item())
            running_map.append(map_at_k(preds, labels).item())

    valid_loss = np.mean(running_loss)
    print(f"Valid (loss): {valid_loss:.3f}", flush=True)
    print(f"Valid (MAP@5): {np.mean(running_map):.3f}", flush=True)

    writer = writer_dict["writer"]
    global_steps = writer_dict["valid_global_steps"]
    writer.add_scalar("valid_loss", valid_loss, global_steps)
    writer_dict["valid_global_steps"] = global_steps + 1

    return valid_loss

def main():

    cfg = get_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, num_classes = get_dataloaders(cfg["db_path"], cfg["loader_cfg"], cfg["val_percent"])

    model = HotelID(num_embedding=4096, num_hotels=num_classes, backbone="efficientnet_b3")
    model = model.to(device)

    if cfg["optim"] == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg["weight_decay"])
    elif cfg["optim"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    else:
        print("Unsupported optimizer, choose either adam or sgd")
        sys.exit(1)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg["epochs"],
        eta_min=0.0
    )

    begin_epoch = 0
    end_epoch = cfg["epochs"]
    if cfg["resume"]:
        try:
            ckpt = torch.load(cfg["ckpt"])
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            begin_epoch = ckpt["epoch"]
        except Exception as e:
            print(e.with_traceback())
            logger.info("Loading had an exception, starting a training from scratch")

    best_loss = 1000
    logger, writer_dict, save_location = init_logging()

    for epoch in range(begin_epoch, end_epoch):
        logger.info(f"Epoch {epoch}")

        train_epoch(model, train_loader, optimizer, writer_dict, device)
        valid_loss = validate(model, val_loader, writer_dict, device)
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        lr_scheduler.step()

        if (epoch + 1) % cfg["save_freq"] == 0:
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()
                },
                is_best,
                save_location,
                "checkpoint_{}.pth".format(epoch),
            )

    writer_dict["writer"].close()

if __name__=="__main__":
    main()
