import os
import json
import torch

from argparse import ArgumentParser
from glob import glob
from yaml import safe_load
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import HotelID
from data_pipeline import ImageDataset, TEST_TRANSFORM


def get_model(num_embedding, num_classes, backbone_name, checkpoint_path, device):
    model = HotelID(num_embedding, num_classes, backbone_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    
    return model

def parse_args():
    parser = ArgumentParser(description="Embeddings precomputation")
    parser.add_argument(
        "--config",
        required=True,
        type=str
    )
    return parser.parse_args()

def get_cfg():

    args = parse_args()
    with open(args.config, "r") as fin:
        cfg = safe_load(fin)
    return cfg

def main():

    cfg = get_cfg()
    train_folder = os.path.join(cfg["db_path"], "train_images")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("id2label.json", "r") as fin:
        id2label = json.load(fin)

    num_classes = len(id2label)

    dset = ImageDataset(
        sorted(glob(os.path.join(train_folder, "**", "*.jpg"), recursive=True)),
        transform=TEST_TRANSFORM,
        ids=list(id2label.keys()),
        labels=True
    )
    base_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)

    out = {}
    for model_cfg in cfg["ensemble"]:

        base_embeddings = torch.tensor([], device=device)
        base_hotel_ids = torch.tensor([], device=device)

        model = get_model(num_embedding=4096, num_classes=num_classes, device=device, **model_cfg)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(base_loader, desc=f"Gather embeddings with {model_cfg['backbone_name']}"):
                batch = {k:v.to(device) for k, v in batch.items()}
                inputs = batch["img"]
                ids = batch["id"]
                base_embeddings = torch.cat((base_embeddings, model(inputs)))
                base_hotel_ids = torch.cat((base_hotel_ids, ids))

        out[model_cfg["backbone_name"]] = base_embeddings
        out["hotel_ids"] = base_hotel_ids
        del model

    torch.save(out, "index_set.pt")



if __name__=="__main__":
    main()
