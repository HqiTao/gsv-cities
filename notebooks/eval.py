import sys
sys.path.append('..') # append parent directory, we need it

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from tqdm import tqdm
from utils.validation import get_validation_recalls
from main import VPRModel
from dataloaders.val.MapillaryDataset import MSLS
from dataloaders.val.PittsburghDataset import PittsburghDataset
from dataloaders.val.GXUDataset import GXUDataset

MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]

IM_SIZE = (320, 320)

def input_transform(image_size=IM_SIZE):
    return T.Compose([
        # T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
		T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
        
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def get_val_dataset(dataset_name, input_transform=input_transform()):
    dataset_name = dataset_name.lower()
    
    
    if 'msls' in dataset_name:
        ds = MSLS(input_transform = input_transform)

    if 'gxu' in dataset_name:
        ds = GXUDataset(input_transform = input_transform)

    elif 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform = input_transform)
    else:
        raise ValueError
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, 'Calculating descritptors...'):
            imgs, labels = batch
            output = model(imgs.to(device)).cpu()
            descriptors.append(output)

    return torch.cat(descriptors)

# define which device you'd like run experiments on (cuda:0 if you only have one gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VPRModel(
        #-------------------------------
        #---- Backbone architecture ----
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        
        #---------------------
        #---- Aggregator -----
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 512,
        #             'out_dim': 512},
        # agg_arch='GeM',
        # agg_config={'p': 3},
        
        agg_arch='ConvAP',
        agg_config={'in_channels': 2048,
                    'out_channels': 1024,
                    's1' : 2,
                    's2' : 2},

        #-----------------------------------
        #---- Training hyperparameters -----
        #
        lr=0.0002, # 0.03 for sgd
        optimizer='adam', # sgd, adam or adamw
        weight_decay=0, # 0.001 for sgd or 0.0 for adam
        momentum=0.9,
        warmpup_steps=600,
        milestones=[5, 10, 15, 25],
        lr_mult=0.3,
        
        #---------------------------------
        #---- Training loss function -----
        # see utils.losses.py for more losses
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        #
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

state_dict = torch.load('../LOGS/resnet50/lightning_logs/version_4/checkpoints/resnet50_epoch(26)_step(33777)_R1[0.9197]_R5[0.9763].ckpt', map_location=torch.device('cpu')) # link to the trained weights
# model.load_state_dict(state_dict)
model.load_state_dict(state_dict['state_dict'])
model.eval()
model = model.to(device)

val_dataset_name = 'GXU'
batch_size = 1

val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_dataset_name)
val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size)

descriptors = get_descriptors(model, val_loader, device)
print(f'Descriptor dimension {descriptors.shape[1]}')

# now we split into references and queries
r_list = descriptors[ : num_references].cpu()
q_list = descriptors[num_references : ].cpu()
recalls_dict, preds = get_validation_recalls(r_list=r_list,
                                    q_list=q_list,
                                    k_values=[1, 5, 10],
                                    gt=ground_truth,
                                    print_results=True,
                                    save_topn=True,
                                    dataset_name=val_dataset_name,
                                    )