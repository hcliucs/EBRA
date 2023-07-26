# EBRA
A secret removal attack against image deep hiding 

## Step 1: generate color maps and edge maps for the two extractors
First, put all clean images, containing no secret, in the same directory. Second, call the functions of "**produce_edge_map**" and "**produce_color_map**" in data.py to generate edge maps and color maps respectively. 

## Step 2: train the two extractors
Dataloader:
```python
from data import Dataloader_Ext
data=Dataloader_Ext(source_img_path, edge_path, color_path)
from torch.utils.data import DataLoader
dataloader=DataLoader(data, batch_size)
```

Create Model and train：
```python
from extractor import Extractor
model=Extractor()
model.train(dataloader, epoch)
```

## Step 3: train the inpaintor
Dataloader:
```python
from data import Dataloader_EBRA
data=Dataloader_EBRA(source_img_path)
from torch.utils.data import DataLoader
dataloader=DataLoader(data, batch_size)
```

Create Model and train：
```python
from ebra import EBRA
model=EBRA()
model.train(dataloader, epoch)
```

## Step 4: attack deep hiding models
Download the source code of [UDH](https://github.com/ChaoningZhang/Universal-Deep-Hiding), [DS](https://github.com/zllrunning/Deep-Steganography), [ISGAN](https://github.com/Marcovaldong/ISGAN), [HCVS](https://github.com/muziyongshixin/pytorch-Deep-Steganography
), and [MIS](https://github.com/m607stars/MultiImageSteganography), and train these hiding models. 

Attack example:
```python
secret_img=...
cover_img=...
DS_hide_model=...
EBRA_model=...
container_img=DS_hide_model(cover_img,secret_img)
# erase
masked_container_img=container_img*(1-mask)
# repair
processed_container_img=EBRA_model(masked_container_img)
```
