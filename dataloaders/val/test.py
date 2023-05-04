import numpy as np

GT_ROOT = '/root/autodl-tmp/gsv-cities/datasets/'

which_ds = 'pitts30k_val'

# dbImages = np.load(GT_ROOT+'msls_val/msls_val_dbImages.npy')

# dbImages = np.load(GT_ROOT+'msls_val/msls_val_pIdx.npy', allow_pickle=True)
dbImages = np.load(GT_ROOT+f'Pittsburgh/{which_ds}_gt.npy', allow_pickle=True)
print(type(dbImages))
print(len(dbImages))