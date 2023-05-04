import numpy as np
import time
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable


txt_path = '../LOGS/results.txt'

def write(x):
    with open(txt_path, "a") as f:
        f.write(x)


def results2txt(q_index, mask, pred, which_ds = 'GXU'):

    GT_ROOT = '/root/autodl-tmp/gsv-cities/datasets/'

    qImages = np.load(GT_ROOT+f'{which_ds}/{which_ds}_qImages.npy')
    dbImages = np.load(GT_ROOT+f'{which_ds}/{which_ds}_dbImages.npy')

    write("|%-12s| " % qImages[q_index])

    num = 0
    for m in mask:
        if m:
            write("%-15s(√)| " % (dbImages[pred[num].numpy()]))
            num += 1
        else:
            write("%-15s(x)| " % (dbImages[pred[num].numpy()]))
            num += 1

    write("\n")
    
    for i in range(len(mask)+1):
        if i == 0:
            write("+------------+")
        else:
            write("-------------------+")

    write(" \n")



def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, save_topn=False, faiss_gpu=False, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))

        # save the image paths of top-n
        if save_topn == True:           
            n = 5
            assert n <= k_values[-1], "The value of n must be less than or equal to the maximum value in k_values"
            current_timestamp = time.strftime("%c")
            write("Results of top-%d(%s)\n" % (n, current_timestamp))
            for i in range(n+1):
                if i == 0:
                    write("+------------+")
                else:
                    write("-------------------+")
            write(" \n")

            for i in range(n+1):
                if i == 0:
                    write("|%-12s|" % 'query')
                else:
                    write(" %-s%-d%-14s|" % ('top', i, ' '))
            write(" \n")

            for i in range(n+1):
                if i == 0:
                    write("+------------+")
                else:
                    write("-------------------+")
            write(" \n")

            for q_idx, pred in enumerate(predictions):
                mask = np.in1d(pred[:n], gt[q_idx])
                results2txt(q_idx, mask, pred)

            write("\n")

        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print('\n') # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performance on {dataset_name}"))
        
        return d, predictions
