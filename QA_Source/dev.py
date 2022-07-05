#%%
from fastai.text.all import get_text_files
import datasets

#Update it to be more suitable to the particular application using more training.

#Make sure to use

src = r'C:\Users\chris\PycharmProjects\qa\Data\lambada-dataset\train-novels'
files = get_text_files(src)
datasets.load_dataset(src, '.txt', data_files=files )