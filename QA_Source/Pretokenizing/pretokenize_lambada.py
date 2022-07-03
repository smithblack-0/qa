from fastai.text import core
from fastai.text import data
import os

max_char_length = 1000000
source_dir = r'C:\Users\chris\PycharmProjects\fastai\Data\lambada-dataset'



### Begin by ensuring that everything that will be included is of an acceptable char length. Those which
#are not must be excluded by deletion

files = core.get_text_files(source_dir)
for file in files:
    with open(file, errors='ignore') as f:
        length = len(f.read())
    if length > max_char_length:
        os.remove(file)

### All files are now garunteed to be of the correct length ###
### Tokenize the folder.

core.tokenize_folder(source_dir)
