import enum
import copy
from collections import namedtuple
from typing import Optional, List
from lib.chunklib import ChunkConfig

## Chunk declarations ##

lambada_pretrain_chunking = {
    'source_dir': r'C:\Users\chris\PycharmProjects\fastai\Data\lambada-dataset\train-novels',
    'destination_dir': r'C:\Users\chris\PycharmProjects\fastai\Chunks\Lambada',
    'destination_names': ["Pretrain"],
    'destination_splits': [1],
    'max_example_length' : 5000,
    'max_file_length' : 2**18
}

lambada_main_chunking = {
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Data\lambada-dataset',
    'destination_dir': r'C:\Users\chris\PycharmProjects\fastai\Chunks\Lambada',
    'destination_names' : ["Train", "Test", "Validation"],
    'destination_splits' : [100, 1, 1],
    'include' : ['lambada_control_test_data_plain_text.txt',
                 'lambada_development_plain_text.txt',
                 'lambada_development_plain_text.txt'
                 ]
}

squad_chunking = {
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\source\squad',
    'destination_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Squad',
    'destination_names': ["Train", "Test", "Validation"],
    'destination_splits': [100, 1, 1],
    'max_file_length' : 2**18
}


### pipeline configs ###


lambada_pretraining_pipeline = {
    'shuffle_buffer' : 2**12,
    'readahead_buffer' : 1000,
    'prefetch_buffer' : 7,
    'batch_size' : 16,
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Lambada\Pretrain',
    'mask_rate' : 0.1,
    'feature_name' : 'data'
}


lambada_main_pipeline = {
    'shuffle_buffer' : 2**12,
    'readahead_buffer' : 1000,
    'prefetch_buffer' : 7,
    'batch_size' : 16,
}

lambada_train_pipeline = {
    **lambada_main_pipeline,
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Lambada\Train'
}

lambada_test_pipeline = {
    **lambada_main_pipeline,
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Lambada\Test'
}

lambada_validation_pipeline = {
    **lambada_main_pipeline,
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Lambada\Validation'

}

squad_main_pipeline = {
    'shuffle_buffer': 2 ** 10,
    'readahead_buffer': 100,
    'prefetch_buffer': 7,
    'batch_size': 16,
    'pipe_ignore' :  ['meta']
}

squad_train_pipeline = {
    **squad_main_pipeline,
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Squad\Train',
    'training_mode' : True
}

squad_test_pipeline = {
    **squad_main_pipeline,
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Squad\Test'
}

squad_val_pipeline = {
    **squad_main_pipeline,
    'source_dir' : r'C:\Users\chris\PycharmProjects\fastai\Chunks\Squad\Validation'

}

class mutable():
    def __init__(self, item, value):
        setattr(self, item, value)
