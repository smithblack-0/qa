
import json
import pathlib
import unicodedata
from typing import List, Dict

from lib import chunklib
from lib.chunklib import ChunkConfig
from config import lambada_pretrain_chunking
from config import lambada_main_chunking
from multiprocessing import spawn

class Pretrain_Loader(chunklib.AbstractLoader):
    def __init__(self):
        super().__init__()

    def load_chunks_from_file(self, path, config) -> List[Dict[str, str]]:
        path = pathlib.Path(path)
        with open(path, errors='ignore') as f:
            lines = f.readlines()

        # Figure out what char lengths I need to aim for
        line_lengths = [len(line) for line in lines]
        net_length = sum([len(line) for line in lines])
        if net_length > config.kwargs['max_example_length']:
            partitions = net_length // config.kwargs["max_example_length"] + 1
        else:
            partitions = 1
        target_charlengths = net_length / partitions

        cum_length = 0
        slice_start = 0
        index = 0
        slices = []
        for length in line_lengths:
            # Figure out how long the slices should be
            # to get around the same char length each time.
            cum_length += length
            index += 1
            if cum_length > target_charlengths:
                slices.append((slice_start, index))
                slice_start = index
                cum_length = 0

        # Go and make the examples
        examples = []
        name = pathlib.Path(path).stem
        for i, slice in enumerate(slices):
            example = lines[slice[0]:slice[1]]
            example = "\n".join(example)
            example = {"data": example, "name": name + "c" + str(i), "length": len(example)}
            examples.append(example)

        # Return
        return examples

class Main_Loader(chunklib.AbstractLoader):
    def __init__(self):
        super().__init__()
    def load_chunks_from_file(self, path, config) -> List[Dict[str, str]]:
        with open(path, errors='ignore') as f:
            chunks = f.readlines()
        name = pathlib.Path(path).stem
        output = [{"name" : name + str(i), "data": item, 'length' : len(item),} for i, item in enumerate(chunks)]
        return output


class Saver(chunklib.AbstractSaver):
    def __init__(self):
        super().__init__()

    def save_file(self, path, data, config):
        path = path + ".json"
        print(path)
        with open(path, 'w') as f:
            json.dump(data, f)

def chunk_pretraining():
    loader = Pretrain_Loader()
    saver = Saver()
    config = ChunkConfig(loader=loader, saver=saver, **lambada_pretrain_chunking)
    chunklib.chunk(config)

def chunk_main():
    loader = Main_Loader()
    saver = Saver()
    config = ChunkConfig(loader=loader, saver=saver, **lambada_main_chunking)
    chunklib.chunk(config)

if __name__ == '__main__':
    spawn.freeze_support()
    print("chunking pretraining")
    chunk_pretraining()
    print("chunking main")
    chunk_main()
    print("Done")
