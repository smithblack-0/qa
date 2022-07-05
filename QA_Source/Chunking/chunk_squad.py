import json
import pathlib
from typing import List, Dict


from QA_Source.lib import chunklib
from QA_Source.lib import pipelinelib as pipelib
from QA_Source.config import squad_chunking


class Loader(chunklib.AbstractLoader):
    """
    This defines how to get the name, instances, and lengths
    out of a file.
    """

    def __init__(self):
        super().__init__()

    def load_chunks_from_file(self, path, config) -> List[Dict[str, str]]:
        path = pathlib.Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
            data = data["data"]
            final_data = []
            name = path.stem
            for i, item in enumerate(data):

                #Get length of string content
                lengths = []
                def fetch_length(item):
                    lengths.append(len(item))
                pipelib.map_across_strings(item, fetch_length)
                length = sum(lengths)

                #format output
                final = {"data" : item, 'name' : name + str(i), "length" : length }
                final_data.append(final)
        return final_data

class Saver(chunklib.AbstractSaver):
    """

    This defines how to save my data

    """

    def __init__(self):
        super().__init__()
    def save_file(self, path, data, config):
        path = path + '.json'
        print(path)
        with open(path, 'w') as f:
            json.dump(data, f)


def chunk_main():
    loader = Loader()
    saver = Saver()
    config = chunklib.ChunkConfig(loader=loader, saver=saver, **squad_chunking)
    chunklib.chunk(config)

print("Chunking squad data")
chunk_main()
