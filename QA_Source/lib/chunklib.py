"""

#Chunking


I chunk the source file. The results are placed in the chunk directory,
in three locations. There is a test folder, a validation folder, and
a train folder.

In order to use the library, one must do a few things. First, import ChunkConfig
and develop an appropriate configuration for your system. Second,

"""
from typing import List, Optional, Dict

import numpy as np
import unicodedata
import re
import os
import pathlib
import random
import multiprocessing
import lib.pipelinelib as pipelib
import json

###



### Config stuff
class AbstractLoader():
    """
    The abstract loader feature.

    In order to utilize this, you must
    impliment the "load_chunks_from_file" method.

    It must be initialized with the config

    """
    #
    def __init__(self):
        pass
    def load_chunks_from_file(self, path, config)-> List[Dict[str, str]]:
        """

        The "to_chunks" method accepts a
        file, and then returns a list of
        chunks. A chunk is an item in
        the format

        {'data' : any, 'name' str: , length : int}

        data: should be whatever you want to use in the model
        name: what the filename should be. Will be cleaned up later
        length: how long the chunk is. Whatever that means.

        It is entirely up to the loader how to
        create chunks from the stream ; however,
        it should be noted that access to the config
        can be gained through .config

        :param path:
        :return:
        """
        raise NotImplementedError("load_chunks_from_file is not implimented")
    def __call__(self, file, config):
        return self.load_chunks_from_file(file, config)

class AbstractSaver():
    """
    Capable of saving information, the saver
    will take in a list of examples in a
    manifest, data format and must then save it somewhere
    """
    def save_file(self, path, data, config):
        raise NotImplementedError("Save chunks must be implimented")
    def __call__(self, path, data, config):
        return self.save_file(path, data, config)


class ChunkConfig():
    """
    How a chunk config file should be structured.

    Four folders may be produced after chunking. These
    folders are Pretraining, Train, Test, and Val

    The pretrain parameters control the pretraining generation,
    which is expected to be large text corpuses. If
    left alone, no such file is generated.

    The labeled items control train, test, val
    chunk generation. You may also define
    how splitting is being performed here

    If the files parameter is filled, the
    code will act in whitelist mode. Else, it will include

    Anything which is left blank will not be used.

    """
    @property
    def source_dir(self)-> str:
        return self._source_dir
    @property
    def destination_dir(self) -> str:
        return self._destination_dir
    @property
    def destination_names(self) -> List[str]:
        return self._destination_names
    @property
    def destination_splits(self) -> List[int]:
        return self._destination_splits
    @property
    def max_file_length(self) -> int:
        return self._max_file_length
    @property
    def include(self) -> Optional[List[str]]:
        return self._include
    @property
    def exclude(self) -> Optional[List[str]]:
        return self._exclude
    @property
    def loader(self) -> AbstractLoader:
        return self._loader
    @property
    def saver(self) -> AbstractSaver:
        return self._saver
    @property
    def kwargs(self):
        return self._kwargs
    def __init__(self,
                 source_dir: str,
                 destination_dir: str,
                 destination_names: List[str],
                 destination_splits: List[int],
                 loader: AbstractLoader,
                 saver: AbstractSaver,
                 max_file_length: int = 2**16,
                 include: Optional[List[str]] = None,
                 exclude: Optional[List[str]] = None,
                 **kwargs
                 ):
        self._source_dir = source_dir
        self._destination_dir = destination_dir
        self._destination_names = destination_names
        self._destination_splits = destination_splits
        self._max_file_length = max_file_length
        self._include = include
        self._exclude = exclude
        self._loader = loader
        self._saver = saver
        self._kwargs = kwargs


#helper functions
def split_list_by_weights(lst, weights):
    """ Splits a list up by the given weights"""
    weights = np.array(weights)
    percentages = weights/weights.sum()
    cumpercent = percentages.cumsum()
    splits = (len(lst)*cumpercent).round()[:-1]
    splits = np.round(splits).astype(int)
    return np.split(lst, splits)



#File management
def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def recursive_child_files(path)-> List[pathlib.Path]:
    """

    Find, and return, a path to all child files
    hidden under this folder

    :param path: The path to the folder
    :return: a list of the file paths
    """
    path = pathlib.Path(path)
    children = os.listdir(path)
    files = []
    for item in children:
        child_path = path.joinpath(item)
        if child_path.is_file():
            files.append(child_path)
        if child_path.is_dir():
            recurse = recursive_child_files(child_path)
            files = files + recursive_child_files(child_path)
    return files


def get_files(path,
              include: Optional[List[str]] = None,
              exclude: Optional[List[str]] = None)-> List[pathlib.Path]:
    """

    Get file paths in a directory, recursively.
    Use a whitelist and blacklist if desired

    :param path: the path to seek on
    :param include: files to include. Whitelist mode
    :param exclude: files to exclude. Blacklist mode
    :return: A list of matching files
    """

    # Verify path exists.
    path = pathlib.Path(path)
    assert os.path.exists(path)
    # Fetch relevant files
    files = recursive_child_files(path)
    if include is not None:
        files = [file for file in files if file.name in include]
    if exclude is not None:
        files = [file for file in files if file.name not in exclude]
    return files

def setup_directory(path):
    """

    Ensures that a place exists for files
    to land in at path.

    :param path: The directory we wish to place things in
    :return: None
    """
    #Creates the structure to the folder if needed
    path = pathlib.Path(path)
    parents = []
    if not path.exists():
        while path.exists() is False:
            parents.append(path.name)
            path = path.parent
        while len(parents) > 0:
            path = path.joinpath(parents.pop())
            os.mkdir(path)

    #Clean out any current occupants of the folder
    for file in os.listdir(path):
        file_path = path.joinpath(file)
        os.remove(file_path)

def _chunk(transfer_dict):

    file_paths = transfer_dict['files']
    config = transfer_dict['config']



def chunk(config: ChunkConfig):
    """

    Responsible for actually making the chunks. Pass it
    a valid ChunkConfig, and it will attempt to assemble
    chunk directories based on the provided instructions.


    :param file_paths: The file paths to load from
    :param config: The config for the chunk
    :return:
    """
    file_paths = get_files(config.source_dir, config.include, config.exclude)
    loader = config.loader
    saver = config.saver
    max_file_length = config.max_file_length
    destinations = [config.destination_dir + "\\" + name for name in config.destination_names]
    splitweights = config.destination_splits

    def save_chunks(chunks, file_counts):
        """

        Saves chunks between files according to splits.

        :param chunks: The chunks to be saved
        :return: chunk_counts
        """

        dir_chunks = split_list_by_weights(chunks, splitweights)
        for i, chunkset in enumerate(dir_chunks):
            save_dir = destinations[i]
            save_name = config.destination_names[i]
            length = 0
            examples = []
            for chunk in chunkset:
                # Get length of string content
                lengths = []

                def fetch_length(item):
                    lengths.append(len(item))

                pipelib.map_across_strings(chunk['data'], fetch_length)
                length += sum(lengths)

                # Append. Save if time
                examples.append(chunk)
                if length > max_file_length:
                    file_location = save_dir + "\\" + save_name + "chunk" + str(file_counts[i])

                    # Save data
                    file_counts[i] += 1
                    saver(file_location, examples, config)

                    # Reset bins
                    length = 0
                    examples = []
            # Finish up



            file_location = save_dir + "\\" + save_name + "chunk" + str(file_counts[i])
            file_counts[i] += 1
            saver(file_location, examples, config)

    # Prepare the destination directories. Create target char length
    for directory in destinations:
        setup_directory(directory)

    char_length_target = max_file_length * sum(splitweights)

    # Get, and split up, the data from the source directory.
    file_counts = [0] * len(destinations)
    chunks = []
    disc_pointers = []
    length = 0
    for file in file_paths:
        # Get chunks
        new_chunks = loader(file, config)
        new_pointers = [{'path': file, 'chunk': i} for i in range(len(new_chunks))]
        chunks = chunks + new_chunks
        disc_pointers = disc_pointers + new_pointers
        length += sum([example["length"] for example in new_chunks])
        # Save if appropriate, and update cache
        if length > char_length_target:
            save_chunks(chunks, file_counts)
            chunks = []
            length = 0
    # Finish
    save_chunks(chunks, file_counts)
