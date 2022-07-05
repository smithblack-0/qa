"""

Chunklib

The chunklib is a libarary oriented towards premangling of text
and nlp files. It is designed to enable, particularly, the division
of large files into much smaller ones.

"""
from __future__ import annotations
from collections import namedtuple
from typing import List, Optional, Dict
from filesplit.split import Split

import numpy as np
import unicodedata
import re
import os
import pathlib
import QA_Source.lib.pipelinelib as pipelib
import filesplit
###

Chunk = namedtuple('Chunk', ['Chunk', 'Name', 'Length'])


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
    def load_chunks_from_file(self, path, config: ChunkConfig)-> List[Chunk]:
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
    def __call__(self, file, config: ChunkConfig):
        return self.load_chunks_from_file(file, config)

class AbstractSaver():
    """
    Capable of saving information, the saver
    will take in a list of examples in a
    manifest, data format and must then save it somewhere
    """
    def save_file(self, path, data: List[Chunk], config: ChunkConfig):
        raise NotImplementedError("Save chunks must be implimented")
    def __call__(self, path, data: List[Chunk], config: ChunkConfig):
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
    def max_file_length(self) -> int:
        return self._max_file_length
    @property
    def include(self) -> Optional[List[str]]:
        return self._include
    @property
    def exclude(self) -> Optional[List[str]]:
        return self._exclude
    @property
    def kwargs(self):
        return self._kwargs
    def __init__(self,
                 source_dir: str,
                 destination_dir: str,
                 max_file_length: int = 2**16,
                 include: Optional[List[str]] = None,
                 exclude: Optional[List[str]] = None,
                 **kwargs
                 ):
        self._source_dir = source_dir
        self._destination_dir = destination_dir
        self._max_file_length = max_file_length
        self._include = include
        self._exclude = exclude
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



def chunk(source_dir,
          destination_dir,
          max_file_length,
          include = None,
          exclude = None,
          ):



    """

    Responsible for actually making the chunks. Pass it
    a valid ChunkConfig, and it will attempt to assemble
    chunk directories based on the provided instructions.


    :param file_paths: The file paths to load from
    :param config: The config for the chunk
    :return:
    """

    file_paths = get_files(source_dir, include, exclude)
    max_file_length = max_file_length
    destination = destination_dir
    setup_directory(destination)
    for file in file_paths:
        splitrep = Split(str(file), outputdir=destination)
        splitrep.bysize(max_file_length, True, True)
