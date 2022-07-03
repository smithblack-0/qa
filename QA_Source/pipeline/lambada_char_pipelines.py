import json
import os
import random
from typing import List
from lib import pipelinelib as pipelib
from lib import chunklib
from infinibatch import iterators as it
from infinibatch import datasets as db
from fastai.text import core
import numpy as np

"""
Creates the loaders for the lambada pretraining pipeline
"""

class PretrainPipeline():
    """
    The pipeline for char based pretraining
    """
    def decode(self, example):
        """ Converts a dict batch from tensor format to char format"""
        return pipelib.map_across_tensors(example, pipelib.char_detensorize)
    @staticmethod
    def key(raw):
        """Tells us how long the example is"""
        return raw['length']

    @staticmethod
    def _load_file(path):
        """

        Accepts a file path, then goes and
        loads it's contents. Converts this
        content into a sequence of items
        in a list, known as "chunks"

        Attempts to keep the line length
        a reasonable length all the while

        A list of chunks is what is returned.

        :param path: The path to the file with chunks
        :return: A list of chunks. All of approximately the same lenght
        """
        with open(path, 'r', errors='ignore') as f:
            data = json.load(f)
        for item in data:
            yield item


    def labelmaker(self, example, feature_name, mask_rate):
        features = example[feature_name]
        tokens = pipelib.map_across_strings(features, lambda x: list(self.tkn(x)))
        def masker(stringlist):
            mask_items = random.sample(list(range(len(stringlist))), int(len(stringlist)*mask_rate))
            output = [chr(0)*len(item) if i in mask_items else item for i, item in enumerate(stringlist)]
            output = output.tolist()
            return output
        masked_tokens = pipelib.map_across_stringlists(tokens, masker)
        features = pipelib.map_across_stringlists(tokens, lambda x : ' '.join(x))
        labels = pipelib.map_across_stringlists(masked_tokens, lambda x : ' '.join(x))

        output = example.copy()
        output['labels'] = labels
        output[feature_name] = features
        return output

    def __init__(self,
        shuffle_buffer: int,
        readahead_buffer: int,
        prefetch_buffer: int,
        batch_size: int,
        source_dir: str,
        mask_rate: float,
        feature_name: str,
        vocab_build_examples: int = 100):

        """

        Sets up the class. This consists of making the loader head,
        and attaching the label manufactoring pipeline onto it.

        :param shuffle_buffer:
        :param readahead_buffer:
        :param prefetch_buffer:
        :param batch_size:
        :param source_dir:
        :param mask_rate:
        """

        ## Setup the primary head. This loads the data and places it together into a batch.

        chunks = [source_dir + "\\" + item for item in os.listdir(source_dir)]
        pipe = db.chunked_dataset_iterator(chunks,
                                           self.load_file,
                                           shuffle_buffer,
                                           True)
        pipe = it.BucketedReadaheadBatchIterator(pipe,
                                                 readahead_buffer,
                                                 self.key,
                                                 batch_size,
                                                 shuffle=True)
        pipe = it.MapIterator(pipe, pipelib.batch_merge)

        #Setup the tokenizer. Then restore the pipeline.
        checkpoint = pipe.getstate()
        vocab_examples = [next(pipe) for _ in range(vocab_build_examples)]
        wtk = core.WordTokenizer()
        self.tkn = core.Tokenizer(wtk)
        self.tkn.setup(vocab_examples)
        pipe.setstate(checkpoint)

        #Label, convert to tensor the remaining content, and prefetch the pipe

        pipe = it.MapIterator(pipe, lambda x : self.labelmaker(x, feature_name, mask_rate))
        pipe = it.MapIterator(pipe, lambda x : pipelib.map_across_stringlists(x, pipelib.char_tensorize_stringlist))
        pipe = it.PrefetchIterator(pipe, prefetch_buffer)

        #Store

        self._pipe = pipe
    def __next__(self):
        return next(self._pipe)




###############################
### Main Pipeline       #####
############################

def make_main_pipeline(
        shuffle_buffer: int,
        readahead_buffer: int,
        prefetch_buffer: int,
        batch_size: int,
        source_dir: str):
    def load_file(path):
        """

        Accepts a file path, then goes and
        loads it's contents. Converts this
        content into a sequence of items
        in a list, known as "chunks"

        Attempts to keep the line length
        a reasonable length all the while

        A list of chunks is what is returned.

        :param path: The path to the file with chunks
        :return: A list of chunks. All of approximately the same lenght
        """
        with open(path, 'r', errors='ignore') as f:
            data = json.load(f)
        for item in data:
            yield item


    def merge_batch(pipeline):
        """ merges a batch together """
        return it.MapIterator(pipeline, pipelib.batch_merge)


    def tokenize(pipeline):
        """ Tokenize the text using the fastai pipeline"""
        wt = core.WordTokenizer()
        tkn = core.Tokenizer(wt)
        def token_converter(example):
            tokens = tkn(example)
            tokens = list(tokens)
            return tokens
        applier = lambda x : pipelib.map_across_strings(x, token_converter)
        return it.MapIterator(pipeline, applier)

    def make_labels(pipeline):
        """ Generate labels by masking words at random """
        def masker(stringlist : List[str]):

            output = stringlist.copy()
            output[-1] = "TK_MASK"
            return output
        def label_maker(batch):
            output = batch.copy()
            features = batch["data"]
            labels = pipelib.map_across_stringlists(features, masker)
            output['feature'] = features
            output['labels'] = labels
            return output

        return it.MapIterator(pipeline, label_maker)

    def decode(pipeline):
        """ Put everything back together the way it originally was"""
        def join(example):
            output = " ".join(example)
            return output
        def decoder(example):
            output = pipelib.map_across_stringlists(example, join)
            return output
        return it.MapIterator(pipeline, decoder)

    def tensorize(pipeline):
        def mapper(example):


            return pipelib.map_across_stringlists(example, pipelib.char_tensorize_stringlist)
        return it.MapIterator(pipeline, mapper)


    def key(raw):
        """Tells us how long the example is"""
        return raw['length']

    chunks = [source_dir + "\\" + item for item in os.listdir(source_dir)]
    pipe = db.chunked_dataset_iterator(chunks,
                                       load_file,
                                       shuffle_buffer,
                                       True)
    pipe = it.BucketedReadaheadBatchIterator(pipe,
                                             readahead_buffer,
                                             key,
                                             batch_size,
                                             shuffle=True)
    pipe = merge_batch(pipe)
    pipe = tokenize(pipe)
    pipe = make_labels(pipe)
    pipe = decode(pipe)
    pipe = tensorize(pipe)
    pipe = it.PrefetchIterator(pipe, prefetch_buffer)
    return pipe
