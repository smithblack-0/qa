"""

Pipeline

This contains the code needed to create the pipeline
and data loader from the squad QA dataset.

"""

import json
import os
from typing import List
from infinibatch import iterators as it
from infinibatch import datasets as db
from fastai.text import core
from fastai.text import data
from lib import pipelinelib as pipelib

#Utility functions are located here

def make_pipeline(source_dir,
                  training_mode,
                  shuffle_buffer,
                  readahead_buffer,
                  prefetch_buffer,
                  batch_size,
                  pipe_ignore):


    def ignore_metainfo(batch: dict, function):
        """

        Ignores the metainfo channel

        :param batch:
        :return:
        """
        update = batch.copy()
        meta = {}
        for key in pipe_ignore:
            meta['key'] = update.pop(key)
        update = function(update)
        for key, value in meta:
            update[key] = value
        return update

    #Pipeline starts here.
    def load_chunk(path):
        """

        This function takes a path
        to a json chunk. It takes
        the chunk, and turns it
        into a list of feature
        dictionaries.

        Each dictionary consists of features given as
        {topic: str,
        passage: str,
        paragraph: int
        question: str,
         answers: list[complex]}

        The passage is produced by concatenating
        all the paragraphs together. making a passage.

        The return is

        "id", "features": {"topic", "passage", "question"}, "labels" : {answer, impossible, plausable}

        :param path: The path on the file system to the data
        :return: A list of examples.
        """
        with open(path) as f:
            file = json.load(f)
        for data in file:
            #Get data out of chunk
            data = data['data']
            topic = data["title"]
            paragraphs = data["paragraphs"]
            passage = ""
            for paragraph in paragraphs:
                passage = passage + paragraph['context'] + " \n\n "

            #Create the features
            for i, paragraph in enumerate(paragraphs):
                for qa in paragraph['qas']:
                    feature = {'topic' : topic, 'passage' : passage, 'question' : qa['question']}
                    label = {'answers' : qa['answers'], 'impossible' : qa['is_impossible']}
                    meta = {'id' : qa['id']}
                    if 'plausable_answers' in qa:
                        label['plausable'] = qa['plausable_answers']
                    example = {'meta' : meta, 'labels' : label, 'features' : feature }
                    yield example

    def batch_merge(pipeline):
        """ Merges the batch together """
        return it.MapIterator(pipeline, pipelib.batch_merge)
    def tokenize_text(pipeline):
        """

        Turns text into tokens.

        :param pipeline:
        :return:
        """

    def clean_text(pipeline):
        """

        Cleans the text up a bit,
        and places extra tokens where
        needed.

        :param pipeline:
        :return:
        """
        fastai_cleaner = core.Tokenizer(lambda x : x)
        def cleaner(example):
            mapper = lambda x : pipelib.map_across_strings(x, fastai_cleaner)
            ignore_info = lambda x : pipelib.ignore_keys(x, mapper, ['meta'])
            update = ignore_info(example)
            return update
        return it.MapIterator(pipeline, cleaner)

    def Numericalize(pipeline):
        converter = data.Numericalize()
        def convert(example):


    def convert_to_tensors(pipeline):
        """ Converts strings in the pipe to tensors"""
        def converter(example):
            tensorizer = lambda x : pipelib.map_across_stringlists(x, pipelib.char_tensorize_stringlist)
            output = pipelib.ignore_keys(example, tensorizer, pipe_ignore)
            return output
        return it.MapIterator(pipeline, converter)


    def key(example):
        lengths = [len(item) for item in example['features'].values()]
        return sum(lengths)

    chunks = [source_dir + "/" + file for file in os.listdir(source_dir)]
    pipeline = db.chunked_dataset_iterator(chunks,
                                           load_chunk,
                                           buffer_size=shuffle_buffer,
                                           train=training_mode,
                                           shuffle=training_mode,)
    pipeline = it.BucketedReadaheadBatchIterator(
        pipeline,
        readahead_buffer,
        key,
        batch_size=batch_size)
    pipeline = batch_merge(pipeline)
    pipeline = clean_text(pipeline)
    pipeline = convert_to_tensors(pipeline)
    pipeline = (pipeline)
    pipeline = it.PrefetchIterator(pipeline, prefetch_buffer)
    return pipeline


