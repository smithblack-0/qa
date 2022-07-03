import torch



def traverse(examples, function, seek_condition):
    """

    Descends within the given datastructure until the seek condition is met,
    then apply the function and reassemble the datastructure. Supports
    dicts and lists. if ignore condition is met, ignore that example
    completely, and reassemble as is.


    :param examples: the examples to travers
    :param function: the function to apply
    :seek_condition: A function which accepts a recursive example. Returns true if match reached
    :ignore_condition: A function which accepts a recursive example. Returns true if ignore reached.
    :return:
    """

    subtraverse = lambda x : traverse(x, function, seek_condition)
    if seek_condition(examples):
        return function(examples)
    if isinstance(examples, list):
        update = []
        for subitem in examples:
            update.append(subtraverse(subitem))
        return update
    if isinstance(examples, dict):
        update = examples.copy()
        for key, value in examples.items():
            update[key] = subtraverse(value)
        return update
    return examples

def map_across_strings(example, function):
    """

    Finds every string in the given structure,
    and applies a function to it

    :param example:
    :param function:
    :return:
    """

    def seeking(example):
        if isinstance(example, str):
            return True
        else:
            return False
    return traverse(example, function, seeking)

def map_across_stringlists(stream, function):
    """
    Find lists of string, and apply a function to each one.

    :param stream:
    :param function:
    :return:
    """
    def seeking(example):
        if isinstance(example, list) and isinstance(example[0], str):
            return True
        else:
            return False
    return traverse(stream, function, seeking)


def map_across_tensors(stream, function):
    """

    Find tensors. Apply function. Return stream

    :param example:
    :param function:
    :return:
    """
    def seeking(substream):
        if isinstance(substream, torch.Tensor):
            return True
        else:
            return False
    return traverse(stream, function, seeking)


#################################3

def batch_merge(batch):
    """

    Collects all the items across a single
    batch into lists. For example,
    rather than a batch having structure [{'id' : 1, ..}, {'id' :2, ..}]
    we have ['id' : [1, 2]]


    :param batch: The batch to transform
    :return: The transformed batch
    """

    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        update = dict(zip(keys, [[] for _ in keys]))
        for example in batch:
            for key, value in example.items():
                update[key].append(value)

        for key, value in update.items():
            update[key] = batch_merge(update[key])
        return update
    if isinstance(batch[0], list):
        update = [[]*len(batch[0])]
        for example in batch:
            for item, batchitem in zip(update, example):
                item.append(batchitem)
        for i, item in enumerate(update):
            update[i] = batch_merge(item)
        return update
    return batch

def ignore_keys(example: dict, function, keys):
    """ Applies a function while ignoring dict keys"""
    output = example.copy()
    bypass = [output.pop(key) for key in keys]
    bypass = dict(zip(keys, bypass))
    output: dict = function(output)
    output.update(bypass)
    return output

def char_tensorize_stringlist(strings: list):
    """ Converts a stringlist to a tensor"""
    # calculate max length. Pad all strings to be of this length
    max_length = max([len(item) for item in strings])
    padded_strings = [string + chr(26) * (max_length - len(string)) for string in strings]

    # Convert the strings into integers
    str_converter = lambda string: [ord(char) for char in string]
    converted_strings = [str_converter(item) for item in padded_strings]

    # Convert the integers into tensors
    return torch.tensor(converted_strings)

def char_detensorize(tensor: torch.Tensor):
    """ Converts a tensor representing unicode back into a sequence of stringlists"""
    tensor_lists = tensor.unbind(0)
    str_converter = lambda tensor: "".join([chr(char) for char in tensor if bool(char != 26)])
    converted_strings = [str_converter(item) for item in tensor_lists]
    return converted_strings
