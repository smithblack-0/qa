"""

Preps lambada dataset for loading
by dividing it into chunks of reasonable length

"""
from QA_Source.lib import chunklib

source_dir = r'C:\Users\chris\PycharmProjects\qa\Data\lambada-dataset\train-novels'
destination_dir = r'C:\Users\chris\PycharmProjects\qa\Chunks\Lambada\Pretrain'
length = 200000
chunklib.chunk(source_dir,
               destination_dir,
               length)