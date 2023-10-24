import argparse
import os
import tqdm
import numpy as np
from numpy.lib.format import open_memmap
import torch

from rlaif.annotators import ScoreAnnotator, ReturnAnnotator, LanguageModelAnnotator
from rlaif.dataset import PairsDataset, dict_collate_fn
from rlaif.llms import AnnotationIdx


parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default="./motif_dataset",
                    help="Directory of the dataset of pairs.")
parser.add_argument('--annotator-type', type=str, default='llama',
                    choices=['score', 'return', 'llama'], help="Type of annotator to use.")
parser.add_argument('--custom-annotator-string', type=str, default=None,
                    help="Custom tag to be used for the annotation, overriding the default one.")
# Parameters used only for the llama annotator
parser.add_argument('--prompt-version', type=str, default='default',
                    choices=['default', 'reworded'], help="Version of the prompt to use.")
parser.add_argument('--goal-key', type=str, default='defaultgoal',
                    choices=['defaultgoal', 'zeroknowledge', 'combat', 'gold', 'stairs'],
                    help="Key for the behavior-specification string to be added to the prompt.")
parser.add_argument('--use-messages', type=bool, default=True,
                    help="Whether to use messages as encodings for the observation.")
parser.add_argument('--use-blstats', type=bool, default=False,
                    help="Whether to use blstats as encodings for the observation.")
parser.add_argument('--llm-size', type=int, default=70, choices=[7, 13, 70],
                    help="Size of the language model to use.")
parser.add_argument('--num-gpus', type=int, default=8,
                    help="Number of GPUs to use for the language model.")
parser.add_argument('--logdir', type=str, default=None,
                    help="Name of the directory to log the conversations of the LLM.")
# "System" parameters
parser.add_argument('--batch-size', type=int, default=2000,
                    help="Number of prompts that will be processed continuously.")
parser.add_argument('--num-workers', type=int, default=80,
                    help="Number of workers for the dataloader.")
parser.add_argument('--n-annotation-chunks', type=int, default=1,
                    help="Number of chunks to split the dataset into.")
parser.add_argument('--chunk-number', type=int, default=0,
                    help="Chunk number that this instance of the script will process.")
parser.add_argument('--flushing-freq', type=int, default=1,
                    help='Number of batches after which the annotations will be flushed to disk.')
parser.add_argument('--ignore-existing', action='store_true',
                    help='Whether to ignore existing annotations (with same options) and overwrite them.')
flags = parser.parse_args()

# Setup annotator
if flags.annotator_type == 'score':
    annotator = ScoreAnnotator(batch_size=flags.batch_size)
elif flags.annotator_type == 'return':
    annotator = ReturnAnnotator(batch_size=flags.batch_size)
elif flags.annotator_type == 'llama':
    annotator = LanguageModelAnnotator(batch_size=flags.batch_size,
                                       model_name=f'meta-llama/Llama-2-{flags.llm_size}b-chat-hf',
                                       use_messages=flags.use_messages, use_blstats=flags.use_blstats,
                                       logdir=flags.logdir, prompt_version=flags.prompt_version,
                                       goal_key=flags.goal_key, num_gpus=flags.num_gpus)
else:
    raise ValueError("Annotator type {} not recognized".format(flags.annotator_type))

or_dataset = PairsDataset(directory=flags.directory, data_keys=annotator.data_keys,
                          info_keys=annotator.info_keys, preference_keys=None,
                          transform=annotator.transform)

if flags.custom_annotator_string is None:
    if flags.annotator_type != 'llama':
        annotator_string = flags.annotator_type
    else:
        msg_str = "msg_" if flags.use_messages else ""
        bl_str = "minbl_" if flags.use_blstats else ""
        annotator_string = f"llama{flags.llm_size}b_{msg_str}{bl_str}{flags.goal_key}_{flags.prompt_version}"
else:
    annotator_string = flags.custom_annotator_string
filename = os.path.join(flags.directory, 'preference', annotator_string + ".npy")

if not os.path.exists(filename) or flags.ignore_existing:
    annotation_array = open_memmap(filename, dtype=np.int8, mode='w+', shape=(len(or_dataset),))
    annotation_array[:] = AnnotationIdx.UNKOWN
else:
    annotation_array = open_memmap(filename, dtype=np.int8, mode='r+')

assert len(or_dataset) % flags.n_annotation_chunks == 0

# Restrict the dataset to the portion that (1) is part of this chunk and (2) has the mask at False
low_idx = flags.chunk_number*len(or_dataset)//flags.n_annotation_chunks
high_idx = (flags.chunk_number+1)*len(or_dataset)//flags.n_annotation_chunks
indices = np.arange(low_idx, high_idx)[annotation_array[low_idx:high_idx] == AnnotationIdx.UNKOWN]
dataset = torch.utils.data.Subset(or_dataset, indices)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size,
                                          collate_fn=dict_collate_fn, shuffle=False,
                                          num_workers=flags.num_workers)

print("Annotating chunk {} of {}".format(flags.chunk_number, flags.n_annotation_chunks))
for i, batch_data in enumerate(tqdm.tqdm(data_loader, total=(len(dataset)//flags.batch_size))):
    curr_idx = i * flags.batch_size
    end_idx = min((i+1) * flags.batch_size, len(dataset))
    annotation = annotator(batch=batch_data, logging_indices=indices[curr_idx:end_idx])
    annotation_array[indices[curr_idx:end_idx]] = annotation
    if i % flags.flushing_freq == 0:
        annotation_array.flush()
annotation_array.flush()
print("Done annotating chunk")
