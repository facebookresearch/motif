from abc import ABC, abstractmethod
import itertools
from typing import Dict, List, Callable, Optional, Tuple, Sequence

import numpy as np
import torchvision

from rlaif.annotators_transforms import BlstatsTransform, MessageTransform
from rlaif.prompts import system_prompts, prompt_templates, goal_strings, regexes, retry_prompts
from rlaif.llms import LocalLanguageModel, AnnotationIdx


class Annotator(ABC):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int]) -> np.array:
        """General method which takes two sequences and returns whether the second element
        is better than the first one, for each batch element,
        together with a mask of the valid/invalid elements.

        Args:
            batch: Dictionary of arrays containing the data for the two sequences (bs, 2, subepisode_length, dims)
            logging_indices: a list of indices for logging info about computation for each element
        Return:
            annotations: int array of shape (bs,) where each element is out of (first, second, tie, invalid)
        """
        pass

    @property
    @abstractmethod
    def data_keys(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def info_keys(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def transform(self) -> Optional[Callable]:
        pass


class LanguageModelAnnotator(Annotator):
    """Annotator that annotates based on the output of a language model."""
    def __init__(self, batch_size: int, model_name: str,
                 use_messages: bool, use_blstats: bool,
                 num_gpus: int = 8, logdir: Optional[str] = None,
                 prompt_version: str = 'v0',
                 goal_key: str = '') -> None:
        assert use_messages or use_blstats
        self.use_messages = use_messages
        self.use_blstats = use_blstats
        self.blstats_keys = [
           'NLE_BL_DEPTH', 'NLE_BL_GOLD', 'NLE_BL_HP',
           'NLE_BL_HPMAX', 'NLE_BL_XP', 'NLE_BL_HUNGER'
        ]
        self.llm = LocalLanguageModel(system_prompt=system_prompts[prompt_version],
                                      answer_regex=regexes[prompt_version],
                                      retry_prompt=retry_prompts[prompt_version],
                                      model_name=model_name, num_gpus=num_gpus,
                                      logdir=logdir)
        self.prompt_template = prompt_templates[prompt_version]
        self.goal_key = goal_key
        super().__init__(batch_size)

    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int] = None) -> np.ndarray:
        messages, time_indices = self.preprocess_messages(batch['message']) if self.use_messages else None
        blstats = self.preprocess_blstats(batch['blstats'], time_indices) if self.use_blstats else None
        prompts, preserved_indices = self.prepare_prompts(messages, blstats)
        results = self.llm.generate(prompts,
                                    np.array(logging_indices)[preserved_indices] if logging_indices is not None else None)
        recomposed_results = np.full(len(batch[list(batch.keys())[0]]), AnnotationIdx.TIE)
        recomposed_results[preserved_indices] = results
        return recomposed_results

    def preprocess_messages(self, batched_messages):
        # Expects (bs, 2, seq_len)
        # Returns a list of lists of lists
        # In which the first one is the batch dimension, the second one is 2
        # the third one is the "message sequence" length
        condensed_messages = []
        message_indices = []

        for pair in batched_messages:
            condensed_pair, pair_indices = [], []
            for messages in pair:
                condensed, indices = self.group_messages(messages)
                condensed_pair.append(condensed)
                pair_indices.append(indices)
            condensed_messages.append(condensed_pair)
            message_indices.append(pair_indices)

        return condensed_messages, message_indices

    def group_messages(self, messages: List[str]) -> Tuple[List[str], List[int]]:
        groups = []
        indices = []
        current_index = 0
        for key, group in itertools.groupby(messages):
            # Exclude empty strings
            if key == "":
                current_index += len(list(group))
                continue
            count = len(list(group))
            groups.append(f'{key} (x{count})' if count > 1 else key)
            indices.append(current_index)
            current_index += count
        return groups, indices if indices != [] else [0]

    def preprocess_blstats(self, blstats: List[List[str]],
                           message_indices: List[List[int]]) -> Tuple[List[str]]:
        selected_blstats = []
        for pair, message_index in zip(blstats, message_indices):
            selected_pair = []
            for j, list_indices in enumerate(message_index):
                selected = [pair[j][k] for k in list_indices]
                selected_pair.append(selected)
            selected_blstats.append(selected_pair)
        return selected_blstats

    def combine_message_blstats(self, message: Optional[str], blstats: Optional[str]):
        filler = ' ' if message is not None and blstats is not None else ''
        msg_str = '' if message is None else message
        bl_str = '' if blstats is None else blstats
        return str(msg_str) + str(filler) + str(bl_str)

    def prepare_prompts(self, batched_messages: List[List[str]],
                        batched_blstats: List[List[str]]) -> Tuple[List[str], List[int]]:
        if batched_messages is not None and batched_blstats is not None:
            assert len(batched_messages) == len(batched_blstats)
        if batched_messages is None:
            batched_messages = [[None, None]]*len(batched_blstats)
        if batched_blstats is None:
            batched_blstats = [[None, None]]*len(batched_messages)

        preserved_indices = []
        prompts = []
        for prompt_idx, (messages, blstats) in enumerate(zip(batched_messages, batched_blstats)):
            seq_1 = []
            for i in range(max(len(messages[0]) if self.use_messages else 0,
                               len(blstats[0]) if self.use_blstats else 0)):
                message = messages[0][i] if self.use_messages and len(messages[0]) > 0 else None
                blstat = blstats[0][i] if self.use_blstats and len(blstats[0]) > 0 else None
                seq_1.append(self.combine_message_blstats(message, blstat))
            seq_1 = "\n".join(seq_1)
            seq_2 = []
            for i in range(max(len(messages[1]) if self.use_messages else 0,
                               len(blstats[1]) if self.use_blstats else 0)):
                message = messages[1][i] if self.use_messages and len(messages[1]) > 0 else None
                blstat = blstats[1][i] if self.use_blstats and len(blstats[1]) > 0 else None
                seq_2.append(self.combine_message_blstats(message, blstat))
            seq_2 = "\n".join(seq_2)
            if seq_1 != seq_2:
                preserved_indices.append(prompt_idx)
                prompts.append(self.prompt_template.format(goal_strings[self.goal_key], seq_1, seq_2))
        return prompts, preserved_indices

    @property
    def data_keys(self) -> List[str]:
        needed_keys = []
        if self.use_messages:
            needed_keys.append('message')
        if self.use_blstats:
            needed_keys.append('blstats')
        return needed_keys

    @property
    def info_keys(self) -> Optional[List[str]]:
        return None

    @property
    def transform(self):
        transforms = []
        if self.use_messages:
            transforms.append(MessageTransform())
        if self.use_blstats:
            transforms.append(BlstatsTransform(self.blstats_keys))
        return torchvision.transforms.Compose(transforms)


class RandomAnnotator(Annotator):
    """Annotator that annotates randomly."""
    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int] = None) -> np.array:
        return np.random.choice([0, 1, 2], size=(self.batch_size,), dtype=np.int8)

    @property
    def data_keys(self) -> Optional[List[str]]:
        return None

    @property
    def info_keys(self) -> Optional[List[str]]:
        return None

    @property
    def transform(self):
        return None


class ScoreAnnotator(Annotator):
    """Annotator that annotates based on the score from which the subepisodes were sampled."""
    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int] = None) -> np.array:
        pairs = np.array((batch['score'][:, 0], batch['score'][:, 1]))
        out = pairs.argmax(axis=0)
        out[pairs[0] == pairs[1]] = AnnotationIdx.TIE
        return out

    @property
    def data_keys(self) -> Optional[List[str]]:
        return None

    @property
    def info_keys(self) -> Optional[List[str]]:
        return ['score']

    @property
    def transform(self):
        return None


class ReturnAnnotator(Annotator):
    """Annotation based on the return of the subepisodes."""
    def __call__(self, batch: Dict[str, np.ndarray], logging_indices: Sequence[int] = None) -> np.array:
        rets1 = batch['rewards'][:, 0].sum(axis=-1)
        rets2 = batch['rewards'][:, 1].sum(axis=-1)
        pairs = np.array((rets1, rets2))
        out = pairs.argmax(axis=0)
        out[pairs[0] == pairs[1]] = AnnotationIdx.TIE
        return out

    @property
    def data_keys(self) -> Optional[List[str]]:
        return ['rewards']

    @property
    def info_keys(self) -> Optional[List[str]]:
        return None

    @property
    def transform(self):
        return None
