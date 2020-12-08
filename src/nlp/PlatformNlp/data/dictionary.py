import os
from collections import Counter
from multiprocessing import Pool

from PlatformNlp import utils


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        begin="[CLS]",
        pad="[PAD]",
        sep="[SEP]",
        unk="[UNK]",
        mask="[MASK]",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.sep_word, self.begin_word, self.mark_word = unk, pad, sep, begin, mask
        self.symbols = []
        self.count = []
        self.indices = {}
        self.begin_index = self.add_symbol(begin)
        self.pad_index = self.add_symbol(pad)
        self.sep_index = self.add_symbol(sep)
        self.unk_index = self.add_symbol(unk)
        self.mask_index = self.add_symbol(mask)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def finalize(self, threshold=-1, nwords=-1, padding_factor=4):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices



    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        for line in f:
            try:
                line = line.strip("\n")
                line = line.strip("\r")
                line = line.strip(" ")
                word = line
                self.add_symbol(word, n=1, overwrite=True)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    @staticmethod
    def _add_file_to_dictionary(filename, dict, tokenize):
        counter = Counter()
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                line = line.strip("\r")
                for word in tokenize(line):
                    counter.update([word])
        for w, c in sorted(counter.items()):
            dict.add_symbol(w, c)


