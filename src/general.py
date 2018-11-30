"""
General utilities.

@author: Yishai Feldman, IBM Research - Haifa

(c) 2017, IBM Corporation
"""
import codecs
import csv
import unicodedata
import numbers
from bisect import bisect_right, bisect_left
from contextlib import contextmanager

import itertools
import pprint
import re
import string
import sys

import dill
from functools import reduce


class Range:
    """
    A range in a text; beginning is inclusive, end is exclusive.
    """

    def __init__(self, begin, end):
        if not isinstance(begin, int):
            raise ValueError('Begin value "%s" is not an integer' % repr(begin))
        if not isinstance(end, int):
            raise ValueError('End value "%s" is not an integer' % end)
        self.begin = begin
        self.end = end

    @staticmethod
    def exclusive_range(begin, end):
        return Range(begin, end)

    @staticmethod
    def inclusive_range(begin, end):
        return Range(begin, end + 1)

    def apply(self, string):
        """Get this range from the given string"""
        return string[self.begin:self.end]

    def is_legal(self):
        return 0 <= self.begin <= self.end

    def overlaps(self, other):
        """Does this range ovelap the other?"""
        return (not (self.end <= other.begin or other.end <= self.begin)
                # corner case: equal but empty ranges
                or self.begin == self.end == other.begin == other.end)

    def contains(self, other):
        """Does this range contain the other?"""
        return self.begin <= other.begin and self.end >= other.end

    def __call__(self, string):
        return self.apply(string)

    def __eq__(self, other):
        return self.begin == other.begin and self.end == other.end

    def __ne__(self, other):
        return self.begin != other.begin or self.end != other.end

    def __lt__(self, other):
        return self.begin < other.begin or self.begin == other.begin and self.end < other.end

    def __le__(self, other):
        return self.begin < other.begin or self.begin == other.begin and self.end <= other.end

    def __gt__(self, other):
        return self.begin > other.begin or self.begin == other.begin and self.end > other.end

    def __ge__(self, other):
        return self.begin > other.begin or self.begin == other.begin and self.end <= other.end

    def __repr__(self):
        return 'Range.exclusive_range(%s, %s)' % (self.begin, self.end)

    def __hash__(self):
        return self.begin + 0x10001 * self.end


def flatten_tree(tree, field):
    """Recursively yield all elements in 'tree' with field name 'field'."""
    if not tree:
        return
    if type(tree) is list:
        for sub in tree:
            for res in flatten_tree(sub, field):
                yield res
    elif type(tree) is dict:
        # perhaps nested object
        f = tree.get(field, None)
        if f:
            for res in flatten_tree(f, field):
                yield res
        else:
            # leaf node
            yield tree
            return
    else:
        # leaf node
        yield tree


def flatten_list_of_lists(top):
    return [element for sublist in top for element in sublist]


def find_prev_in_sorted(seq, x, key):
    """Find rightmost value in seq less than or equal to x, using an optional key function."""
    if key is not None:
        search_seq = list(map(key, seq))
    else:
        search_seq = seq
    index = bisect_right(search_seq, x)
    if index:
        return seq[index - 1]
    raise IndexError


def find_in_sorted(seq, x, key):
    """
    Find value in seq with key x, using an optional key function.

    :returns value if found, None otherwise
    """
    if key is not None:
        search_seq = list(map(key, seq))
    else:
        search_seq = seq
    index = bisect_left(search_seq, x)
    if index < len(seq) and search_seq[index] == x:
        return seq[index]
    return None


def gen_all_occurrences(substring, string):
    """
    Find all occurrences of a word-delimited substring in a string, including overlaps.

    :param substring: substring to search for
    :param string: string to search in
    :return: a generator for all starting positions of substring in string
    """
    pat = str(substring)
    if str.isalpha(substring[0]):
        pat = r'\b' + pat
    if str.isalpha(substring[-1]):
        pat += r'\b'
    cpat = re.compile(pat, re.UNICODE)
    start = 0
    while True:
        match = cpat.search(string, start)
        if not match:
            return
        index = match.start()
        yield index
        start = index + 1


def marker_string(markers, char='^'):
    """
    Return a string containing markers in the given sequence of positions

    :param markers: positions to put markers in
    :param char: marker character
    :return: string with marker character in given positions
    """
    s = sorted(markers)
    prev = -len(char)
    spaces = []
    for marker in s:
        spaces.append(' ' * (marker - prev - len(char)))
        prev = marker
    return char.join(spaces) + char


def prune_nulls(seq):
    """Filter all empty (false) elements from a sequence"""
    return list(itertools.compress(seq, seq))


def unique_word_in_string(sub, full):
    """
    Find starting position of 'sub' in 'full', provided it only appears once.
    Search for 'sub' only at word boundaries.
    Return -1 if 'sub' doesn't appear at all, -2 if it appears multiple times.

    :param sub: substring to search for
    :param full: string to search in
    :return: starting position if unique, -1 if none, -2 if multiple times
    """
    # N.B. can't use '\b' since 'sub' may not begin or end with a word character
    pat = re.compile(r'(?:^|\W)(' + sub + r')(?:$|\W)', re.UNICODE)
    match = pat.search(full)
    if not match:
        return -1
    begin = match.start(1)
    second = pat.search(full, begin + 1)
    if second < 0:
        return begin
    return -2


def column_to_index(col):
    """Translate a numeric or character column name into a 1-based index"""
    if isinstance(col, numbers.Number):
        return col
    try:
        num = int(col)
        return num
    except ValueError:
        return reduce(lambda res, digit: res * 26 + digit,
                      (ord(ch.upper()) - ord('A') + 1 for ch in col))


def index_to_column(index):
    """Translate a 1-based column index to a character column name"""
    letters = string.letters[26:]
    result = ''
    while index:
        mod = (index - 1) % 26
        result = letters[mod] + result
        index = (index - mod) // 26
    return result


def print_dill(dill_file, out_file=None, objects=1, skip=0):
    out = sys.stdout
    if out_file:
        out = open(out_file, 'w')
    with open(dill_file, 'rb') as f:
        for i in range(skip):
            dill.load(f)
        for i in range(objects):
            print(pprint.pformat(dill.load(f), width=250), file=out)
    if out_file:
        out.close()


def split_iterator(sequence, blocksize):
    it = iter(sequence)
    while True:
        chunk = list(itertools.islice(it, blocksize))
        if not chunk:
            break
        yield chunk


def remove_suffix_after(string, marker):
    i = string.find(marker)
    if i < 0:
        return string
    return string[:i]


UNICODE_BOM = str(codecs.BOM_UTF8, encoding='utf-8')


def filter_printable(text):
    if isinstance(text, str):
        return [c for c in text if c != UNICODE_BOM
                                and unicodedata.category(c) not in ('Cc', 'Cn', 'Co')]
    return text


def contract_text(text):
    """
    Return a contraction of the given text, useful as a key

    :param text: any text
    :return: a contracted form that can be used as a key
    """
    key = ''.join([ch for ch in str(text) if str.isalpha(ch)])
    return key


def split_after(values, splits):
    """
    Split a sequence into sub-sequences immediately after locations indicated by the sequence of splits.

    :param values: sequence to split
    :param splits: boolean sequence of split locations; should be equal in length to `values`
    :return: iterator of lists, each of which is a sub-sequence
    """
    subseq = []
    for value, split in zip(values, splits):
        subseq.append(value)
        if split:
            yield subseq
            subseq = []
    if subseq:
        yield subseq


def invert_multiple_valued_dict(d):
    """
    Given a dict whose values are sequences, return a dict mapping from each value to the set of corresponding keys

    :param d: input dict
    :return: inverse dict, values are sets of keys
    """
    result = {}
    for key, values in list(d.items()):
        for v in values:
            result.setdefault(v, set()).add(key)
    return result


@contextmanager
def labeled_break():
    """
    Create a label (using 'with'); use raise with that label to quit the scope of the label.
    """

    class LabeledBreakException(Exception):
        pass

    try:
        yield LabeledBreakException
    except LabeledBreakException:
        pass


if __name__ == '__main__':
    pass


def compare_sets(seq1, seq2, name1, name2):
    """
    Print the symmetric difference of two sequences, viewed as sets.

    :param seq1: first sequence
    :param seq2: second sequence
    :param name1: name of first sequence
    :param name2: name of second sequence
    """
    set1 = set(seq1)
    set2 = set(seq2)
    print('%s: %s' % (name1, sorted(list(set1 - set2))))
    print('%s: %s' % (name2, sorted(list(set2 - set1))))


def non_overriding_update(d1, d2):
    """
    Update a dict `d1` with default key/value pairs from `d2` if the keys don't already exist in d1.

    :param d1: original dict
    :param d2: default dict
    :return: nothing; modifies `d1`; doesn't modify `d2`
    """
    tmp = {}
    tmp.update(d2)
    tmp.update(d1)
    d1.update(tmp)


def non_overriding_union(d1, d2):
    """
    Return a new dict with the pairs from `d1` and default key/value pairs from `d2` if the keys don't already exist in d1.

    :param d1: original dict
    :param d2: default dict
    :return: new dict containing pairs from `d1`, as well as pairs from `d2` whose keys are not in `d1`
    """

    result = {}
    result.update(d2)
    result.update(d1)
    return result
