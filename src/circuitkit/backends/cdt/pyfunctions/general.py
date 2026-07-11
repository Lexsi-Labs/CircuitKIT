# -*- coding: utf-8 -*-
import collections
import json
import os

import numpy as np
import torch


import logging

logger = logging.getLogger(__name__)

def createDirectory(path):
    """
    * Create a directory at a given path if it does not exist
    """
    if not os.path.exists(path):
        os.mkdir(path)


def lst2str(lst):
    """
    * Join the elements of a list to a string and return the string
    """
    s = " ".join(lst)
    return s


def elements2str(lst):
    """
    * Maps sublists of a list to a string
    """
    processed = []
    for element in lst:
        processed.append(lst2str(element))
    return processed


def stripString(line, stripChars, replaceStr):
    """
    * Strip and replace all characters in stripChars in a given string
    * and return the procesesd string
    """
    for c in stripChars:
        line = line.replace(c, replaceStr)
    return line


def saveJson(path, obj):
    """
    * Save a python object at a given path
    """
    with open(path, "w") as outfile:
        json.dump(obj, outfile)


def readJson(path):
    """
    * Load a json file at a given path and return the python object
    """
    with open(path, "r") as fp:
        data = json.load(fp)
        return data


def readDocument(path):
    """
    * Read in a document at a given path and return the document as string
    """
    f = open(path, "r")
    document = f.readlines()
    return document


def moveFiles(names, originalPath, newPath):
    """
    * Given a list of file names, move them from origial path to
    * the new path
    """
    for name in names:
        os.rename(originalPath + name + ".json", newPath + name + ".json")


def extractListfromTuples(tuples, index):
    """
    * Given a list of tuples, return a list containing the elements of
    * the tuples at a given index
    """
    lst = []

    for element in tuples:
        lst.append(element[index])
    return lst


def extractSubsetList(lst, indices):
    """
    * Given a list of lists, return a list of sublists at a given set
    * of indices
    """
    sublst = []
    for element in lst:
        sub_element = np.array(element)[indices].tolist()
        sublst.append(sub_element)
    return sublst


def getUniqueColValues(labels):
    """
    * Given a list of lists, return a list of lists that contain the
    * unique values of each list element
    """
    columns = [[] for i in range(len(labels[0]))]

    for patient in labels:
        for x, label in enumerate(patient):
            if label not in columns[x]:
                columns[x].append(label)
    return columns


def getNumMaxOccurrences(lst):
    """
    * Given a list, return the number of times the mode appears
    """
    max_occur = max(lst, key=lst.count)
    return lst.count(max_occur)


def getClassIndices(y):
    """
    * Given a list of values, find all the unique values that occur
    * and return a list containing lists that contain the indices
    * that containo each unique value
    """
    vals = np.unique(y)
    lsts = []

    for val in vals:
        lsts.append(np.where(y == val)[0].tolist())
    return lsts


def combineListElements(lst1, lst2, combineChar="_"):
    """
    * Given two lists of strings, return a single list containing
    * the concatenation of the strings in each list
    """
    combined = []
    for i in range(len(lst1)):
        combined.append(lst1[i] + combineChar + lst2[i])
    return combined


def listFilesType(path, fileType):
    """
    * Given a path, list all files that are of type fileType
    """
    lst = []
    for f in os.listdir(path):
        if f.endswith(fileType):
            lst.append(f)
    return lst


def mapList(lst, mapping):
    """
    * Given a list, return a list containing the mapped values of the
    * elements given a mapping (python dictionary)
    """
    processed = []
    for elem in lst:
        processed.append(mapping[elem])
    return processed


def saveTxtFile(path, text):
    f = open(path, "a")
    f.write(str(text))
    f.close()


def hasNumeric(word):
    for i in range(10):
        if str(i) in word:
            return True
    return False


def hasLetter(word):
    letters = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    for letter in letters:
        if letter in word:
            return True
    return False


def extractListFromDic(lst, key, additional_key=None):
    extracted = []

    for i in range(len(lst)):
        if additional_key is None:
            extracted.append(lst[i][key])
        else:
            extracted.append(lst[i][key][additional_key])

    return extracted


def find_sub_list(sl, lyr):
    sll = len(sl)
    for ind in (i for i, e in enumerate(lyr) if e == sl[0]):
        if lyr[ind : ind + sll] == sl:
            return ind, ind + sll - 1


def plot_identified_heads_att_matrix(att_probs, identified_heads):
    for level, h in identified_heads:
        att_m = att_probs[level, h, :, :]
        s = sns.heatmap(att_m, cmap="Blues")  # noqa: F821 - pre-existing vendored bug
        s.set(xlabel="Pos", ylabel="Pos")
        plt.title(f"Attention Prob {level}.{h}")  # noqa: F821 - pre-existing vendored bug
        plt.grid(True)  # noqa: F821 - pre-existing vendored bug
        plt.show()  # noqa: F821 - pre-existing vendored bug


def print_highlight_text(encoding, pos_lst, use_print=True):
    formatted = []
    for i in range(512):
        t = tokenizer.decode(encoding["input_ids"][:, i])  # noqa: F821 - pre-existing vendored bug
        if i in pos_lst:
            formatted.append("\033[103m " + t + "\033[49m")
        else:
            formatted.append(t)

    result = " ".join(formatted)
    if use_print:
        logger.info(result)

    return result


# for examining attn maps
def combine_token_attn(interval_dict, avg_att_m):
    word_cnt = len(interval_dict)
    new_att_m = np.zeros(word_cnt)
    for i in range(word_cnt):
        t_idx_lst = interval_dict[i + 1]
        if len(t_idx_lst) == 1:
            new_att_m[i] = avg_att_m[t_idx_lst[0]]
        else:
            new_att_m[i] = np.sum(avg_att_m[t_idx_lst[0] : t_idx_lst[-1] + 1])
    return new_att_m


def compute_word_intervals(encoding, tokenizer):
    word_cnt = 0
    interval_dict = collections.defaultdict(list)

    pretok_sent = ""
    for i in range(512):
        tok = tokenizer.decode(encoding["input_ids"][:, i])
        if tok.startswith("##"):
            interval_dict[word_cnt].append(i)
            pretok_sent += tok[2:]
        else:
            word_cnt += 1
            interval_dict[word_cnt].append(i)
            pretok_sent += " " + tok
    pretok_sent = pretok_sent[1:]
    word_lst = pretok_sent.split(" ")

    assert len(interval_dict) == len(word_lst)

    return interval_dict, word_lst


def compare_same(a, b, atol=1e-4, rtol=1e-3):
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    comparison = np.isclose(a, b, atol, rtol)
    proportion = comparison.sum() / comparison.size
    logger.info(f"{proportion:.2%} of the values are equal")
    return proportion
