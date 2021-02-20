import numpy as np

from util.defragTrees import DefragModel


def getTextsAndLabels(iterator):
    texts = []
    labels = []
    inputs = list()
    for text, label in iterator:
        text, text_length = text
        texts.append(text)
        labels.append(label)
    return np.array([t.numpy() for t in texts]), np.array([l.numpy() for l in labels])


def indexToString(index_arr, TEXT):
    string = TEXT.vocab.itos[index_arr[0]]
    return string if 7 < TEXT.vocab.freqs[string] and string != 'UNK' else ''


def convertTexts(texts, TEXT):
    for text in texts:
        yield " ".join(list(indexToString(t, TEXT) for t in text))


def getCorpus(texts, TEXT):
    return  [text for text in convertTexts(texts, TEXT)]


def getZ(matrix, splitter):
    rnum, cnum = matrix.shape[0], splitter.shape[0]
    Z = np.zeros((rnum, cnum))
    for i in tqdm(range(rnum)):
        flg = np.zeros(cnum)
        for idx, s in enumerate(splitter):
            sidx = int(s[0])
            midx = np.argwhere(matrix[i].indices == sidx)
            if len(midx) == 0:
                continue
            if matrix[i].data[midx[0]] >= s[1]:
                flg[idx] = 1
        Z[i] = flg
    return Z
