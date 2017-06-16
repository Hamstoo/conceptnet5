import pandas as pd
import numpy as np
import logging
import gzip
import struct
import pickle
from .transforms import l1_normalize_columns, l2_normalize_rows, standardize_row_labels
from .oocframe import OOCFrame
from conceptnet5.uri import uri_prefix
from conceptnet5.vectors import standardized_uri


def load_hdf(filename):
    """
    Load a semantic vector space from an HDF5 file.

    HDF5 is a complex format that can contain many instances of different kinds
    of data. The convention we use is that the file contains one labeled
    matrix, named "mat".
    """
    return pd.read_hdf(filename, 'mat', encoding='utf-8')


def save_hdf(table, filename):
    """
    Save a semantic vector space into an HDF5 file, following the convention
    of storing it as a labeled matrix named 'mat'.
    """
    return table.to_hdf(filename, 'mat', mode='w', encoding='utf-8')


def save_labels_and_npy(table, vocab_filename, matrix_filename):
    """
    Save a semantic vector space in two files: a NumPy .npy file of the matrix,
    and a text file with one label per line. We use this for exporting the
    Luminoso background space.
    """
    np.save(matrix_filename, table.values)
    save_index_as_labels(table.index, vocab_filename)


def vec_to_text_line(label, vec):
    """
    Output a labeled vector as a line in a fastText-style text format.
    """
    cells = [label] + ['%4.4f' % val for val in vec]
    return ' '.join(cells)


def export_text(frame, filename, filter_language=None):
    """
    Save a semantic vector space as a fastText-style text file.

    If `filter_language` is set, it will output only vectors in that language.
    """
    vectors = frame.values
    index = frame.index
    if filter_language is not None:
        start_idx = index.get_loc('/c/%s/#' % filter_language, method='bfill')
        try:
            end_idx = index.get_loc('/c/%s0' % filter_language, method='bfill')
        except KeyError:
            end_idx = frame.shape[0]
        frame = frame.iloc[start_idx:end_idx]
        index = frame.index

    with gzip.open(filename, 'wt') as out:
        dims = "%s %s" % frame.shape
        print(dims, file=out)
        for i in range(frame.shape[0]):
            label = index[i]
            if filter_language is not None:
                label = label.split('/', 3)[-1]
            vec = vectors[i]
            print(vec_to_text_line(label, vec), file=out)


def convert_glove(glove_filename, output_filename, nrows):
    """
    Convert GloVe data from a gzipped text file to an HDF5 dataframe.
    """
    glove_raw = load_glove(glove_filename, nrows)
    glove_std = standardize_row_labels(glove_raw, forms=False)
    del glove_raw
    glove_normal = l2_normalize_rows(l1_normalize_columns(glove_std))
    del glove_std
    save_hdf(glove_normal, output_filename)


def convert_fasttext(fasttext_filename, output_filename, nrows, language):
    """
    Convert FastText data from a gzipped text file to an HDF5 dataframe.
    """
    ft_raw = load_fasttext(fasttext_filename, nrows)
    ft_std = standardize_row_labels(ft_raw, forms=False, language=language)
    del ft_raw
    # TODO: it doesn't seem to make sense to l1_normalize_columns across different
    # TODO: languages which presumably all get fit separately--the dimensions won't
    # TODO: have the same meanings
    ft_normal = l2_normalize_rows(l1_normalize_columns(ft_std))
    del ft_std
    save_hdf(ft_normal, output_filename)


def convert_fasttext_2_oocframe(fasttext_filename, output_path, nrows, language=None, prefix_required=True):
    """
    Convert FastText data from a gzipped text file to an OOCFrame which is basically
    a directory on disk containing separate files for each label so that the entire
    data structure doesn't have to be held in memory all at once.  It's not necessary
    to have the entire data structure in memory because all that's really needed to
    query it is its index of terms.
    """
    if language is None or len(language) != 2:
        raise ValueError('Unsupported language: {}'.format(language))
    prefix = '/c/{}/'.format(language)

    oocframe = OOCFrame(output_path)
    with gzip.open(fasttext_filename, 'rt') as infile:
        nrows_str, ncols_str = infile.readline().rstrip().split()
        nrows = min(int(nrows_str), nrows)
        _ = int(ncols_str)
        i = 0  # ensure that nrows is applied to the specified language
        for line in infile:
            if nrows > 0 and i >= nrows:
                break
            items = line.rstrip().split(' ')
            label = items[0]

            # setting `include_punctuation=True` in `tokens.simple_tokenize` doesn't properly handle '#' chars
            # the same was as if the prefixes are already present (it ends up putting an underscore between them
            # and any following chars, e.g. '###mm_mortar' gets converted to '###_mm_mortar') and it would take
            # a lot to fix the regexes that function uses so just punch a prefix here (even though that's part
            # of the purpose of `standardized_uri`) so that we don't have to deal with it
            if not prefix_required and not label.startswith(prefix):
                label = prefix + label

            if label.startswith(prefix):

                new_label = uri_prefix(standardized_uri(language, label))
                if new_label != label:
                    print('{} changed to {}'.format(label, new_label))
                    label = new_label

                values = [float(x) for x in items[1:]]
                oocframe.insert(label, values, weight=1.0 / (i + 1))
                i += 1

    logging.debug('oocframe.combine_weights()')
    oocframe.combine_weights()

    # TODO: implement support for the real DataFrame interface
    # TODO: i.e. oocframe = l2_normalize_rows(l1_normalize_columns(oocframe))
    logging.debug('oocframe.l1_normalize_columns()')
    oocframe.l1_normalize_columns()
    logging.debug('oocframe.l2_normalize_rows()')
    oocframe.l2_normalize_rows()


def convert_word2vec(word2vec_filename, output_filename, nrows, language='en'):
    """
    Convert word2vec data from its gzipped binary format to an HDF5
    dataframe.
    """
    w2v_raw = load_word2vec_bin(word2vec_filename, nrows)
    w2v_std = standardize_row_labels(w2v_raw, forms=False, language=language)
    del w2v_raw
    w2v_normal = l2_normalize_rows(l1_normalize_columns(w2v_std))
    del w2v_std
    save_hdf(w2v_normal, output_filename)


def convert_polyglot(polyglot_filename, output_filename, language):
    """
    Convert Polyglot data from its pickled format to an HDF5 dataframe.
    """
    pg_raw = load_polyglot(polyglot_filename)
    pg_std = standardize_row_labels(pg_raw, language, forms=False)
    del pg_raw
    save_hdf(pg_std, output_filename)


def load_glove(filename, max_rows=1000000):
    """
    Load a DataFrame from the GloVe text format, which is the same as the
    fastText format except it doesn't tell you up front how many rows and
    columns there are.
    """
    arr = None
    label_list = []
    with gzip.open(filename, 'rt') as infile:
        for i, line in enumerate(infile):
            if i >= max_rows:
                break
            items = line.rstrip().split(' ')
            label_list.append(items[0])
            if arr is None:
                ncols = len(items) - 1
                arr = np.zeros((max_rows, ncols), 'f')
            values = [float(x) for x in items[1:]]
            arr[i] = values

    if len(label_list) < max_rows:
        arr = arr[:len(label_list)]
    return pd.DataFrame(arr, index=label_list, dtype='f')


def load_fasttext(filename, max_rows=1000000):
    """
    Load a DataFrame from the fastText text format.

    Load the numberbatch-17.04.txt.gz file--the one that's downloadable from
    https://github.com/commonsense/conceptnet-numberbatch.
    """
    arr = None
    labels = []
    with gzip.open(filename, 'rt') as infile:
        nrows_str, ncols_str = infile.readline().rstrip().split()
        nrows = min(int(nrows_str), max_rows)
        ncols = int(ncols_str)
        arr = np.zeros((nrows, ncols), dtype='f')
        for i, line in enumerate(infile):
            if i >= nrows:
                break
            items = line.rstrip().split(' ')
            labels.append(items[0])
            values = [float(x) for x in items[1:]]
            arr[i] = values

    return pd.DataFrame(arr, index=labels, dtype='f')


def _read_until_space(file):
    chars = []
    while True:
        newchar = file.read(1)
        if newchar == b'' or newchar == b' ':
            break
        chars.append(newchar[0])
    return bytes(chars).decode('utf-8', 'replace')


def _read_vec(file, ndims):
    fmt = 'f' * ndims
    bytes_in = file.read(4 * ndims)
    values = list(struct.unpack(fmt, bytes_in))
    return np.array(values)


def load_word2vec_bin(filename, nrows):
    """
    Load a DataFrame from word2vec's binary format. (word2vec's text format
    should be the same as fastText's, but it's less efficient to load the
    word2vec data that way.)
    """
    label_list = []
    arr = None
    with gzip.open(filename, 'rb') as infile:
        header = infile.readline().rstrip()
        nrows_str, ncols_str = header.split()
        nrows = min(int(nrows_str), nrows)
        ncols = int(ncols_str)
        arr = np.zeros((nrows, ncols), dtype='f')
        while len(label_list) < nrows:
            label = _read_until_space(infile)
            vec = _read_vec(infile, ncols)
            if label == '</s>':
                # Skip the word2vec sentence boundary marker, which will not
                # correspond to anything in other data
                continue
            idx = len(label_list)
            arr[idx] = vec
            label_list.append(label)

    return pd.DataFrame(arr, index=label_list, dtype='f')


def load_polyglot(filename):
    """
    Load a pickled matrix from the Polyglot format.
    """
    labels, arr = pickle.load(open(filename, 'rb'), encoding='bytes')
    label_list = list(labels)
    return pd.DataFrame(arr, index=label_list, dtype='f')


def load_labels_and_npy(label_file, npy_file):
    """
    Load a semantic vector space from two files: a NumPy .npy file of the matrix,
    and a text file with one label per line.
    """
    label_list = [line.rstrip('\n') for line in open(label_file, encoding='utf-8')]
    arr = np.load(npy_file)
    return pd.DataFrame(arr, index=label_list, dtype='f')


def load_labels_as_index(label_filename):
    """
    Load a set of labels (with no attached vectors) from a text file, and
    represent them in a pandas Index.
    """
    labels = [line.rstrip('\n') for line in open(label_filename, encoding='utf-8')]
    return pd.Index(labels)


def save_index_as_labels(index, label_filename):
    """
    Save a pandas Index as a text file of labels.
    """
    with open(label_filename, 'w', encoding='utf-8') as out:
        for label in index:
            print(label, file=out)
