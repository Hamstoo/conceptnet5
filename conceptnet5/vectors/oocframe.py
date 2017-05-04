import os
import re
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from conceptnet5.vectors import standardized_uri


class OOCFrame(object):
    """Out-of-core pandas.DataFrame-like object.  The index is in-core, but the data
    are stored in files on disk.
    TODO: not sure if this should derive from DataFrame or not.
    """
    def __init__(self, path):
        self._path = path

        # if the path already exists then we'll assume the data has already
        # been (fully) written to disk
        if os.path.exists(self._path):
            fpaths = [y for x in os.walk(self._path) for y in glob(os.path.join(x[0], '*.npy'))]
            self._index = [self._filepath2label(f) for f in fpaths]
            self.disable_insert()
        else:
            # these members support the `combine_weights` method
            self._fileroots2ids = defaultdict(lambda: 0)
            self._filenames2weights = {}
            self._index = []

    N_CHAR_DIRS = 3

    # don't use [^\w#] because it doesn't catch e.g. u with umlaut and Greeks
    NON_WORD_CHAR_RE = re.compile(r'[^A-Za-z0-9_#]')

    def _label2path(self, label):

        # remove last item after split and replace it with its first few chars
        # (but then add it back) to reduce the number of files in each directory,
        # note that we'll typically be splitting a string like '/c/en/term' but
        # this is implemented in a way that's also intended to work if the string
        # contains some other number (e.g. 0) of slashes
        splitted = label.split('/')
        term = splitted[-1]
        splitted = splitted[:-1]  # remove last

        for i in range(OOCFrame.N_CHAR_DIRS):
            ch_i = term[i] if len(term) > i else '_'
            # handle labels like '/c/en/a.a', '/c/en/1,4_naphthoquinone' (yes,
            # lots of terms have commas), and others with weird unicode chars
            if OOCFrame.NON_WORD_CHAR_RE.match(ch_i):
                ch_i = '_'
            splitted.append(ch_i)

        splitted.append(term)  # append last
        return os.path.join(self._path, *splitted)

    def _filepath2label(self, fpath):
        splitted = fpath.split('/')
        language = splitted[-OOCFrame.N_CHAR_DIRS - 2]
        term = '.'.join(splitted[-1].split('.')[:-1])  # remove .npy extension
        return standardized_uri(language, term)

    def insert(self, label, data, weight=1.0):
        """Insert a term label along with its data vector into the OOCFrame.  This
        method writes the data vector to disk and records the label in a
        """
        if self._fileroots2ids is None:
            raise ValueError('Insert has been disabled for this OOCFrame; if the intent is to overwrite it, then delete its directory first')
        self._index.append(label)
        fpathroot = self._label2path(label)
        fname = '{}.npy'.format(fpathroot)

        # insert a differentiating id into the filename if not unique
        if os.path.isfile(fname):  #if fname in self._filenames2weights faster?
            self._fileroots2ids[fpathroot] += 1
            fid = self._fileroots2ids[fpathroot]
            fname = '{}.{}.npy'.format(fpathroot, fid)

        if os.path.isfile(fname):
            raise ValueError('File already exists: {}'.format(fname))
        self._filenames2weights[fname] = weight

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as fp:
            np.save(fp, data, allow_pickle=False)

    def combine_weights(self):
        """Effectively does the same thing as `standardize_row_labels`, which groups
        identical terms into a single, weighted average vector.
        """
        if not self._filenames2weights:
            raise ValueError('`_filenames2weights` is empty; `combine_weights` should only be called once after inserting all data')

        # for each file root with multiple files
        for fpathroot, maxid in self._fileroots2ids.items():

            weightsum = 0.0

            for fid in range(maxid + 1):
                fname = '{}{}.npy'.format(fpathroot, '.{}'.format(fid) if fid else '')
                weight = self._filenames2weights[fname]
                weightsum += weight
                data = np.load(fname)
                if fid == 0: weighted_avg  = data * weight
                else:        weighted_avg += data * weight
                os.remove(fname)

            # save a new weighted average file with no file id
            weighted_avg /= weightsum
            fname = '{}.npy'.format(fpathroot)
            np.save(fname, weighted_avg, allow_pickle=False)

        self.disable_insert()

    def disable_insert(self):
        """Either all inserts have been completed and `combine_weights` has been
        called or this OOCFrame already existed on disk at time of construction.
        """
        # setting these to None will cause an exception in `insert` if attempted
        self._fileroots2ids = None
        self._filenames2weights = None

        # construct an Index, just like a normal DataFrame
        self._index = pd.Index(self._index)

    @property
    def index(self):
        return self._index

    def l1_normalize_columns(self):
        col_norms = None
        for label in self.index:
            fname = '{}.npy'.format(self._label2path(label))
            if col_norms is None: col_norms  = np.load(fname)
            else:                 col_norms += np.load(fname)
        for label in self.index:
            fname = '{}.npy'.format(self._label2path(label))
            normalized = np.load(fname) / col_norms
            np.save(fname, normalized, allow_pickle=False)  # overwrite

    def l2_normalize_rows(self):
        for label in self.index:
            fname = '{}.npy'.format(self._label2path(label))
            data = np.load(fname)
            row_norm = np.sqrt(np.sum(np.power(data, 2)))
            normalized = data / row_norm
            np.save(fname, normalized, allow_pickle=False)  # overwrite