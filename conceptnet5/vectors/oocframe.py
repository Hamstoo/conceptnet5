import os
import numpy as np
from glob import glob
from collections import defaultdict


class OOCFrame(object):
    """Out-of-core pandas.DataFrame-like object.  The index is in-core, but the data
    are stored in files on disk.
    TODO: not sure if this should derive from DataFrame or not.
    """
    def __init__(self, path):
        self._path = path
        if os.path.exists(self._path):
            raise ValueError('Path already exists: {}'.format(self._path))

        # these members support the `combine_weights` method
        self._fileroots2ids = defaultdict(lambda: 0)
        self._filenames2weights = {}

    def insert(self, label, data, weight=1.0):
        """Insert a term label along with its data vector into the OOCFrame.  This
        method writes the data vector to disk and records the label in a
        """
        # remove leading '/' which would o/w cause os.path.join to ignore self.path
        froot = '{}'.format(label[1:])
        fpathroot = os.path.join(self._path, froot)
        fname = '{}.npy'.format(fpathroot)

        # insert a differentiating id into the filename if not unique
        if os.path.isfile(fname):  #if fname in self._filenames2weights faster?
            self._fileroots2ids[froot] += 1
            fid = self._fileroots2ids[froot]
            fname = '{}.{}.npy'.format(fpathroot, fid)

        if os.path.isfile(fname):
            raise ValueError('File already exists: {}'.format(fname))
        self._filenames2weights[fname] = weight  # TODO: no need to store full path fname here

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
        for froot, maxid in self._fileroots2ids.items():
            fpathroot = os.path.join(self._path, froot)

            weightsum = 0.0

            for fid in range(maxid + 1):
                fname = '{}{}.npy'.format(fpathroot, '.{}'.format(fid) if fid else '')
                weight = self._filenames2weights[fname]
                weightsum += weight
                data = np.load(fname)
                if fid == 0: agg  = data * weight
                else:        agg += data * weight
                os.remove(fname)

            # save a new weighted average file with no file id
            agg /= weightsum
            fname = '{}.npy'.format(fpathroot)
            with open(fname, 'wb') as fp:
                np.save(fp, agg, allow_pickle=False)

        self._fileroots2ids = None
        self._filenames2weights = None

    def l1_normalize_columns(self):
        fnames = [y for x in os.walk(self._path) for y in glob(os.path.join(x[0], '*.npy'))]
        col_norms = None
        for fname in fnames:
            if col_norms is None: col_norms  = np.load(fname)
            else:                 col_norms += np.load(fname)
        for fname in fnames:
            normalized = np.load(fname) / col_norms
            np.save(fname, normalized, allow_pickle=False)  # overwrite

    def l2_normalize_rows(self):
        fnames = [y for x in os.walk(self._path) for y in glob(os.path.join(x[0], '*.npy'))]
        for fname in fnames:
            data = np.load(fname)
            row_norm = np.sqrt(np.sum(np.power(data, 2)))
            normalized = data / row_norm
            np.save(fname, normalized, allow_pickle=False)  # overwrite