# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Parser for the ESC-10 and ESC-50 data sets"""
import os
from pathlib import Path
from typing import Optional, Mapping, Sequence

from pandas import read_csv

from audeep.backend.data.data_set import Split
from audeep.backend.log import LoggingMixin
from audeep.backend.parsers.base import Parser, _InstanceMetadata


class KoeParser(LoggingMixin, Parser):
    """
    Parser for Koe-based dataset.
    All audio files are stored in one folder, named after their IDs.
    A tsv file named 'metadata.tsv' must exist containing the mapping from ID to label and folds
    """

    def __init__(self, basedir: Path):
        """
        Creates and initializes a new KoeParser for the specified data set base directory.

        Parameters
        ----------
        basedir: pathlib.Path
            The data set base directory
        """
        super(KoeParser, self).__init__(basedir)
        self._data = None
        self._metadata = None
        self._label_map_cache = None
        self._can_parse_cache = None
        self._numfolds = None

    def can_parse(self) -> bool:
        """
        Check that all files are .wav and their ids exist in the tsv file

        Returns
        -------
        bool
            True, if this parser can parse the directory structure in the data set base directory
        """
        if self._can_parse_cache is None:
            files = sorted([file for file in self._basedir.glob("*")
                                  if file.is_file() and file.name.lower().endswith(".wav")], key=lambda x: x.name)
            file_names = [x.name for x in files]
            meta_file = self._basedir.joinpath('metadata.tsv')

            if not os.path.isfile(meta_file):
                self.log.debug('cannot parse: metadata.tsv doesn\'t exist')
                self._can_parse_cache = False
                return False

            self._metadata = read_csv(meta_file, sep='\t')
            header = list(self._metadata.columns)
            if header != ['id', 'filename', 'label', 'label_enum', 'fold']:
                self.log.debug('cannot parse: invalid header in metadata.tsv')
                self._can_parse_cache = False
                return False

            metadata_file_names = list(self._metadata['filename'])
            if set(file_names) != set(metadata_file_names):
                # for ind, (x, y) in enumerate(zip(file_names, metadata_file_names)):
                #     if x != y:
                #         print('At {} {} != {}'.format(ind, x, y))

                self.log.debug('cannot parse: mismatch between wav files and IDs stored in metadata.tsv')
                self._can_parse_cache = False
                return False

            self._data = []
            self._label_map_cache = {}
            self._numfolds = 0
            # for i in range(len(metadata_file_names)):
            for sid, filename, label, label_enum, fold in self._metadata.values:
                self._data.append((sid, filename, label, label_enum, fold))

                if label not in self._label_map_cache:
                    self._label_map_cache[label] = label_enum

                if fold > self._numfolds:
                    self._numfolds = fold

            self._numfolds += 1

            self._can_parse_cache = True

            return True
        else:
            return self._can_parse_cache

    @property
    def num_instances(self) -> int:
        """
        Returns the number of instances in the data set.

        Returns
        -------
        int
            The number of instances in the data set

        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse Koe dataset at {}".format(self._basedir))

        return len(self._data)

    @property
    def num_folds(self) -> int:
        """
        Returns the number of cross-validation folds, which is 10 for this parser.

        Returns
        -------
        int
            Five

        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse Koe dataset at {}".format(self._basedir))

        return self._numfolds

    @property
    def label_map(self) -> Optional[Mapping[str, int]]:
        """
        Returns the mapping of nominal to numeric labels for this data set.

        Nominal labels are assigned integer indices in order of the three digits prepended to the class directories.
        For example, the first three class directories of the ESC-10 data set are '001 - Dog bark', '002 - Rain', and
        '003 - Sea waves', which would result in the label map {'Dog bark': 0, 'Rain': 1, 'Sea waves': 2}.

        Returns
        -------
        map of str to int
            The mapping of nominal to numeric labels for this data set

        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse Koe dataset at {}".format(self._basedir))

        return self._label_map_cache

    def parse(self) -> Sequence[_InstanceMetadata]:
        """
        Parses the instances contained in this data set.

        For each instance, metadata is computed and stored in an _InstanceMetadata object. Instances are parsed in the
        order of the class directories, and in alphabetical order within class directories.

        Returns
        -------
        list of _InstanceMetadata
            A list of _InstanceMetadata containing one entry for each parsed audio file

        Raises
        ------
        IOError
            If the data set cannot be parsed
        """
        if not self.can_parse():
            raise IOError("unable to parse Koe dataset at {}".format(self._basedir))

        meta_list = []

        for sid, filename, label, label_enum, fold in self._data:
            filepath = Path(os.path.join(self._basedir, filename))
            cv_folds = [Split.TRAIN] * self._numfolds
            cv_folds[fold] = Split.VALID

            instance_metadata = _InstanceMetadata(path=filepath,
                                                  filename=str(filename),
                                                  label_nominal=label,
                                                  label_numeric=label_enum,
                                                  cv_folds=cv_folds,
                                                  partition=None)
            meta_list.append(instance_metadata)

        return meta_list
