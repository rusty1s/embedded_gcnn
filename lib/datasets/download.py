from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
from six.moves import urllib


def maybe_download_and_extract(url, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    # Only download if file doesn't exist.
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading {} {:.2f}%'
                .format(filename, 100 * count * block_size / total_size))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        size = os.stat(filepath).st_size

        print()
        print('Successfully downloaded {} ({} bytes).'.format(filename, size))

    mode = 'r:gz' if filename.split('.')[-1] == 'gz' else 'r'
    archive = tarfile.open(filepath, mode)

    # Get the top level directory in the tar file.
    extracted_dir = os.path.join(data_dir,
                                 os.path.commonprefix(archive.getnames()))

    # Only extract if file doesn't exist.
    if not os.path.exists(extracted_dir):
        sys.stdout.write(
            '>> Extracting {} to {}...'.format(filename, extracted_dir))
        sys.stdout.flush()

        tarfile.open(filepath, mode).extractall(data_dir)

        print(' Done!')

    return extracted_dir
