"""Download the pinned CSV dataset and verify both published checksums."""
import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

REVISION = '9f0d8a65a42a87eb0944a67fecc03934bace2d20'
URL = f'https://raw.githubusercontent.com/fabriciojoc/brazilian-malware-dataset/{REVISION}/brazilian-malware.zip'
ZIP_SHA256 = '657136309532868b78b646b14716013d5c36681f5d4bd008536523c1912ea7b7'
CSV_SHA256 = 'd13e54cf1970ffedbf1043196c135da3187c90a82e397f681d843ed320249242'


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {path.name}')


def fetch(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv = output / 'brazilian-malware.csv'
    if csv.exists():
        verify(csv, CSV_SHA256)
        print(f'Existing dataset verified: {csv}')
        return
    archive = output / 'brazilian-malware.zip'
    # Temporary files remain on the target filesystem and disappear on failure.
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary = Path(temporary)
        downloaded = temporary / archive.name
        if archive.exists():
            verify(archive, ZIP_SHA256)
            source = archive
        else:
            with urlopen(URL, timeout=120) as response, downloaded.open('wb') as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(chunk)
            verify(downloaded, ZIP_SHA256)
            source = downloaded
        extracted = temporary / csv.name
        with ZipFile(source) as bundle:
            # Read only the named CSV; never extract arbitrary archive paths.
            with bundle.open(csv.name) as incoming, extracted.open('wb') as stream:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                    stream.write(chunk)
        verify(extracted, CSV_SHA256)
        if source == downloaded:
            downloaded.replace(archive)
        extracted.replace(csv)
    print(f'Dataset downloaded and verified: {csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/raw')
    fetch(parser.parse_args().output)
