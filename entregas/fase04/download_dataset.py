"""Download and verify the complete REHAB24-6 Zenodo record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

RECORD_ID = "13305826"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"
DOWNLOAD_BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files"
DATA_DIR = Path(__file__).parent / "data" / "raw" / "rehab24-6"
CHUNK_SIZE = 1024 * 1024


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_record_files() -> list[dict[str, object]]:
    request = Request(API_URL, headers={"User-Agent": "FIAP-Fase4-Dataset-Downloader/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Zenodo returned no files for the dataset record.")
    return entries


def download_file(entry: dict[str, object], destination: Path) -> None:
    filename = entry.get("key")
    checksum = entry.get("checksum")
    expected_size = entry.get("size")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid file name in Zenodo metadata: {filename!r}")
    if not isinstance(checksum, str) or not checksum.startswith("md5:"):
        raise ValueError(f"Missing MD5 checksum for {filename}.")
    if not isinstance(expected_size, int):
        raise ValueError(f"Missing file size for {filename}.")

    target = destination / filename
    expected_md5 = checksum.removeprefix("md5:")
    if target.is_file() and target.stat().st_size == expected_size:
        if file_md5(target) == expected_md5:
            print(f"Verified; skipping: {filename}")
            return

    temporary_file = target.with_name(f"{target.name}.part")
    url = f"{DOWNLOAD_BASE_URL}/{filename}?download=1"
    request = Request(url, headers={"User-Agent": "FIAP-Fase4-Dataset-Downloader/1.0"})
    digest = hashlib.md5()
    received_size = 0
    try:
        with urlopen(request, timeout=60) as response, temporary_file.open("wb") as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                received_size += len(chunk)
        if received_size != expected_size:
            raise RuntimeError(
                f"{filename}: expected {expected_size} bytes, received {received_size}."
            )
        if digest.hexdigest() != expected_md5:
            raise RuntimeError(f"{filename}: MD5 checksum does not match Zenodo.")
        temporary_file.replace(target)
    except (OSError, URLError, RuntimeError):
        temporary_file.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified: {filename} ({received_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        dest="filenames",
        help="download only this Zenodo filename; may be supplied more than once",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = fetch_record_files()
    if args.filenames:
        available = {entry.get("key"): entry for entry in entries}
        unknown = set(args.filenames) - available.keys()
        if unknown:
            parser.error(f"unknown Zenodo file(s): {', '.join(sorted(unknown))}")
        entries = [available[filename] for filename in args.filenames]

    for entry in entries:
        download_file(entry, DATA_DIR)
    print(f"Requested REHAB24-6 files are verified in {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
