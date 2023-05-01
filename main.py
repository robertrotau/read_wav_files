import json
import pathlib
from pathlib import Path

import numpy as np
import numpy.typing as npt


def read_meta_file(file_path: str | pathlib.Path) -> dict[str, dict]:
    """This functions reads the .txt file in the data-directory and turns it into a python dictionary.

    Args:
        file_path (str | pathlib.Path): path to data-directory

    Returns:
        dict[str, dict]: Python Dictionary to loop through in later functions/methods
    """
    meta_data = list(Path(file_path).glob("*.txt"))
    with Path.open(meta_data[0]) as text:
        return json.load(text)


def read_wavfile(file_path: str | pathlib.Path) -> "WAVFile":
    """This functions reads the .wav file using numpy.fromfile function. Creates attributes for the "WAVFile" object.
    For more information on numpy.fromfiles go to https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html

    Args:
        file_path (str | pathlib.Path): string specifying a path or pathlib.Path to .wav file

    Returns:
        WAVFile: returns an object of class "WAVFile"
    """
    with Path.open(Path(file_path), "rb") as file:
        n_channels = int(np.fromfile(file, dtype="uint16", count=1, offset=22))
        fs = int(np.fromfile(file, dtype="uint32", count=1))
        bits_per_sample = int(np.fromfile(file, dtype="uint16", count=1, offset=6))

        if bits_per_sample == 16:
            data = np.fromfile(file, dtype=np.int16, count=-1, offset=8)

        if bits_per_sample == 32:
            data = np.fromfile(file, dtype=np.int32, count=-1, offset=8)

        if bits_per_sample == 64:
            data = np.fromfile(file, dtype=np.int64, count=-1, offset=8)

        duration = float(data.size / fs)

        return WAVFile(
            bits_per_sample, duration, Path(file_path).name, fs, n_channels, data
        )


class WAVFile:
    def __init__(
        self,
        bits_per_sample: int,
        duration: float,
        file_name: str,
        fs: int,
        n_channels: int,
        data: npt.NDArray[np.int_],
    ):
        self.bits_per_sample = bits_per_sample
        self.duration = duration
        self.file_name = file_name
        self.fs = fs
        self.n_channels = n_channels
        self.data = data


def read_recording(file_path: str | pathlib.Path) -> "Recording":
    """This function creates a "Recording" object by reading out information from the file name (see Args).
    Creates attributes for the "Recording" object.

    Args:
        file_path (str | pathlib.Path): string specifying a path or pathlib.Path to .wav file

    Returns:
        Recording: returns an object of class "Recording"
    """
    digit = int(Path(file_path).name[0])
    speaker = str(Path(file_path).name[2:4])
    index = int(Path(file_path).name[5 : Path(file_path).name.find(".")])

    return Recording(digit, index, speaker, read_wavfile(file_path))


class Recording:
    def __init__(self, digit: int, index: int, speaker: str, wav: WAVFile):
        self.digit = digit
        self.index = index
        self.speaker = speaker
        self.wav = wav


def make_corpus(dataset_path: str | pathlib.Path) -> "Corpus":
    """This function creates an empty object from class "Corpus".
    Inside this object, we can add different speakers/recordings by using methods from class "Corpus".

    Args:
        dataset_path (str | pathlib.Path): string specifying a path or pathlib.Path to .wav file

    Returns:
        Corpus: returns an EMPTY object of class "Corpus"
    """
    recordings: list[Recording] = []
    speakers: set[str] = set()
    dataset_path = Path(dataset_path)

    return Corpus(dataset_path, recordings, speakers)


class Corpus:
    def __init__(
        self,
        dataset_path: pathlib.Path,
        recordings: list[Recording],
        speakers: set[str],
    ):
        self.dataset_path = dataset_path
        self.recordings = recordings
        self.speakers = speakers

    def add_speakers(self, *speakers: str) -> "Corpus":
        """Add speaker(s) to a new object of class "Corpus" by using Python Set union() method.
        Before return, all recordings from all speakers in new_corpus are added to the object.

        Returns:
            Corpus: object of class "Corpus"
        """
        new_corpus = make_corpus(self.dataset_path)

        new_corpus.speakers = self.speakers.union(speakers)

        for speaker in new_corpus.speakers:
            pathlist = Path(self.dataset_path, speaker).glob("*.wav")
            for path in pathlist:
                recording = read_recording(path)
                new_corpus.recordings.append(recording)

        return Corpus(
            new_corpus.dataset_path, new_corpus.recordings, new_corpus.speakers
        )

    def add_accent(self, accent: str) -> "Corpus":
        """Add speaker(s) to a new object of class "Corpus" with a unique accent specified in argument "accent: str".
        The method loops through the dictionary and searches for the accent and adds the speaker alias to a new set.
        The speaker aliases are added to the "speakers" attribute.
        Before return, all recordings from all speakers in new_corpus are added to the object.

        Returns:
            Corpus: object of class "Corpus"
        """
        new_corpus = make_corpus(self.dataset_path)

        meta_dict = read_meta_file(self.dataset_path)

        new_speakers = set()
        for speaker_alias in meta_dict.keys():
            speakers_accent = list(meta_dict[speaker_alias].values())[0]
            speakers_accent = speakers_accent.split("/")
            speakers_accent_cap = [accents.capitalize() for accents in speakers_accent]
            accent_cap = accent.capitalize()
            if accent_cap in speakers_accent_cap:
                new_speakers.add(speaker_alias)

        new_corpus.speakers = self.speakers.union(new_speakers)

        for speaker in new_corpus.speakers:
            pathlist = Path(self.dataset_path, speaker).glob("*.wav")
            for path in pathlist:
                recording = read_recording(path)
                new_corpus.recordings.append(recording)

        return Corpus(
            new_corpus.dataset_path, new_corpus.recordings, new_corpus.speakers
        )

    def add_gender(self, gender: str) -> "Corpus":
        """Add speaker(s) to a new object of class "Corpus" with a gender specified in argument "gender: str".
        The method loops through the dictionary and searches for the gender and adds the speaker alias to a new set.
        The speaker aliases are added to the "speakers" attribute.
        Before return, all recordings from all speakers in new_corpus are added to the object.

        Returns:
            Corpus: object of class "Corpus"
        """
        new_corpus = make_corpus(self.dataset_path)

        meta_dict = read_meta_file(self.dataset_path)

        new_speakers = set()
        for speaker_alias in meta_dict.keys():
            speakers_gender = list(meta_dict[speaker_alias].values())[2]
            if speakers_gender == gender:
                new_speakers.add(speaker_alias)

        new_corpus.speakers = self.speakers.union(new_speakers)

        for speaker in new_corpus.speakers:
            pathlist = Path(self.dataset_path, speaker).glob("*.wav")
            for path in pathlist:
                recording = read_recording(path)
                new_corpus.recordings.append(recording)

        return Corpus(
            new_corpus.dataset_path, new_corpus.recordings, new_corpus.speakers
        )

    def __len__(self) -> int:
        """
        Returns:
            int: returns how many recordings are inside an object of class "Corpus"
        """
        return len(self.speakers)

    def __and__(self, other: "Corpus") -> "Corpus":
        """This method creates a new instance of class "Corpus" containing the intersection of two objects from type "Corpus".

        Returns:
            Corpus: object of class "Corpus"
        """
        new_corpus = make_corpus(self.dataset_path)
        if type(other) != Corpus:
            return NotImplemented

        new_corpus.speakers = self.speakers & other.speakers

        for speaker in new_corpus.speakers:
            pathlist = Path(self.dataset_path, speaker).glob("*.wav")
            for path in pathlist:
                recording = read_recording(path)
                new_corpus.recordings.append(recording)

        return Corpus(
            new_corpus.dataset_path, new_corpus.recordings, new_corpus.speakers
        )
