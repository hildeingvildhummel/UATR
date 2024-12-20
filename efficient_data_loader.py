"""https://music-classification.github.io/tutorial/part3_supervised/tutorial.html"""
from torch.utils import data
from torch.utils.data import Dataset

from Augmentation_functions_torch import *
from random import choice
import math
import soundfile as sf
from scipy import signal
import numpy as np
import torch

class Augmentation_functions(object):
    def __init__(self, sample_rate, recording, start_time, parentdir):
        self.sample_rate = sample_rate
        self.recording = recording
        self.start_time = start_time
        self.parentdir = parentdir

    def _no_augmentation(self, wav):
        return wav

    def __call__(self, audio):
        # fns = [PolarityInversion(), Noise(min_snr=0.3, max_snr=0.5), Gain(),
        #        PitchShift(n_samples=num_samples, sample_rate=self.sample_rate), Reverb(sample_rate=self.sample_rate),
        #        self._no_augmentation, MixUp(parentdir=self.parentdir, recording_len=300, sample_len=int(audio.shape[1]/self.sample_rate), samplerate=self.sample_rate, sigma=50, recording=self.recording, start_time=self.start_time),
        #                LowPassFilter(sample_rate=self.sample_rate, freq_low=10, freq_high=1000)]

        fns = [Noise(min_snr=0.3, max_snr=0.5), MixUp(parentdir=self.parentdir, recording_len=300, sample_len=int(audio.shape[1]/self.sample_rate), samplerate=self.sample_rate, sigma=50, recording=self.recording, start_time=self.start_time),
               LowPassFilter(sample_rate=self.sample_rate, freq_low=10, freq_high=1000),
               self._no_augmentation]

        augmentation = choice(fns)
        augmented_audio = augmentation(audio)
        return augmented_audio


class ReadAudioSections():

    def __init__(self, sample_len, samplerate):
        self.sample_len = sample_len
        self.sample_rate = samplerate
    def _read_audio_section(self, filename, start_time, stop_time, augmentation=False):
        track = sf.SoundFile(filename)

        can_seek = track.seekable()  # True
        if not can_seek:
            raise ValueError("Not compatible with seeking")

        sr = track.samplerate
        sample_len = stop_time - start_time
        start_frame = int(sr * start_time)

        frames_to_read = int(sr * (sample_len))

        track.seek(start_frame)
        audio_section = track.read(frames_to_read)
        if augmentation:
            new_time = choice([i for i in range(0, int(track.frames/sr) - sample_len) if i not in range(start_time,
                                                                                           start_time + sample_len)])
            track.seek(new_time)
            audio_section_aug = track.read(frames_to_read)
            return audio_section, sr, audio_section_aug
        return audio_section, sr

    def _reduce_sr(self, audio, sr, new_sr, sample_size):
        if sr == new_sr:
            return audio
        else:
            resampled = signal.resample(audio, int(new_sr * sample_size))
            return resampled

    def extract_as_clip(self, input_filename, start_time, stop_time, sample_size = None):
        if sample_size is None:
            sample_size = self.sample_len
        audio_extract, sr = self._read_audio_section(input_filename, start_time, stop_time)
        audio_extract = self._reduce_sr(audio_extract, sr, self.sample_rate, sample_size)
        audio_extract = torch.from_numpy(np.array(audio_extract, dtype='float32')).unsqueeze(0)
        return audio_extract, sr

class DeepShipLoader():
    def __init__(self, sample_len):
        self.sample_len = sample_len

    def _extract_duration_audio(self, input_filename):
        track = sf.SoundFile(input_filename)
        num_frames = track.frames
        sr = track.samplerate
        return num_frames / (self.sample_len * sr)

    def extract_label_and_duration(self, filename):
        labels = {}
        for root, dirs, files in os.walk(filename):
            for input_filename in files:
                if input_filename.endswith('.wav'):
                    subdir = root.split('/')[-1]
                    time_ext = self._extract_duration_audio(os.path.join(root, input_filename))
                    duration = math.floor(time_ext)

                    labels[os.path.join(root, input_filename)] = [duration, subdir]
        return labels

    def return_recording_index(self, label_dictionary):
        values = np.array(list(label_dictionary.values()))
        durations = values[:, 0].astype(int)
        index_list = []
        for idx, x in enumerate(durations):
            if idx == 0:
                index_list.append(0)
                continue
            previous_value = index_list[idx - 1]
            new_value = durations[idx - 1] + previous_value
            index_list.append(new_value)
        return index_list

    def get_total_duration(self, label_dictionary):
        values = np.array(list(label_dictionary.values()))
        total_duration = sum(values[:, 0].astype(int))
        return total_duration


class EfficientDataSet(Dataset):
    def __init__(self, parent_dir, recording_len, sample_len, samplerate, sigma, augmentation=True, labeled_data=False, return_recording=False):
        super().__init__()
        self.parentdir = parent_dir
        self.recording_len = recording_len
        self.sample_len = sample_len
        self.samplerate = samplerate
        self.augmentation = augmentation
        self.labeled_data = labeled_data
        # self.Label_translation = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        # #  self.Label_translation = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        self.Label_translation = self._get_Label_translation()
        if self.labeled_data:
            self.index_functions = self._labeled_index_to_start_time
            self.labeled_fuctions = DeepShipLoader(self.sample_len)
            self.label_dict = self.labeled_fuctions.extract_label_and_duration(self.parentdir)
        else:
            self.index_functions = self._index_to_start_time
        self.excluded_index = np.array([], dtype=int)
        self.counter = 0
        self.sigma = sigma
        self.return_recording = return_recording

    def _get_Label_translation(self):
        label_dict = {}
        list_of_dirs = [f.name for f in os.scandir(self.parentdir) if f.is_dir()]
        counter = 0
        for class_name in list_of_dirs:
            label_dict[class_name] = counter
            counter += 1
        return label_dict

    def _extract_wav_files(self):
        list_of_dirs = self._get_subdirs(self.parentdir)
        wav_files = []
        for directory in list_of_dirs:
            wav_files.extend([os.path.join(directory, fi) for fi in os.listdir(directory) if fi.endswith(".wav")])
            wav_files.extend([os.path.join(directory, fi) for fi in os.listdir(directory) if fi.endswith(".flac")])
        return wav_files

    def _get_subdirs(self, directory):
        fu = [x[0] for x in os.walk(directory)]
        return fu

    def _index_to_start_time(self, index):
        recordings = self._extract_wav_files()
        num_samples = self.recording_len / self.sample_len
        recording_index = np.arange(len(recordings)) * math.floor(num_samples)
        recording_index = recording_index.astype(int)
        index_of_interest = [i for i, e in enumerate(recording_index) if e <= index][-1]
        recording_of_interest = recordings[index_of_interest]

        residual = index - recording_index[index_of_interest]
        start_time = residual * self.sample_len

        return recording_of_interest, int(start_time)

    def _labeled_index_to_start_time(self, index):
        label_dict = self.label_dict
        recordings = list(label_dict.keys())

        recording_index = self.labeled_fuctions.return_recording_index(label_dict)
        index_of_interest = [i for i, e in enumerate(recording_index) if e <= index][-1]
        recording_of_interest = recordings[index_of_interest]
        previous_num_samples = 0
        for i in range(index_of_interest):
            previous_recording = recordings[i]
            previous_num_samples += label_dict[previous_recording][0]

        start_time = (index - previous_num_samples) * self.sample_len

        return recording_of_interest, int(start_time)



    def _len_function(self):
        recordings = self._extract_wav_files()
        num_wavs = len(recordings)
        durations = math.floor(self.recording_len / self.sample_len) * num_wavs
        return durations

    def _labeled_len_function(self):
        label_dict = self.label_dict
        durations = self.labeled_fuctions.get_total_duration(label_dict)
        if self.augmentation:
            durations = 3*durations
        return durations

    def _SimCLR_Augmentation(self,audio, recording, start_time, parentdir):
        augment_func = Augmentation_functions(self.samplerate, recording, start_time, parentdir)
        positive_pair = augment_func(audio)
        augment_func = Augmentation_functions(self.samplerate, recording, start_time, parentdir)
        audio = augment_func(audio)
        return audio, positive_pair


    def __getitem__(self, item):
        if self.augmentation and self.labeled_data:
            item /= 3

        recording, start_time = self.index_functions(item)
        read_audio = ReadAudioSections(self.sample_len, self.samplerate)

        audio, sr = read_audio.extract_as_clip(recording, start_time, start_time + self.sample_len)

        if not self.labeled_data:
            anchor, positive_pair = self._SimCLR_Augmentation(audio, recording, start_time, self.parentdir)

            if self.return_recording:
                sub_rec = recording.split('/')
                wav = sub_rec[-1]
                type = sub_rec[-2]
                recording = type + '/' + wav
                return anchor, positive_pair, recording, audio
            else:
                return anchor, positive_pair, 0, audio

        else:
            label_dict = self.label_dict
            label = self.Label_translation[label_dict[recording][1]]
            if self.augmentation:
                spec_masked, positive_pair = self._SimCLR_Augmentation(audio, recording, start_time, self.parentdir)
            else:
                spec_masked = audio
                positive_pair = audio
                if self.return_recording:
                    audio = recording
                else:
                    positive_pair = audio
            return spec_masked, positive_pair, label, audio

    def __len__(self):
        if self.labeled_data:
            return self._labeled_len_function()
        else:
            return self._len_function()
def get_dataloader(recording_path, recording_len_sec=300,
                   sample_len_sec=2, sample_rate=16000, batch_size=32, augmentation=True, labeled_data=False, sigma=5, shuffle=True,return_recording=False):
    data_loader = data.DataLoader(dataset=EfficientDataSet(parent_dir=recording_path, recording_len=recording_len_sec,
                                                              sample_len=sample_len_sec,
                                                              samplerate=sample_rate, augmentation=augmentation, labeled_data=labeled_data,
                                                              sigma= sigma, return_recording=return_recording),
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      drop_last=False,
                                      num_workers=6)
    return data_loader