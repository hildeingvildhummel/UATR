import torch
import random
from torchaudio.transforms import Vol
from julius.filters import highpass_filter, lowpass_filter
import numpy as np
import augment
from scipy import signal
import datetime
import os
import math
import soundfile as sf


class TNCSampler():
    """This class selects another snippet of the same recording of a specified length. This sample is selected
    by a normal distribution. The normal distribution is layed over time, with the center-time of the original
    recording as mu and a manually defined sigma. This function is inspired by the original paper about TNC
    sampling:
    Tonekaboni, Sana, Danny Eytan, and Anna Goldenberg.
    "Unsupervised representation learning for time series with temporal neighborhood coding."
    arXiv preprint arXiv:2106.00750 (2021).

    Input:
        - parentdir: The root directory containing the recordings
        - recording_len: The duration of the selected recording (sec)
        - sample_len: The duration of the windowed sample (2 secs in paper)
        - sigma: The sigma of the normal distribution
    """

    def __init__(self, parentdir, recording_len, sample_len, samplerate, sigma):
        self.parentdir = parentdir
        self.recording_len = recording_len
        self.sample_len = sample_len
        self.samplerate = samplerate
        self.sigma = sigma

    def _get_subdirs(self, directory):
        """Extracts all subdirectories in a list
        Input:
            - directory: The path to the directory of interest
        Output:
            - dirs: List of subdirectories"""
        dirs = [x[0] for x in os.walk(directory)]
        return dirs

    def _extract_wav_files(self):
        """Lists all wav and flac files within the parentdir
        Output:
            - wav_files: List of paths to all individual wav and flac files"""
        list_of_dirs = self._get_subdirs(self.parentdir)
        wav_files = []
        for directory in list_of_dirs:
            wav_files.extend([os.path.join(directory, fi) for fi in os.listdir(directory) if fi.endswith(".wav")])
            wav_files.extend([os.path.join(directory, fi) for fi in os.listdir(directory) if fi.endswith(".flac")])
        return wav_files

    def _from_filename_to_datetimeObject(self, filename):
        """Converts a filename to a dateTime object
        Input:
            - filename: The file of interest
        Output:
            - timeObj: The dateTime object corresponding to the selected file"""
        timeStr = filename.split('_')[-1][:-4].split('.')[0]
        timeObj = datetime.datetime.strptime(timeStr, '%Y%m%dT%H%M%S')
        return timeObj

    def _from_datetimeObject_to_filename(self, datetimeObj):
        """Converts a dateTime object to the corresponding wav file name. This function returns None
        if it cannot find the file.
        Input:
            Input:
                - datetimeObj: The dateTime object of interest
            Output:
                - wav_file_of_interest[0]: The corresponding wav file name"""
        wav_files = self._extract_wav_files()
        datetimeStr = datetime.datetime.strftime(datetimeObj, '%Y%m%dT%H%M%S')
        wav_file_of_interest = [s for s in wav_files if datetimeStr in s]
        if len(wav_file_of_interest) == 0:
            return None
        else:
            return wav_file_of_interest[0]

    def _define_previous_recording(self, recording):
        """Extracts the file name of the recording in the past connected to the selected recording.
        Input:
            - recording: The filename of interest
        Output:
            - filename: The connected wav recording file name"""
        time_Obj = self._from_filename_to_datetimeObject(recording)
        previous_timeObj = time_Obj - datetime.timedelta(seconds=self.recording_len)
        filename = self._from_datetimeObject_to_filename(previous_timeObj)
        return filename

    def _define_next_recording(self, recording):
        """Extracts the file name of the recording in the future connected to the selected recording.
        Input:
            - recording: The filename of interest
        Output:
            - filename: The connected wav recording file name"""
        time_Obj = self._from_filename_to_datetimeObject(recording)
        previous_timeObj = time_Obj + datetime.timedelta(seconds=self.recording_len)
        filename = self._from_datetimeObject_to_filename(previous_timeObj)
        return filename

    def _get_recording_and_startTime(self, mu, sigma, recording):
        sample_start_time = np.random.normal(mu, sigma, 1)
        if sample_start_time < 0:
            recording_of_interest = self._define_previous_recording(recording)
            sample_start_time = self.recording_len + sample_start_time
        elif sample_start_time > self.recording_len:
            recording_of_interest = self._define_next_recording(recording)
            sample_start_time = sample_start_time - self.recording_len
        else:
            recording_of_interest = recording
        return math.floor(sample_start_time[0]), recording_of_interest

    def _extract_time_range_data(self):
        time_stamps = []
        for root, sub, files  in os.walk(self.parentdir):
            for file in files:
                time_stamps.append(self._from_filename_to_datetimeObject(os.path.join(self.parentdir, file)))
        youngest = min(time_stamps)
        oldest = max(time_stamps)
        return youngest, oldest

    def pick_random_time(self, start_time, recording):
        import scipy.stats as stats
        mu = start_time + (self.sample_len / 2)
        sigma = self.sigma
        recording_of_interest = None
        while recording_of_interest is None:
            try:
                sample_start_time, recording_of_interest = self._get_recording_and_startTime(mu, sigma, recording)
            except:
                recording_of_interest = recording
                lower = 0
                upper = self.recording_len
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                sample_start_time = int(X.rvs())
        return recording_of_interest, sample_start_time

class MixUp(torch.nn.Module):
    def __init__(self, parentdir, recording_len, sample_len, samplerate, sigma, recording, start_time):
        super().__init__()
        self.sample_len = sample_len
        self.sample_rate = samplerate
        self.sampler = TNCSampler(parentdir=parentdir, recording_len=recording_len,
                                     sample_len=sample_len, samplerate=samplerate, sigma=sigma)
        self.recording = recording
        self.start_time = start_time
        self.HighPass = HighPassFilter(freq_low=1000, freq_high=8000, sample_rate=self.sample_rate)
        self.LowPass = LowPassFilter(freq_low=0, freq_high=1000, sample_rate=self.sample_rate)


    def _read_audio_section(self, filename, start_time, stop_time):
        track = sf.SoundFile(filename)

        can_seek = track.seekable()  # True
        if not can_seek:
            raise ValueError("Not compatible with seeking")

        sr = track.samplerate
        start_frame = int(sr * start_time)

        frames_to_read = int(sr * (stop_time - start_time))

        track.seek(start_frame)
        audio_section = track.read(frames_to_read)
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

    def forward(self, audio):
        positive_recording, positive_start_time = self.sampler.pick_random_time(self.start_time, self.recording)
        positive_pair, sr = self.extract_as_clip(positive_recording, positive_start_time,
                                                       positive_start_time + self.sample_len)
        orig = self.LowPass(audio)
        high_mix = self.HighPass(positive_pair)

        combined = orig + high_mix

        return combined

"""Speech-related augmentation functions originate from: https://github.com/Spijkervet/torchaudio-augmentations/tree/master
Go to this Github Page for more information."""

class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio

class FrequencyFilter(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float,
        freq_high: float,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high

    def cutoff_frequency(self, frequency: float) -> float:
        return frequency / self.sample_rate

    def sample_uniform_frequency(self):
        return random.uniform(self.freq_low, self.freq_high)


class HighPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 200,
        freq_high: float = 1200,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        audio = highpass_filter(audio, cutoff=cutoff)
        return audio


class LowPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 2200,
        freq_high: float = 4000,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        audio = lowpass_filter(audio, cutoff=cutoff)
        return audio


class HighLowPass(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        lowpass_freq_low: float = 2200,
        lowpass_freq_high: float = 4000,
        highpass_freq_low: float = 200,
        highpass_freq_high: float = 1200,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        self.high_pass_filter = HighPassFilter(
            sample_rate, highpass_freq_low, highpass_freq_high
        )
        self.low_pass_filter = LowPassFilter(
            sample_rate, lowpass_freq_low, lowpass_freq_high
        )

    def forward(self, audio):
        highlowband = random.randint(0, 1)
        if highlowband == 0:
            audio = self.high_pass_filter(audio)
        else:
            audio = self.low_pass_filter(audio)
        return audio


class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

        return audio + noise


class PitchShift(torch.nn.Module):
    def __init__(
        self, n_samples, sample_rate, pitch_shift_min=-7.0, pitch_shift_max=7.0
    ):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.pitch_shift_cents_min = int(pitch_shift_min * 100)
        self.pitch_shift_cents_max = int(pitch_shift_max * 100)
        self.src_info = {"rate": self.sample_rate}

    def process(self, x):
        n_steps = random.randint(self.pitch_shift_cents_min, self.pitch_shift_cents_max)
        effect_chain = augment.EffectChain().pitch(n_steps).rate(self.sample_rate)
        num_channels = x.shape[0]
        target_info = {
            "channels": num_channels,
            "length": self.n_samples,
            "rate": self.sample_rate,
        }
        y = effect_chain.apply(x, src_info=self.src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        if y.shape[1] != x.shape[1]:
            if y.shape[1] > x.shape[1]:
                y = y[:, : x.shape[1]]
            else:
                y0 = torch.zeros(num_channels, x.shape[1]).to(y.device)
                y0[:, : y.shape[1]] = y
                y = y0
        return y

    def __call__(self, audio):
        if audio.ndim == 3:
            for b in range(audio.shape[0]):
                audio[b] = self.process(audio[b])
            return audio
        else:
            return self.process(audio)




class PolarityInversion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio):
        audio = torch.neg(audio)
        return audio




class Reverb(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        reverberance_min=0,
        reverberance_max=100,
        dumping_factor_min=0,
        dumping_factor_max=100,
        room_size_min=0,
        room_size_max=100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max
        self.src_info = {"rate": self.sample_rate}
        self.target_info = {
            "channels": 1,
            "rate": self.sample_rate,
        }

    def forward(self, audio):
        reverberance = torch.randint(
            self.reverberance_min, self.reverberance_max, size=(1,)
        ).item()
        dumping_factor = torch.randint(
            self.dumping_factor_min, self.dumping_factor_max, size=(1,)
        ).item()
        room_size = torch.randint(
            self.room_size_min, self.room_size_max, size=(1,)
        ).item()

        num_channels = audio.shape[0]
        effect_chain = (
            augment.EffectChain()
            .reverb(reverberance, dumping_factor, room_size)
            .channels(num_channels)
        )

        audio = effect_chain.apply(
            audio, src_info=self.src_info, target_info=self.target_info
        )

        return audio




