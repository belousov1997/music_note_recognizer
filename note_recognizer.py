# Standard imports
import wave

import librosa
# Imports für Darstellung eines Audio-Signals
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.playback import play


# Import für Filterung eines Audio-Signals


def play_file():
    wav_file = AudioSegment.from_file(file="redemption_song.wav", format="wav")
    # play(wav_file)
    # TODO Herumspielen mit Highpassfilter
    filtered_version = wav_file.high_pass_filter(50, order=4)
    # TODO Herumspielen mit Lowpassfilter
    # filtered_version = wav_file.low_pass_filter(80, order=1)

    # TODO exportieren der Datei in eine neue Datei
    filtered_version.export("./2.wav", format='wav')

    # TODO nochmal gefiltert
    filtered_version = AudioSegment.from_file(file="2.wav", format="wav")
    filtered_version2 = filtered_version.low_pass_filter(14000, order=1)
    play(filtered_version2)


def read_audio_original(file) -> None:
    """
    This function should read the file's basic informations and content such as the Audio-Signal curve.
    Output gets plotted with matplotlib
    :return:
    """
    # Audio-Datei in ein Numpy Array konvertieren
    audio_data = np.array(file.get_array_of_samples())
    # print("length of audiodata in np.array", len(audio_data))

    # Entnehme sample-rate und Anzahl der Kanäle der Audio-Datei
    sample_rate = file.frame_rate
    num_channels = file.channels

    # Errechne die Dauer der Audio-Datei in Sekunden
    # len(audio_data) = Gesamtanzahl der Samples
    # sample_rate = Abtastrate, wird beim digitalisieren festgelegt und ist im Sample immer gleich (Samples pro Sekunde)
    # num_channels = Anzahl an Kanäle
    # Die Berechnung pro Sekunde ergibt sich aus Gesamtsample / (Anzahl an Samples pro Sek * Anzahl der Kanäle)
    duration = len(audio_data) / float(sample_rate * num_channels)

    # Zeitachse erstellen, dabei 0=startpunkt, duration=endpunkt, und len(audio_data) ist die Anzahl der gesamten Samples
    time_axis = np.linspace(0, duration, len(audio_data))

    # Plotten des Audio-Signals
    # Amplitude ist die Intensität eines Signals! Je höher die Amplitude, desto Stärker das Signal wird wahrgenommen
    plt.plot(time_axis, audio_data)
    plt.xlabel("Zeit (sek)")
    plt.ylabel("Amplitude")
    plt.show()


def librosa_vs_original_samplesize() -> None:
    """
    Compares the Versions of Librosas loaded audio_file with a set sample_rate
    and the orginal sample_rate from the meta data
    The Original sample_rate of 48000 kHz (48 Hz) is loaded by setting sr=None in the parameters
    The Audio-files are compared in a plot
    The Default sample_rate is set to 22050 kHz (22.050 Hz)
    :return:
    """

    # Default
    audio_data_with_sr, sample_rate1 = librosa.load("redemption_song.wav")
    # sd.play(audio_data_with_sr, sample_rate1)
    duration = librosa.get_duration(y=audio_data_with_sr, sr=sample_rate1)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Abtastrate: {sample_rate1} Samples in seconds")
    print(f"Length of file_without_sr: {len(audio_data_with_sr)} Samples")

    # Original
    audio_data_without_sr, sample_rate2 = librosa.load("redemption_song.wav", sr=None)
    # sd.play(audio_data_without_sr, sample_rate2)
    duration2 = librosa.get_duration(y=audio_data_without_sr, sr=sample_rate2)
    print(f"Duration: {duration2:.2f} seconds")
    print(f"Abtastrate: {sample_rate2} Samples in seconds")
    print(f"Length of file_with_sr: {len(audio_data_without_sr)} Samples")

    # Erstelle Zeitachsen
    time_with_sr = np.arange(0, len(audio_data_with_sr)) / sample_rate1
    time_without_sr = np.arange(0, len(audio_data_without_sr)) / sample_rate2

    # Zwei Plots gezeichnet
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(time_with_sr, audio_data_with_sr)
    axs[0].set_title('With SR')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('Time (seconds)')
    axs[1].plot(time_without_sr, audio_data_without_sr)
    axs[1].set_title('Without SR')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Amplitude')

    plt.show()


def frequency_analysis() -> None:
    """
    This function should analyse the frequency of the audio_file using Fast Fourier Transform as a spectrum analyzer.
    The FFT calculates the amplitude of each frequency component in the signal and displays the results as a graph.
    The frequencyrange is determined by the lowest and highest frequency present in the signal
    :return:
    """
    # Wichtigsten Daten der audio_Datei anzeigen:
    audio_data, sample_rate = librosa.load("redemption_song.wav")
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Length of file: {len(audio_data)} Samples")

    # Kontrolle wieviele Kanäle es gibt in der Audiofile
    with wave.open('redemption_song.wav', 'r') as wav_file:
        num_channels = wav_file.getnchannels()
        print('Number of channels:', num_channels)

    # Erstellen eines Zeitvektors
    # Bei num wird die Anzahl an Samples angegeben. Hier kann man entweder die Gesamtanzahl an Samples errechnen,
    # indem man Dauer * Anzahl an Samples pro Sekunde multipliziert
    # Oder man nimmt direkt len(audio_data) als Samplewert
    time_vector = np.linspace(start=0, stop=duration, num=len(audio_data), endpoint=False)
    print(len(time_vector))

    # TODO Fast Fourier Transformation (Siehe Glossar)


def filter_noise() -> None:
    """
    This function should filter the noise of an inputfile according its frequency using a bandpassfilter.
    1. First determine frequency range of the audio signal
    2. Choose the limits (low and upper)
    3. Implementing the filters using scipy.signal.butter to design a filter and cisp.signal.filtfilt to apply
    the filter to an audio-signal.
    NOTED: .wav file is required
    :return:
    """


def edit_file(inputpath) -> None:
    """

    :param inputpath:
    :return:
    """
    # Filter lower frequencies, um noise zu reduzieren über 80hz
    # audio_file = audio_file.high_pass_filter(80, order=4)
    # segments_ms = 50
    # volume = [segment.dBFS for segment in audio_file[::segments_ms]]

    # print("size", volume.__sizeof__())
    # print(volume)

    # Array mit den ungefähren Timestamps für jede gespielte Note im Song
    # TODO Dies muss eigentlich automatisch passieren.
    # actual_notes = \
    #  [2.6, 3.5, 4.1, 4.8, 5.8, 6.6, 7.4, 8.2, 9.1, 10.2,
    #  10.6, 11.3, 11.6, 12.1, 12.5, 12.9, 13.6]
    # x_axis = np.arange(len(volume)) * (segments_ms / 1000)
    # for s in actual_notes:
    #   plt.axvline(x=s, color='r', linewidth=0.5, linestyle="-")
    # länge volume array wird sortiert und mal der Segmentierung /1000 genommen. Ergibt 0,05 Schritte
    # print(x_axis)  # Die X-Achse vorbereitet für volume
    # plt.plot(x_axis, volume)
    # plt.show()

    # print(volume)


# filtered_version2.export("./3.wav",format="wav")
# filtered_version2 = AudioSegment.from_file(file="3.wav", format="wav")
# play(filtered_version)
# play(filtered_version2)
if __name__ == "__main__":
    # TODO Die behandelte Datei
    audio_file = AudioSegment.from_file(file="redemption_song.wav")
    read_audio_original(audio_file)
    librosa_vs_original_samplesize()
    frequency_analysis()
