from pydub import AudioSegment
from pydub.silence import split_on_silence
import sys, glob, os, logging
logging.basicConfig(level=logging.INFO)

def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def split_on_long_silences(audio):
    ''' Detect long silences. In our setting, a silence longer than 5 seconds
        means that the computer speaker was starting another ImageNet label
        entirely. '''
    #split track where silence is 2 seconds or more and get chunks
    chunks = split_on_silence(audio,
        min_silence_len=4900,

        # consider it silent if quieter than -16 dBFS
        #Adjust this per requirement
        silence_thresh=-32
    )
    return chunks

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 1:
        print('Usage: split-audio.py [audio-file] (just one please!)')
        sys.exit(1)
    audio_file = glob.glob(os.path.join('.', args[0]))
    if audio_file == []:
        print('Error: file not found!')
        sys.exit(1)

    audio_file = audio_file[0]
    logging.info('Loading input file...')
    audio = AudioSegment.from_file(audio_file)
    logging.info('Done.')
    logging.info('Splitting chunks on long silences...')
    chunks = split_on_long_silences(audio)
    logging.info(len(chunks) +' chunks found.')
    #Process each chunk per requirements
    for i, chunk in enumerate(chunks):
        ##Create 0.5 seconds silence chunk
        #silence_chunk = AudioSegment.silent(duration=500)

        ##Add 0.5 sec silence to beginning and end of audio chunk
        #audio_chunk = silence_chunk + chunk + silence_chunk

        #Normalize each audio chunk
        normalized_chunk = match_target_amplitude(chunk, -20.0)

        filename = './{0}/full-{0}.mp3'.format(i)
        #Export audio chunk with new bitrate
        logging.info("Exporting " + filename)
        #Create folder for new label audio, if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            normalized_chunk.export(filename, bitrate='192k', format="mp3")


