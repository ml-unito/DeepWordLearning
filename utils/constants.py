import os, logging

class Constants():
    # this actually refers to the python working dir
    # so you have to take care to run scripts from the
    # root folder - which is DeepWordLearning/
    ROOT_FOLDER = os.path.abspath('.')
    DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
    AUDIO_DATA_FOLDER = os.path.join(DATA_FOLDER, 'audio')
    TIMIT_DATA_FOLDER = os.path.join(DATA_FOLDER, 'timit')
    AVAILABLE_SPEAKERS = ['tom', 'allison', 'daniel', 'ava', 'lee', 'susan', 'tom-130', 'allison-130', 'daniel-130', \
                          'ava-130', 'lee-130', 'susan-130']
    LOGGING_LEVEL = logging.DEBUG

