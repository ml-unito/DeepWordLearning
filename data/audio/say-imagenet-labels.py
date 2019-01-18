import logging, os, sys, json, subprocess

def tokenize_label_words(imagenet_words_string):
    '''Each ImageNet label has one or more words that are associated with it. Each
       one should be used and say-d. This function tokenizes a string representing
       all words that refer to an ImageNet label.'''
    return imagenet_words_string.split(',')

if __name__ == '__main__':
    speakers = ['tom', 'karen', 'lee', 'moira', 'tessa', 'daniel', 'kate', 'oliver', 'serena', 'kathy', 'samantha', 'vicki', 'victoria']
    args = sys.argv[1:]
    if len(args) > 3:
        print('Usage: python3 say-imagenet-labels [imagenet-label-file] [dataset-output-path] [rate]')
        sys.exit(1)
    if len(args) == 3:
        rate = args[2]
    else:
        rate = 'default'
    with open(args[0]) as imagenet_labels_file:
        labels_dict = json.load(imagenet_labels_file)

    basepath = args[1]

    for speaker in speakers:
        print('Processing {}'.format(speaker))
        speaker_dir = os.path.join(basepath, speaker)
        os.makedirs(speaker_dir)
        for key, value in labels_dict.items():
            # Create a new directory for the labels audio files, if needed
            target_dir = os.path.join(speaker_dir, str(key))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            word_list = tokenize_label_words(value)
            for i, word in enumerate(word_list):
                if rate != 'default':
                    subprocess.run(['say', '-v', str(speaker), word, '-o', os.path.join(target_dir, str(i) + '.aiff'), '-r', rate])
                else:
                    subprocess.run(['say', '-v', str(speaker), word, '-o', os.path.join(target_dir, str(i) + '.aiff')])
                i += 1
