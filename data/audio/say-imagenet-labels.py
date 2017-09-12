import logging, os, sys, json, subprocess

def tokenize_label_words(imagenet_words_string):
    '''Each ImageNet label has one or more words that are associated with it. Each
       one should be used and say-d. This function tokenizes a string representing
       all words that refer to an ImageNet label.'''
    return imagenet_words_string.split(',')

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 3:
        print('Usage: python3 say-imagenet-labels [imagenet-label-file] [dataset-output-path] [optional: speaker name]')
        sys.exit(1)
    with open(args[0]) as imagenet_labels_file:
        labels_dict = json.load(imagenet_labels_file)

    basepath = args[1]
    if len(args)> 2:
        speaker = args[2]
    else:
        speaker = 'tom'
    print('Chosen speaker: '+ speaker)

    for key, value in labels_dict.items():
        # Create a new directory for the labels audio files, if needed
        target_dir = os.path.join(basepath, str(key))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        word_list = tokenize_label_words(value)
        i = 0
        for word in word_list:
            subprocess.run(['say', '-v', str(speaker), word, '-o', os.path.join(target_dir, str(i) + '.aiff'), '-r', str(130)])
            i += 1


