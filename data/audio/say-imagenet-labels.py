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
    if len(args)> 3:
        speaker = args[3]
    else:
        speaker = 'tom'
    print('Chosen speaker: '+ speaker)

    for key, value in labels_dict.items():
        if int(key) < 1000:
            continue
        # Create a new directory for the labels audio files, if needed
        target_dir = os.path.join(basepath, str(key))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        word_list = tokenize_label_words(value)
        i = 1000
        for word in word_list:
            if args[2] != 'default':
                subprocess.run(['say', '-v', str(speaker), word, '-o', os.path.join(target_dir, str(i) + '.aiff'), '-r', str(args[2])])
            else:
                subprocess.run(['say', '-v', str(speaker), word, '-o', os.path.join(target_dir, str(i) + '.aiff')])
            i += 1
