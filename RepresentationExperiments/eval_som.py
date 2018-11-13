from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_csv_visual_100classes, from_csv, to_csv

audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '100classes',
                               'audio100classes.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '100classes',
                                'VisualInputTrainingSet.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a SOM.')
    parser.add_argument('path', type=str, help='The path to the trained SOM model')
    parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Random generator seed')
    parser.add_argument('--data', metavar='data', type=str, default='audio')
    parser.add_argument('--neurons1', type=int, default=50,
                        help='Number of neurons for SOM, first dimension')
    parser.add_argument('--neurons2', type=int, default=50,
                        help='Number of neurons for SOM, second dimension')
    parser.add_argument('--subsample', action='store_true', default=False)
    parser.add_argument('--rotation', action='store_true', default=False)

    if args.data == 'audio':
        xs, ys, _ = from_csv_with_filenames(audio_data_path)
    elif args.data == 'video':
        xs, ys = from_csv_visual_100classes(visual_data_path)
    else:
        raise ValueError('--data argument not recognized')

    som = SOM(args.neurons1, args.neurons2, dim, n_iterations=args.epochs, alpha=args.alpha,
                 tau=0.1, threshold=0.6, batch_size=args.batch, data=args.data, sigma=args.sigma)

    ys = np.array(ys)
    xs = np.array(xs)

    if args.subsample:
        xs, _, ys, _ = train_test_split(xs, ys, test_size=0.6, stratify=ys, random_state=args.seed)

    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, stratify=ys,
                                                            random_state=args.seed)

    xs_train, xs_val, ys_train, ys_val = train_test_split(xs, ys, test_size=0.5, stratify=ys,
                                                            random_state=args.seed)

    xs_train, xs_test = transform_data(xs_train, xs_val, rotation=args.rotation)

    som.restore_trained()

    print('Computing compactness...')
    compactness = som.class_compactness(xs_val, ys_val)
    print('Compactness: {}'.format(compactness))
    print('Computing neuron collapse...')
    collapse_ratio_examples, collapse_ratio_neurons = neuron_collapse(xs_val, ys_val)
    print('Computing class-wise neuron collapse...')
    classwise_ratio_examples, classwise_ratio_neurons = neuron_collapse_classwise(xs_val, ys_val)
