import argparse
import linecache
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directories', nargs='+',
                        default=['../../data/model_evaluations/metrics_history'],
                        help='Directories with model evaluations.')
    args = parser.parse_args()
    input_directories = args.input_directories

    # plot metric values over epochs (model training)
    # input_directory should here be something like 'metric_history'
    val_mae = read_eval(Path(args.input_directories[0], 'val_mae.txt'))
    train_mae = read_eval(Path(args.input_directories[0], 'train_mae.txt'))
    plot_metric_over_epochs(val_mae, train_mae, metric='MAE')

    # compare metric values over epochs for (multiple) model(s)
    compare_model_histories(input_directories)


def compare_model_histories(input_directories):
    model_eval_folders = input_directories
    model_evals = []
    train_mae_histories = []
    val_mae_histories = []
    for eval_folder in model_eval_folders:
        model_eval = {}
        model_eval['model_name'] = str(eval_folder)

        p = str(Path(eval_folder, 'eval_test.txt'))
        model_eval['test_mae'] = float(linecache.getline(p, 2))

        p = str(Path(eval_folder, 'eval_train.txt'))
        model_eval['train_mae'] = float(linecache.getline(p, 2))

        p = Path(eval_folder, 'val_mae.txt')
        model_eval['val_mae_history'] = []
        with open(p) as txt_file:
            for line in txt_file:
                model_eval['val_mae_history'].append(float(line))
        val_mae_histories.append(model_eval['val_mae_history'])

        p = Path(eval_folder, 'train_mae.txt')
        model_eval['train_mae_history'] = []
        with open(p) as txt_file:
            for line in txt_file:
                model_eval['train_mae_history'].append(float(line))
        train_mae_histories.append(model_eval['train_mae_history'])

        model_evals.append(model_eval)

    for i in range(len(val_mae_histories)):
        plt.plot(val_mae_histories[i], label=model_evals[i]['model_name'])
    plt.legend()
    plt.xlim(0, 30)
    plt.xlabel('epoch')
    plt.ylabel('MAE validation data')
    plt.show()

    for i in range(len(train_mae_histories)):
        plt.plot(train_mae_histories[i], label=model_evals[i]['model_name'])
    plt.legend()
    plt.xlim(0, 30)
    plt.xlabel('epoch')
    plt.ylabel('MAE train data')
    plt.show()

    model_evals = pd.DataFrame(model_evals)
    print(model_evals[['model_name', 'test_mae', 'train_mae']])
    model_evals.to_csv('model_comparison.csv')


def plot_metric_over_epochs(metric_values_val,
                            metric_values_train,
                            metric='MAE',
                            label_size=21, ticks_size=17):
    plt.plot(metric_values_val, label='validation')
    plt.plot(metric_values_train, label='train')
    plt.xlabel('Epochs', fontsize=label_size)
    plt.ylabel(metric, fontsize=label_size)
    plt.legend()
    plt.title(metric + ' of Model')
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.show()


def read_eval(file_path):
    diffs = []
    with open(file_path, 'r') as txt_file:
        for line in txt_file:
            diffs.append(float(line))
    return diffs


if __name__ == '__main__':
    main()
