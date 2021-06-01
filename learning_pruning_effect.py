import os
import json
import argparse
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from plot_prune import read_baseline
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error

LANGs = ["en", "zh", "de", "es", "fr", "ja"]
LANGs2ID = {lang: i for i, lang in enumerate(LANGs)}
NUM_LAYER = 12
NUM_HEADS = 16
NUM_HEAD_STRUCTURE = 3
IDX2STRUCTURE = {0: "encoder", 1: "cross", 2: "decoder"}

TARGET_CONFIGS = LANGs
OUTPUT_DIR_FORMAT = "marc_{}_mbart"


def create_dataset(args, source_configs, metric="accuracy", one_hot_feature=True):
    """
    Create the supervised-learning dataset of <features of configurations of pruned head> as X and
    <change in performance> as Y. All the pruning experiments are assumed to be done on
    "facebook/mbart-large-cc25"
    Currently, <features...> includes
    source langauges (as one-hot encoding): if we train model on English, this will be represented by
                                            , e.g., [1, 0, 0, 0, 0, 0]; this vector always has length 6
                                            because we have 6 languages in total. If we use
                                            different combintaions of langauges, corresponding positions
                                            will be 1.
    target langauges (as one-hot encoding or int): this is similar to source langauge features, except we only
                                            consider testing on mono-lingual corpus.
    position of heads (as one-hot encofing or int): the position consists of three parts and
                                            all of them were represented by one-hot encoding:
                                            <structural-wise location>, <layer-wise location>, <subspace-wise location>
                                            <structure-wise location>: is the head belongs to encoder-only,
                                                                        decoder-only, or cross (i.e. decoder uses to
                                                                        receive input from encoder)
                                            <layer-wise location>: which layer (regardless of encoder-stack
                                                                   or decoder-stack) is the head located. In
                                                                   mbart-large-cc25, this vector should have length 12.
                                            <subspace-wise location>: within one layer, which head is this (0-15)
    :param metric:
    :return: np.array
    """
    Xs, ys = [], []

    for source_config in tqdm(source_configs, desc="Reading sources"):
        model_output_dir = OUTPUT_DIR_FORMAT.format(source_config)
        if source_config == "{}":
            source_feature = np.ones(len(LANGs))
        else:
            source_lang_id = LANGs2ID[source_config]
            source_feature = np.eye(len(LANGs))[source_lang_id]

        for target_config in TARGET_CONFIGS:
            target_lang_id = LANGs2ID[target_config]
            target_feature = np.eye(len(LANGs))[target_lang_id] if one_hot_feature else [target_lang_id]

            baseline = read_baseline(model_output_dir, target_config, metric)
            eval_output_dir = os.path.join(model_output_dir, target_config)

            for layer in range(2 * NUM_LAYER):
                heads = range(2 * NUM_HEADS) if layer > NUM_LAYER - 1 else range(NUM_HEADS)
                i = layer % NUM_LAYER
                layer_feature = np.eye(NUM_LAYER)[i] if one_hot_feature else [i]

                for head in heads:
                    structure_idx = 0
                    if layer >= NUM_LAYER:
                        # if we are in decoder, [0...NUM_HEADS) is self_attn, [NUM_HEADS...2 * NUM_HEADS) is cross_attn
                        structure_idx += 1
                        if head < NUM_HEADS:
                            structure_idx += 1

                    subspace_idx = head % NUM_HEADS
                    subspace_feature = np.eye(NUM_HEADS)[subspace_idx] if one_hot_feature else [subspace_idx]
                    structure_feature = np.eye(NUM_HEAD_STRUCTURE)[structure_idx] if one_hot_feature else [structure_idx]

                    X = np.concatenate([source_feature,
                                        target_feature,
                                        structure_feature,
                                        layer_feature,
                                        subspace_feature])
                    if one_hot_feature:
                        assert len(X) == NUM_LAYER + NUM_HEADS + NUM_HEAD_STRUCTURE + len(LANGs) * 2
                    else:
                        assert len(X) == 1 + 1 + 1 + 1 + len(LANGs)
                    Xs.append(X)
                    # read change in performance
                    with open(f"{eval_output_dir}/layer{layer}_head{head}/{args.eval_file}", "r") as f:
                        eval_dict = json.load(f)
                        y = (eval_dict[f"eval_{metric}"] - baseline) * 100
                    ys.append(y)
    assert len(Xs) == len(ys)
    Xs = np.stack(Xs)
    ys = np.stack(ys)
    # shuffle the read dataset
    rand_idx = np.arange(len(Xs))
    np.random.shuffle(rand_idx)
    Xs = Xs[rand_idx]
    ys = ys[rand_idx]
    return Xs, ys


def plot_weight(feature_weights, feature_names, plot_path, plot_title):
    plt.figure(figsize=(10, 5))
    assert len(feature_weights) == len(feature_names)
    plt.bar(feature_names, feature_weights, align='center')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(rotation='vertical')
    plt.xlabel("Feature names")
    plt.ylabel("Feature coefficient")
    plt.title(plot_title)
    plt.savefig(plot_path)
    plt.show()


def linear_models(args, non_test: Tuple, test: Tuple, feature_names):
    Xs, ys = non_test
    Xtest, ytest = test

    # Linear regression
    regressor = LinearRegression().fit(Xs, ys)
    y_hat = regressor.predict(Xtest)
    test_score = np.mean(np.abs(ytest - y_hat))
    test_score_std = np.std(np.abs(ytest - y_hat))
    plot_dir = os.path.join(args.plot_source_dir, "linear_reg")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    linear_plot_path = os.path.join(plot_dir, f"weight_plot"
                                              f"{'_AllOneHot' if args.use_one_hot_feature else '_SourceOneHotOnly'}.jpg"
                                    )
    linear_plot_title = f"Linear Regression, test_MAE: {test_score :.3f} +/- {test_score_std :.3f}"

    plot_weight(regressor.coef_, feature_names, linear_plot_path, linear_plot_title)

    # Lasso regression: we use k-fold to choose the best
    lasso_alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    regressor = LassoCV(alphas=lasso_alphas,
                        cv=args.k_fold,).fit(Xs, ys)
    y_hat = regressor.predict(Xtest)
    test_score = np.mean(np.abs(ytest - y_hat))
    test_score_std = np.std(np.abs(ytest - y_hat))
    best_hyper = regressor.alpha_
    plot_dir = os.path.join(args.plot_source_dir, "lasso_reg")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    lasso_plot_path = os.path.join(plot_dir, f"alphas{lasso_alphas}_weight_plot"
                                             f"{'_AllOneHot' if args.use_one_hot_feature else '_SourceOneHotOnly'}.jpg"
                                   )
    lasso_plot_title = f"Lasso Regression, test_MAE: {test_score:.3f} +/- {test_score_std:.3f}, best_alpha: {best_hyper}"
    plot_weight(regressor.coef_, feature_names, lasso_plot_path, lasso_plot_title)

    # Ridge regression: we use k-fold to choose the best
    ridge_alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    regressor = RidgeCV(alphas=ridge_alphas,
                        cv=args.k_fold,).fit(Xs, ys)
    y_hat = regressor.predict(Xtest)
    test_score = np.mean(np.abs(ytest-y_hat))
    test_score_std = np.std(np.abs(ytest-y_hat))
    best_hyper = regressor.alpha_
    plot_dir = os.path.join(args.plot_source_dir, "ridge_reg")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    ridge_plot_path = os.path.join(plot_dir, f"alphas{ridge_alphas}_weight_plot"
                                             f"{'_AllOneHot' if args.use_one_hot_feature else '_SourceOneHotOnly'}.jpg"
                                   )
    ridge_plot_title = f"Ridge Regression, test_MAE: {test_score:.3f} +/- {test_score_std:.3f}, best_alpha: {best_hyper}"
    plot_weight(regressor.coef_, feature_names, ridge_plot_path, ridge_plot_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", help="Name of the evaluation file results to be read",
                        type=str, default="eval_results_None.json")
    parser.add_argument("--metric", help="Performance metric used to measure change",
                        type=str, default="mae")
    parser.add_argument("--plot_source_dir", help="Performance metric used to measure change",
                        type=str, default="plot_output")
    parser.add_argument("--k_fold", help="Propostion of training among all non-test examples",
                        type=float, default=5)
    parser.add_argument("--use_one_hot_feature", help="Whether use one hot feature for some of the features",
                        action="store_true")
    args = parser.parse_args()

    # we always use results from jointly trained model as the test set
    Xtest, ytest = create_dataset(args, source_configs=["{}"], metric=args.metric,
                                  one_hot_feature=args.use_one_hot_feature)
    Xs, ys = create_dataset(args, source_configs=LANGs, metric=args.metric,
                            one_hot_feature=args.use_one_hot_feature)
    feature_names = ["Source_" + lang for lang in LANGs]
    if args.use_one_hot_feature:
        feature_names += ["Target_" + lang for lang in LANGs] + \
                         [IDX2STRUCTURE[i] for i in range(NUM_HEAD_STRUCTURE)] + \
                         [f"Layer {str(i)}" for i in range(NUM_LAYER)] + \
                         [f"Head {str(i)}" for i in range(NUM_HEADS)]
    else:
        feature_names += ["target_lang_idx", "structure_idx", "layer_idx", "subspace_idx"]
    # n_train = int(len(Xs) * args.train_portion)
    # Xtrain, ytrain = Xs[:n_train], ys[:n_train]
    # Xval, yval = Xs[n_train:], ys[n_train:]
    linear_models(args, non_test=(Xs, ys), test=(Xtest, ytest), feature_names=feature_names)
