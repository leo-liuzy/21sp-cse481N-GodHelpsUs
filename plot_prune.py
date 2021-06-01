import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json

eval_file = "eval_results_None.json"
trained_lang = "{}"
# tested_lang = "ja"
# basline, metric = 0.4748, "mae"
metric = "mae"


NUM_LAYER = 12
NUM_HEADS = 16

tested_langs = ["en", "zh", "de", "es", "fr", "ja"]


def read_baseline(model_output_dir, tested_lang, metric):
    eval_file = f"{model_output_dir}/eval_results_{tested_lang}.json"
    with open(eval_file, "r") as f:
        eval_dict = json.load(f)
        assert f"eval_{metric}" in eval_dict
        return eval_dict[f"eval_{metric}"]


def read_result(eval_output_dir, baseline, metric):
    results = np.zeros((NUM_LAYER, NUM_HEADS * 3))
    for layer in range(2 * NUM_LAYER):
        heads = range(2 * NUM_HEADS) if layer > NUM_LAYER - 1 else range(NUM_HEADS)
        i = layer % NUM_LAYER

        for head in heads:
            multiplier = 0
            if layer >= NUM_LAYER:
                # if we are in decoder, [0...NUM_HEADS) is self_attn, [NUM_HEADS...2 * NUM_HEADS) is cross_attn
                multiplier += 1
                if head < NUM_HEADS:
                    multiplier += 1

            j = head % NUM_HEADS + NUM_HEADS * multiplier
            head_eval_file = f"{eval_output_dir}/layer{layer}_head{head}/{eval_file}"

            with open(head_eval_file, "r") as f:
                eval_dict = json.load(f)
                results[i, j] = (eval_dict[f"eval_{metric}"] - baseline) * 100
    return results


def heatmap(eval_output_dir, plot_output_dir, plot_title, baseline, metric):
    # the output layout is:
    # [encoder_attn_results, cross_attn_results, decoder_attn_results]
    results = read_result(eval_output_dir, baseline, metric)
    fig, ax = plt.subplots(figsize=(15, 5))
    yticks = [i + 1 for i in range(NUM_LAYER)]
    yticklabels = [str(i + 1) for i in range(NUM_LAYER)]
    xtick_labels = [str(i + 1) for i in range(NUM_HEADS)] * 3
    plt.title(plot_title)
    ax = sns.heatmap(results,
                     linewidth=0.5, square=True, yticklabels=yticklabels, xticklabels=xtick_labels)
    # ax.annotate("", xy=(0, 0), arrowprops=dict(arrowstyle='<->', facecolor='red'), annotation_clip=False)
    # ax.set_yticks([str(i + 1) for i in range(NUM_LAYER)])
    ax.invert_yaxis()
    plt.savefig(f"{plot_output_dir}/{plot_output_title}.jpg")
    plt.show()


def layer_wise_print(heads_data):
    overall_means = []
    overall_std = []
    overall_max = []
    overall_min = []

    for layer_idx in range(NUM_LAYER - 1, -1, -1):
        layer_data = heads_data[layer_idx]
        layer_mean = np.mean(layer_data)
        layer_std = np.std(layer_data)
        layer_max = np.max(layer_data)
        layer_min = np.min(layer_data)
        print(f"{layer_idx + 1}th layer ---\t mean: {layer_mean} "
              f"\t std: {layer_std}\t max: {layer_max}"
              f"\t min: {layer_min}")
        overall_means.append(layer_mean)
        overall_std.append(layer_std)
        overall_max.append(layer_max)
        overall_min.append(layer_min)

    print(f"Average(layer) ---\t mean: {np.mean(overall_means)}, \t std: {np.mean(overall_std)}"
          f"\t max: {np.mean(overall_max)} \t min: {np.mean(overall_min)}")
    print()


def subspace_wise_print(heads_data):
    overall_means = []
    overall_std = []
    overall_max = []
    overall_min = []

    for subspace_idx in range(NUM_HEADS):
        subspace_data = heads_data[:, subspace_idx]
        subspace_mean = np.mean(subspace_data)
        subspace_std = np.std(subspace_data)
        subspace_max = np.max(subspace_data)
        subspace_min = np.min(subspace_data)
        print(f"{subspace_idx + 1}th subspace ---\t mean: {subspace_mean} "
              f"\t std: {subspace_std}\t max: {subspace_max}"
              f"\t min: {subspace_min}")
        overall_means.append(subspace_mean)
        overall_std.append(subspace_std)
        overall_max.append(subspace_max)
        overall_min.append(subspace_min)

    print(f"Average(subspace) ---\t mean: {np.mean(overall_means)}, \t std: {np.mean(overall_std)}"
          f"\t max: {np.mean(overall_max)} \t min: {np.mean(overall_min)}")
    print()


def cross_lingual_helpfulness_analysis(multi_lang_results):
    helfulness_dict = {"helpful": multi_lang_results > 0,
                       "harmful": multi_lang_results < 0,
                       "neutral": multi_lang_results == 0}

    for helfulness, helfulness_matrix in helfulness_dict.items():
        print("*" * 10 + f" Cross-lingual statistics for masking to be {helfulness} " + "*" * 10)
        multi_lang_sum = np.sum(helfulness_matrix, axis=0)
        some_helpful_masking = (multi_lang_sum > 0)
        all_helpful_masking = (multi_lang_sum == len(tested_langs))

        print(f"{helfulness} heads (at least in one lang) percentage: {np.mean(some_helpful_masking) * 100} "
              f"({np.sum(some_helpful_masking)})")
        print(f"{helfulness} heads (in all lang) percentage: {np.mean(all_helpful_masking) * 100} "
              f"({np.sum(all_helpful_masking)})")
        # print(all_helpful_masking.astype(int))
        structure_dict = {"encoder_heads": multi_lang_sum[:, :NUM_HEADS],
                          "cross_heads": multi_lang_sum[:, NUM_HEADS:NUM_HEADS * 2],
                          "decoder_heads": multi_lang_sum[:, NUM_HEADS * 2:]}
        for structure_name, heads_data_by_structure in structure_dict.items():
            print(f"{structure_name}: ")
            layer_wise_print(heads_data_by_structure)
            subspace_wise_print(heads_data_by_structure)
            # print()
        print("*" * 50)



if __name__ == "__main__":
    multi_lang_results = []
    for tested_lang in tested_langs:
        model_output_dir = f"marc_{trained_lang}_mbart"
        eval_output_dir = os.path.join(model_output_dir, tested_lang)
        plot_output_dir = "plot_output"
        if not os.path.exists(plot_output_dir):
            os.makedirs(plot_output_dir)
        plot_output_title = f"source{trained_lang.upper()}_test{tested_lang.upper()}_metric{metric.upper()[:3]}"
        baseline = read_baseline(model_output_dir, tested_lang, metric)
        mono_lang_results = read_result(eval_output_dir, baseline, metric)
        multi_lang_results.append(mono_lang_results)
        # heatmap(eval_output_dir, plot_output_dir, plot_output_title, baseline, metric)
    multi_lang_results = np.stack(multi_lang_results)
    cross_lingual_helpfulness_analysis(multi_lang_results)


    print()

