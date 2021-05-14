import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json

eval_file = "eval_results_None.json"
trained_lang = "en"
tested_lang = "es"
# basline, metric = 0.4748, "mae"
basline, metric = 0.5922, "accuracy"

NUM_LAYER = 12
NUM_HEADS = 16


def heatmap(eval_output_dir, plot_output_dir, plot_title, baseline, metric):
    # the output layout is:
    # [encoder_attn_results, cross_attn_results, decoder_attn_results]
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



if __name__ == "__main__":
    model_output_dir = f"marc_{trained_lang}_mbart"
    eval_output_dir = os.path.join(model_output_dir, tested_lang)
    plot_output_dir = "plot_output"
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)
    plot_output_title = f"source{trained_lang.upper()}_test{tested_lang.upper()}_metric{metric.upper()[:3]}"
    heatmap(eval_output_dir, plot_output_dir, plot_output_title, basline, metric)
