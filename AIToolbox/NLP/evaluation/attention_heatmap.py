import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import seaborn as sns

from AIToolbox.experiment_save.core_metrics.abstract_metric import AbstractBaseMetric


class AttentionHeatMap(AbstractBaseMetric):
    def __init__(self, attention_matrices, source_sentences, target_sentences, plot_save_dir):
        """

        Args:
            attention_matrices (numpy.array or list):
            source_sentences (list):
            target_sentences (list):
            plot_save_dir (str):
        """
        if len(attention_matrices) != len(source_sentences) != len(target_sentences):
            raise ValueError(f'Lengths of attention_matrices, source_sentences and target_sentences are not the same. '
                             f'Their lengths are: {len(attention_matrices)}, {len(source_sentences)}, {len(target_sentences)}')

        self.attention_matrices = attention_matrices
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.plot_save_dir = plot_save_dir
        AbstractBaseMetric.__init__(self, None, None, metric_name='Attention_HeatMap', np_array=False)

    def calculate_metric(self):
        dir_path = self.prepare_folder_for_saving(self.plot_save_dir)
        output_plot_paths = []

        for i, (attn_matrix, source_sent, target_sent) in enumerate(zip(self.attention_matrices, self.source_sentences, self.target_sentences)):
            plot_file_path = os.path.join(dir_path, f'attn_plot_{i}.png')
            output_plot_paths.append(plot_file_path)

            self.plot_sentence_attention(attn_matrix, source_sent, target_sent, plot_file_path)

        self.metric_result = output_plot_paths

    @staticmethod
    def plot_sentence_attention(attention_matrix, sentence_source, sentence_target, plot_file_path=None):
        """

        Args:
            attention_matrix (np.array):
            sentence_source (list):
            sentence_target (list):
            plot_file_path (str):

        Returns:

        """
        # alpha_arr /= np.max(np.abs(alpha_arr),axis=0)
        fig = plt.figure()
        fig.set_size_inches(8, 8)

        gs = gridspec.GridSpec(2, 2, width_ratios=[12, 1], height_ratios=[12, 1])

        ax = plt.subplot(gs[0])
        ax_c = plt.subplot(gs[1])

        cmap = sns.light_palette((200, 75, 60), input="husl", as_cmap=True)
        # prop = FontProperties(fname='fonts/IPAfont00303/ipam.ttf', size=12)
        ax = sns.heatmap(attention_matrix, xticklabels=sentence_source, yticklabels=sentence_target, 
                         ax=ax, cmap=cmap, cbar_ax=ax_c)

        ax.xaxis.tick_top()
        ax.yaxis.tick_right()

        ax.set_xticklabels(sentence_target, minor=True, rotation=60, size=12)
        
        for label in ax.get_xticklabels(minor=False):
            label.set_fontsize(12)
            # label.set_font_properties(prop)

        for label in ax.get_yticklabels(minor=False):
            label.set_fontsize(12)
            label.set_rotation(-90)
            label.set_horizontalalignment('left')

        ax.set_xlabel("Source", size=20)
        ax.set_ylabel("Hypothesis", size=20)

        if plot_file_path:
            fig.savefig(plot_file_path, format="png")

    @staticmethod
    def prepare_folder_for_saving(output_plot_dir):
        """

        Args:
            output_plot_dir (str):

        Returns:
            str:
        """
        if os.path.exists(output_plot_dir):
            shutil.rmtree(output_plot_dir)

        os.mkdir(output_plot_dir)
        dir_path = os.path.join(output_plot_dir, 'attention_heatmaps')
        os.mkdir(dir_path)
        return dir_path
