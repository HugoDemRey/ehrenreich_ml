from abc import ABC, abstractmethod
from src.audio_features.features import Chromagram, Feature
import src.libfmp, src.libfmp.c3, src.libfmp.b

import librosa
import numpy as np

class FeatureAligner(ABC):

    @abstractmethod
    def align(self, ref: Feature, query: Feature, output_type='time'):
        pass


class ChromagramAligner(FeatureAligner):

    def __init__(self, sigma = np.array([[2, 1], [1, 2], [1, 1]])):
        self.sigma = sigma

    def align(self, ref: Chromagram, query: Chromagram, output_type='time') -> tuple:
        """
        Aligns two chromagrams using Dynamic Time Warping (DTW) and returns the start and end of the optimal alignment path in the reference feature.
        Parameters:
            ref (Chromagram): The reference chromagram feature.
            query (Chromagram): The query chromagram feature.
            output_type (str): The type of output to return. Either 'time' for seconds (default) or 'frame' for frame indices.
        Returns:
            start_s (float): The start time of the optimal alignment path in seconds.
            end_s (float): The end time of the optimal alignment path in seconds.
        """


        C_FMP = src.libfmp.c3.compute_cost_matrix(query.data(), ref.data(), 'euclidean')
        D_librosa, P_librosa = librosa.sequence.dtw(C=C_FMP, subseq=True, backtrack=True, step_sizes_sigma=self.sigma)
        # cmap = src.libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
        # src.libfmp.c3.plot_matrix_with_points(D_librosa, P_librosa, cmap=cmap, 
        #                                 xlabel='Ehrenreich', ylabel=f'{i}th Naxos preview', 
        #                                 title="DTW Cost Matrix with Optimal Path (Librosa)",
        #                                 marker='o', linestyle='-')
        first_ref_frame_index = P_librosa[-1, 1]
        last_ref_frame_index = P_librosa[0, 1]

        if output_type == 'frame':
            return first_ref_frame_index, last_ref_frame_index

        start_s = first_ref_frame_index / ref.sampling_rate()
        end_s = last_ref_frame_index / ref.sampling_rate()

        return start_s, end_s
