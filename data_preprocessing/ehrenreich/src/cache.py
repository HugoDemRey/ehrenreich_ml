import numpy as np

import hashlib
import pickle


class Cache:

    @staticmethod
    def request_resource(expr):
        """
        Evaluates the callable expr (e.g., a lambda or function), but first checks if the result is cached.
        The cache key is computed from the code and all referenced variable values (globals and closure).
        If cached, returns cached result. Otherwise, computes, caches, and returns result.

        Usage:
            Manager.request_resource(lambda: sum([1, 2]))
        Now also stores a string representation of the expression in the cache for easier navigation.
        """
        import inspect
        try:
            code = expr.__code__.co_code
            # Get closure variable values
            closure = tuple(cell.cell_contents for cell in expr.__closure__) if expr.__closure__ else ()
            closure_names = expr.__code__.co_freevars if expr.__closure__ else ()
            closure_dict = {name: value for name, value in zip(closure_names, closure)}

            # Get global variable values referenced by the function
            global_names = expr.__code__.co_names
            global_dict = {name: expr.__globals__[name] for name in global_names if name in expr.__globals__}

            # Try to get the source code for better expr_repr
            try:
                src = inspect.getsource(expr).strip()
            except Exception:
                src = repr(expr)

            # Compose key data from code, closure, and globals
            key_data = (code, tuple(sorted(closure_dict.items())), tuple(sorted(global_dict.items())))
            expr_repr = f"{src} | closure: {closure_dict} | globals: {global_dict}"
        except AttributeError:
            # Fallback for non-lambda callables
            key_data = repr(expr)
            expr_repr = repr(expr)
        key_bytes = pickle.dumps(key_data)
        cache_key = hashlib.md5(key_bytes).hexdigest()
        cache_path = f"data/cache/request_resource_{cache_key}.pkl"

        try:
            with open(cache_path, "rb") as f:
                print(f"🟢 \033[92mCache hit ({cache_key})\033[0m")
                cached_obj = pickle.load(f)
                # If the cache contains the new format, return only the result
                if isinstance(cached_obj, dict) and 'expr_repr' in cached_obj:
                    if cached_obj.get('result_type') == 'npy' and 'npy_path' in cached_obj:
                        return np.load(cached_obj['npy_path'], allow_pickle=True)
                    elif 'result' in cached_obj:
                        return cached_obj['result']
                # Backward compatibility
                return cached_obj
        except FileNotFoundError:
            print(f"🟠 \033[93mCache miss ({cache_key})\033[0m")
            result = expr()

            # If result is a large numpy array, save it separately to avoid MemoryError
            if isinstance(result, np.ndarray) and result.nbytes > 100 * 1024 * 1024:  # 100MB threshold
                npy_path = cache_path.replace('.pkl', '.npy')
                np.save(npy_path, result)
                cache_data = {'expr_repr': expr_repr, 'result_type': 'npy', 'npy_path': npy_path}
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                return result
            else:
                cache_data = {'expr_repr': expr_repr, 'result': result}
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                return result
                pickle.dump(cache_data, f)
            return result


    # @staticmethod
    # def plot_spectrogram(signal: Signal) -> None:
    #     Spectrogram.plot_spectrogram(signal)

    # @staticmethod
    # def compare_silences_to_ground_truth(signal: Signal) -> None:
    #     if (not CacheOld.resource_exists(signal.id, CacheToken.GROUND_TRUTH)):
    #         raise ValueError(f"Ground truth for signal {signal.id} not found in cache. Please add it manually before comparing.")
    #     with open(CacheOld.resource_path(signal.id, CacheToken.GROUND_TRUTH), 'r') as f:
    #         ground_truth = [float(line.strip()) for line in f if line.strip()]
    #     Spectrogram.compare_silences_to_ground_truth(signal, ground_truth)

    # @staticmethod
    # def get_stft(signal: Signal, force_cache=False) -> Tuple[np.ndarray, np.ndarray]:
    #     if not force_cache and CacheOld.resource_exists(signal.id, CacheToken.STFT):
    #         stft = np.load(CacheOld.resource_path(signal.id, CacheToken.STFT), allow_pickle=True)
    #         X, Y_db = stft['X'], stft['Y_db']
    #     else:
    #         X, Y_db = FeatureExtractor.compute_stft(signal)
    #         stft = {'X': X, 'Y_db': Y_db}
    #         np.savez(CacheOld.resource_path(signal.id, CacheToken.STFT), **stft)
    #     return X, Y_db
    
    # @staticmethod
    # def get_spectrogram_log_freq(signal: Signal, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    #     id = signal.id + f"_ws{window_size}"
    #     if CacheOld.resource_exists(id, CacheToken.SPECTROGRAM_LOG_FREQ):
    #         spectrogram = np.load(CacheOld.resource_path(id, CacheToken.SPECTROGRAM_LOG_FREQ), allow_pickle=True)
    #         return spectrogram['Spec_Features'], spectrogram['Pitch_Values']
    #     else:
    #         Spec_Features, Pitch_Values = FeatureExtractor.compute_spectrogram_log_freq(signal, window_size)
    #         spectrogram = {'Spec_Features': Spec_Features, 'Pitch_Values': Pitch_Values}
    #         np.savez(CacheOld.resource_path(id, CacheToken.SPECTROGRAM_LOG_FREQ), **spectrogram)
    #         return Spec_Features, Pitch_Values
        
    # @staticmethod
    # def get_chromagram(signal: Signal, window_size: int) -> np.ndarray:
    #     return FeatureExtractor.compute_chromagram(signal, window_size)


    # @staticmethod
    # def get_silence_splits(signal: Signal, force_cache=False) -> List[Tuple[float, float]]:
    #     silenceSegmenter: Segmenter = SilenceSegmenter()
    #     if not force_cache and CacheOld.resource_exists(signal.id, CacheToken.SILENCES):
    #         silences = np.load(CacheOld.resource_path(signal.id, CacheToken.SILENCES), allow_pickle=True)
    #     else:
    #         silences = SilenceSegmenter.librosa_split(signal)
    #         np.save(CacheOld.resource_path(signal.id, CacheToken.SILENCES), silences)
    #     return silences
