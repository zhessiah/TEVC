REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_robust import Robust_ParallelRunner
REGISTRY["parallel_robust"] = Robust_ParallelRunner