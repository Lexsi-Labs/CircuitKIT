from functools import partial

from tqdm.auto import tqdm

tqdm = partial(
    tqdm,
    dynamic_ncols=True,
    bar_format="{desc}{bar}{r_bar}",
    leave=None,
    delay=0,
)
"""
Wrapper around `tqdm` with default settings for the project.
"""
