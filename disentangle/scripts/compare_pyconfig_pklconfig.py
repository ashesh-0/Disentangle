"""
Here, we compare a .py config file with a .pkl config file which gets generated from a training.
"""
import os.path

# from absl import flags
from ml_collections.config_flags import config_flags
from disentangle.scripts.compare_configs import (get_comparison_df, get_df_column_name, get_commit_key,
                                                 get_changed_files, display_changes)
from disentangle.config_utils import load_config

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("py_config", None, "Python config file", lock_config=True)
flags.DEFINE_string("pkl_config", None, "Work directory.")

if __name__ == '__main__':
    config1 = FLAGS.py_config
    config2 = load_config(FLAGS.pkl_config
                          )
    df = get_comparison_df(config1, config2, os.path.basename(FLAGS.py_config.name),
                           get_df_column_name(FLAGS.pkl_config))

    changed_files = get_changed_files(*list(df.loc[get_commit_key()].values))
    display_changes(df, changed_files)
