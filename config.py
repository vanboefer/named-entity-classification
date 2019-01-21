"""
This script serves two purposes:
- When ran directly, it creates the config.ini file, which stores all the project paths. This needs to be done at the beginning of the project.
- When imported, it gives you access to the ProjectPaths class. This class makes all the paths defined in config.ini accessible.
"""

import configparser
from pathlib import Path


class ProjectPaths:
    config = configparser.ConfigParser()
    config.read('config.ini')

    def __init__(self):
        self.paths = dict(
            work_dir = Path(self.config['PATHS']['work_dir']),
            data_raw = Path(self.config['PATHS']['data_raw']),
            data_processed = Path(self.config['PATHS']['data_processed']),
            results = Path(self.config['PATHS']['results']),
            word_embeddings = Path(self.config['PATHS']['word_embeddings']),
            templates = Path(self.config['PATHS']['templates']),
                          )

    def __getitem__(self, key):
        return self.paths[key]

if __name__ == '__main__':
    config = configparser.ConfigParser()

    config['PATHS'] = {}
    work_dir = Path(__file__).resolve().parent

    data_raw_dir = work_dir / 'data_raw'
    data_processed_dir = work_dir / 'data_processed'
    results_dir = work_dir / 'results'
    embeddings_dir = work_dir / 'word_embeddings_EN'
    templates_dir = work_dir / 'templates'

    data_paths = [data_raw_dir,
                  data_processed_dir,
                  results_dir,
                  embeddings_dir]
    for path in data_paths:
        if not path.exists():
            path.mkdir(parents=True)

    config['PATHS']['work_dir'] =  str(work_dir)
    config['PATHS']['data_raw'] = str(data_raw_dir)
    config['PATHS']['data_processed'] = str(data_processed_dir)
    config['PATHS']['results'] = str(results_dir)
    config['PATHS']['word_embeddings'] = str(embeddings_dir)
    config['PATHS']['templates'] = str(templates_dir)

    config_file = work_dir / 'config.ini'

    with open(config_file, 'w') as f:
        config.write(f)
