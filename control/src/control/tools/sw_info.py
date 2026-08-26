#! /usr/bin/env python3
import json

from control.utils import config_file
from control.utils.paths import PanoPaths


def get_sw_info() -> None:
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        commit = repo.head.commit.hexsha
        author = repo.head.commit.author.name
        branch = repo.active_branch.name
        commit_date = repo.head.commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
        sw_info={'commit':commit,\
                 'author':author,\
                 'branch':branch,
                 'commit_date':commit_date }
    except Exception as e:
        sw_info={'commit':'unknown',\
                 'author':'unknown',\
                 'branch':'unknown',\
                 'commit_date':'unknown',
                 'error': str(e)}

    tmp_dir = PanoPaths.tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_dir / config_file.sw_info_filename, 'w') as f:
        json.dump(sw_info, f, indent=4)

if __name__ == '__main__':
    get_sw_info()
