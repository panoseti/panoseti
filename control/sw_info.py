#! /usr/bin/env python3
import git
import json

def get_sw_info():
    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.commit.hexsha
    author = repo.head.commit.author.name
    branch = repo.active_branch.name
    commit_date = repo.head.commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
    sw_info={'commit':commit,'author':author,'branch':branch,'commit_date':commit_date}
    with open('sw_info.json','w') as f:
        json.dump(sw_info, f)

if __name__ == '__main__':
    get_sw_info()
