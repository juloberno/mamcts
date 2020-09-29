import sys
import os

cwd = os.getcwd()
workspace_folder = cwd
repo_paths = ["mamcts/python/bindings", "mamcts"]

executed_file = sys.argv[0]
tmp = executed_file.replace("mamcts", "mamcts/bazel-bin")
runfiles_dir = tmp.replace(".py", ".runfiles")

sys.path.append(runfiles_dir)
for repo in repo_paths:
    full_path = os.path.join(runfiles_dir, repo)
    print("adding python path: {}".format(full_path))
    sys.path.append(full_path)