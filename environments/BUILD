cc_library(
    name = "crossing_state",
    hdrs = [
        "crossing_state.h",
        "viewer.h",
        "crossing_state_episode_runner.h"
    ],
    srcs = [
        "crossing_state.cpp",
        "crossing_state_episode_runner.cpp",
    ],
    deps = [
        "//mcts:mamcts",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "pyviewer",
    srcs = ["pyviewer.py"],
    data = ['//python/bindings:mamcts.so'],
    imports = ['../../python/bindings'],
    visibility = ["//visibility:public"],
)