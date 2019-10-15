cc_library(
    name = "crossing_state",
    hdrs = [
        "crossing_state.h",
        "viewer.hpp"
    ],
    srcs = [
        "crossing_state.cpp",
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