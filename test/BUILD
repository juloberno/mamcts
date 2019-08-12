cc_test(
    name = "mcts_test",
    srcs = [
        "mcts_test.cc",
        "mcts_test_class.h",
        "simple_state.h"
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//mcts:mcts",
        "@gtest//:main",
    ],
)