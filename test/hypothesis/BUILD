cc_test(
    name = "hypothesis_test",
    srcs = [
        "hypothesis_test.cc",
        "hypothesis_simple_state.h"
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//mcts:mcts",
        "@gtest//:main",
    ],
)