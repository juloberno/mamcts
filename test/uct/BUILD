cc_test(
    name = "uct_test",
    srcs = [
        "uct_test.cc",
        "uct_test_class.h",
        "simple_state.h"
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//mcts:mcts",
        "@gtest//:main",
    ],
)