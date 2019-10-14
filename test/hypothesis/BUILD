cc_test(
    name = "crossing_state_test",
    srcs = [
        "crossing_state_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//environments:crossing_state",
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)

cc_test(
    name = "belief_tracker_test",
    srcs = [
        "belief_tracker_test.cc",
        "belief_tracker_test_state.h"
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)

cc_test(
    name = "hypothesis_statistic_test",
    srcs = [
        "hypothesis_statistic_test.cc",
        "hypothesis_statistic_test_state.h"
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)