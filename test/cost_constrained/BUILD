cc_test(
    name = "cost_constrained_test_single_cost",
    srcs = [
        "cost_constrained_test_single_cost.cc",
        "cost_constrained_statistic_test_state_single_cost.h"
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)

cc_test(
    name = "cost_constrained_statistic_test_state_test",
    srcs = [
        "cost_constrained_statistic_test_state_test.cc",
        "cost_constrained_statistic_test_state_single_cost.h"
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)

cc_test(
    name = "cost_constrained_test_multiple_cost",
    srcs = [
        "cost_constrained_test_multiple_cost.cc",
        "cost_constrained_statistic_test_state_multiple_cost.h",
        "cost_constrained_statistic_test_state_single_cost.h"
    ],
    copts = ["-Iexternal/gtest/include", "-DNDEBUG"],
    deps = [
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)

cc_test(
    name = "cost_constrained_test_action_values",
    srcs = [
        "cost_constrained_test_action_values.cc",
        "cost_constrained_statistic_test_state_multiple_cost.h",
        "cost_constrained_statistic_test_state_single_cost.h"
    ],
    copts = ["-Iexternal/gtest/include", "-DNDEBUG"],
    deps = [
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)

cc_library(
    name ="cost_constrained_test",
    hdrs = [
        "cost_constrained_statistic_test_state_multiple_cost.h",
        "cost_constrained_statistic_test_state_single_cost.h"
    ],
    visibility = ["//visibility:public"],
)