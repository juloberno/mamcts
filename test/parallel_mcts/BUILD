cc_test(
    name = "cost_constrained_parallel_test",
    srcs = [
        "cost_constrained_parallel_test.cc"
    ],
    copts = ["-Iexternal/gtest/include", "-DNDEBUG"],
    deps = [
        "//mcts:mamcts",
        "@gtest//:main",
        "//test/cost_constrained:cost_constrained_test",
    ],
)
