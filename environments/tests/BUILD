py_test(
  name = "py_crossing_state_test",
  srcs = ["py_crossing_state_test.py"],
  data = ['//python/bindings:mamcts.so'],
  imports = ['../../python/bindings'],
  deps=[
    "//environments:pyviewer"
  ]
)

cc_test(
    name = "crossing_state_int_test",
    srcs = [
        "crossing_state_int_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//environments:crossing_state",
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)

cc_test(
    name = "crossing_state_float_test",
    srcs = [
        "crossing_state_float_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//environments:crossing_state",
        "//mcts:mamcts",
        "@gtest//:main",
    ],
)
