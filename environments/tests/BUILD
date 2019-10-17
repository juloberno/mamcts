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
