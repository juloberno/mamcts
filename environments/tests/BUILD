py_test(
  name = "py_crossing_state_int_test",
  srcs = ["py_crossing_state_int_test.py"],
  data = ['//python/bindings:mamcts.so'],
  imports = ['../../python/bindings'],
  deps=[
    "//environments:pyviewer",
    "//util:debug_settings"
  ]
)

py_test(
  name = "py_crossing_state_float_test",
  srcs = ["py_crossing_state_float_test.py"],
  data = ['//python/bindings:mamcts.so'],
  imports = ['../../python/bindings'],
  deps=[
    "//environments:pyviewer",
    "//util:debug_settings"
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
