package(default_visibility = ["//visibility:public"])


# the following rules are only for testing purposes
cc_binary(
  name = "mamcts.so",
  srcs = ["module.cpp",
          "common.hpp",
          "define_mamcts.hpp",
          "define_mamcts.cpp",
          "define_environments.hpp",
          "define_environments.cpp",
  ],
  deps = [
    "//mcts:mamcts",
    "//environments:crossing_state",
    "@python_linux//:python-lib",
    "@pybind11//:pybind11",
    "@com_github_eigen_eigen//:eigen", 
  ],
  linkshared = 1,
  linkstatic = 1
)

py_test(
  name = "python_import_test",
  srcs = ["python_import_test.py"],
  data = ['//python/bindings:mamcts.so'],
  imports = ['../bindings'],
)
