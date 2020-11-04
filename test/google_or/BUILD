cc_test(
    name = "google_or_example",
    srcs = [
        "google_or_example.cc",
    ],
    copts = ["-Iexternal/gtest/include", "-DNDEBUG"],
    deps = [
        "@google_or//ortools/linear_solver:linear_solver",
        "@gtest//:main",
    ],
)