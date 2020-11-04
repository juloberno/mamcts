load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")


def google_or_dependencies():
    _maybe(
    git_repository,
    name = "com_google_protobuf",
    commit = "fde7cf7",  # release v3.13.0
    remote = "https://github.com/protocolbuffers/protobuf.git",
    )

    _maybe(
    git_repository,
    name = "google_or",
    commit = "ddd049dee8e3f912a70ae06b8c234bb31d505f12",  # release v3.13.0
    remote = "https://github.com/google/or-tools.git",
    )

    _maybe(
    http_archive, 
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://mirror.bazel.build/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
    )

    _maybe(
    http_archive, 
    name = "scip",
    build_file = "@google_or//bazel:scip.BUILD",
    patches = [ "@google_or//bazel:scip.patch" ],
    sha256 = "033bf240298d3a1c92e8ddb7b452190e0af15df2dad7d24d0572f10ae8eec5aa",
    url = "https://github.com/google/or-tools/releases/download/v7.7/scip-7.0.1.tgz",
    )

    _maybe(
    git_repository,
    name = "com_github_gflags_gflags",
    commit = "e171aa2",  # release v2.2.2
    remote = "https://github.com/gflags/gflags.git",
    )   

    _maybe(
    git_repository,
    name = "com_github_glog_glog",
    commit = "96a2f23",  # release v0.4.0
    remote = "https://github.com/google/glog.git",
    )

    _maybe(
    git_repository,
    name = "bazel_skylib",
    commit = "e59b620",  # release 1.0.2
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    )

    _maybe(
    git_repository,
    name = "com_google_absl",
    commit = "b56cbdd", # release 20200923
    remote = "https://github.com/abseil/abseil-cpp.git",
    )

    _maybe(
    http_archive, 
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "//test/external:gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
    )

    _maybe(
    http_archive, 
    name = "glpk",
    build_file = "@google_or//bazel:glpk.BUILD",
    sha256 = "4281e29b628864dfe48d393a7bedd781e5b475387c20d8b0158f329994721a10",
    url = "http://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz",
    )

    _maybe(
    http_archive, 
    name = "bliss",
    build_file = "@google_or//bazel:bliss.BUILD",
    patches = ["@google_or//bazel:bliss-0.73.patch"],
    sha256 = "f57bf32804140cad58b1240b804e0dbd68f7e6bf67eba8e0c0fa3a62fd7f0f84",
    url = "http://www.tcs.hut.fi/Software/bliss/bliss-0.73.zip",
    )




def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)