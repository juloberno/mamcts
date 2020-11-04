workspace(name="mamcts")


load("//util:deps.bzl", "mamcts_dependencies")
mamcts_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()



# Google or tools -----------------------------
load("//util:deps_or.bzl", "google_or_dependencies")
google_or_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
# Load common dependencies.
protobuf_deps()