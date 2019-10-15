// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "python/bindings/define_mamcts.hpp"
#include "python/bindings/define_environments.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mamcts, m) {
  define_mamcts(m);
  define_environments(m);
}