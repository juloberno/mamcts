// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef PYTHON_DEFINE_ENVIRONMENTS_HPP_
#define PYTHON_DEFINE_ENVIRONMENTS_HPP_
#include "python/common.hpp"

namespace py = pybind11;

void define_environments(py::module m);

#endif   // PYTHON_DEFINE_ENVIRONMENTS_HPP_