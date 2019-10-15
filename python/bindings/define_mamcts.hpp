// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef PYTHON_DEFINE_MAMCTS_HPP_
#define PYTHON_DEFINE_MAMCTS_HPP_
#include "python/bindings/common.hpp"

namespace py = pybind11;

void define_mamcts(py::module m);

#endif   // PYTHON_DEFINE_MAMCTS_HPP_