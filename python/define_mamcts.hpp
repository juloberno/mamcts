// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef PYTHON_PYTHON_PLANNER_UCT_HPP_
#define PYTHON_PYTHON_PLANNER_UCT_HPP_
#include "python/common.hpp"

namespace py = pybind11;

void define_mamcts(py::module m);

#endif   // PYTHON_PYTHON_PLANNER_UCT_HPP_