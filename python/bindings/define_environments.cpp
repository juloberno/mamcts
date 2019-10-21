// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "python/bindings/define_environments.hpp"
#include "python/bindings/define_crossing_state.hpp"
#include "mcts/mcts.h"

namespace py = pybind11;
using namespace mcts;

void define_environments(py::module m)
{
    py::class_<Viewer,
             PyViewer,
             std::shared_ptr<Viewer>>(m, "Viewer")
      .def(py::init<>())
      .def("drawPoint", &Viewer::drawPoint)
      .def("drawLine", &Viewer::drawLine);

    define_crossing_state<int>(m, "Int");
    define_crossing_state<float>(m, "Float");
}