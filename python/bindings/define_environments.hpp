// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef PYTHON_DEFINE_ENVIRONMENTS_HPP_
#define PYTHON_DEFINE_ENVIRONMENTS_HPP_

#include "python/bindings/common.hpp"
#include "environments/viewer.hpp"

namespace py = pybind11;

class PyViewer : public mcts::Viewer {
 public:
  using Viewer::Viewer;

  void drawPoint(float x, float y, float size,  mcts::Color color) override {
    PYBIND11_OVERLOAD_PURE(
        void,
        mcts::Viewer,
        drawPoint,
        x,
        y,
        size,
        color);
  }

  void drawLine(std::pair<float,float> x, std::pair<float,float> y, float linewidth,  mcts::Color color) override {
    PYBIND11_OVERLOAD_PURE(
        void,
        mcts::Viewer,
        drawLine,
        x,
        y,
        linewidth,
        color);
  }
};



void define_environments(py::module m);

#endif   // PYTHON_DEFINE_ENVIRONMENTS_HPP_