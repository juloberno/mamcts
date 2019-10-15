// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================
#ifndef MCTS_VIEWER_HPP_
#define MCTS_VIEWER_HPP_

#include <tuple>

namespace mcts {

typedef std::tuple<float,float,float ,float> Color; // < R;G;B;ALPHA

class Viewer {
  public:
    Viewer() {}
    virtual ~Viewer() {}
    virtual void drawPoint(float x, float y, float size,  Color color) = 0;
    virtual void drawLine(float x, float y, float linewidth,  Color color) = 0;
};


}

#endif // MCTS_VIEWER_HPP_