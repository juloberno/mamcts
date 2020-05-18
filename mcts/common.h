// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_COMMON_H
#define MCTS_COMMON_H

#include "glog/logging.h"

namespace mcts {

class UctTest;

#ifdef UNIT_TESTING
#define MCTS_TEST friend class UctTest;
#else
#define MCTS_TEST
#endif

#ifdef UCT_COST_CONSTRAINED_STATISTIC_H
#define FRIEND_COST_CONSTRAINED_STATISTIC friend class CostConstrainedStatistic;
#else
#define FRIEND_COST_CONSTRAINED_STATISTIC
#endif


#ifdef CRTP_DYNAMIC_INTERFACE
#define CRTP_INTERFACE(type) inline Implementation& impl()  { \
        auto derivedptr = dynamic_cast<type*>(this); \
        MCTS_EXPECT_TRUE(derivedptr!=nullptr); \
        return *derivedptr; \
        }
#else
#define CRTP_INTERFACE(type) inline Implementation& impl() {\
        return *static_cast<type*>(this); \
        }
#endif

#ifdef CRTP_DYNAMIC_INTERFACE
#define CRTP_CONST_INTERFACE(type) inline const Implementation& impl() const { \
        auto const derivedptr = dynamic_cast<const type*>(this); \
        MCTS_EXPECT_TRUE(derivedptr!=nullptr); \
        return *derivedptr; \
    }
#else
#define CRTP_CONST_INTERFACE(type) inline const Implementation& impl() const { \
        return *static_cast<const type*>(this); \
        }
#endif




#ifndef NDEBUG
#   define MCTS_EXPECT_TRUE(condition, ...) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ". " << #__VA_ARGS__ << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define MCTS_EXPECT_TRUE(condition, message) do { } while (false)
#endif

struct RequiresHypothesis 
{};

struct RequiresCost 
{};

} // namespace mcts
#endif
