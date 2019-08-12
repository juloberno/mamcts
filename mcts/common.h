// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_COMMON_H
#define MCTS_COMMON_H


namespace mcts {

class MctsTest;

#ifdef UNIT_TESTING
#define MCTS_TEST friend class MctsTest;
#else
#define MCTS_TEST
#endif


#ifdef CRTP_DYNAMIC_INTERFACE
#define CRTP_INTERFACE(type) inline Implementation& impl()  { \
        auto derivedptr = dynamic_cast<type*>(this); \
        TEST_ASSERT(derivedptr!=nullptr); \
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
        TEST_ASSERT(derivedptr!=nullptr); \
        return *derivedptr; \
    }
#else
#define CRTP_CONST_INTERFACE(type) inline const Implementation& impl() const { \
        return *static_cast<const type*>(this); \
        }
#endif

#define TEST_ASSERT(cond) assert(#cond)

} // namespace mcts
#endif
