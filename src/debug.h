// debug.h
#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <sstream>

// Define debug levels
#define DEBUG_LEVEL_NONE 0
#define DEBUG_LEVEL_INFO 1
#define DEBUG_LEVEL_WARNING 2
#define DEBUG_LEVEL_ERROR 3

// Set the current debug level
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 2
#endif

// Base debug macro using variadic arguments
// add back in __FILE__ << ":" << __LINE__ if needed
#define DEBUG_PRINT(level_str, ...) \
    do { \
        std::cout << "[" << level_str << "] ";  \
        std::cout << __VA_ARGS__ << std::endl; \
    } while (0)

// Specific macros for each debug level with compile-time removal
#if DEBUG_LEVEL <= DEBUG_LEVEL_INFO
    #define DEBUG_INFO(...)    DEBUG_PRINT("INFO", __VA_ARGS__)
#else
    #define DEBUG_INFO(...)    do { } while (0)
#endif

#if DEBUG_LEVEL <= DEBUG_LEVEL_WARNING
    #define DEBUG_WARNING(...) DEBUG_PRINT("WARNING", __VA_ARGS__)
#else
    #define DEBUG_WARNING(...) do { } while (0)
#endif

#if DEBUG_LEVEL <= DEBUG_LEVEL_ERROR
    #define DEBUG_ERROR(...)   DEBUG_PRINT("ERROR", __VA_ARGS__)
#else
    #define DEBUG_ERROR(...)   do { } while (0)
#endif

#endif // DEBUG_H
