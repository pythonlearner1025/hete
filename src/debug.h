// debug.h
#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include "constants.h"

// Define debug levels
#define DEBUG_LEVEL_NONE 0
#define DEBUG_LEVEL_INFO 1
#define DEBUG_LEVEL_WARNING 2
#define DEBUG_LEVEL_ERROR 3
#define DEBUG_LEVEL_WRITE 4

// Base debug macro using variadic arguments
#define DEBUG_PRINT(level_str, ...) \
    do { \
        std::cout << "[" << level_str << "] " << __VA_ARGS__ << std::endl; \
    } while (0)

// Existing macros for each debug level with compile-time removal
#if DEBUG_LEVEL == DEBUG_LEVEL_NONE
    #define DEBUG_NONE(...) DEBUG_PRINT("LOG", __VA_ARGS__)
#else
    #define DEBUG_NONE(...)    do { } while (0)
#endif

#if DEBUG_LEVEL <= DEBUG_LEVEL_INFO && DEBUG_LEVEL != DEBUG_LEVEL_NONE
    #define DEBUG_INFO(...)    DEBUG_PRINT("INFO", __VA_ARGS__)
#else
    #define DEBUG_INFO(...)    do { } while (0)
#endif

#if DEBUG_LEVEL <= DEBUG_LEVEL_WARNING && DEBUG_LEVEL != DEBUG_LEVEL_NONE
    #define DEBUG_WARNING(...) DEBUG_PRINT("WARNING", __VA_ARGS__)
#else
    #define DEBUG_WARNING(...) do { } while (0)
#endif

#if DEBUG_LEVEL <= DEBUG_LEVEL_ERROR && DEBUG_LEVEL != DEBUG_LEVEL_NONE
    #define DEBUG_ERROR(...)   DEBUG_PRINT("ERROR", __VA_ARGS__)
#else
    #define DEBUG_ERROR(...)   do { } while (0)
#endif

// New macro to write to a log file specified by path
#if DEBUG_LEVEL == DEBUG_LEVEL_WRITE //|| DEBUG_LEVEL == 0
    #define DEBUG_WRITE(log_file_path, ...) \
        do { \
            std::ofstream log_file(log_file_path, std::ios::app); \
            if (log_file.is_open()) { \
                log_file << __VA_ARGS__ << std::endl; \
                log_file.close(); \
            } else { \
                std::cerr << "Failed to open log file: " << log_file_path << std::endl; \
            } \
        } while (0)
#else
    #define DEBUG_WRITE(log_file_path, ...) do { } while (0)
#endif

#endif // DEBUG_H