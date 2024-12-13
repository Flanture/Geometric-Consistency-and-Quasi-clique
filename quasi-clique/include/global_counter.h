#ifndef GLOBAL_COUNTER_H
#define GLOBAL_COUNTER_H

#include <mutex>
#include <string>
#include <sstream>
#include <atomic>

extern int global_counter;
extern std::mutex mtx;
extern std::atomic<int> iteration_count;
extern const int MAX_ITERATIONS;
void resetIterationCount();
std::string getFileName(const std::string& prefix);
std::string getFileName(const std::string& base_path, const std::string& suffix);
#endif // GLOBAL_COUNTER_H
