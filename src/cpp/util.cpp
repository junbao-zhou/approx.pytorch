#include "util.h"

double exec_time(std::function<void(void)> func)
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();

    duration<double, std::milli> ms_double = end - start;

    return ms_double.count();
}