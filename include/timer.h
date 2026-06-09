/* timer.h — portable monotonic wall-clock seconds (Windows + POSIX). */
#ifndef MC_TIMER_H
#define MC_TIMER_H

#if defined(_WIN32)
#  include <windows.h>
static inline double mc_wtime(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#  include <time.h>
static inline double mc_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#endif /* MC_TIMER_H */
