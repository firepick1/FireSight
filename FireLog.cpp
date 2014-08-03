#include "FireLog.h"
#include "version.h"
#include <iostream>

#include <errno.h>
#include <time.h>
#ifdef WIN32
  #include <Winsock2.h>
  #include <Windows.h>
#else
  #define LOG_THREAD_ID
  #include <sys/time.h>
#endif
#include <string.h>

#ifdef LOG_THREAD_ID
#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#endif
#include <unistd.h>
#include <sys/syscall.h>
static bool logTID = 1;
#else
static bool logTID = 0;
#endif

#define LOGMAX 255

using namespace std;

FILE *logFile = NULL;
int logLevel = FIRELOG_WARN;
static char lastMessage[5][LOGMAX+1];


int firelog_init(const char *path, int level) {
  logLevel = level;
  logFile = fopen(path, "w");
  if (!logFile) {
    return errno;
  }
  char version[32];
  snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  LOGINFO2("FireLog %s version %s", path, version);
  firelog_lastMessageClear();
  return 0;
}

int firelog_destroy() {
  int rc = 0;
  if (logFile) {
    rc = fclose(logFile);
  }
  return rc;
}

/* Last message up to given level*/
const char * firelog_lastMessage(int level) {
  int i;
  for (i = 0; i<FIRELOG_TRACE; i++) {
    if (lastMessage[i]) {
      return lastMessage[i];
    }
  }
  return "";
}

void firelog_show_thread_id(int show) {
  logTID = show;
}

void firelog_lastMessageClear() {
  memset(lastMessage, 0, sizeof(lastMessage));
}

int firelog_level(int newLevel) {
  int oldLevel = logLevel;
  logLevel = newLevel;
  switch (newLevel) {
    case FIRELOG_ERROR:
      LOGDEBUG1("firelog_level(%s)", "FIRELOG_ERROR");
      break;
    case FIRELOG_WARN:
      LOGDEBUG1("firelog_level(%s)", "FIRELOG_WARN");
      break;
    case FIRELOG_INFO:
      LOGDEBUG1("firelog_level(%s)", "FIRELOG_INFO");
      break;
    case FIRELOG_DEBUG:
      LOGDEBUG1("firelog_level(%s)", "FIRELOG_DEBUG");
      break;
    case FIRELOG_TRACE:
      LOGDEBUG1("firelog_level(%s)", "FIRELOG_TRACE");
      break;
    default:
      LOGERROR1("firelog_level(unknown level %d)", newLevel);
      break;
  }
  return oldLevel;
}

void firelog(const char *msg, int level) {
#ifdef WIN32
 SYSTEMTIME st;
 GetSystemTime(&st);
 int now_hour = st.wHour;
 int now_min = st.wMinute;
 int now_sec = st.wSecond;
 int now_ms =  st.wMilliseconds;
#else
  timeval tp;
  gettimeofday(&tp, 0);
  time_t curtime = tp.tv_sec;
  struct tm *pLocalNow = localtime(&curtime);
  int now_hour = pLocalNow->tm_hour;
  int now_min = pLocalNow->tm_min;
  int now_sec = pLocalNow->tm_sec;
  int now_ms = tp.tv_usec/1000;
#endif
  int tid = 0;
#ifdef LOG_THREAD_ID
  tid = syscall(SYS_gettid);
#endif
  const char * levelStr = "?";

  switch (level) {
    case FIRELOG_ERROR: levelStr = " ERROR "; break;
    case FIRELOG_WARN: levelStr = " W "; break;
    case FIRELOG_INFO: levelStr = " I "; break;
    case FIRELOG_DEBUG: levelStr = " D "; break;
    case FIRELOG_TRACE: levelStr = " T "; break;
  }

  if (logTID) {
    snprintf(lastMessage[level], LOGMAX, "%02d:%02d:%02d.%03d %d %s %s", 
        now_hour, now_min, now_sec, now_ms, tid, levelStr, msg);
  } else {
    snprintf(lastMessage[level], LOGMAX, "%02d:%02d:%02d.%03d %s %s", 
        now_hour, now_min, now_sec, now_ms, levelStr, msg);
  }

  if (logFile) {
    fprintf(logFile, "%s", lastMessage[level]);
    fprintf(logFile, "\n");
    fflush(logFile);
  }
#ifdef __cplusplus
  else {
    cerr << lastMessage[level] << endl;
    cerr.flush();
  }
#endif
}
