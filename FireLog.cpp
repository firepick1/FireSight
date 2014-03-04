#include "FireLog.h"
#include "version.h"
#include <iostream>

#include <errno.h>
#include <time.h>
#include <string.h>

#ifdef LOG_THREAD_ID
#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#endif
#include <unistd.h>
#include <sys/syscall.h>
#endif

#define LOGMAX 255

using namespace std;

FILE *logFile = NULL;
int logLevel = FIRELOG_WARN;
static bool logTID = 0;
static char lastMessage[5][LOGMAX+1];


int firelog_init(const char *path, int level) {
  logLevel = level;
  logFile = fopen(path, "w");
  if (!logFile) {
    return errno;
  }
	char version[32];
	snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  LOGINFO2("FireLog %s versio %s", path, version);
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

void firelog_show_thread_id(bool show) {
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
	time_t now = time(NULL);
	struct tm *pLocalNow = localtime(&now);
#ifdef LOG_THREAD_ID
	int tid = syscall(SYS_gettid);
#endif

  if (logFile) {
    fprintf(logFile, "%02d:%02d:%02d ", pLocalNow->tm_hour, pLocalNow->tm_min, pLocalNow->tm_sec);
    switch (level) {
      case FIRELOG_ERROR: fprintf(logFile, " ERROR "); break;
      case FIRELOG_WARN: fprintf(logFile, " W "); break;
      case FIRELOG_INFO: fprintf(logFile, " I "); break;
      case FIRELOG_DEBUG: fprintf(logFile, " D "); break;
      case FIRELOG_TRACE: fprintf(logFile, " T "); break;
      default: fprintf(logFile, "?%d? ", level); break;
    }
#ifdef LOG_THREAD_ID
		if (logTID) {
		  fprintf(logFile, "%d ", tid);
		}
#endif

		snprintf(lastMessage[level], LOGMAX, "%s", msg);
    fprintf(logFile, "%s", msg);
    fprintf(logFile, "\n");
    fflush(logFile);
  }
#ifdef __cplusplus
  else {
		cerr << pLocalNow->tm_hour << ":" << pLocalNow->tm_min << ":" << pLocalNow->tm_sec;
    switch (level) {
      case FIRELOG_ERROR: cerr << " ERROR " ; break;
      case FIRELOG_WARN: cerr << " W " ; break;
      case FIRELOG_INFO: cerr << " I " ; break;
      case FIRELOG_DEBUG: cerr << " D " ; break;
      case FIRELOG_TRACE: cerr << " T " ; break;
      default: cerr << "?" << level << "? " ; break;
    }
#ifdef LOG_THREAD_ID
		if (logTID) {
		  cerr << tid << " ";
		}
#endif
		snprintf(lastMessage[level], LOGMAX, "%s", msg);
		cerr << lastMessage[level] << endl;
	}
#endif
}
