#include "FireLog.h"
#include "version.h"

#include <errno.h>
#include <time.h>
#include <string.h>
#include <sys/syscall.h>

#include <boost/format.hpp>

#define LOGMAX 255

using namespace std;

FILE *logFile = NULL;
int logLevel = FIRELOG_WARN;
char lastMessage[5][LOGMAX+1];


int firelog_init(const char *path, int level) {
  logLevel = level;
  logFile = fopen(path, "w");
  if (!logFile) {
    return errno;
  }
  LOGINFO3("FireLog %s %d.%d", path, VERSION_MAJOR, VERSION_MINOR);
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

void firelog_lastMessageClear() {
	memset(lastMessage, 0, sizeof(lastMessage));
}

int firelog_level(int newLevel) {
  int oldLevel = logLevel;
  logLevel = newLevel;
	switch (newLevel) {
	  case FIRELOG_ERROR:
			LOGINFO1("firelog_level(%s)", "FIRELOG_ERROR");
			break;
	  case FIRELOG_WARN:
			LOGINFO1("firelog_level(%s)", "FIRELOG_WARN");
			break;
	  case FIRELOG_INFO:
			LOGINFO1("firelog_level(%s)", "FIRELOG_INFO");
			break;
	  case FIRELOG_DEBUG:
			LOGINFO1("firelog_level(%s)", "FIRELOG_DEBUG");
			break;
	  case FIRELOG_TRACE:
			LOGINFO1("firelog_level(%s)", "FIRELOG_TRACE");
			break;
		default:
			LOGINFO1("firelog_level(unknown level %d)", newLevel);
			break;
	}
  return oldLevel;
}

void firelog(const char *fmt, int level, const void * value1, const void * value2, const void * value3) {
	time_t now = time(NULL);
	struct tm *pLocalNow = localtime(&now);
	int tid = syscall(SYS_gettid);
	char logBuf[LOGMAX+1];

  if (logFile) {
    fprintf(logFile, "%02d:%02d:%02d ", pLocalNow->tm_hour, pLocalNow->tm_min, pLocalNow->tm_sec);
    switch (level) {
      case FIRELOG_ERROR: fprintf(logFile, "ERROR %d ", tid); break;
      case FIRELOG_WARN: fprintf(logFile, "WARN %d ", tid); break;
      case FIRELOG_INFO: fprintf(logFile, "INFO %d ", tid); break;
      case FIRELOG_DEBUG: fprintf(logFile, "DEBUG %d ", tid); break;
      case FIRELOG_TRACE: fprintf(logFile, "TRACE %d ", tid); break;
      default: fprintf(logFile, "?%d? %d ", level, tid); break;
    }
		sprintf(lastMessage[level], fmt, value1, value2, value3);
    fprintf(logFile, fmt, value1, value2, value3);
    fprintf(logFile, "\n", tid);
    fflush(logFile);
  }
#ifdef __cplusplus
  else {
		cout << pLocalNow->tm_hour << ":" << pLocalNow->tm_min << ":" << pLocalNow->tm_sec;
    switch (level) {
      case FIRELOG_ERROR: cout << "ERROR " ; break;
      case FIRELOG_WARN: cout << "WARN " ; break;
      case FIRELOG_INFO: cout << "INFO " ; break;
      case FIRELOG_DEBUG: cout << "DEBUG " ; break;
      case FIRELOG_TRACE: cout << "TRACE " ; break;
      default: cout << "?" << level << "? " ; break;
    }
		sprintf(lastMessage[level], fmt, value1, value2, value3);
		cout << tid << " " << lastMessage[level] << endl;
	}
#endif
}
