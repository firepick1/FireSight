#ifndef FIRELOG_H
#define FIRELOG_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

#define FIRELOG_ERROR 0
#define FIRELOG_WARN 1
#define FIRELOG_INFO 2
#define FIRELOG_DEBUG 3
#define FIRELOG_TRACE 4

#define LOGERROR3(fmt,v1,v2,v3) if (logLevel >= FIRELOG_ERROR) {firelog(fmt, FIRELOG_ERROR, (void *)v1, (void *)v2, (void *)v3);}
#define LOGERROR2(fmt,v1,v2) if (logLevel >= FIRELOG_ERROR) {firelog(fmt, FIRELOG_ERROR, (void *)v1, (void *)v2, fmt);}
#define LOGERROR1(fmt,v1) if (logLevel >= FIRELOG_ERROR) {firelog(fmt, FIRELOG_ERROR, (void *)v1, fmt, fmt);}
#define LOGERROR(fmt) if (logLevel >= FIRELOG_ERROR) {firelog(fmt, FIRELOG_ERROR, fmt, fmt, fmt);}
#define LOGWARN3(fmt,v1,v2,v3) if (logLevel >= FIRELOG_WARN) {firelog(fmt, FIRELOG_WARN, (void *)v1, (void *)v2, (void *)v3);}
#define LOGWARN2(fmt,v1,v2) if (logLevel >= FIRELOG_WARN) {firelog(fmt, FIRELOG_WARN, (void *)v1, (void *)v2, fmt);}
#define LOGWARN1(fmt,v1) if (logLevel >= FIRELOG_WARN) {firelog(fmt, FIRELOG_WARN, (void *)v1, fmt, fmt);}
#define LOGWARN(fmt,v1,v2,v3) if (logLevel >= FIRELOG_WARN) {firelog(fmt, FIRELOG_WARN, fmt, fmt, fmt);}
#define LOGINFO3(fmt,v1,v2,v3) if (logLevel >= FIRELOG_INFO) {firelog(fmt, FIRELOG_INFO, (void *)v1, (void *)v2, (void *)v3);}
#define LOGINFO2(fmt,v1,v2) if (logLevel >= FIRELOG_INFO) {firelog(fmt, FIRELOG_INFO, (void *)v1, (void *)v2, fmt);}
#define LOGINFO1(fmt,v1) if (logLevel >= FIRELOG_INFO) {firelog(fmt, FIRELOG_INFO, (void *)v1, fmt, fmt);}
#define LOGINFO(fmt) if (logLevel >= FIRELOG_INFO) {firelog(fmt, FIRELOG_INFO, fmt, fmt, fmt);}
#define LOGDEBUG3(fmt,v1,v2,v3) if (logLevel >= FIRELOG_DEBUG) {firelog(fmt, FIRELOG_DEBUG, (void *)v1, (void *)v2, (void *)v3);}
#define LOGDEBUG2(fmt,v1,v2) if (logLevel >= FIRELOG_DEBUG) {firelog(fmt, FIRELOG_DEBUG, (void *)v1, (void *)v2, fmt);}
#define LOGDEBUG1(fmt,v1) if (logLevel >= FIRELOG_DEBUG) {firelog(fmt, FIRELOG_DEBUG, (void *)v1, fmt, fmt);}
#define LOGDEBUG(fmt) if (logLevel >= FIRELOG_DEBUG) {firelog(fmt, FIRELOG_DEBUG, fmt, fmt, fmt);}
#define LOGTRACE3(fmt,v1,v2,v3) if (logLevel >= FIRELOG_TRACE) {firelog(fmt, FIRELOG_TRACE, (void *)v1, (void *)v2, (void *)v3);}
#define LOGTRACE2(fmt,v1,v2) if (logLevel >= FIRELOG_TRACE) {firelog(fmt, FIRELOG_TRACE, (void *)v1, (void *)v2, fmt);}
#define LOGTRACE1(fmt,v1) if (logLevel >= FIRELOG_TRACE) {firelog(fmt, FIRELOG_TRACE, (void *)v1, fmt, fmt);}
#define LOGTRACE(fmt) if (logLevel >= FIRELOG_TRACE) {firelog(fmt, FIRELOG_TRACE, fmt, fmt, fmt);}

#define LOGRC(rc, msg,stmt) \
  if (rc == 0){\
    rc = stmt; \
    if (rc){\
      LOGERROR2("%s result:%d", msg, rc); \
    }else{\
      LOGINFO1("%s ok",msg);\
    }\
  }

extern int logLevel;
extern FILE *logFile;

/**
 * By default, logging output is sent to cout. You can also call
 * firelog_init() to send log output to logfile specified by path.
 * If you do call firelog_init(), you must also call firelog_destroy().
 * @param path to logfile
 * @param level logging level
 * @return 0 for success
 */
int firelog_init(const char *path, int level = FIRELOG_WARN);

/**
 * Release resources allocated by firelog_init().
 */
int firelog_destroy();

/**
 * Change logging level.
 * @param newLevel E.g., FIRELOG_TRACE
 * @return former logging level
 */
int firelog_level(int newLevel);

/**
 * Return last message
 * @param level logging level
 * @return Last message logged for given level
 */
const char * firelog_lastMessage(int level);

/**
 * (INTERNAL)
 * Clear cache of logging messages
 */
void firelog_lastMessageClear();

/**
 * (INTERNAL)
 * Do not call main logging function directly.
 * Use logging defines instead.
 */
void firelog(const char *fmt, int level, const void * value1, const void * value2, const void * value3);

#ifdef __cplusplus
}
#endif
#endif
