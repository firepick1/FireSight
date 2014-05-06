#ifndef FIRELOG_H
#define FIRELOG_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#ifdef _MSC_VER
#include "winjunk.hpp"
#else
#define CLASS_DECLSPEC
#endif

#define FIRELOG_ERROR 0
#define FIRELOG_WARN 1
#define FIRELOG_INFO 2
#define FIRELOG_DEBUG 3
#define FIRELOG_TRACE 4

#define FIRELOG_LINESIZE 200
#define FIRELOG4(lvl,fmt,v1,v2,v3,v4) if (logLevel>=lvl){\
	char log_buf[FIRELOG_LINESIZE]; \
	snprintf(log_buf,sizeof(log_buf),fmt,v1,v2,v3,v4); \
	firelog(log_buf,lvl);\
	}
#define FIRELOG3(lvl,fmt,v1,v2,v3) FIRELOG4(lvl,fmt,v1,v2,v3,"")
#define FIRELOG2(lvl,fmt,v1,v2,v3) FIRELOG4(lvl,fmt,v1,v2,"","")
#define FIRELOG1(lvl,fmt,v1,v2,v3) FIRELOG4(lvl,fmt,v1,"","","")

#define LOGERROR4(fmt,v1,v2,v3,v4) FIRELOG4(FIRELOG_ERROR,fmt,v1,v2,v3,v4)
#define LOGERROR3(fmt,v1,v2,v3) FIRELOG4(FIRELOG_ERROR,fmt,v1,v2,v3,"")
#define LOGERROR2(fmt,v1,v2) FIRELOG4(FIRELOG_ERROR,fmt,v1,v2,"","")
#define LOGERROR1(fmt,v1) FIRELOG4(FIRELOG_ERROR,fmt,v1,"","","")
#define LOGERROR(fmt) FIRELOG4(FIRELOG_ERROR,fmt,"","","","")

#define LOGWARN4(fmt,v1,v2,v3,v4) FIRELOG4(FIRELOG_WARN,fmt,v1,v2,v3,v4)
#define LOGWARN3(fmt,v1,v2,v3) FIRELOG4(FIRELOG_WARN,fmt,v1,v2,v3,"")
#define LOGWARN2(fmt,v1,v2) FIRELOG4(FIRELOG_WARN,fmt,v1,v2,"","")
#define LOGWARN1(fmt,v1) FIRELOG4(FIRELOG_WARN,fmt,v1,"","","")
#define LOGWARN(fmt) FIRELOG4(FIRELOG_WARN,fmt,"","","","")

#define LOGDEBUG4(fmt,v1,v2,v3,v4) FIRELOG4(FIRELOG_DEBUG,fmt,v1,v2,v3,v4)
#define LOGDEBUG3(fmt,v1,v2,v3) FIRELOG4(FIRELOG_DEBUG,fmt,v1,v2,v3,"")
#define LOGDEBUG2(fmt,v1,v2) FIRELOG4(FIRELOG_DEBUG,fmt,v1,v2,"","")
#define LOGDEBUG1(fmt,v1) FIRELOG4(FIRELOG_DEBUG,fmt,v1,"","","")
#define LOGDEBUG(fmt) FIRELOG4(FIRELOG_DEBUG,fmt,"","","","")

#define LOGINFO4(fmt,v1,v2,v3,v4) FIRELOG4(FIRELOG_INFO,fmt,v1,v2,v3,v4)
#define LOGINFO3(fmt,v1,v2,v3) FIRELOG4(FIRELOG_INFO,fmt,v1,v2,v3,"")
#define LOGINFO2(fmt,v1,v2) FIRELOG4(FIRELOG_INFO,fmt,v1,v2,"","")
#define LOGINFO1(fmt,v1) FIRELOG4(FIRELOG_INFO,fmt,v1,"","","")
#define LOGINFO(fmt) FIRELOG4(FIRELOG_INFO,fmt,"","","","")

#define LOGTRACE4(fmt,v1,v2,v3,v4) FIRELOG4(FIRELOG_TRACE,fmt,v1,v2,v3,v4)
#define LOGTRACE3(fmt,v1,v2,v3) FIRELOG4(FIRELOG_TRACE,fmt,v1,v2,v3,"")
#define LOGTRACE2(fmt,v1,v2) FIRELOG4(FIRELOG_TRACE,fmt,v1,v2,"","")
#define LOGTRACE1(fmt,v1) FIRELOG4(FIRELOG_TRACE,fmt,v1,"","","")
#define LOGTRACE(fmt) FIRELOG4(FIRELOG_TRACE,fmt,"","","","")

#define LOGRC(rc, msg,stmt) \
  if (rc == 0){\
    rc = stmt; \
    if (rc){\
      LOGERROR2("%s result:%d", msg, rc); \
    }else{\
      LOGINFO1("%s ok",msg);\
    }\
  }

extern CLASS_DECLSPEC int logLevel;
extern CLASS_DECLSPEC FILE *logFile;

/**
 * By default, logging output is sent to cout. You can also call
 * firelog_init() to send log output to logfile specified by path.
 * If you do call firelog_init(), you must also call firelog_destroy().
 * @param path to logfile
 * @param level logging level
 * @return 0 for success
 */
CLASS_DECLSPEC int firelog_init(const char *path, int level);

/**
 * Release resources allocated by firelog_init().
 */
CLASS_DECLSPEC int firelog_destroy();

/**
 * Change logging level.
 * @param newLevel E.g., FIRELOG_TRACE
 * @return former logging level
 */
CLASS_DECLSPEC int firelog_level(int newLevel);

/**
 * Return last message
 * @param level logging level
 * @return Last message logged for given level
 */
CLASS_DECLSPEC const char * firelog_lastMessage(int level);

/**
 * Include thread id on each line logged
 * @param show thread id if true
 */
void firelog_show_thread_id(int show);

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
CLASS_DECLSPEC void firelog(const char *msg, int level);

#ifdef __cplusplus
}
#endif
#endif
