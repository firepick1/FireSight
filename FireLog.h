/*
FireLog.h https://github.com/firepick1/FirePick/wiki

Copyright (C) 2013  Karl Lew, <karl@firepick.org>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

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

int firelog_init(char *path, int level);
int firelog_destroy();
int firelog_level(int newLevel);
const char * firelog_lastMessage(int level);
void firelog_lastMessageClear();
void firelog(const char *fmt, int level, const void * value1, const void * value2, const void * value3);

#ifdef __cplusplus
}
#endif
#endif
