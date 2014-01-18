/*
FireStep.cpp https://github.com/firepick1/FirePick/wiki

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

#include <sys/stat.h> 
#include <sys/types.h> 
#include <stdlib.h> 
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include "FireStep.h"
#include "FireLog.h"

pthread_t tidReader;

int fdrTinyG = -1;
int fdwTinyG = -1;

#define CMDMAX 255
#define WRITEBUFMAX 100
#define INBUFMAX 3000
#define JSONMAX 3000
char jsonBuf[JSONMAX+3]; // +nl, cr, EOS
int jsonLen = 0;
int jsonDepth = 0;

char inbuf[INBUFMAX+1]; // +EOS
int inbuflen = 0;
int inbufEmptyLine = 0;

static void * firestep_reader(void *arg);
static int firestep_writeCore(const char *buf, size_t bufsize);

static int callSystem(char *cmdbuf) {
  int rc = 0;
  rc = system(cmdbuf);
  if (rc == -1) {
    LOGERROR2("callSystem(%s) -> %d", cmdbuf, rc);
    return rc;
  }
  if (WIFEXITED(rc)) {
    if (WEXITSTATUS(rc)) {
      LOGERROR2("callSystem(%s) -> EXIT %d", cmdbuf, WEXITSTATUS(rc));
      return rc;
    }
  } else if (WIFSTOPPED(rc)) {
      LOGERROR2("callSystem(%s) -> STOPPED %d", cmdbuf, WSTOPSIG(rc));
      return rc;
  } else if (WIFSIGNALED(rc)) {
      LOGERROR2("callSystem(%s) -> SIGNALED %d", cmdbuf, WTERMSIG(rc));
      return rc;
  }
  LOGINFO1("callSystem(%s)", cmdbuf);
  return 0;
}

static int firestep_config() {
  int rc = 0;
  char cmdbuf[CMDMAX+1];

  LOGINFO("Configure TinyG");

  sprintf(cmdbuf, "{\"jv\":5,\"sv\":2, \"tv\":0}\n");
	rc = firestep_writeCore(cmdbuf, strlen(cmdbuf));
  if (rc) { return rc; }

  sprintf(cmdbuf, "{\"sr\":{\"mpox\":t,\"mpoy\":t,\"mpoz\":t,\"vel\":t,\"stat\":t}}\n");
	rc = firestep_writeCore(cmdbuf, strlen(cmdbuf));
  if (rc) { return rc; }

	const char *yInit = "{\"y\":{\"am\":1,\"vm\":35000,\"fr\":40000,\"tm\":400,\"jm\":20000000000,\"jh\":40000000000,\"jd\":0.050,\"sn\":3,\"sx\":0,\"sv\":3000,\"lv\":1000,\"lb\":2,\"zb\":1}";
	sprintf(cmdbuf, "%s", yInit);
	rc = firestep_writeCore(cmdbuf, strlen(cmdbuf));
  if (rc) { return rc; }

	return rc;
}

int firestep_init(){
  if (fdrTinyG >= 0) {
    return 0; // already started
  }

  const char * path = "/dev/ttyUSB0";
  char cmdbuf[CMDMAX+1];
	int rc;

  sprintf(cmdbuf, "stty 115200 -F %s", path);
  rc = callSystem(cmdbuf);
  if (rc) { return rc; }

  sprintf(cmdbuf, "stty 1400:4:1cb2:a00:3:1c:7f:15:4:1:1:0:11:13:1a:0:12:f:17:16:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0 -F %s", path);
  rc = callSystem(cmdbuf);
  if (rc) { return rc; }

  fdwTinyG = fdrTinyG = open(path, O_RDWR | O_ASYNC | O_NONBLOCK);
  if (fdrTinyG < 0) {
    rc = errno;
    LOGERROR2("Cannot open %s (errno %d)", path, rc);
    return rc;
  }
  LOGINFO1("firestep_init %s (open for write) ", path);

  LOGRC(rc, "pthread_create(firestep_reader) -> ", pthread_create(&tidReader, NULL, &firestep_reader, NULL));

	//firestep_config();

  return rc;
}

const char * firestep_json() {
	int wait = 0;
	while (jsonDepth > 0) {
		LOGDEBUG1("firestep_json() waiting for JSON %d", wait++);
		sched_yield(); // wait for completion
		if (wait > 10) {
			LOGERROR("firestep_json() unterminated JSON");
			return "{\"error\":\"unterminated JSON\"}";
		}
	}
	jsonBuf[jsonLen] = 0;
	if (jsonLen > 0) {
		jsonBuf[jsonLen++] = '\n';
		jsonBuf[jsonLen++] = 0;
	}
	return jsonBuf;
}

void firestep_destroy() {
  LOGINFO("firestep_destroy");
  if (fdrTinyG >= 0) {
    close(fdrTinyG);
    fdrTinyG = -1;
  }
}

static int firestep_writeCore(const char *buf, size_t bufsize) {
  char message[WRITEBUFMAX+4];
  if (bufsize > WRITEBUFMAX) {
    memcpy(message, buf, WRITEBUFMAX);
    message[WRITEBUFMAX] = '.'; 
    message[WRITEBUFMAX+1] = '.'; 
    message[WRITEBUFMAX+2] = '.'; 
    message[WRITEBUFMAX+3] = 0;
  } else {
    memcpy(message, buf, bufsize);
    message[bufsize] = 0;
  }
  char *s;
  for (s = message; *s; s++) {
    switch (*s) {
      case '\n':
      case '\r':
        *s = ' ';
				break;
    }
  }
  LOGDEBUG1("firestep_write %s start", message);
  ssize_t rc = write(fdwTinyG, buf, bufsize);
  if (rc == bufsize) {
    LOGINFO2("firestep_write %s (%ldB)", message, bufsize);
  } else {
    LOGERROR2("firestep_write %s -> [%ld]", message, rc);
  }
	return rc < 0 ? rc : 0;
}

int firestep_write(const char *buf, size_t bufsize) {
  if (strncmp("config", buf, 6) == 0) {
	  firestep_config();
	} else {
		return firestep_writeCore(buf, bufsize);
	}
}

// Add the given character to jsonBuf if it is the inner part of the json response 
#define ADD_JSON(c) \
			if (jsonDepth > 1) {\
				jsonBuf[jsonLen++] = c; \
				if (jsonLen >= JSONMAX) { \
					LOGWARN1("Maximum JSON length is %d", JSONMAX); \
					return 0; \
				} \
				jsonBuf[jsonLen] = 0;\
			}

static int firestep_readchar(int c) {
	switch (c) {
		case EOF:
      inbuf[inbuflen] = 0;
      inbuflen = 0;
      LOGERROR1("firestep_readchar %s[EOF]", inbuf);
      return 0;
		case '\n':
      inbuf[inbuflen] = 0;
      if (inbuflen) { // discard blank lines
				if (strncmp("{\"sr\"",inbuf, 5) == 0) {
					LOGDEBUG2("firestep_readchar %s (%dB)", inbuf, inbuflen);
				} else {
					LOGINFO2("firestep_readchar %s (%dB)", inbuf, inbuflen);
				}
      } else {
        inbufEmptyLine++;
				if (inbufEmptyLine % 1000 == 0) {
					LOGWARN1("firestep_readchar skipped %ld blank lines", inbufEmptyLine);
				}
			}
      inbuflen = 0;
			break;
		case '\r':
      // skip
			break;
		case 'a': case 'A':
		case 'b': case 'B':
		case 'c': case 'C':
		case 'd': case 'D':
		case 'e': case 'E':
		case 'f': case 'F':
		case 'g': case 'G':
		case 'h': case 'H':
		case 'i': case 'I':
		case 'j': case 'J':
		case 'k': case 'K':
		case 'l': case 'L':
		case 'm': case 'M':
		case 'n': case 'N':
		case 'o': case 'O':
		case 'p': case 'P':
		case 'q': case 'Q':
		case 'r': case 'R':
		case 's': case 'S':
		case 't': case 'T':
		case 'u': case 'U':
		case 'v': case 'V':
		case 'w': case 'W':
		case 'x': case 'X':
		case 'y': case 'Y':
		case 'z': case 'Z':
		case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
		case '.': case '-': case '_': case '/':
		case '{': case '}':
		case '(': case ')':
		case '[': case ']':
		case '<': case '>':
		case '"': case '\'': case ':': case ',':
		case ' ': case '\t':
			if (c == '{') {
				if (jsonDepth++ <= 0) {
					jsonLen = 0;
				}
				ADD_JSON(c);
			} else if (c == '}') {
				ADD_JSON(c);
				if (--jsonDepth < 0) {
					LOGWARN1("Invalid JSON %s", jsonBuf);
					return 0;
				}
			} else {
				ADD_JSON(c);
			}
      if (inbuflen >= INBUFMAX) {
				inbuf[INBUFMAX] = 0;
        LOGERROR1("firestep_readchar overflow %s", inbuf);
				break;
      } else {
        inbuf[inbuflen] = c;
        inbuflen++;
				LOGTRACE2("firestep_readchar %x %c", (int) c, (int) c);
      }
			break;
		default:
		  // discard unexpected character (probably wrong baud rate)
			LOGTRACE2("firestep_readchar %x ?", (int) c, (int) c);
		  break;
	}
	return 1;
}

static void * firestep_reader(void *arg) {
#define READBUFLEN 100
  char readbuf[READBUFLEN];

  LOGINFO("firestep_reader listening...");

	if (fdrTinyG >= 0) {
		char c;
		char loop = true;
		while (loop) {
			int rc = read(fdrTinyG, readbuf, READBUFLEN);
			if (rc < 0) {
				if (errno == EAGAIN) {
					sched_yield();
					continue;
				}
				LOGERROR2("firestep_reader %s [ERRNO:%d]", inbuf, errno);
				break;
			}
			if (rc == 0) {
				sched_yield(); // nothing available to read
				continue;
			} else  {
			  int i;
				for (i = 0; i < rc; i++) {
					if (!firestep_readchar(readbuf[i])) {
						loop = false;
						break;
					}
				}
			}
		}
	}
  
  LOGINFO("firestep_reader exit");
  return NULL;
}

