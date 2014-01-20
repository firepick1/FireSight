cmake_minimum_required(VERSION 2.8)

project( FireSight )
SET(COMPILE_DEFINITIONS -Werror)
SET(firesight_VERSION_MAJOR 0)
SET(firesight_VERSION_MINOR 1)
set(firesight_VERSION_PATCH 1)
set(firesight_VERSION_STRING ${firesight_VERSION_MAJOR}.${firesight_VERSION_MINOR}.${firesight_VERSION_PATCH})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_FILE_OFFSET_BITS=64")

include (InstallRequiredSystemLibraries)
SET(CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
SET(CPACK_PACKAGE_GENERATED_WARNING "GENERATED FILE!!! DO NOT CHANGE!!!")
SET(CPACK_PACKAGE_VERSION_MAJOR "${firesight_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${firesight_VERSION_MINOR}")
include (CPack)

configure_file (
	"${PROJECT_SOURCE_DIR}/version.h.in"
	"${PROJECT_BINARY_DIR}/version.h"
	)

find_package( OpenCV REQUIRED )

add_library(firesight SHARED FireLog.c FireSight.cpp )
target_link_libraries(firesight ${OpenCV_LIBS} )
set_target_properties(firesight PROPERTIES VERSION ${firesight_VERSION_STRING} SOVERSION ${firesight_VERSION_MAJOR})

add_executable(firesightmain FireSightMain.cpp )
target_link_libraries(firesightmain ${FireSight_BINARY_DIR}/libfiresight.so ${OpenCV_LIBS} )

INSTALL(TARGETS firesight DESTINATION lib)
INSTALL(FILES FireSight.hpp DESTINATION include)

