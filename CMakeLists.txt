cmake_minimum_required(VERSION 2.8)

project( firesight )
SET(COMPILE_DEFINITIONS -Werror)
SET(PROJECT_VERSION_MAJOR 0)
SET(PROJECT_VERSION_MINOR 4)
set(PROJECT_VERSION_PATCH 6)
set(PROJECT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_FILE_OFFSET_BITS=64")

include (InstallRequiredSystemLibraries)
SET(CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
SET(CPACK_PACKAGE_GENERATED_WARNING "GENERATED FILE!!! DO NOT CHANGE!!!")
SET(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
include (CPack)

SET(BUILD_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
SET(LIBRARY_OUTPUT_PATH "${BUILD_TARGET_DIR}")
SET(EXECUTABLE_OUTPUT_PATH "${BUILD_TARGET_DIR}")
SET(RUNTIME_OUTPUT_NAME firesight)

configure_file (
	"${PROJECT_SOURCE_DIR}/version.h.in"
	"${BUILD_TARGET_DIR}/version.h"
	)

find_package( OpenCV REQUIRED )

include_directories( ${PROJECT_SOURCE_DIR} ${BUILD_TARGET_DIR} )
link_directories( ${BUILD_TARGET_DIR} )

add_library(_firesight SHARED 
	dft.cpp 
	FireLog.cpp 
	generated_code.cpp
	HoleRecognizer.cpp 
	jo_util.cpp 
	MatchedRegion.cpp 
	MatUtil.cpp
	Model.cpp 
	MSER.cpp 
	Pipeline.cpp 
	proto.cpp 
	warpRing.cpp
	)
target_link_libraries(_firesight libjansson.so ${OpenCV_LIBS} )
set_target_properties(_firesight PROPERTIES VERSION ${PROJECT_VERSION_STRING} SOVERSION ${PROJECT_VERSION_MAJOR})

add_executable(gencode gencode.cpp)

add_executable(firesight FireSight.cpp)
target_link_libraries(firesight libjansson.so lib_firesight.so ${OpenCV_LIBS} )

add_executable(test test/test.cpp)
target_link_libraries(test libjansson.so lib_firesight.so ${OpenCV_LIBS} )

INSTALL(TARGETS _firesight DESTINATION lib)
INSTALL(TARGETS firesight DESTINATION bin)
INSTALL(FILES FireLog.h FireSight.hpp DESTINATION include)

