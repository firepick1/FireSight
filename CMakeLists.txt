cmake_minimum_required(VERSION 2.8)

project( firesight )

SET(PROJECT_VERSION_MAJOR 0)
SET(PROJECT_VERSION_MINOR 8)
set(PROJECT_VERSION_PATCH 6)
set(PROJECT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

IF(WIN32)
	MESSAGE(STATUS "Detecting WINDOWS build")
	list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
	# By default, do not warn when built on machines using only VS Express:
	IF(NOT DEFINED CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS)
		SET(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS ON)
	ENDIF()
	SET(JANSSON_DIR "${CMAKE_SOURCE_DIR}/jansson/msvc" CACHE STRING "Jansson directory")
	SET(JANSSON_INCLUDE_DIRS "${JANSSON_DIR}/include")
	SET(JANSSON_LIB "${JANSSON_DIR}/lib/Debug/jansson_d.lib")
	SET(JANSSON_DLL "${JANSSON_DIR}/bin/Debug/jansson_d.dll")
	MESSAGE(STATUS "JANSSON_DLL:${JANSSON_DLL}")
	MESSAGE(STATUS "JANSSON_LIB:${JANSSON_LIB}")
	SET(FIRESIGHT_LIB "_firesight.lib")
	SET(FIRESIGHT_DLL "_firesight.dll")
ELSE(WIN32)
	MESSAGE(STATUS "Detecting LINUX build")
	SET(JANSSON_LIB "libjansson.so")
	SET(FIRESIGHT_LIB "lib_firesight.so")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")
	SET(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -g")
ENDIF(WIN32)

SET(COMPILE_DEFINITIONS -Werror)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_FILE_OFFSET_BITS=64")

include (InstallRequiredSystemLibraries)
SET(CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
SET(CPACK_PACKAGE_GENERATED_WARNING "GENERATED FILE!!! DO NOT CHANGE!!!")
SET(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
include (CPack)

SET(BUILD_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target" CACHE STRING "Put all binary output in target")
SET(LIBRARY_OUTPUT_PATH "${BUILD_TARGET_DIR}")
SET(EXECUTABLE_OUTPUT_PATH "${BUILD_TARGET_DIR}")
SET(RUNTIME_OUTPUT_NAME firesight)

configure_file (
	"${PROJECT_SOURCE_DIR}/version.h.in"
	"${BUILD_TARGET_DIR}/version.h"
	)

find_package( OpenCV REQUIRED )
if (OpenCV_LIBS) 
	MESSAGE(STATUS "OpenCV_LIBS:${OpenCV_LIBS}")
else(OpenCV_LIBS)
	MESSAGE("find_package(OpenCV) failed => OpenCV_LIBS is not defined")
endif(OpenCV_LIBS)

include_directories( ${PROJECT_SOURCE_DIR} ${BUILD_TARGET_DIR} ${JANSSON_INCLUDE_DIRS})
link_directories( ${BUILD_TARGET_DIR} )

file(GLOB FIRESIGHT_LIB_FILES
	bgSub.cpp
	calcHist.cpp
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

add_library(_firesight SHARED ${FIRESIGHT_LIB_FILES})
target_link_libraries(_firesight ${JANSSON_LIB} ${OpenCV_LIBS} )
set_target_properties(_firesight PROPERTIES 
		VERSION ${PROJECT_VERSION_STRING} 
		SOVERSION ${PROJECT_VERSION_MAJOR}
		)
if(WIN32)
	add_custom_command(TARGET _firesight PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${JANSSON_LIB}"      
        "${PROJECT_SOURCE_DIR}/target/Debug/" 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${JANSSON_DLL}"      
        "${PROJECT_SOURCE_DIR}/target/" 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${JANSSON_DLL}"      
        "${PROJECT_SOURCE_DIR}/target/Debug/" 
		)
	add_custom_command(TARGET _firesight POST_BUILD    
		COMMAND cd ${PROJECT_SOURCE_DIR}/target/Debug
		COMMAND copy ${FIRESIGHT_DLL} ..
		COMMAND copy ${FIRESIGHT_DLL} ..
		COMMAND cd ${PROJECT_SOURCE_DIR}/msvc
		)
endif(WIN32)
	
add_executable(gencode gencode.cpp)

add_executable(firesight FireSight.cpp)
add_dependencies(firesight _firesight)
target_link_libraries(firesight ${JANSSON_LIB} ${FIRESIGHT_LIB} ${OpenCV_LIBS} )
if(WIN32)
	add_custom_command(TARGET firesight POST_BUILD    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${PROJECT_SOURCE_DIR}/target/Debug/firesight.exe"      
        "${PROJECT_SOURCE_DIR}/target/firesight.exe" 
				)
endif(WIN32)

add_executable(test 
	test/test_warpAffine.cpp 
	test/test_warpRing.cpp 
	test/test_regionKeypoint.cpp 
	test/test_matMaxima.cpp 
	test/test_jo_util.cpp
	test/test.cpp)
add_dependencies(test _firesight)
target_link_libraries(test ${JANSSON_LIB} ${FIRESIGHT_LIB} ${OpenCV_LIBS} )
if(WIN32)
	add_custom_command(TARGET test POST_BUILD    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${PROJECT_SOURCE_DIR}/target/Debug/test.exe"      
        "${PROJECT_SOURCE_DIR}/target/test.exe" 
				)
endif(WIN32)

INSTALL(TARGETS _firesight DESTINATION lib)
INSTALL(TARGETS firesight DESTINATION bin)
INSTALL(FILES FireLog.h FireSight.hpp DESTINATION include)

