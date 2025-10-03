cmake_minimum_required(VERSION 3.28)

project(RenderGraph LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
include(FetchContent)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Disable GLFW examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Disable GLFW tests")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Disable GLFW docs")
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 4afa227a056681d2628894b0893527bf69496a41
)

FetchContent_Declare(
  thread-pool
  GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
  GIT_TAG        cabb3df5876c9a6824b07fcb0ff73d4a0e506ca0
)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
)

FetchContent_Declare(
  vk-bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
  GIT_TAG        v1.3.302
)

set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk
  GIT_TAG e767d0ee782709f2bd2df148927ddab02f2dbfe4
)

FetchContent_Declare(
  vulkan-memory-allocator
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.2.1
)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(glfw glm thread-pool vk-bootstrap volk vulkan-memory-allocator googletest)

# need to build as module
add_library(glm-module)
target_sources(glm-module PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS ${glm_SOURCE_DIR}/glm
    FILES ${glm_SOURCE_DIR}/glm/glm.cppm
)
target_link_libraries(glm-module PUBLIC glm::glm)
target_compile_features(glm-module PRIVATE cxx_std_23)
if (MSVC)
  #enable multiple cores compilation for VS
  target_compile_options(glm-module PRIVATE "/MP")
endif()
set_target_properties(glm-module PROPERTIES CXX_SCAN_FOR_MODULES ON)
# MSVC recognizes only .ixx, so need to specify this for .cppm
set_source_files_properties(${glm_SOURCE_DIR}/glm/glm.cppm PROPERTIES
    COMPILE_OPTIONS "/interface"
)

##############################

file(GLOB_RECURSE app_interface_vulkan "include/*.ixx")
file(GLOB_RECURSE app_implementation_vulkan "src/*.c*")
source_group("include" FILES ${app_interface_vulkan})
source_group("src" FILES ${app_implementation_vulkan})

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(${PROJECT_NAME})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME} PUBLIC
                               FILE_SET CXX_MODULES
                               FILES ${app_interface_vulkan}
                               PRIVATE ${app_implementation_vulkan})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)
###############################

if (MSVC)
  #enable multiple cores compilation for VS
  target_compile_options(${PROJECT_NAME} PRIVATE "/MP")
endif()

find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
# don't link loader explicitly
#target_link_libraries(${PROJECT_NAME} PRIVATE ${Vulkan_LIBRARIES})


target_include_directories(RenderGraph
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${glfw_SOURCE_DIR}/include
        ${vk-bootstrap_SOURCE_DIR}/src
        ${volk_SOURCE_DIR}
        ${vulkan-memory-allocator_SOURCE_DIR}/include
    PRIVATE
        ${thread-pool_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PUBLIC glfw vk-bootstrap::vk-bootstrap volk::volk GPUOpen::VulkanMemoryAllocator glm-module)
target_compile_definitions(${PROJECT_NAME} PRIVATE VK_NO_PROTOTYPES GLFW_INCLUDE_VULKAN)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_SCAN_FOR_MODULES ON)

enable_testing()

file(GLOB_RECURSE tests_source "tests/*.c*")
add_executable(${PROJECT_NAME}Tests ${tests_source})
target_link_libraries(${PROJECT_NAME}Tests PRIVATE gtest_main ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME}Tests PRIVATE cxx_std_23)
set_target_properties(${PROJECT_NAME}Tests PROPERTIES CXX_SCAN_FOR_MODULES ON)
if (MSVC)
  #enable multiple cores compilation for VS
  target_compile_options(${PROJECT_NAME}Tests PRIVATE "/MP")
endif()

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}Tests)