cmake_minimum_required(VERSION 3.16)
project(drobotics_pi05_worker CXX)

# Build on the S600 board against the D-Robotics LLM SDK v2.0.0:
#   cmake -S apx-drobotics/worker -B apx-drobotics/build/worker \
#     -DOELLM_ROOT=/opt/d-robotics-llm/2.0.0/D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600
#   cmake --build apx-drobotics/build/worker
# apx-drobotics/scripts/deploy.sh runs these steps.

set(OELLM_ROOT "" CACHE PATH "Path to the SDK's oellm_runtime_s600 directory")
if(NOT EXISTS "${OELLM_ROOT}/include/oellm_runtime_basic/oellm_runtime.h")
  message(FATAL_ERROR "OELLM_ROOT must point at oellm_runtime_s600; got '${OELLM_ROOT}'")
endif()

set(PI0_DIST "${OELLM_ROOT}/examples/vla_demo/pi0/dist")
set(OELLM_LIB "${OELLM_ROOT}/lib")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(drobotics_pi05_worker pi05_worker.cc)
target_compile_options(drobotics_pi05_worker PRIVATE -Wall -Wextra)
target_include_directories(drobotics_pi05_worker PRIVATE
  "${OELLM_ROOT}/include"
  "${OELLM_ROOT}/3rdparty")
# libvla_pi0_sdk registers the "Pi0Sdk" runtime type from a static initializer,
# so it must stay linked even though no symbol in this file references it.
# libvla_sdk.so in SDK v2.0.0 references oellm::component::Tokenizer::~Tokenizer(),
# which no SDK library exports. The official demo runs with it unresolved, so the
# link allows it rather than failing; the loader binds functions lazily.
target_link_libraries(drobotics_pi05_worker PRIVATE
  -Wl,--no-as-needed
  "${PI0_DIST}/libvla_pi0_sdk.so"
  "${OELLM_LIB}/libvla_sdk.so"
  "${OELLM_LIB}/liboellm_runtime.so"
  -Wl,--as-needed
  "-Wl,-rpath-link,${OELLM_LIB}"
  -Wl,--allow-shlib-undefined
  -Wl,-z,lazy
  dl pthread)
# The SDK libraries load further SDK libraries (libchat_template.so and others)
# without a search path of their own. RUNPATH covers only this executable's
# direct dependencies, so emit an RPATH, which also covers theirs, as the
# official demo does.
target_link_options(drobotics_pi05_worker PRIVATE -Wl,--disable-new-dtags)
set_target_properties(drobotics_pi05_worker PROPERTIES
  BUILD_RPATH "${OELLM_LIB};${PI0_DIST}")
