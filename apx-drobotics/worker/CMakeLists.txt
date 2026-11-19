cmake_minimum_required(VERSION 3.16)
project(drobotics_pi05_worker CXX)

# Build on the S600 board against the D-Robotics LLM SDK v2.0.0:
#   cmake -S apx-drobotics/worker -B apx-drobotics/build/worker \
#     -DOELLM_ROOT=/opt/d-robotics-llm/2.0.0/D-Robotics_LLM_2.0.0_S600_S100_SDK/oellm_runtime_s600
#   cmake --build apx-drobotics/build/worker
# See doc/tutorial-pi0.5-on-drobotics.md for the full deployment steps.

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
# Do not add -Wl,--allow-shlib-undefined. The first SDK v2.0.0 build had a dangling
# reference to oellm::component::Tokenizer::~Tokenizer() in libvla_sdk.so and needed
# it; the fixed build links cleanly. The flag is global, so it would hide any real
# unresolved symbol. After an SDK upgrade, run `ldd -r` on the built worker.
target_link_libraries(drobotics_pi05_worker PRIVATE
  -Wl,--no-as-needed
  "${PI0_DIST}/libvla_pi0_sdk.so"
  "${OELLM_LIB}/libvla_sdk.so"
  "${OELLM_LIB}/liboellm_runtime.so"
  -Wl,--as-needed
  "-Wl,-rpath-link,${OELLM_LIB}"
  -Wl,-z,lazy
  dl pthread)
# The SDK libraries load further SDK libraries (libchat_template.so and others)
# without a search path of their own. RUNPATH covers only this executable's
# direct dependencies, so emit an RPATH, which also covers theirs, as the
# official demo does.
target_link_options(drobotics_pi05_worker PRIVATE -Wl,--disable-new-dtags)
set_target_properties(drobotics_pi05_worker PROPERTIES
  BUILD_RPATH "${OELLM_LIB};${PI0_DIST}")
