# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "JSEP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_JSEP=1)

  file(GLOB_RECURSE onnxruntime_providers_js_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/js/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/js/*.cc"
  )
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_js_contrib_ops_cc_srcs})
    list(APPEND onnxruntime_providers_js_cc_srcs ${onnxruntime_js_contrib_ops_cc_srcs})
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_providers_js_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_js ${onnxruntime_providers_js_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_js
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers Boost::mp11 Eigen3::Eigen
  )

  add_dependencies(onnxruntime_providers_js ${onnxruntime_EXTERNAL_DEPENDENCIES})
