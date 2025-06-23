// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <iosfwd>

#include "core/providers/webgpu/webgpu_external_header.h"

#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;

enum class BufferCacheMode {
  Disabled,
  LazyRelease,
  Simple,
  Bucket,
  DynamicBucket
};
std::ostream& operator<<(std::ostream& os, BufferCacheMode mode);

//
// IBufferCacheManager is an interface for buffer cache management.
//
// By implementing this interface, we can have different buffer cache management strategies.
// Currently, we have 4 strategies:
// - Disabled: no cache. always allocate a new buffer and release it immediately after use.
// - LazyRelease: no cache. the difference from Disabled is that it delays the release of buffers until the next refresh.
// - Simple: a simple cache that always keeps buffers. when a buffer is requested, it tries to find a buffer in the cache.
// - Bucket: a cache that keeps buffers in different buckets based on the buffer size, with a maximum number of buffers in each bucket.
// - DynamicBucket: a variation bucket cache that dynamically adjusts bucket sizes based on usage patterns in real-time requests and previous sessions.
class IBufferCacheManager {
 public:
  virtual ~IBufferCacheManager() = default;

  // calculate actual buffer size to allocate based on the requested size.
  virtual size_t CalculateBufferSize(size_t request_size) = 0;

  // return a buffer if available in cache. otherwise empty.
  virtual WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) = 0;

  // register a newly created buffer
  virtual void RegisterBuffer(WGPUBuffer buffer, size_t request_size) = 0;

  // release a buffer
  virtual void ReleaseBuffer(WGPUBuffer buffer) = 0;

  // when a stream refresh is requested
  virtual void OnRefresh() = 0;

  // Track start of session run
  virtual void OnRunStart() {}

  // Track end of session run and update memory patterns
  virtual void OnRunEnd() {}
};

//
// BufferManager manages operations on buffers.
//
class BufferManager {
 public:
  BufferManager(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode);

  void Upload(void* src, WGPUBuffer dst, size_t size);
  void MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size);
  WGPUBuffer Create(size_t size, wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);
  // Create a buffer mapped for writing.
  WGPUBuffer CreateUMA(size_t size, wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                                              wgpu::BufferUsage::CopyDst);
  void Release(WGPUBuffer buffer);
  void Download(WGPUBuffer src, void* dst, size_t size);
  void RefreshPendingBuffers();

  void OnRunStart() {
    if (storage_cache_) storage_cache_->OnRunStart();
    if (uniform_cache_) uniform_cache_->OnRunStart();
    if (query_resolve_cache_) query_resolve_cache_->OnRunStart();
    if (default_cache_) default_cache_->OnRunStart();
  }

  void OnRunEnd() {
    if (storage_cache_) storage_cache_->OnRunEnd();
    if (uniform_cache_) uniform_cache_->OnRunEnd();
    if (query_resolve_cache_) query_resolve_cache_->OnRunEnd();
    if (default_cache_) default_cache_->OnRunEnd();
  }

 private:
  IBufferCacheManager& GetCacheManager(wgpu::BufferUsage usage) const;
  IBufferCacheManager& GetCacheManager(WGPUBuffer buffer) const;

  WebGpuContext& context_;
  std::unique_ptr<IBufferCacheManager> storage_cache_;
  std::unique_ptr<IBufferCacheManager> uniform_cache_;
  std::unique_ptr<IBufferCacheManager> query_resolve_cache_;
  std::unique_ptr<IBufferCacheManager> default_cache_;
};

class BufferManagerFactory {
 public:
  static std::unique_ptr<BufferManager> Create(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode);

 private:
  BufferManagerFactory() {}
};

// Structure to track memory usage patterns
struct MemoryUsagePattern {
  size_t request_size;
  size_t frequency;

  MemoryUsagePattern(size_t size = 0, size_t freq = 0)
      : request_size(size), frequency(freq) {}
};

// Base class for bucket cache managers that includes shared logging functionality
class BucketCacheManagerBase : public IBufferCacheManager {
protected:
  // Cache statistics tracking
  struct CacheStats {
    size_t hits{0};             // Number of cache hits
    size_t misses{0};           // Number of cache misses
    size_t total_requests{0};   // Total number of requests
    uint64_t miss_bytes{0};       // Total bytes missed
    uint64_t hit_bytes{0};        // Total bytes hit
    uint64_t total_requested_size{0};  // Total requested size across all requests
    uint64_t total_normalized_size{0}; // Total normalized size across all requests
  };

  // Memory metrics
  int64_t total_memory_{0};      // Current total allocated memory
  int64_t peak_memory_{0};       // Peak memory usage observed
  int64_t active_buffers_{0};    // Number of buffers currently in use
  int64_t total_buffers_{0};     // Total number of buffers (active + cached)

  // Session tracking
  int64_t session_id_{-1};         // Current session ID
  int64_t total_cache_hit_{0};     // Cache hit buffer size in current session
  int64_t total_cache_miss_{0};    // Cache miss buffer size in current session

  std::unordered_map<size_t, CacheStats> session_stats_;  // Stats per buffer size for current session
  std::ofstream cache_stats_file_;     // File for cache statistics
  std::ofstream memory_metrics_file_;   // File stream for logging metrics

  virtual void OpenFiles(const char* memory_metrics_filename, const char* cache_stats_filename);
  void LogCacheStats();
  void LogMemoryMetrics();
  void UpdateMemoryMetrics(bool is_allocation, size_t buffer_size, bool is_from_destructor = false, bool skip_active_buffers_update = false);

  BucketCacheManagerBase();
  virtual ~BucketCacheManagerBase();
};

}  // namespace webgpu
}  // namespace onnxruntime
