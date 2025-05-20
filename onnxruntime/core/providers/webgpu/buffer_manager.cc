// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>

namespace onnxruntime {
namespace webgpu {

namespace {
// Buffer cache configuration.
// Cache configuration: Buffers that haven't been used for BUFFER_TIMEOUT milliseconds will be released
constexpr std::chrono::milliseconds BUFFER_TIMEOUT{1000};
constexpr const char* METRICS_FILE = "webgpu_memory_metrics_1000ms.csv";
constexpr const char* METRICS_HEADER = "Timestamp,TotalMemory(MB),PeakMemory(MB),ActiveBuffers,TotalBuffers,TimeoutMs\n";

struct CachedBuffer {
  WGPUBuffer buffer;
  std::chrono::steady_clock::time_point last_used;
};

constexpr size_t
NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}

void EnforceBufferUnmapped(WebGpuContext& context, WGPUBuffer buffer) {
  if (context.ValidationMode() > ValidationMode::Basic) {
    ORT_ENFORCE(wgpuBufferGetMapState(buffer) == WGPUBufferMapState_Unmapped, "Buffer is still mapped.");
  }
}

}  // namespace

class DisabledCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    // always return empty buffer
    return nullptr;
  }
  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }
  void ReleaseBuffer(WGPUBuffer buffer) override {
    wgpuBufferRelease(buffer);
  }

  void OnRefresh() override {
    // no-op
  }
};

class LazyReleaseCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh() override {
    Release();
    pending_buffers_.clear();
  }

 public:
  ~LazyReleaseCacheManager() {
    Release();
  }

 protected:
  void Release() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
  }

  std::vector<WGPUBuffer> pending_buffers_;
};

class SimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh() override {
    for (auto& buffer : pending_buffers_) {
      buffers_[static_cast<size_t>(wgpuBufferGetSize(buffer))].emplace_back(buffer);
    }
    pending_buffers_.clear();
  }

 public:
  ~SimpleCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<WGPUBuffer> pending_buffers_;
};

class BucketCacheManager : public IBufferCacheManager {
 private:
  // Constants for dynamic bucket management
  static constexpr size_t MIN_BUCKET_SIZE = 64;        // Minimum bucket size (must be multiple of 16)
  static constexpr size_t MAX_BUCKET_COUNT = 500;      // Maximum number of buckets to prevent unbounded growth
  static constexpr size_t INITIAL_BUCKET_LIMIT = 100;  // Initial limit for new buckets

  // Memory metrics
  int64_t total_memory_{0};     // Current total allocated memory
  int64_t peak_memory_{0};      // Peak memory usage observed
  int64_t active_buffers_{0};   // Number of buffers currently in use
  int64_t total_buffers_{0};    // Total number of buffers (active + cached)
  std::ofstream metrics_file_;  // File stream for logging metrics
 public:
  BucketCacheManager() {
    Initialize();
    OpenMetricsFile();
  }
  BucketCacheManager(std::unordered_map<size_t, size_t>&& buckets_limit) : buckets_limit_{buckets_limit} {
    Initialize();
    OpenMetricsFile();
  }

  ~BucketCacheManager() {
    // Release remaining buffers
    for (auto& pair : buckets_) {
      for (auto& cached_buffer : pair.second) {
        if (cached_buffer.buffer != nullptr) {
          UpdateMetrics(false, wgpuBufferGetSize(cached_buffer.buffer), true);
          wgpuBufferRelease(cached_buffer.buffer);
        }
      }
    }

    if (metrics_file_.is_open()) {
      metrics_file_.close();
    }
  }

  size_t CalculateBufferSize(size_t request_size) override {
    size_t bucket_size = NormalizeBufferSize(request_size);

    // Initialize a new bucket if we don't have one for this size
    if (buckets_.find(bucket_size) == buckets_.end() && buckets_.size() < MAX_BUCKET_COUNT) {
      buckets_.emplace(bucket_size, std::vector<CachedBuffer>());
      buckets_limit_.emplace(bucket_size, INITIAL_BUCKET_LIMIT);
    }

    return bucket_size;
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto cached_buffer = it->second.back();
      it->second.pop_back();
      // Buffer reactivated from cache
      UpdateMetrics(true, 0);
      return cached_buffer.buffer;
    }
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer buffer, size_t request_size) override {
    const auto buffer_size = wgpuBufferGetSize(buffer);
    UpdateMetrics(true, buffer_size);
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    if (buffer != nullptr) {
      const auto buffer_size = wgpuBufferGetSize(buffer);
      auto it = buckets_.find(buffer_size);
      if (it != buckets_.end() && it->second.size() < buckets_limit_[buffer_size]) {
        CachedBuffer cached_buffer{
            buffer,
            std::chrono::steady_clock::now()};
        it->second.push_back(cached_buffer);
        // Buffer moved to cache, still counts in total memory but not active
        UpdateMetrics(false, 0);
      } else {
        // Buffer truly released, update both counters
        UpdateMetrics(false, buffer_size);
        wgpuBufferRelease(buffer);
      }
    }
  }

  void OnRefresh() override {
    auto now = std::chrono::steady_clock::now();

    // Use a vector to store keys that need to be erased to avoid iterator invalidation
    std::vector<size_t> sizes_to_erase;

    // Clean up timed out buffers in each bucket
    for (auto& bucket_pair : buckets_) {
      const auto size = bucket_pair.first;
      auto& bucket_vec = bucket_pair.second;

      // Remove timed out buffers from the back
      while (!bucket_vec.empty()) {
        auto& cached_buffer = bucket_vec.back();

        // Check if buffer timeout has elapsed
        if (now - cached_buffer.last_used < BUFFER_TIMEOUT) {
          break;
        }

        // Release the buffer and update metrics
        if (cached_buffer.buffer != nullptr) {
          // Don't decrease active buffer count since it's already decreased in ReleaseBuffer when moving to cache
          // But we still need to update total memory
          UpdateMetrics(false, wgpuBufferGetSize(cached_buffer.buffer), false, false);
          wgpuBufferRelease(cached_buffer.buffer);
          cached_buffer.buffer = nullptr;  // Prevent potential double-free
        }

        bucket_vec.pop_back();
      }

      // If bucket is empty after cleanup, mark it for removal
      if (bucket_vec.empty()) {
        sizes_to_erase.push_back(size);
      }
    }

    // Remove empty buckets
    for (const auto& size : sizes_to_erase) {
      buckets_.erase(size);
      buckets_limit_.erase(size);
    }
  }

 protected:
  void Initialize() {
    buckets_.reserve(MAX_BUCKET_COUNT);
#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(buckets_limit_.size() <= MAX_BUCKET_COUNT, "Bucket limit sizes cannot be greater than MAX_BUCKET_COUNT.");
#endif
    for (const auto& pair : buckets_limit_) {
      buckets_.emplace(pair.first, std::vector<CachedBuffer>());
    }
  }

  void OpenMetricsFile() {
    metrics_file_.open(METRICS_FILE, std::ios::out | std::ios::trunc);
    if (metrics_file_.is_open()) {
      metrics_file_ << METRICS_HEADER;
      metrics_file_.flush();
    }
  }

  void LogMetrics() {
    if (!metrics_file_.is_open()) return;

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    metrics_file_ << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << ","
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(total_memory_) / (1024 * 1024) << ","  // Convert to MB
                  << static_cast<double>(peak_memory_) / (1024 * 1024) << ","
                  << active_buffers_ << ","
                  << total_buffers_ << ","
                  << BUFFER_TIMEOUT.count() << std::endl;
    metrics_file_.flush();
  }

  void UpdateMetrics(bool is_allocation, size_t buffer_size, bool is_from_destructor = false, bool update_active_buffers = true) {
    if (is_allocation) {
      total_memory_ += buffer_size;
      if (update_active_buffers) {
        active_buffers_++;
      }
      if (buffer_size > 0) {
        total_buffers_++;
      }
      peak_memory_ = std::max(peak_memory_, total_memory_);
    } else {
      total_memory_ -= buffer_size;
      if (update_active_buffers) {
        if (is_from_destructor) {
          active_buffers_ = std::max(active_buffers_ - 1, 0LL);
        } else {
          active_buffers_--;
        }
      }
      // Only decrement total_buffers when truly releasing a buffer (buffer_size > 0)
      // For cache operations where buffer_size = 0, we keep the total count
      if (buffer_size > 0) {
        total_buffers_--;
      }
    }
    LogMetrics();
  }
  std::unordered_map<size_t, size_t> buckets_limit_;
  std::unordered_map<size_t, std::vector<CachedBuffer>> buckets_;
};

std::unique_ptr<IBufferCacheManager> CreateBufferCacheManager(BufferCacheMode cache_mode) {
  switch (cache_mode) {
    case BufferCacheMode::Disabled:
      return std::make_unique<DisabledCacheManager>();
    case BufferCacheMode::LazyRelease:
      return std::make_unique<LazyReleaseCacheManager>();
    case BufferCacheMode::Simple:
      return std::make_unique<SimpleCacheManager>();
    case BufferCacheMode::Bucket:
      return std::make_unique<BucketCacheManager>();
    default:
      ORT_NOT_IMPLEMENTED("Unsupported buffer cache mode");
  }
}

std::ostream& operator<<(std::ostream& os, BufferCacheMode mode) {
  switch (mode) {
    case BufferCacheMode::Disabled:
      os << "Disabled";
      break;
    case BufferCacheMode::LazyRelease:
      os << "LazyRelease";
      break;
    case BufferCacheMode::Simple:
      os << "Simple";
      break;
    case BufferCacheMode::Bucket:
      os << "Bucket";
      break;
    default:
      os << "Unknown(" << static_cast<int>(mode) << ")";
  }
  return os;
}

BufferManager::BufferManager(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode)
    : context_{context},
      storage_cache_{CreateBufferCacheManager(storage_buffer_cache_mode)},
      uniform_cache_{CreateBufferCacheManager(uniform_buffer_cache_mode)},
      query_resolve_cache_{CreateBufferCacheManager(query_resolve_buffer_cache_mode)},
      default_cache_{CreateBufferCacheManager(BufferCacheMode::Disabled)} {
}

void BufferManager::Upload(void* src, WGPUBuffer dst, size_t size) {
  // If the buffer is mapped, we can directly write to it.
  void* mapped_data = wgpuBufferGetMappedRange(dst, 0, WGPU_WHOLE_MAP_SIZE);  // ensure the buffer is mapped
  if (mapped_data) {
    memcpy(mapped_data, src, size);
    wgpuBufferUnmap(dst);
    return;
  }

  // Otherwise, we need to use a staging buffer to upload data.
  auto buffer_size = NormalizeBufferSize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
  desc.mappedAtCreation = true;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  mapped_data = staging_buffer.GetMappedRange();
  memcpy(mapped_data, src, size);
  staging_buffer.Unmap();

  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(staging_buffer, 0, dst, 0, buffer_size);
  context_.Flush();
}

void BufferManager::MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) {
  ORT_ENFORCE(src != dst, "Source and destination buffers must be different.");
  EnforceBufferUnmapped(context_, src);
  EnforceBufferUnmapped(context_, dst);

  auto buffer_size = NormalizeBufferSize(size);
  auto src_size = static_cast<size_t>(wgpuBufferGetSize(src));
  auto dst_size = static_cast<size_t>(wgpuBufferGetSize(dst));
  ORT_ENFORCE(buffer_size <= src_size && buffer_size <= dst_size,
              "Source and destination buffers must have enough space for the copy operation. src_size=",
              src_size, ", dst_size=", dst_size, ", copy_size=", buffer_size, ".");

  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(src, 0, dst, 0, buffer_size);
}

WGPUBuffer BufferManager::Create(size_t size, wgpu::BufferUsage usage) {
  auto& cache = GetCacheManager(usage);
  auto buffer_size = cache.CalculateBufferSize(size);

  auto buffer = cache.TryAcquireCachedBuffer(buffer_size);
  if (buffer) {
    return buffer;
  }

  // cache miss, create a new buffer
  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = usage;
  buffer = context_.Device().CreateBuffer(&desc).MoveToCHandle();

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", uint64_t(usage), ".");

  cache.RegisterBuffer(buffer, size);
  return buffer;
}

WGPUBuffer BufferManager::CreateUMA(size_t size, wgpu::BufferUsage usage) {
  ORT_ENFORCE(usage & wgpu::BufferUsage::Storage, "UMA buffer must be a storage buffer.");
  auto& cache = GetCacheManager(usage);
  auto buffer_size = cache.CalculateBufferSize(size);

  // Ensure the buffer is mapped for writing at creation.
  usage |= wgpu::BufferUsage::MapWrite;

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = usage;
  desc.mappedAtCreation = true;
  auto buffer = context_.Device().CreateBuffer(&desc).MoveToCHandle();

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", uint64_t(usage), ".");

  cache.RegisterBuffer(buffer, size);
  return buffer;
}

void BufferManager::Release(WGPUBuffer buffer) {
  EnforceBufferUnmapped(context_, buffer);
  GetCacheManager(buffer).ReleaseBuffer(buffer);
}

void BufferManager::Download(WGPUBuffer src, void* dst, size_t size) {
  EnforceBufferUnmapped(context_, src);
  auto buffer_size = NormalizeBufferSize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(src, 0, staging_buffer, 0, buffer_size);
  context_.Flush();

  // TODO: revise wait in whole project

  ORT_ENFORCE(context_.Wait(staging_buffer.MapAsync(wgpu::MapMode::Read, 0, buffer_size, wgpu::CallbackMode::WaitAnyOnly, [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
    ORT_ENFORCE(status == wgpu::MapAsyncStatus::Success, "Failed to download data from buffer: ", std::string_view{message});
  })) == Status::OK());

  auto mapped_data = staging_buffer.GetConstMappedRange();
  memcpy(dst, mapped_data, size);
}

void BufferManager::RefreshPendingBuffers() {
  storage_cache_->OnRefresh();
  uniform_cache_->OnRefresh();
  query_resolve_cache_->OnRefresh();
  default_cache_->OnRefresh();
}

IBufferCacheManager& BufferManager::GetCacheManager(wgpu::BufferUsage usage) const {
  if (usage & wgpu::BufferUsage::Storage) {
    return *storage_cache_;
  } else if (usage & wgpu::BufferUsage::Uniform) {
    return *uniform_cache_;
  } else if (usage & wgpu::BufferUsage::QueryResolve) {
    return *query_resolve_cache_;
  } else {
    return *default_cache_;
  }
}

IBufferCacheManager& BufferManager::GetCacheManager(WGPUBuffer buffer) const {
  auto usage = static_cast<wgpu::BufferUsage>(wgpuBufferGetUsage(buffer));
  return GetCacheManager(usage);
}

std::unique_ptr<BufferManager> BufferManagerFactory::Create(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode) {
  return std::make_unique<BufferManager>(context, storage_buffer_cache_mode, uniform_buffer_cache_mode, query_resolve_buffer_cache_mode);
}

}  // namespace webgpu
}  // namespace onnxruntime
