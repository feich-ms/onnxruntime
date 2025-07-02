// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

#include <chrono>
#include <ctime>
#include <iomanip>

namespace onnxruntime {
namespace webgpu {

namespace {
constexpr const char* MEMORY_METRICS_FILE_OF_OPTIMIZATION_WITH_DYNAMIC_BUCKET = "memory_result_of_buffer_memory_optimization_with_dynamic_bucket.csv";
constexpr const char* MEMORY_METRICS_FILE_OF_NO_OPTIMIZATION = "memory_result_of_no_optimization.csv";
constexpr const char* MEMORY_METRICS_HEADER = "Timestamp,Session,TotalMemory(MB),PeakMemory(MB),ActiveBuffers,TotalBuffers,CacheHit(MB),CacheMiss(MB)\n";
constexpr const char* CACHE_STATS_FILE_OF_OPTIMIZATION_WITH_DYNAMIC_BUCKET = "cache_result_of_buffer_memory_optimization_with_dynamic_bucket.csv";
constexpr const char* CACHE_STATS_FILE_OF_NO_OPTIMIZATION = "cache_result_of_no_optimization.csv";
constexpr const char* CACHE_STATS_HEADER = "Session,BufferSize,Requests,TotalRequestedSize,TotalNormalizedSize,Hits,HitBytes,HitRate,Misses,MissBytes,MissRate\n";

constexpr size_t NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}

void EnforceBufferUnmapped(WebGpuContext& context, WGPUBuffer buffer) {
  if (context.ValidationMode() > ValidationMode::Basic) {
    ORT_ENFORCE(wgpuBufferGetMapState(buffer) == WGPUBufferMapState_Unmapped, "Buffer is still mapped.");
  }
}

}  // namespace

class DisabledCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size, bool is_before_init) override {
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
  size_t CalculateBufferSize(size_t request_size, bool is_before_init) override {
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
  size_t CalculateBufferSize(size_t request_size, bool is_before_init) override {
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

// BucketCacheManagerBase implementation
BucketCacheManagerBase::BucketCacheManagerBase() : session_id_(-1) {}

BucketCacheManagerBase::~BucketCacheManagerBase() {
  if (memory_metrics_file_.is_open()) {
    memory_metrics_file_.close();
  }
  if (cache_stats_file_.is_open()) {
    cache_stats_file_.close();
  }
}

void BucketCacheManagerBase::OpenFiles(const char* memory_metrics_filename, const char* cache_stats_filename) {
  memory_metrics_file_.open(memory_metrics_filename, std::ios::out | std::ios::trunc);
  if (memory_metrics_file_.is_open()) {
    memory_metrics_file_ << MEMORY_METRICS_HEADER;
    memory_metrics_file_.flush();
  }

  cache_stats_file_.open(cache_stats_filename, std::ios::out | std::ios::trunc);
  if (cache_stats_file_.is_open()) {
    cache_stats_file_ << CACHE_STATS_HEADER;
    cache_stats_file_.flush();
  }
}

void BucketCacheManagerBase::LogCacheStats() {
  if (!cache_stats_file_.is_open()) return;

  // Sort buffer sizes for consistent output
  std::vector<size_t> sizes;
  for (const auto& pair : session_stats_) {
    sizes.push_back(pair.first);
  }
  std::sort(sizes.begin(), sizes.end());

  for (size_t size : sizes) {
    const auto& stats = session_stats_[size];
    if (stats.total_requests == 0) continue;

    float hit_rate = static_cast<float>(stats.hits) / stats.total_requests * 100;
    float miss_rate = static_cast<float>(stats.misses) / stats.total_requests * 100;
    cache_stats_file_ << session_id_ << ","
                     << size << ","
                     << stats.total_requests << ","
                     << stats.total_requested_size << ","
                     << stats.total_normalized_size << ","
                     << stats.hits << ","
                     << stats.hit_bytes << ","
                     << std::fixed << std::setprecision(2) << hit_rate << "%,"
                     << stats.misses << ","
                     << stats.miss_bytes << ","
                     << std::fixed << std::setprecision(2) << miss_rate << "%"
                     << std::endl;
  }
  cache_stats_file_.flush();
}

void BucketCacheManagerBase::LogMemoryMetrics() {
  if (!memory_metrics_file_.is_open()) return;

  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_now;
#ifdef _WIN32
  localtime_s(&tm_now, &time_t_now);
#else
  localtime_r(&time_t_now, &tm_now);
#endif
  memory_metrics_file_ << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << ","
                     << session_id_ << ","
                     << std::fixed << std::setprecision(2)
                     << static_cast<double>(total_memory_) / (1024 * 1024) << ","  // Convert to MB
                     << static_cast<double>(peak_memory_) / (1024 * 1024) << ","
                     << active_buffers_ << ","
                     << total_buffers_ << ","
                     << static_cast<double>(total_cache_hit_) / (1024 * 1024) << "," // Convert to MB
                     << static_cast<double>(total_cache_miss_) / (1024 * 1024)  // Convert to MB
                     << std::endl;
  memory_metrics_file_.flush();
}

void BucketCacheManagerBase::UpdateMemoryMetrics(bool is_allocation, size_t buffer_size, bool is_from_destructor, bool skip_active_buffers_update) {
  if (is_allocation) {
    total_memory_ += buffer_size;
    if (!skip_active_buffers_update) {
      active_buffers_++;
    }
    if (buffer_size > 0) {
      total_buffers_++;
    }
    peak_memory_ = std::max(peak_memory_, total_memory_);
  } else {
    total_memory_ -= buffer_size;
    if (!skip_active_buffers_update) {
      if (is_from_destructor) {
        active_buffers_ = std::max(active_buffers_ - 1, 0LL);
      } else {
        active_buffers_--;
      }
    }
    if (buffer_size > 0) {
      total_buffers_--;
    }
  }
  LogMemoryMetrics();
}

// TODO: maybe use different bucket size for storage and uniform buffers?
constexpr std::initializer_list<std::pair<const size_t, size_t>> BUCKET_DEFAULT_LIMIT_TABLE = {
    {64, 250},
    {128, 200},
    {256, 200},
    {512, 200},
    {2048, 230},
    {4096, 200},
    {8192, 50},
    {16384, 50},
    {32768, 50},
    {65536, 50},
    {131072, 50},
    {262144, 50},
    {524288, 50},
    {1048576, 50},
    {2097152, 30},
    {4194304, 20},
    {8388608, 10},
    {12582912, 10},
    {16777216, 10},
    {26214400, 15},
    {33554432, 22},
    {44236800, 2},
    {58982400, 6},
    // we don't want to cache the bucket sizes below but not caching them
    // results in some major performance hits for models like sd-turbo.
    {67108864, 6},
    {134217728, 6},
    {167772160, 6},
};

class BucketCacheManager : public BucketCacheManagerBase {
 public:
  BucketCacheManager() : buckets_limit_{BUCKET_DEFAULT_LIMIT_TABLE} {
    Initialize();
    OpenFiles(MEMORY_METRICS_FILE_OF_NO_OPTIMIZATION, CACHE_STATS_FILE_OF_NO_OPTIMIZATION);
  }
  BucketCacheManager(std::unordered_map<size_t, size_t>&& buckets_limit) : buckets_limit_{buckets_limit} {
    Initialize();
  }

  void OnRunStart() override {
    ++session_id_;
  }

  void OnRunEnd() override {
    // Log cache statistics for the session
    LogCacheStats();

    // Clear session stats for next session
    session_stats_.clear();
  }

  size_t CalculateBufferSize(size_t request_size, bool is_before_init) override {
    size_t normalized_size = NormalizeBufferSize(request_size);

    if (!is_before_init) {
      // binary serch size
      auto it = std::lower_bound(buckets_keys_.begin(), buckets_keys_.end(), request_size);
      if (it != buckets_keys_.end()) {
        normalized_size = *it;
      }
    }

    auto& stats = session_stats_[normalized_size];
    stats.total_requested_size += request_size;
    stats.total_normalized_size += normalized_size;

    return normalized_size;
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    // Update cache statistics
    auto& stats = session_stats_[buffer_size];
    stats.total_requests++;

    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      stats.hits++;
      stats.hit_bytes += buffer_size;
      total_cache_hit_ += buffer_size;
      UpdateMemoryMetrics(true, 0);
      return buffer;
    }

    // Record cache miss
    stats.misses++;
    stats.miss_bytes += buffer_size;
    total_cache_miss_ += buffer_size;
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer buffer/*buffer*/, size_t /*request_size*/) override {
    const auto buffer_size = wgpuBufferGetSize(buffer);
    UpdateMemoryMetrics(true, buffer_size);
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    auto buffer_size = static_cast<size_t>(wgpuBufferGetSize(buffer));
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && it->second.size() < buckets_limit_[buffer_size]) {
      it->second.emplace_back(buffer);
      UpdateMemoryMetrics(false, 0);
    } else {
      UpdateMemoryMetrics(false, buffer_size);
      wgpuBufferRelease(buffer);
    }
  }

  void OnRefresh() override {
    // no-op
  }

  ~BucketCacheManager() {
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        UpdateMemoryMetrics(false, wgpuBufferGetSize(buffer), true);
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  void Initialize() {
    buckets_keys_.reserve(buckets_limit_.size());
    buckets_.reserve(buckets_limit_.size());
    for (const auto& pair : buckets_limit_) {
      buckets_keys_.push_back(pair.first);
      buckets_.emplace(pair.first, std::vector<WGPUBuffer>());
    }
    std::sort(buckets_keys_.begin(), buckets_keys_.end());

#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(std::all_of(buckets_keys_.begin(), buckets_keys_.end(), [](size_t size) { return size % 16 == 0; }),
                "Bucket sizes must be multiples of 16.");

    for (size_t i = 1; i < buckets_keys_.size(); ++i) {
      ORT_ENFORCE(buckets_keys_[i] > buckets_keys_[i - 1], "Bucket sizes must be in increasing order.");
    }
#endif
  }
  std::unordered_map<size_t, size_t> buckets_limit_;
  std::unordered_map<size_t, std::vector<WGPUBuffer>> buckets_;
  std::vector<size_t> buckets_keys_;
};

class DynamicBucketCacheManager : public BucketCacheManagerBase {
public:
  DynamicBucketCacheManager() {
    OpenFiles(MEMORY_METRICS_FILE_OF_OPTIMIZATION_WITH_DYNAMIC_BUCKET, CACHE_STATS_FILE_OF_OPTIMIZATION_WITH_DYNAMIC_BUCKET);
  }

  ~DynamicBucketCacheManager() {
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        UpdateMemoryMetrics(false, wgpuBufferGetSize(buffer), true);
        wgpuBufferRelease(buffer);
      }
    }
  }

  void OnRunStart() override {
    current_run_usage_.clear();
    ++session_id_;
  }

  void OnRunEnd() override {
    // Log cache statistics for the session
    LogCacheStats();

    // Clear session stats for next session
    session_stats_.clear();

    // Update memory patterns based on this session run.
    for (const auto& usage : current_run_usage_) {
      auto& pattern = memory_patterns_[usage.first];
      pattern.request_size = usage.first;
      pattern.frequency = usage.second;
    }

    // Adjust buckets based on the collected memory patterns every 2 runs.
    // The reason for this is to allow the cache to adapt to the memory usage patterns
    // of previous runs of last completed token generation session.
    if ((session_id_ + 1) % 2 == 0) {
      AdjustBuckets();
    }
  }

  size_t CalculateBufferSize(size_t request_size, bool is_before_init) override {
    size_t normalized_request_size = NormalizeBufferSize(request_size);

    // Track usage for the current run
    current_run_usage_[normalized_request_size]++;
    auto& stats = session_stats_[normalized_request_size];
    stats.total_requested_size += request_size;
    stats.total_normalized_size += normalized_request_size;

    // Check if we already have a bucket for this size. If not, create a new bucket so that it can cache buffers of
    // this size in the current session run if the buffer is quickly released in the same session.
    if (buckets_.find(normalized_request_size) == buckets_.end()) {
      buckets_.emplace(normalized_request_size, std::vector<WGPUBuffer>());
      buckets_keys_.push_back(normalized_request_size);
      std::sort(buckets_keys_.begin(), buckets_keys_.end());
    }

    return normalized_request_size;
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto& stats = session_stats_[buffer_size];
    stats.total_requests++;

    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      stats.hits++;
      stats.hit_bytes += buffer_size;
      total_cache_hit_ += buffer_size;
      UpdateMemoryMetrics(true, 0);
      return buffer;
    }

    stats.misses++;
    stats.miss_bytes += buffer_size;
    total_cache_miss_ += buffer_size;
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer buffer, size_t request_size) override {
    const auto buffer_size = wgpuBufferGetSize(buffer);
    UpdateMemoryMetrics(true, buffer_size);
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    auto buffer_size = static_cast<size_t>(wgpuBufferGetSize(buffer));

    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end()) {
      it->second.emplace_back(buffer);
      UpdateMemoryMetrics(false, 0);
    } else {
      UpdateMemoryMetrics(false, buffer_size);
      wgpuBufferRelease(buffer);
    }
  }

  void OnRefresh() override {
    // no-op
  }

  // Analyze memory patterns and adjust bucket sizes.
  void AdjustBuckets() {
    // Store old buckets to handle transitions.
    auto old_buckets = std::move(buckets_);

    // Clear and recreate buckets structure.
    buckets_keys_.clear();
    buckets_.clear();

    // Create new buckets based on patterns.
    for (const auto& pattern : memory_patterns_) {
      // The request size here is already normalized, so we can use it directly as the bucket size key.
      size_t bucket_size = pattern.second.request_size;
      buckets_keys_.push_back(bucket_size);

      // Initialize bucket vector.
      auto& bucket = buckets_[bucket_size];

      auto old_bucket_it = old_buckets.find(bucket_size);
      if (old_bucket_it != old_buckets.end()) {
        // Transfer buffers from old to new bucket.
        bucket = std::move(old_bucket_it->second);
        old_bucket_it->second.clear();
        old_buckets.erase(old_bucket_it);
      }
    }

    // Sort bucket sizes.
    std::sort(buckets_keys_.begin(), buckets_keys_.end());

    // Release any remaining buffers in old buckets that were not hit by the memory usage patterns.
    for (auto& pair : old_buckets) {
      for (auto& buffer : pair.second) {
        UpdateMemoryMetrics(false, wgpuBufferGetSize(buffer), false, true);
        wgpuBufferRelease(buffer);
      }
      pair.second.clear();
    }
    old_buckets.clear();

    // Clear patterns for next adjustment period.
    memory_patterns_.clear();
  }

 private:
  std::unordered_map<size_t, size_t> current_run_usage_;            // Tracks usage in current session run.
  std::unordered_map<size_t, MemoryUsagePattern> memory_patterns_;  // Tracks patterns across session runs.
  std::unordered_map<size_t, std::vector<WGPUBuffer>> buckets_;
  std::vector<size_t> buckets_keys_;
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
    case BufferCacheMode::DynamicBucket:
      return std::make_unique<DynamicBucketCacheManager>();
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
    case BufferCacheMode::DynamicBucket:
      os << "DynamicBucket";
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

WGPUBuffer BufferManager::CreateBeforeSessionInit(size_t size, wgpu::BufferUsage usage) {
  auto& cache = GetCacheManager(usage);
  auto buffer_size = cache.CalculateBufferSize(size, true);

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
  auto buffer_size = cache.CalculateBufferSize(size, true);

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
