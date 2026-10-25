module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <vector>
#include <vk_mem_alloc.h>
#include <volk.h>

export module ResourceUploader;

import Allocator;
import Buffer;
import Command;
import CommandPool;
import Device;
import Sync;
import Texture;

export namespace RenderGraph {
class ResourceUploader final {
 private:
  struct UploadBatch {
    std::unique_ptr<CommandBuffer> _commandBuffer;
    std::vector<std::unique_ptr<Buffer>> _stagingBuffers;
    uint64_t _completionValue = 0;

    UploadBatch(const CommandPool& commandPool, const Device& device);
  };

  const MemoryAllocator* _memoryAllocator;
  const Device* _device;
  CommandPool _commandPool;
  Semaphore _timelineSemaphore;
  std::unique_ptr<UploadBatch> _recordingBatch;
  std::deque<std::unique_ptr<UploadBatch>> _inFlightBatches;
  uint64_t _nextCompletionValue = 1;

  UploadBatch& _getRecordingBatch();
  void _recordBufferBarrier(VkCommandBuffer commandBuffer,
                            VkBuffer buffer,
                            VkDeviceSize offset,
                            VkDeviceSize size,
                            VkPipelineStageFlags2 sourceStageMask,
                            VkAccessFlags2 sourceAccessMask,
                            VkPipelineStageFlags2 destinationStageMask,
                            VkAccessFlags2 destinationAccessMask);

 public:
  // queueType selects the execution queue; Device falls back to GENERAL when a separate queue is unavailable.
  ResourceUploader(const MemoryAllocator& memoryAllocator,
                   const Device& device,
                   QueueType queueType = QueueType::GENERAL);
  ResourceUploader(const ResourceUploader&) = delete;
  ResourceUploader& operator=(const ResourceUploader&) = delete;
  ResourceUploader(ResourceUploader&&) = delete;
  ResourceUploader& operator=(ResourceUploader&&) = delete;

  void upload(Buffer& destination, std::span<const std::byte> data, VkDeviceSize offset = 0);
  // Upload the base level for every layer; mipmaps are generated separately.
  void upload(Image& destination, std::span<const std::byte> data, std::span<const VkDeviceSize> bufferOffsets);
  // The returned ticket covers this batch's uploads on the selected queue.
  GpuTicket submit();
  void reclaim();

  ~ResourceUploader();
};
}  // namespace RenderGraph
