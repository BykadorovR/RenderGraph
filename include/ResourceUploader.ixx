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
struct UploadTicket {
  VkSemaphore semaphore = nullptr;
  uint64_t value = 0;
};

class ResourceUploader final {
 private:
  struct UploadBatch {
    std::unique_ptr<CommandBuffer> _commandBuffer;
    std::vector<std::unique_ptr<Buffer>> _stagingBuffers;
    uint64_t _completionValue = 0;
  };

  const MemoryAllocator* _memoryAllocator;
  const Device* _device;
  CommandPool _commandPool;
  Semaphore _timelineSemaphore;
  std::unique_ptr<UploadBatch> _recordingBatch;
  std::deque<std::unique_ptr<UploadBatch>> _inFlightBatches;
  uint64_t _nextCompletionValue = 1;

  UploadBatch& _getRecordingBatch();
  void _generateMipmaps(Image& image, VkCommandBuffer commandBuffer);
  void _recordBufferBarrier(VkCommandBuffer commandBuffer,
                            VkBuffer buffer,
                            VkDeviceSize offset,
                            VkDeviceSize size,
                            VkPipelineStageFlags2 sourceStageMask,
                            VkAccessFlags2 sourceAccessMask,
                            VkPipelineStageFlags2 destinationStageMask,
                            VkAccessFlags2 destinationAccessMask);

 public:
  ResourceUploader(const MemoryAllocator& memoryAllocator, const Device& device);
  ResourceUploader(const ResourceUploader&) = delete;
  ResourceUploader& operator=(const ResourceUploader&) = delete;
  ResourceUploader(ResourceUploader&&) = delete;
  ResourceUploader& operator=(ResourceUploader&&) = delete;

  void upload(Buffer& destination, std::span<const std::byte> data, VkDeviceSize offset = 0);
  void upload(Image& destination, std::span<const std::byte> data, std::span<const VkDeviceSize> bufferOffsets);
  UploadTicket submit();
  void reclaim();

  ~ResourceUploader();
};
}  // namespace RenderGraph
