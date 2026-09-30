module;

#include <vector>
#include <volk.h>

export module Sync;

import Device;
import Buffer;

export namespace RenderGraph {
class Semaphore final {
 private:
  const Device* _device;
  VkSemaphore _semaphore;
  VkSemaphoreType _type;

 public:
  Semaphore(VkSemaphoreType type, const Device& device);
  Semaphore(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;
  Semaphore(Semaphore&&) = delete;
  Semaphore& operator=(Semaphore&&) = delete;

  VkSemaphore getSemaphore() const noexcept;
  ~Semaphore();
};

class Barrier final {
 private:
  std::vector<VkBufferMemoryBarrier2> _bufferBarriers;
  std::vector<VkImageMemoryBarrier2> _imageBarriers;

 public:
  void addBuffer(Buffer* buffer,
                 VkPipelineStageFlags2 srcStageMask,
                 VkAccessFlags2 srcAccessMask,
                 VkPipelineStageFlags2 dstStageMask,
                 VkAccessFlags2 dstAccessMask,
                 uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 VkDeviceSize offset = 0,
                 VkDeviceSize size = VK_WHOLE_SIZE);

  void addImage(VkImage image,
                VkPipelineStageFlags2 srcStageMask,
                VkAccessFlags2 srcAccessMask,
                VkPipelineStageFlags2 dstStageMask,
                VkAccessFlags2 dstAccessMask,
                VkImageLayout oldLayout,
                VkImageLayout newLayout,
                VkImageSubresourceRange subresourceRange,
                uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);
  void execute(VkCommandBuffer commandBuffer) const;

  const std::vector<VkBufferMemoryBarrier2>& getBufferBarriers() const noexcept;
  const std::vector<VkImageMemoryBarrier2>& getImageBarriers() const noexcept;
};
}  // namespace RenderGraph
