module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>

export module Buffer;

import Allocator;
import Device;

export namespace RenderGraph {
class Buffer final {
 private:
  const MemoryAllocator* _memoryAllocator;
  VkBuffer _buffer;
  VmaAllocation _allocation;
  VmaAllocationInfo _allocationInfo;
  VkDeviceSize _size;

 public:
  Buffer(VkDeviceSize size,
         VkBufferUsageFlags usage,
         VmaAllocationCreateFlags flags,
         const MemoryAllocator& memoryAllocator);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;

  VkBuffer getBuffer() const noexcept;
  VkDeviceSize getSize() const noexcept;
  const VmaAllocationInfo& getAllocationInfo() const noexcept;
  VmaAllocation getAllocation() const noexcept;
  VkDeviceAddress getDeviceAddress(const Device& device) const noexcept;
  ~Buffer();
};
}  // namespace RenderGraph
