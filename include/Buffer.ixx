module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>
#include <functional>
#include <memory>
#include <vector>

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
         const MemoryAllocator& memoryAllocator,
         const Device& device);
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

class BufferHandler final {
 private:
  std::vector<std::shared_ptr<Buffer>> _buffers;
  std::function<int()> _index;

 public:
  // Index callbacks must support concurrent calls and remain stable throughout Graph::render().
  BufferHandler(std::vector<std::shared_ptr<Buffer>> buffers, std::function<int()> index);
  BufferHandler(std::vector<std::unique_ptr<Buffer>> buffers, std::function<int()> index);
  BufferHandler(const BufferHandler&) = delete;
  BufferHandler& operator=(const BufferHandler&) = delete;
  Buffer& getBuffer() const;
  std::function<int()> getIndexFunction() const;
  int getIndex() const;
  std::vector<Buffer*> getBuffers() const;
};
}  // namespace RenderGraph
