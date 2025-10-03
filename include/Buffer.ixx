module;
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include "vk_mem_alloc.h"
export module Buffer;
import Allocator;
import Command;
import <span>;
import <memory>;
import <stdexcept>;
import <volk.h>;

export namespace RenderGraph {
export VmaAllocationCreateFlagBits;

class Buffer final {
 private:
  const MemoryAllocator* _memoryAllocator;
  VkBuffer _buffer;
  VmaAllocation _allocation;
  VmaAllocationInfo _allocationInfo;
  VkDeviceSize _size;
  std::unique_ptr<Buffer> _bufferStaging;
  static constexpr VkPipelineStageFlags VK_PIPELINE_STAGE_ALL_SHADER_BITS =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |
      VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
 public:
  Buffer(VkDeviceSize size,
         VkBufferUsageFlags usage,
         VmaAllocationCreateFlags flags,
         const MemoryAllocator& memoryAllocator);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;

  void setData(std::span<const std::byte> data, const CommandBuffer& commandBufferTransfer);
  VkBuffer getBuffer() const noexcept;
  VkDeviceSize getSize() const noexcept;
  const VmaAllocationInfo& getAllocationInfo() const noexcept;
  VmaAllocation getAllocation() const noexcept;
  ~Buffer();
};
}  // namespace RenderGraph