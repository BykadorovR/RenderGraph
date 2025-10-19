module;
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
export module Allocator;
import <vk_mem_alloc.h>;
import Device;
import Instance;
import <volk.h>;


export namespace RenderGraph {
class MemoryAllocator final {
 private:
  VmaAllocator _allocator;

 public:
  MemoryAllocator(const Device& device, const Instance& instance);
  MemoryAllocator(const MemoryAllocator&) = delete;
  MemoryAllocator& operator=(const MemoryAllocator&) = delete;
  MemoryAllocator(MemoryAllocator&&) = delete;
  MemoryAllocator& operator=(MemoryAllocator&&) = delete;
  ~MemoryAllocator();

  VmaAllocator getAllocator() const noexcept;
};
}  // namespace RenderGraph