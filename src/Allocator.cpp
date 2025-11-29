module;
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
// it's mandatory to include here for VMA_IMPLEMENTATION, otherwise if we compile Allocator after some other module
// it will be cached without VMA_IMPLEMENTATION defined and cause linker errors.
#include <vk_mem_alloc.h>
module Allocator;
using namespace RenderGraph;

MemoryAllocator::MemoryAllocator(const Device& device, const Instance& instance) {
  const VmaVulkanFunctions vmaVulkanFunctions = {
      vkGetInstanceProcAddr, vkGetDeviceProcAddr, vkGetPhysicalDeviceProperties, vkGetPhysicalDeviceMemoryProperties,
      vkAllocateMemory, vkFreeMemory, vkMapMemory, vkUnmapMemory, vkFlushMappedMemoryRanges,
      vkInvalidateMappedMemoryRanges, vkBindBufferMemory, vkBindImageMemory, vkGetBufferMemoryRequirements,
      vkGetImageMemoryRequirements, vkCreateBuffer, vkDestroyBuffer, vkCreateImage, vkDestroyImage, vkCmdCopyBuffer,
      /// Fetch "vkGetBufferMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetBufferMemoryRequirements2KHR" when using
      /// VK_KHR_dedicated_allocation extension.
      vkGetBufferMemoryRequirements2KHR,
      /// Fetch "vkGetImageMemoryRequirements 2" on Vulkan >= 1.1, fetch "vkGetImageMemoryRequirements2KHR" when using
      /// VK_KHR_dedicated_allocation extension.
      vkGetImageMemoryRequirements2KHR,
      vkBindBufferMemory2KHR,
      /// Fetch "vkBindImageMemory2" on Vulkan >= 1.1, fetch "vkBindImageMemory2KHR" when using VK_KHR_bind_memory2
      /// extension.
      vkBindImageMemory2KHR,
      vkGetPhysicalDeviceMemoryProperties2KHR
  };
  VmaAllocatorCreateInfo createInfo{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = device.getPhysicalDevice(),
      .device = device.getLogicalDevice(),
      .pVulkanFunctions = &vmaVulkanFunctions,
      .instance = instance.getInstance().instance,
  };

  if (vmaCreateAllocator(&createInfo, &_allocator) != VK_SUCCESS) {
    throw std::runtime_error("Can't create vma allocator");
  }
}

VmaAllocator MemoryAllocator::getAllocator() const noexcept { return _allocator; }

MemoryAllocator::~MemoryAllocator() { vmaDestroyAllocator(_allocator); }