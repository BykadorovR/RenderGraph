module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <stdexcept>
#include <string>
#include <vk_mem_alloc.h>
#include <volk.h>

module Buffer;

using namespace RenderGraph;

Buffer::Buffer(VkDeviceSize size,
               VkBufferUsageFlags usage,
               VmaAllocationCreateFlags flags,
               const MemoryAllocator& memoryAllocator)
    : _memoryAllocator(&memoryAllocator) {
  _size = size;

  VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = size,
                                .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VmaAllocationCreateInfo allocCreateInfo = {.flags = flags, .usage = VMA_MEMORY_USAGE_AUTO};

  auto result = vmaCreateBuffer(_memoryAllocator->getAllocator(), &bufferInfo, &allocCreateInfo, &_buffer, &_allocation,
                                &_allocationInfo);
  if (result != VK_SUCCESS) throw std::runtime_error("Can't vmaCreateBuffer " + std::to_string(result));
}

VkDeviceSize Buffer::getSize() const noexcept { return _size; }

const VmaAllocationInfo& Buffer::getAllocationInfo() const noexcept { return _allocationInfo; }

VmaAllocation Buffer::getAllocation() const noexcept { return _allocation; }

VkDeviceAddress Buffer::getDeviceAddress(const Device& device) const noexcept {
  VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, _buffer};
  return vkGetBufferDeviceAddress(device.getLogicalDevice(), &addrInfo);
}

VkBuffer Buffer::getBuffer() const noexcept { return _buffer; }

Buffer::~Buffer() { vmaDestroyBuffer(_memoryAllocator->getAllocator(), _buffer, _allocation); }
