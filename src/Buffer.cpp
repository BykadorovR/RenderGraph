module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <stdexcept>
#include <functional>
#include <memory>
#include <vector>
#include <utility>
#include <string>
#include <vk_mem_alloc.h>
#include <volk.h>

module Buffer;

using namespace RenderGraph;

Buffer::Buffer(VkDeviceSize size,
               VkBufferUsageFlags usage,
               VmaAllocationCreateFlags flags,
               const MemoryAllocator& memoryAllocator,
               const Device& device)
    : _memoryAllocator(&memoryAllocator) {
  _size = size;
  const auto& queueFamilies = device.getResourceQueueFamilies();

  VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = size,
                                .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = device.getResourceSharingMode(),
                                .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size()),
                                .pQueueFamilyIndices = queueFamilies.data()};
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

BufferHandler::BufferHandler(std::vector<std::shared_ptr<Buffer>> buffers, std::function<int()> index)
    : _buffers(std::move(buffers)),
      _index(std::move(index)) {}

BufferHandler::BufferHandler(std::vector<std::unique_ptr<Buffer>> buffers, std::function<int()> index)
    : _index(std::move(index)) {
  _buffers.reserve(buffers.size());
  for (auto& buffer : buffers) _buffers.emplace_back(std::move(buffer));
}

Buffer& BufferHandler::getBuffer() const {
  auto* buffer = _buffers.at(getIndex()).get();
  if (!buffer) throw std::logic_error("BufferHandler selected a null buffer");
  return *buffer;
}

std::function<int()> BufferHandler::getIndexFunction() const { return _index; }

int BufferHandler::getIndex() const { return _index(); }

std::vector<Buffer*> BufferHandler::getBuffers() const {
  std::vector<Buffer*> result;
  result.reserve(_buffers.size());
  for (const auto& buffer : _buffers) result.push_back(buffer.get());
  return result;
}
