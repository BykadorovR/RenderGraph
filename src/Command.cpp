module;

#include <stdexcept>
#include <utility>
#include <volk.h>

module Command;

using namespace RenderGraph;

CommandBuffer::CommandBuffer(const CommandPool& pool, const Device& device) : _pool(&pool), _device(&device) {
  VkCommandBufferAllocateInfo allocInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = pool.getCommandPool(),
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = 1};

  if (vkAllocateCommandBuffers(device.getLogicalDevice(), &allocInfo, &_buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

CommandBuffer::CommandBuffer(CommandBuffer&& commandBuffer) noexcept
    : _buffer(std::exchange(commandBuffer._buffer, nullptr)),
      _pool(std::exchange(commandBuffer._pool, nullptr)),
      _device(std::exchange(commandBuffer._device, nullptr)),
      _active(std::exchange(commandBuffer._active, false)) {
  ;
}

void CommandBuffer::beginCommands() noexcept {
  VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

  vkBeginCommandBuffer(_buffer, &beginInfo);
  _active = true;
}

void CommandBuffer::endCommands() noexcept {
  vkEndCommandBuffer(_buffer);
  _active = false;
}

bool CommandBuffer::getActive() const noexcept { return _active; }

const VkCommandBuffer& CommandBuffer::getCommandBuffer() const noexcept { return _buffer; }

CommandBuffer::~CommandBuffer() {
  _active = false;
  if (_device && _pool && _buffer)
    vkFreeCommandBuffers(_device->getLogicalDevice(), _pool->getCommandPool(), 1, &_buffer);
}
