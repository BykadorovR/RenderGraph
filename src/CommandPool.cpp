module CommandPool;
using namespace RenderGraph;

CommandPool::CommandPool(vkb::QueueType type, const Device& device) : _device(&device) {
  _type = type;
  VkCommandPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                   .queueFamilyIndex = static_cast<uint32_t>(device.getQueueIndex(type))};

  if (vkCreateCommandPool(device.getLogicalDevice(), &poolInfo, nullptr, &_commandPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics command pool!");
  }
}

vkb::QueueType CommandPool::getType() const noexcept { return _type; }

VkCommandPool CommandPool::getCommandPool() const noexcept { return _commandPool; }

CommandPool::~CommandPool() { vkDestroyCommandPool(_device->getLogicalDevice(), _commandPool, nullptr); }
