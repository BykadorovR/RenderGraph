module Sync;
using namespace RenderGraph;

Semaphore::Semaphore(VkSemaphoreType type, const Device& device) : _device(&device) {
  _type = type;

  VkSemaphoreCreateInfo semaphoreInfo{};
  if (type & VK_SEMAPHORE_TYPE_TIMELINE) {
    VkSemaphoreTypeCreateInfo timelineCreateInfo{};
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.pNext = nullptr;
    timelineCreateInfo.semaphoreType = type;
    timelineCreateInfo.initialValue = 0;
    semaphoreInfo.pNext = &timelineCreateInfo;
  }
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphoreInfo.flags = 0;

  if (vkCreateSemaphore(device.getLogicalDevice(), &semaphoreInfo, nullptr, &_semaphore) != VK_SUCCESS)
    throw std::runtime_error("failed to create semaphore!");
}

VkSemaphore Semaphore::getSemaphore() const noexcept { return _semaphore; }

Semaphore::~Semaphore() { vkDestroySemaphore(_device->getLogicalDevice(), _semaphore, nullptr); }