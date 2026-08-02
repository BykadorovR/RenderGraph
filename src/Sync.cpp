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

void Barrier::addBuffer(Buffer* buffer,
                        VkPipelineStageFlags2 srcStageMask,
                        VkAccessFlags2 srcAccessMask,
                        VkPipelineStageFlags2 dstStageMask,
                        VkAccessFlags2 dstAccessMask,
                        uint32_t srcQueueFamilyIndex,
                        uint32_t dstQueueFamilyIndex,
                        VkDeviceSize offset,
                        VkDeviceSize size) {
  _bufferBarriers.push_back({
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .pNext = nullptr,
      .srcStageMask = srcStageMask,
      .srcAccessMask = srcAccessMask,
      .dstStageMask = dstStageMask,
      .dstAccessMask = dstAccessMask,
      .srcQueueFamilyIndex = srcQueueFamilyIndex,
      .dstQueueFamilyIndex = dstQueueFamilyIndex,
      .buffer = buffer->getBuffer(),
      .offset = offset,
      .size = size,
  });
}

void Barrier::addImage(VkImage image,
                       VkPipelineStageFlags2 srcStageMask,
                       VkAccessFlags2 srcAccessMask,
                       VkPipelineStageFlags2 dstStageMask,
                       VkAccessFlags2 dstAccessMask,
                       VkImageLayout oldLayout,
                       VkImageLayout newLayout,
                       VkImageSubresourceRange subresourceRange,
                       uint32_t srcQueueFamilyIndex,
                       uint32_t dstQueueFamilyIndex) {
  _imageBarriers.push_back({
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .pNext = nullptr,
      .srcStageMask = srcStageMask,
      .srcAccessMask = srcAccessMask,
      .dstStageMask = dstStageMask,
      .dstAccessMask = dstAccessMask,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .srcQueueFamilyIndex = srcQueueFamilyIndex,
      .dstQueueFamilyIndex = dstQueueFamilyIndex,
      .image = image,
      .subresourceRange = subresourceRange,
  });
}

void Barrier::execute(VkCommandBuffer commandBuffer) const {
  if (_bufferBarriers.empty() && _imageBarriers.empty()) {
    return;
  }

  const VkDependencyInfo dependencyInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .pNext = nullptr,
      .dependencyFlags = 0,
      .memoryBarrierCount = 0,
      .pMemoryBarriers = nullptr,
      .bufferMemoryBarrierCount = static_cast<uint32_t>(_bufferBarriers.size()),
      .pBufferMemoryBarriers = _bufferBarriers.data(),
      .imageMemoryBarrierCount = static_cast<uint32_t>(_imageBarriers.size()),
      .pImageMemoryBarriers = _imageBarriers.data(),
  };

  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

const std::vector<VkBufferMemoryBarrier2>& Barrier::getBufferBarriers() const noexcept { return _bufferBarriers; }

const std::vector<VkImageMemoryBarrier2>& Barrier::getImageBarriers() const noexcept { return _imageBarriers; }