module;

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <volk.h>

module MipmapGenerator;

using namespace RenderGraph;

MipmapGenerator::GenerationBatch::GenerationBatch(const CommandPool& commandPool, const Device& device)
    : _commandBuffer(std::make_unique<CommandBuffer>(commandPool, device)) {}

MipmapGenerator::MipmapGenerator(const Device& device)
    : _device(&device),
      _commandPool(QueueType::GENERAL, device),
      _timelineSemaphore(VK_SEMAPHORE_TYPE_TIMELINE, device) {}

MipmapGenerator::GenerationBatch& MipmapGenerator::_getRecordingBatch() {
  if (_recordingBatch == nullptr) {
    _recordingBatch = std::make_unique<GenerationBatch>(_commandPool, *_device);
    _recordingBatch->_commandBuffer->beginCommands();
  }
  return *_recordingBatch;
}

void MipmapGenerator::generate(Image& image, VkFilter filter) {
  if (image.getMipMapNumber() <= 1) {
    return;
  }
  const auto requiredUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (image.getAspectMask() != VK_IMAGE_ASPECT_COLOR_BIT || (image.getUsageFlags() & requiredUsage) != requiredUsage) {
    throw std::invalid_argument("Image does not support mipmap generation");
  }
  if (filter != VK_FILTER_LINEAR && filter != VK_FILTER_NEAREST) {
    throw std::invalid_argument("Mipmap generation requires linear or nearest filtering");
  }
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(_device->getPhysicalDevice(), image.getFormat(), &properties);
  const auto blitFeatures = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
  if ((properties.optimalTilingFeatures & blitFeatures) != blitFeatures) {
    throw std::invalid_argument("Image format does not support mipmap blits");
  }
  if (filter == VK_FILTER_LINEAR &&
      (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
    throw std::invalid_argument("Image format does not support linear mipmap filtering");
  }

  const VkCommandBuffer commandBuffer = _getRecordingBatch()._commandBuffer->getCommandBuffer();
  VkImageMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .oldLayout = image.getImageLayout(),
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image.getImage(),
      .subresourceRange =
          {
              .aspectMask = image.getAspectMask(),
              .levelCount = static_cast<uint32_t>(image.getMipMapNumber()),
              .baseArrayLayer = 0,
              .layerCount = static_cast<uint32_t>(image.getLayerNumber()),
          },
  };
  const VkDependencyInfo dependencyInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  };

  // Finish earlier reads and writes of the mip chain before overwriting its levels.
  vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
  image.setImageLayout(VK_IMAGE_LAYOUT_GENERAL);

  barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.subresourceRange.levelCount = 1;

  int mipWidth = image.getResolution().x;
  int mipHeight = image.getResolution().y;
  for (uint32_t mipLevel = 1; mipLevel < static_cast<uint32_t>(image.getMipMapNumber()); ++mipLevel) {
    barrier.subresourceRange.baseMipLevel = mipLevel - 1;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    const VkImageBlit blit{
        .srcSubresource =
            {
                .aspectMask = image.getAspectMask(),
                .mipLevel = mipLevel - 1,
                .baseArrayLayer = 0,
                .layerCount = static_cast<uint32_t>(image.getLayerNumber()),
            },
        .srcOffsets = {{0, 0, 0}, {mipWidth, mipHeight, 1}},
        .dstSubresource =
            {
                .aspectMask = image.getAspectMask(),
                .mipLevel = mipLevel,
                .baseArrayLayer = 0,
                .layerCount = static_cast<uint32_t>(image.getLayerNumber()),
            },
        .dstOffsets = {{0, 0, 0}, {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1}},
    };
    vkCmdBlitImage(commandBuffer, image.getImage(), VK_IMAGE_LAYOUT_GENERAL, image.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                   1, &blit, filter);

    mipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
    mipHeight = mipHeight > 1 ? mipHeight / 2 : 1;
  }
}

GpuTicket MipmapGenerator::submit(std::span<const GpuTicket> waits) {
  if (_recordingBatch == nullptr) {
    throw std::logic_error("Can't submit an empty mipmap batch");
  }

  std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
  waitSemaphores.reserve(waits.size());
  for (const auto& ticket : waits) {
    const auto pending = std::ranges::find(waitSemaphores, ticket.semaphore, &VkSemaphoreSubmitInfo::semaphore);
    if (pending == waitSemaphores.end()) {
      waitSemaphores.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                .semaphore = ticket.semaphore,
                                .value = ticket.value,
                                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
    } else {
      pending->value = std::max(pending->value, ticket.value);
    }
  }

  const VkQueue queue = _device->getQueue(QueueType::GENERAL);
  _inFlightBatches.push_back(std::move(_recordingBatch));
  GenerationBatch& batch = *_inFlightBatches.back();
  batch._commandBuffer->endCommands();
  const VkCommandBufferSubmitInfo commandBufferInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = batch._commandBuffer->getCommandBuffer(),
  };
  const VkSemaphoreSubmitInfo signalSemaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = _timelineSemaphore.getSemaphore(),
      .value = _nextCompletionValue,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkSubmitInfo2 submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = static_cast<uint32_t>(waitSemaphores.size()),
      .pWaitSemaphoreInfos = waitSemaphores.data(),
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &commandBufferInfo,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signalSemaphoreInfo,
  };
  const VkResult result = vkQueueSubmit2(queue, 1, &submitInfo, nullptr);
  if (result != VK_SUCCESS) {
    _inFlightBatches.pop_back();
    throw std::runtime_error("Mipmap generation submission failed: " + std::to_string(result));
  }
  batch._completionValue = _nextCompletionValue++;
  return {.semaphore = _timelineSemaphore.getSemaphore(), .value = batch._completionValue};
}

void MipmapGenerator::reclaim() {
  if (_inFlightBatches.empty()) {
    return;
  }

  uint64_t completedValue = 0;
  const VkResult result = vkGetSemaphoreCounterValue(_device->getLogicalDevice(), _timelineSemaphore.getSemaphore(),
                                                     &completedValue);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Can't query mipmap generation completion: " + std::to_string(result));
  }
  while (!_inFlightBatches.empty() && _inFlightBatches.front()->_completionValue <= completedValue) {
    _inFlightBatches.pop_front();
  }
}

MipmapGenerator::~MipmapGenerator() {
  if (!_inFlightBatches.empty()) {
    const uint64_t completionValue = _inFlightBatches.back()->_completionValue;
    const VkSemaphore semaphore = _timelineSemaphore.getSemaphore();
    const VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &semaphore,
        .pValues = &completionValue,
    };
    vkWaitSemaphores(_device->getLogicalDevice(), &waitInfo, std::numeric_limits<uint64_t>::max());
  }
}
